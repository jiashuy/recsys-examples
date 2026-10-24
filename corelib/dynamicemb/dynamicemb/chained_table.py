# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A key index with value rows apart from the slots: one table.

A key is looked up by chain inside its bucket and carries the number of the row its
value lives in, taken from a pool of rows that has nothing to do with where the key
sits. A step is two calls with whatever the caller needs to decide in between:

    rows = table.find_and_stage(keys)      # hits answered; misses held back
    ...                                    # admission, a lower tier, ...
    out = table.commit(keep)               # misses placed; room made where needed
    ...                                    # read the victims' rows, then
    table.release_rows(out.victim_rows)

See docs/chained_bucket_index_design.md. This is the first stage: one table, the
single-word score, no pinning of its own (the cache's eviction guard is the only
protection), and a bucket that cannot make room is an error.
"""

from typing import NamedTuple, Optional, Tuple

import torch
from dynamicemb_extensions import chained_bucket_ids, chained_find, chained_place

# How find treats the score of a key it finds.
SCORE_NONE = 0  # leave it alone
SCORE_ASSIGN = 1  # set it to the given score
SCORE_ACCUMULATE = 2  # add the given score (1 when none is given)

_POLICIES = {"none": SCORE_NONE, "assign": SCORE_ASSIGN, "accumulate": SCORE_ACCUMULATE}


class CommitResult(NamedTuple):
    # Row of each placed key, in the order the staged keys were given.
    rows: torch.Tensor
    # Position in the keys passed to find_and_stage of each placed key.
    positions: torch.Tensor
    # What was evicted to make room. The rows still hold the victims' values;
    # give them back with release_rows once they have been read.
    victim_keys: torch.Tensor
    victim_rows: torch.Tensor


class ChainedTableFull(RuntimeError):
    """A bucket could not make room: every member it could evict is protected.

    ``partial`` is the CommitResult of the buckets that did place their keys; the rows
    of the keys that were not placed have already been given back.
    """

    def __init__(self, message: str, partial: CommitResult):
        super().__init__(message)
        self.partial = partial


def _as_int64(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.int64:
        return t
    if t.dtype == torch.uint64:
        return t.view(torch.int64)
    raise TypeError(f"keys must be int64 or uint64, got {t.dtype}")


class ChainedTable:
    def __init__(
        self,
        num_buckets: int,
        bucket_capacity: int = 128,
        row_capacity: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        if not 1 <= bucket_capacity <= 256:
            raise ValueError("bucket_capacity must be in [1, 256]")
        if num_buckets < 1:
            raise ValueError("num_buckets must be positive")
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.num_buckets = int(num_buckets)
        self.bucket_capacity = int(bucket_capacity)
        slots = self.num_buckets * self.bucket_capacity
        if row_capacity is None:
            # The rows of the keys of a step that the table has not yet made room
            # for, plus some: a row is held from when a key is placed until its
            # victim is released.
            row_capacity = slots + max(slots // 8, 1024)
        if row_capacity < slots:
            raise ValueError("row_capacity must cover every slot")
        if row_capacity >= 1 << 32:
            raise ValueError("row_capacity must fit in 32 bits")
        self.row_capacity = int(row_capacity)

        dev = self.device
        self.keys = torch.zeros(slots, dtype=torch.int64, device=dev)
        self.scores = torch.zeros(slots, dtype=torch.int64, device=dev)
        self.index = torch.zeros(slots, dtype=torch.int32, device=dev)
        self.next = torch.full((slots,), -1, dtype=torch.int16, device=dev)
        self.head = torch.full((slots,), -1, dtype=torch.int16, device=dev)
        self.bucket_size = torch.zeros(self.num_buckets, dtype=torch.int32, device=dev)

        # The pool of free rows is a stack; _top is how many are on it.
        self._free = torch.arange(self.row_capacity, dtype=torch.int32, device=dev)
        self._top = self.row_capacity

        self._staged_keys: Optional[torch.Tensor] = None
        self._staged_scores: Optional[torch.Tensor] = None
        self._staged_positions: Optional[torch.Tensor] = None

    # -- queries --------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self.num_buckets * self.bucket_capacity

    def size(self) -> int:
        return int(self.bucket_size.sum().item())

    def free_rows(self) -> int:
        return self._top

    def bucket_of(self, keys: torch.Tensor) -> torch.Tensor:
        return chained_bucket_ids(_as_int64(keys), self.num_buckets)

    def _find(self, keys, scores, mode) -> torch.Tensor:
        return chained_find(
            _as_int64(keys),
            self.keys,
            self.scores,
            self.index,
            self.next,
            self.head,
            self.bucket_size,
            self.num_buckets,
            self.bucket_capacity,
            None if scores is None else _as_int64(scores),
            mode,
        )

    def lookup(self, keys: torch.Tensor) -> torch.Tensor:
        """Row of each key, -1 for a key the table does not have. Changes nothing."""
        return self._find(keys, None, SCORE_NONE)

    # -- one step -------------------------------------------------------------

    def find_and_stage(
        self,
        keys: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        policy: str = "assign",
    ) -> torch.Tensor:
        """Answer the keys the table has and hold back the ones it does not.

        ``keys`` must be unique. Returns each key's row, -1 for a key the table does
        not have; those are staged, with ``scores`` (or 0, or 1 for ``accumulate``)
        as the score they will enter with. A hit's score is updated by ``policy``.
        The table is otherwise untouched, so nothing needs undoing if the step is
        abandoned before :meth:`commit`.
        """
        mode = _POLICIES[policy]
        keys = _as_int64(keys)
        rows = self._find(keys, scores, mode)
        miss = (rows < 0).nonzero(as_tuple=True)[0]
        self._staged_positions = miss
        self._staged_keys = keys[miss]
        if scores is None:
            fill = 1 if mode == SCORE_ACCUMULATE else 0
            self._staged_scores = torch.full(
                (miss.numel(),), fill, dtype=torch.int64, device=self.device
            )
        else:
            self._staged_scores = _as_int64(scores)[miss]
        return rows

    @property
    def num_staged(self) -> int:
        return 0 if self._staged_keys is None else int(self._staged_keys.numel())

    def commit(
        self,
        keep: Optional[torch.Tensor] = None,
        guard: Optional[Tuple[int, int]] = None,
        guard_table: int = 0,
    ) -> CommitResult:
        """Place the staged keys.

        ``keep`` is a bool mask over the staged keys (``None`` keeps them all); the
        others are dropped, having never been in the table. A kept key takes a free
        slot of its bucket if there is one; otherwise the bucket's lowest-scored
        member that ``guard`` allows is evicted and the key takes its slot. All the
        keys of a bucket choose their victims together.
        """
        empty = torch.empty(0, dtype=torch.int64, device=self.device)
        if self._staged_keys is None or self._staged_keys.numel() == 0:
            self._clear_staged()
            return CommitResult(empty, empty, empty, empty)
        keys, scores, positions = (
            self._staged_keys,
            self._staged_scores,
            self._staged_positions,
        )
        if keep is not None:
            keys, scores, positions = keys[keep], scores[keep], positions[keep]
        m = int(keys.numel())
        if m == 0:
            self._clear_staged()
            return CommitResult(empty, empty, empty, empty)
        if m > self._top:
            raise RuntimeError(
                f"the pool has {self._top} free rows and {m} keys need one; "
                "increase row_capacity or release the victims of earlier steps"
            )

        rows = self._free[self._top - m : self._top].to(torch.int64)
        self._top -= m

        bucket = self.bucket_of(keys)
        perm = torch.argsort(bucket, stable=True)
        sb = bucket[perm]
        skeys, sscores, srows = keys[perm], scores[perm], rows[perm]
        seg_bucket, counts = torch.unique_consecutive(sb, return_counts=True)
        u = int(seg_bucket.numel())
        seg_begin = torch.cumsum(counts, 0) - counts
        free_slots = self.bucket_capacity - self.bucket_size[seg_bucket].to(torch.int64)
        need = torch.clamp(counts - free_slots, min=0)
        seg_voff = torch.cumsum(need, 0) - need
        total_victims = int(need.sum().item())

        vict_slot = torch.empty(total_victims, dtype=torch.int32, device=self.device)
        vkeys = torch.empty(total_victims, dtype=torch.int64, device=self.device)
        vrows = torch.empty(total_victims, dtype=torch.int64, device=self.device)
        seg_fail = torch.zeros(u, dtype=torch.int32, device=self.device)
        g_ptr, g_n = guard if guard is not None else (0, 0)
        chained_place(
            self.keys,
            self.scores,
            self.index,
            self.next,
            self.head,
            self.bucket_size,
            self.num_buckets,
            self.bucket_capacity,
            skeys,
            sscores,
            srows,
            seg_bucket,
            seg_begin,
            counts,
            seg_voff,
            vict_slot,
            vkeys,
            vrows,
            seg_fail,
            g_ptr,
            g_n,
            guard_table,
        )
        self._clear_staged()

        failed = seg_fail.to(torch.bool)
        if bool(failed.any()):
            seg_of_key = torch.repeat_interleave(
                torch.arange(u, device=self.device), counts
            )
            unplaced = failed[seg_of_key]
            self.release_rows(srows[unplaced])
            ok_victim = torch.repeat_interleave(~failed, need)
            partial = CommitResult(
                srows[~unplaced],
                positions[perm][~unplaced],
                vkeys[ok_victim],
                vrows[ok_victim],
            )
            raise ChainedTableFull(
                f"{int(failed.sum())} bucket(s) could not make room for "
                f"{int(unplaced.sum())} key(s): every member they could evict is "
                "in use by a step in flight. Increase the capacity or reduce the "
                "prefetch pipeline depth.",
                partial,
            )
        return CommitResult(rows, positions, vkeys, vrows)

    def _clear_staged(self) -> None:
        self._staged_keys = None
        self._staged_scores = None
        self._staged_positions = None

    def release_rows(self, rows: torch.Tensor) -> None:
        """Give rows back to the pool, once nothing needs what they hold."""
        r = int(rows.numel())
        if r == 0:
            return
        if self._top + r > self.row_capacity:
            raise RuntimeError("more rows released than were taken")
        self._free[self._top : self._top + r] = rows.to(torch.int32)
        self._top += r

    # -- inspection -----------------------------------------------------------

    def export(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(keys, scores, rows) of everything stored, bucket by bucket."""
        cap = self.bucket_capacity
        occupied = torch.arange(cap, device=self.device)[None, :] < self.bucket_size[
            :, None
        ]
        mask = occupied.reshape(-1)
        return (
            self.keys[mask],
            self.scores[mask],
            self.index[mask].to(torch.int64) & 0xFFFFFFFF,
        )

    def check_invariants(self, pending_rows: int = 0) -> None:
        """Walk every chain and check the structure; for tests, it is slow.

        ``pending_rows`` is the number of rows taken from the pool that are not on a
        slot: victims not yet released.
        """
        cap = self.bucket_capacity
        keys = self.keys.cpu().tolist()
        index = self.index.cpu().tolist()
        nxt = (self.next.cpu().to(torch.int32) & 0xFFFF).tolist()
        head = (self.head.cpu().to(torch.int32) & 0xFFFF).tolist()
        sizes = self.bucket_size.cpu().tolist()
        seen_keys = set()
        seen_rows = set()
        for b in range(self.num_buckets):
            size = sizes[b]
            assert 0 <= size <= cap, f"bucket {b}: size {size}"
            visited = []
            for c in range(cap):
                cur = head[b * cap + c]
                steps = 0
                while cur != 0xFFFF:
                    assert cur < size, f"bucket {b}: chain {c} reaches slot {cur}"
                    key = keys[b * cap + cur]
                    h = hash_py(key)
                    assert h % self.num_buckets == b, f"key {key} in the wrong bucket"
                    assert (h >> 32) % cap == c, f"key {key} on the wrong chain"
                    visited.append(cur)
                    cur = nxt[b * cap + cur]
                    steps += 1
                    assert steps <= cap, f"bucket {b}: chain {c} does not end"
            assert sorted(visited) == list(range(size)), f"bucket {b}: chains miss slots"
            for s in range(size):
                key = keys[b * cap + s]
                assert key not in seen_keys, f"key {key} stored twice"
                seen_keys.add(key)
                row = index[b * cap + s] & 0xFFFFFFFF
                assert row < self.row_capacity, f"row {row} out of range"
                assert row not in seen_rows, f"row {row} used twice"
                seen_rows.add(row)
        assert (
            self._top + len(seen_rows) + pending_rows == self.row_capacity
        ), "rows are lost or duplicated"


def hash_py(key: int) -> int:
    """The table's hash of a key (fmix64, unsigned), for tests."""
    mask = (1 << 64) - 1
    v = key & mask
    v ^= v >> 33
    v = (v * 0xFF51AFD7ED558CCD) & mask
    v ^= v >> 33
    v = (v * 0xC4CEB9FE1A85EC53) & mask
    v ^= v >> 33
    return v
