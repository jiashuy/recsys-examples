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

"""A key index with value rows apart from the slots.

A key is looked up by chain inside its bucket and carries the number of the row its
value lives in, taken from a pool of rows that has nothing to do with where the key
sits. A step is two calls with whatever the caller needs to decide in between:

    rows = tables.find_and_stage(keys, table_ids)   # hits answered; misses held back
    ...                                             # admission, a lower tier, ...
    out = tables.commit(keep)                       # misses placed; room made
    ...                                             # read the victims' rows, then
    tables.release_rows(out.victim_rows, out.victim_tables)

Each table has its own arrays and its own pool of rows; a table can be grown without
touching the others, and growing moves only the keys, never the values. A table that
never evicts (``no_eviction``) grows when a bucket cannot hold its keys. A table that
pins (``track_refs``) keeps a count per row, and a row with a count is never chosen as
a victim; the cache's eviction guard is honoured as well.

See docs/chained_bucket_index_design.md.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from dynamicemb_extensions import (
    chained_apply,
    chained_bucket_ids,
    chained_erase,
    chained_find,
    chained_make_views,
    chained_pin,
    chained_place,
    chained_plan,
    chained_release,
    chained_segment_need,
    chained_stage,
    device_timestamp,
)

# How find treats the score of a key it finds.
SCORE_NONE = 0  # leave it alone
SCORE_ASSIGN = 1  # set it to the given score
SCORE_ACCUMULATE = 2  # add the given score (1 when none is given)
SCORE_LRU_LFU = 3  # (timestamp, frequency): the timestamp is now, the frequency grows

_POLICIES = {
    "none": SCORE_NONE,
    "assign": SCORE_ASSIGN,
    "accumulate": SCORE_ACCUMULATE,
    "lru_lfu": SCORE_LRU_LFU,
}


class CommitResult(NamedTuple):
    # Each placed key: its row, its table, and its position in the keys passed to
    # find_and_stage, in the order the keys were staged.
    rows: torch.Tensor
    positions: torch.Tensor
    tables: torch.Tensor
    # What was evicted to make room. The rows still hold the victims' values;
    # give them back with release_rows once they have been read.
    victim_keys: torch.Tensor
    victim_rows: torch.Tensor
    victim_tables: torch.Tensor
    # The victims' score words, ``[n, num_scores]``: what a tier below can keep them at.
    victim_scores: torch.Tensor


class ChainedTableFull(RuntimeError):
    """A bucket could not make room: every member it could evict is protected, or
    it was given more new keys than it has slots.

    ``partial`` is the CommitResult of the buckets that did place their keys; the rows
    of the keys that were not placed have already been given back.
    """

    def __init__(self, message: str, partial: Optional[CommitResult]):
        super().__init__(message)
        self.partial = partial


def _as_int64(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.int64:
        return t
    if t.dtype == torch.uint64:
        return t.view(torch.int64)
    raise TypeError(f"keys must be int64 or uint64, got {t.dtype}")


def _default_rows(slots: int) -> int:
    # The rows of the keys of a step that the table has not yet made room for, plus
    # some: a row is held from when a key is placed until its victim is released.
    return slots + max(slots // 8, 1024)


class ChainedTables:
    def __init__(
        self,
        num_buckets: Sequence[int],
        bucket_capacity: int = 128,
        row_capacities: Optional[Sequence[int]] = None,
        track_refs: bool = False,
        no_eviction: Union[bool, Sequence[bool]] = False,
        device: Optional[torch.device] = None,
        num_scores: int = 1,
    ):
        if num_scores not in (1, 2):
            raise ValueError("a slot holds one score word, or two: (timestamp, frequency)")
        self.num_scores = int(num_scores)
        self.num_tables = len(num_buckets)
        if self.num_tables < 1 or min(num_buckets) < 1:
            raise ValueError("every table needs at least one bucket")
        self.bucket_capacity = int(bucket_capacity)
        if self.bucket_capacity < 1:
            raise ValueError("bucket_capacity must be positive")
        # A link is an offset in a bucket, and the offsets below 4 * capacity have to
        # fit it with the value that means none: 2 bytes unless a bucket is large.
        if 4 * self.bucket_capacity < 0xFFFF:
            self._link_dtype = torch.int16
        elif 4 * self.bucket_capacity < 0xFFFFFFFF:
            self._link_dtype = torch.int32
        else:
            raise ValueError("a bucket takes at most 2^30 - 1 slots")
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.track_refs = bool(track_refs)
        if isinstance(no_eviction, bool):
            no_eviction = [no_eviction] * self.num_tables
        self.no_eviction = [bool(x) for x in no_eviction]
        assert len(self.no_eviction) == self.num_tables

        self._nb = [int(n) for n in num_buckets]
        rows = (
            [_default_rows(n * self.bucket_capacity) for n in self._nb]
            if row_capacities is None
            else [int(r) for r in row_capacities]
        )
        self._row_cap = rows
        dev = self.device
        self._keys: List[torch.Tensor] = []
        self._scores: List[torch.Tensor] = []
        self._index: List[torch.Tensor] = []
        self._next: List[torch.Tensor] = []
        self._head: List[torch.Tensor] = []
        self._bsize: List[torch.Tensor] = []
        self._refs: List[Optional[torch.Tensor]] = []
        self._free: List[torch.Tensor] = []
        for t, nb in enumerate(self._nb):
            slots = nb * self.bucket_capacity
            if rows[t] < slots:
                raise ValueError("row_capacity must cover every slot")
            if rows[t] >= 1 << 31:
                raise ValueError("row_capacity must fit in 31 bits")
            self._append_arrays(nb)
            self._refs.append(
                torch.zeros(rows[t], dtype=torch.int32, device=dev)
                if self.track_refs
                else None
            )
            # The pool of free rows is a stack; _tops[t] is how many are on it.
            self._free.append(torch.arange(rows[t], dtype=torch.int32, device=dev))
        self._tops = torch.tensor(rows, dtype=torch.int32, device=dev)
        self._version = [0] * self.num_tables
        self._rebuild_views()

        self._staged_keys: Optional[torch.Tensor] = None
        self._staged_tids: Optional[torch.Tensor] = None
        self._staged_scores: Optional[torch.Tensor] = None
        self._staged_positions: Optional[torch.Tensor] = None

    # -- layout ---------------------------------------------------------------

    def _new_arrays(self, nb: int):
        slots = nb * self.bucket_capacity
        dev = self.device
        return (
            torch.zeros(slots, dtype=torch.int64, device=dev),
            torch.zeros(slots * self.num_scores, dtype=torch.int64, device=dev),
            torch.zeros(slots, dtype=torch.int32, device=dev),
            torch.full((slots,), -1, dtype=self._link_dtype, device=dev),
            torch.full((slots,), -1, dtype=self._link_dtype, device=dev),
            torch.zeros(nb, dtype=torch.int32, device=dev),
        )

    def _append_arrays(self, nb: int) -> None:
        keys, scores, index, nxt, head, bsize = self._new_arrays(nb)
        self._keys.append(keys)
        self._scores.append(scores)
        self._index.append(index)
        self._next.append(nxt)
        self._head.append(head)
        self._bsize.append(bsize)

    def _rebuild_views(self) -> None:
        self._views = chained_make_views(
            self._keys,
            self._scores,
            self._index,
            self._next,
            self._head,
            self._bsize,
            self._refs,
            self._free,
            self._tops,
            self._nb,
            self.bucket_capacity,
            self.num_scores,
        )

    def version(self, table: int = 0) -> int:
        """Changes whenever a key of the table moves or goes: what a map from rows to
        slots, built once, is good for."""
        return self._version[table]

    # -- queries --------------------------------------------------------------

    def num_buckets_of(self, table: int) -> int:
        return self._nb[table]

    def row_capacity_of(self, table: int) -> int:
        """Rows the table's value pool has to have."""
        return self._row_cap[table]

    def capacity_of(self, table: int) -> int:
        return self._nb[table] * self.bucket_capacity

    def size(self, table: int = 0) -> int:
        return int(self._bsize[table].sum().item())

    def free_rows(self, table: int = 0) -> int:
        return int(self._tops[table].item())

    def touch(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        policy: str = "assign",
        timestamp: Optional[int] = None,
    ) -> torch.Tensor:
        """Row of each key (-1 if absent), updating the score of the ones that are there
        as :meth:`find_and_stage` does; nothing is staged."""
        mode = _POLICIES[policy]
        ts = int(device_timestamp() if timestamp is None else timestamp) if mode == SCORE_LRU_LFU else 0
        sc = None if scores is None else _as_int64(scores)
        return chained_find(self._views, _as_int64(keys), table_ids, sc, mode, False, ts)

    def lookup(self, keys: torch.Tensor, table_ids: torch.Tensor) -> torch.Tensor:
        """Row of each key, -1 for a key the table does not have. Changes nothing."""
        return chained_find(self._views, _as_int64(keys), table_ids, None, SCORE_NONE, False)

    # -- one step -------------------------------------------------------------

    def find_and_stage(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        policy: str = "assign",
        pin: bool = False,
        timestamp: Optional[int] = None,
    ) -> torch.Tensor:
        """Answer the keys the tables have and hold back the ones they do not.

        ``keys`` must be unique within a table. Returns each key's row, -1 for a key
        its table does not have; those are staged, with ``scores`` (or 0, or 1 for
        ``accumulate``) as the score they will enter with. A hit's score is updated by
        ``policy``, and with ``pin`` its row takes a reference. The tables are
        otherwise untouched, so nothing needs undoing if the step is abandoned before
        :meth:`commit`.
        """
        mode = _POLICIES[policy]
        keys = _as_int64(keys)
        sc = None if scores is None else _as_int64(scores)
        if mode == SCORE_LRU_LFU and self.num_scores != 2:
            raise ValueError("lru_lfu needs tables with two score words")
        ts = int(device_timestamp() if timestamp is None else timestamp) if mode == SCORE_LRU_LFU else 0
        rows = chained_find(self._views, keys, table_ids, sc, mode, pin, ts)
        fill = 1 if mode in (SCORE_ACCUMULATE, SCORE_LRU_LFU) else 0
        m, skeys, stids, sscores, spos = chained_stage(
            rows, keys, table_ids, sc, fill, ts, self.num_scores
        )
        self._staged_keys = skeys[:m]
        self._staged_tids = stids[:m]
        self._staged_scores = sscores[: m * self.num_scores].view(m, self.num_scores)
        self._staged_positions = spos[:m]
        return rows

    @property
    def num_staged(self) -> int:
        return 0 if self._staged_keys is None else int(self._staged_keys.numel())

    def staged_positions(self) -> torch.Tensor:
        """Where, among the keys given to find_and_stage, the staged ones are."""
        if self._staged_positions is None:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return self._staged_positions

    def discard_staged(self) -> None:
        """Drop the staged keys without placing any: a step none of whose misses stays."""
        self._clear_staged()

    def _empty_result(self) -> CommitResult:
        e = torch.empty(0, dtype=torch.int64, device=self.device)
        return CommitResult(
            e, e, e, e, e, e, torch.empty(0, self.num_scores, dtype=torch.int64, device=self.device)
        )

    def commit(
        self,
        keep: Optional[torch.Tensor] = None,
        guard: Optional[Tuple[int, int]] = None,
        pin: bool = False,
    ) -> CommitResult:
        """Place the staged keys.

        ``keep`` is a bool mask over the staged keys (``None`` keeps them all); the
        others are dropped, having never been in a table. A kept key takes a free slot
        of its bucket if there is one; otherwise the bucket's lowest-scored member that
        is neither pinned nor refused by ``guard`` is evicted and the key takes its
        slot. All the keys of a bucket choose their victims together. With ``pin`` the
        placed keys' rows start with a reference.

        A table that never evicts is grown instead, until its buckets hold the keys.
        """
        if self._staged_keys is None or self._staged_keys.numel() == 0:
            self._clear_staged()
            return self._empty_result()
        keys, tids, scores, positions = (
            self._staged_keys,
            self._staged_tids,
            self._staged_scores,
            self._staged_positions,
        )
        self._clear_staged()
        if keep is not None:
            keys, tids, scores, positions = (
                keys[keep],
                tids[keep],
                scores[keep],
                positions[keep],
            )
        m = int(keys.numel())
        if m == 0:
            return self._empty_result()
        dev = self.device
        total_buckets = sum(self._nb)

        # Sort by bucket, growing the tables that must not evict until they fit.
        while True:
            plan = chained_plan(
                self._views,
                keys,
                tids,
                scores.reshape(-1).contiguous(),
                total_buckets,
                self.num_tables,
                self.num_scores,
            )
            (perm, _sgid, skeys, stids, sscores, seg_table, seg_bucket, seg_begin,
             counts, need, voff, num_runs, tstart) = plan  # fmt: skip
            if not any(self.no_eviction):
                break
            grow = sorted(
                {
                    t
                    for t in seg_table[need > 0].unique().tolist()
                    if self.no_eviction[t]
                }
            )
            if not grow:
                break
            for t in grow:
                self.expand(t)
            total_buckets = sum(self._nb)

        g_ptr, g_n = guard if guard is not None else (0, 0)
        (rows, srows, _vict_slot, vkeys, vrows, vtabs, seg_fail, stats, vscores) = chained_apply(
            self._views, perm, skeys, stids, sscores, seg_table, seg_bucket,
            seg_begin, counts, need, voff, num_runs, tstart, g_ptr, g_n, pin,
            self.num_tables, self.num_scores,
        )  # fmt: skip
        for t in range(self.num_tables):
            self._version[t] += 1
        victims, runs, failed_buckets, exhausted = stats.tolist()
        vscores = vscores[: victims * self.num_scores].view(victims, self.num_scores)
        if exhausted:
            per_table = (tstart[1:] - tstart[:-1]).tolist()
            tops = self._tops.tolist()
            short = [(t, tops[t], c) for t, c in enumerate(per_table) if c > tops[t]]
            t, have, need_rows = short[0]
            raise RuntimeError(
                f"table {t} has {have} free rows and {need_rows} keys need one; "
                "increase row_capacity or release the victims of earlier steps"
            )

        if failed_buckets:
            failed = seg_fail[:runs].to(torch.bool)
            seg_of_key = torch.repeat_interleave(
                torch.arange(runs, device=dev), counts[:runs]
            )
            unplaced = failed[seg_of_key]
            self.release_rows(srows[unplaced], stids[unplaced])
            ok_victim = torch.repeat_interleave(~failed, need[:runs])
            partial = CommitResult(
                srows[~unplaced],
                positions[perm][~unplaced],
                stids[~unplaced],
                vkeys[:victims][ok_victim],
                vrows[:victims][ok_victim],
                vtabs[:victims][ok_victim],
                vscores[ok_victim],
            )
            raise ChainedTableFull(
                f"{failed_buckets} bucket(s) could not make room for "
                f"{int(unplaced.sum())} key(s): every member they could evict is "
                "in use by a step in flight, or they were given more new keys than "
                "they have slots. Increase the capacity or reduce the prefetch "
                "pipeline depth.",
                partial,
            )
        return CommitResult(
            rows, positions, tids, vkeys[:victims], vrows[:victims], vtabs[:victims], vscores
        )

    def _clear_staged(self) -> None:
        self._staged_keys = None
        self._staged_tids = None
        self._staged_scores = None
        self._staged_positions = None

    def release_rows(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Give rows back to their pools, once nothing needs what they hold."""
        if rows.numel() == 0:
            return
        if self.num_tables == 1:
            chained_release(self._views, 0, rows)
            return
        for t in range(self.num_tables):
            chained_release(self._views, t, rows[table_ids == t])

    def erase(self, keys: torch.Tensor, table_ids: torch.Tensor) -> torch.Tensor:
        """Remove keys. Their rows go back to the pools; each key's row comes back, -1
        for a key that was not there. Not for a step in flight: a pinned row is erased
        like any other."""
        keys = _as_int64(keys)
        if keys.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = chained_erase(
            self._views, keys, table_ids, sum(self._nb), self.num_tables, self.num_scores
        )
        for t in range(self.num_tables):
            self._version[t] += 1
        found = rows >= 0
        self.release_rows(rows[found], table_ids[found])
        return rows

    def pin(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Take a reference on each row: a row with one is not evicted."""
        assert self.track_refs, "this table does not pin"
        chained_pin(self._views, rows, table_ids, 1)

    def unpin(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        assert self.track_refs, "this table does not pin"
        chained_pin(self._views, rows, table_ids, -1)

    def refs_of(self, table: int) -> torch.Tensor:
        assert self.track_refs, "this table does not pin"
        return self._refs[table]

    # -- growing --------------------------------------------------------------

    def expand(self, table: int, num_buckets: Optional[int] = None) -> int:
        """Rebuild a table with more buckets. Only the keys move: each keeps its row,
        so the values do not. Returns the table's row capacity, which grows with it;
        the value pool has to follow.
        """
        old_nb = self._nb[table]
        new_nb = max(int(num_buckets) if num_buckets else 2 * old_nb, old_nb + 1)
        keys, scores, rows = self.export(table)
        scores = scores.reshape(keys.numel(), self.num_scores)
        cap = self.bucket_capacity
        dev = self.device
        while True:
            arrays = self._new_arrays(new_nb)
            tmp = chained_make_views(
                [arrays[0]], [arrays[1]], [arrays[2]], [arrays[3]], [arrays[4]],
                [arrays[5]], [None], [self._free[table]], self._tops[table : table + 1],
                [new_nb], cap, self.num_scores,
            )  # fmt: skip
            local = chained_bucket_ids(keys, new_nb)
            perm = torch.argsort(local, stable=True)
            slocal = local[perm]
            seg_bucket, counts = torch.unique_consecutive(slocal, return_counts=True)
            seg_table = torch.zeros_like(seg_bucket)
            need = chained_segment_need(tmp, seg_table, seg_bucket, counts)
            if bool((need > 0).any()):
                new_nb *= 2  # a bucket would overflow: spread them further
                continue
            break
        seg_begin = torch.cumsum(counts, 0) - counts
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        chained_place(
            tmp,
            keys[perm],
            scores[perm].reshape(-1).contiguous(),
            rows[perm],
            seg_table,
            seg_bucket,
            seg_begin,
            counts,
            torch.zeros_like(counts),
            torch.empty(0, dtype=torch.int32, device=dev),
            empty,
            empty,
            torch.zeros(seg_bucket.numel(), dtype=torch.int32, device=dev),
            0,
            0,
            False,
        )
        (
            self._keys[table],
            self._scores[table],
            self._index[table],
            self._next[table],
            self._head[table],
            self._bsize[table],
        ) = arrays
        self._nb[table] = new_nb
        self._version[table] += 1

        # The pool has to cover every slot; the new rows go on the free stack.
        slots = new_nb * cap
        if slots > self._row_cap[table]:
            old_r = self._row_cap[table]
            new_r = _default_rows(slots)
            top = int(self._tops[table].item())
            free = torch.empty(new_r, dtype=torch.int32, device=dev)
            free[:top] = self._free[table][:top]
            added = new_r - old_r
            free[top : top + added] = torch.arange(
                old_r, new_r, dtype=torch.int32, device=dev
            )
            self._free[table] = free
            self._tops[table] = top + added
            self._row_cap[table] = new_r
            if self.track_refs:
                refs = torch.zeros(new_r, dtype=torch.int32, device=dev)
                refs[:old_r] = self._refs[table]
                self._refs[table] = refs
        self._rebuild_views()
        return self._row_cap[table]

    # -- inspection -----------------------------------------------------------

    def export(self, table: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(keys, scores, rows) of everything a table stores, bucket by bucket."""
        cap = self.bucket_capacity
        occupied = torch.arange(cap, device=self.device)[None, :] < self._bsize[table][
            :, None
        ]
        mask = occupied.reshape(-1)
        scores = self._scores[table].view(-1, self.num_scores)[mask]
        return (
            self._keys[table][mask],
            scores[:, 0] if self.num_scores == 1 else scores,
            self._index[table][mask].to(torch.int64) & 0xFFFFFFFF,
        )

    def bucket_of(self, keys: torch.Tensor, table: int = 0) -> torch.Tensor:
        return chained_bucket_ids(_as_int64(keys), self._nb[table])

    def check_invariants(self, pending_rows: Optional[Sequence[int]] = None) -> None:
        """Walk every chain and check the structure; for tests, it is slow.

        ``pending_rows`` is, per table, the number of rows taken from the pool that are
        not on a slot: victims not yet released.
        """
        pending = list(pending_rows) if pending_rows is not None else [0] * self.num_tables
        cap = self.bucket_capacity
        for t in range(self.num_tables):
            keys = self._keys[t].cpu().tolist()
            index = self._index[t].cpu().tolist()
            nxt = self._next[t].cpu().to(torch.int64).tolist()
            head = self._head[t].cpu().to(torch.int64).tolist()
            sizes = self._bsize[t].cpu().tolist()
            seen_keys = set()
            seen_rows = set()
            for b in range(self._nb[t]):
                size = sizes[b]
                assert 0 <= size <= cap, f"table {t} bucket {b}: size {size}"
                visited = []
                for c in range(cap):
                    cur = head[b * cap + c]
                    steps = 0
                    while cur != -1:
                        assert cur < size, f"bucket {b}: chain {c} reaches slot {cur}"
                        key = keys[b * cap + cur]
                        h = hash_py(key)
                        assert h % self._nb[t] == b, f"key {key} in the wrong bucket"
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
                    assert row < self._row_cap[t], f"row {row} out of range"
                    assert row not in seen_rows, f"row {row} used twice"
                    seen_rows.add(row)
            assert (
                int(self._tops[t].item()) + len(seen_rows) + pending[t] == self._row_cap[t]
            ), f"table {t}: rows are lost or duplicated"


class ChainedTable(ChainedTables):
    """One table: the same thing without the table ids."""

    def __init__(
        self,
        num_buckets: int,
        bucket_capacity: int = 128,
        row_capacity: Optional[int] = None,
        device: Optional[torch.device] = None,
        track_refs: bool = False,
        no_eviction: bool = False,
        num_scores: int = 1,
    ):
        super().__init__(
            [num_buckets],
            bucket_capacity,
            None if row_capacity is None else [row_capacity],
            track_refs,
            [no_eviction],
            device,
            num_scores,
        )

    @property
    def num_buckets(self) -> int:
        return self._nb[0]

    @property
    def row_capacity(self) -> int:
        return self._row_cap[0]

    @property
    def capacity(self) -> int:
        return self._nb[0] * self.bucket_capacity

    def _zeros(self, n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.int64, device=self.device)

    def lookup(self, keys: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().lookup(keys, self._zeros(keys.numel()))

    def find_and_stage(  # type: ignore[override]
        self,
        keys: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        policy: str = "assign",
        pin: bool = False,
        timestamp: Optional[int] = None,
    ) -> torch.Tensor:
        return super().find_and_stage(
            keys, self._zeros(keys.numel()), scores, policy, pin, timestamp
        )

    def release_rows(self, rows: torch.Tensor, table_ids=None) -> None:  # type: ignore[override]
        super().release_rows(rows, self._zeros(rows.numel()))

    def erase(self, keys: torch.Tensor, table_ids=None) -> torch.Tensor:  # type: ignore[override]
        return super().erase(keys, self._zeros(keys.numel()))

    def pin(self, rows: torch.Tensor, table_ids=None) -> None:  # type: ignore[override]
        super().pin(rows, self._zeros(rows.numel()))

    def unpin(self, rows: torch.Tensor, table_ids=None) -> None:  # type: ignore[override]
        super().unpin(rows, self._zeros(rows.numel()))

    def check_invariants(self, pending_rows: int = 0) -> None:  # type: ignore[override]
        super().check_invariants([pending_rows])


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
