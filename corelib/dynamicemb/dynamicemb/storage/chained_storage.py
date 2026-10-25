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

"""A table of rows over the chained bucket index.

The training storage -- the backing behind a cache, or the HBM-direct table. The keys
are in a :class:`ChainedTables`, so a key's row is not where its hash slot is. What
prefetch needs from it is the index (``find_and_stage`` / ``commit``), the value
buffers (``state``) and the tier below it (``lower``); checkpoints, exports, incremental
dump and replay are here too, and :class:`ChainedHybridStorage` pairs an HBM tier with
a host tier.

"""

import math
from typing import Iterator, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from dynamicemb_extensions import EvictStrategy, ScorePolicy, device_timestamp, flagged_compact

from dynamicemb.storage.chained_table import ChainedTables
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    EvictedItemMode,
    ReplayContent,
)
from dynamicemb.checkpoint.files import _iter_batches_from_files
from dynamicemb.storage.row_io import (
    _all_gather_dumped_columns,
    _dump_table,
    _load_key_values,
    _replay_write_values,
    _split_value_row,
    _validate_load_meta,
    export_keys_values_iter,
)
from dynamicemb.storage.eviction_sink import Discard, EvictionSink, Retain, ToLower, ToTier
from dynamicemb.storage.lower_tier import LowerTier
from dynamicemb.storage.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.checkpoint.row_chunk import RowChunk
from dynamicemb.storage.score import ScoreRules, is_lru_lfu
from dynamicemb.storage.table_state import (
    DynamicEmbTableState,
    _append_erased,
    _pop_state_erased_keys,
    create_table_state,
)
from dynamicemb.types import SCORE_TYPE, CopyMode, MultiTableAdmitter, ReplayStats, Storage

_SUPPORTED = (
    DynamicEmbScoreStrategy.TIMESTAMP,
    DynamicEmbScoreStrategy.STEP,
    DynamicEmbScoreStrategy.CUSTOMIZED,
    DynamicEmbScoreStrategy.LFU,
    DynamicEmbScoreStrategy.NO_EVICTION,
)


class _ChainedIndexView:
    """The part of a key index that the checkpoint code reads, answered from a
    :class:`ChainedStorage`: the keys of a table with their scores and rows, and an
    insert that returns the rows. What those helpers call a slot index is, here, a row.
    """

    score_fn_key_ = 0

    def __init__(self, storage: "ChainedStorage"):
        self._s = storage
        self.num_scores_ = storage.index.num_scores
        self._inverse = None  # (table, version) -> row -> slot
        self.key_type = storage._state.options_list[0].index_type
        self.score_names_ = [storage._state.score_policy.name]
        self.bucket_capacity_ = storage.index.bucket_capacity

    def incremental_dump(
        self,
        score_threshold,
        table_id: int,
        batch_size: int = 65536,
        pg=None,
        return_index: bool = False,
    ):
        """Keys, scores and rows of what scores at least the threshold, on the host."""
        names = [n for n in score_threshold if n in self.score_names_]
        thresholds = [score_threshold[n] for n in names]
        keys, scores, rows = [], {n: [] for n in names}, []
        for k, named, r in self._batched_export_keys_scores(
            names, self._s._state.device, table_id, thresholds, batch_size, True
        ):
            keys.append(k.cpu())
            rows.append(r.cpu())
            for n in names:
                scores[n].append(named[n].cpu())

        def cat(parts, dtype):
            return torch.cat(parts) if parts else torch.empty(0, dtype=dtype)

        out_scores = {n: cat(scores[n], SCORE_TYPE) for n in names}
        return (
            cat(keys, torch.int64),
            out_scores,
            cat(rows, torch.int64) if return_index else None,
        )

    @property
    def per_table_capacity_(self) -> List[int]:
        return [self._s.index.capacity_of(t) for t in range(self._s.num_tables)]

    def capacity(self, table_id: Optional[int] = None) -> int:
        if table_id is None:
            return sum(self.per_table_capacity_)
        return self._s.index.capacity_of(table_id)

    def size(self, table_id: Optional[int] = None) -> int:
        if table_id is None:
            return sum(self._s.index.size(t) for t in range(self._s.num_tables))
        return self._s.index.size(table_id)

    def _batched_export_keys_scores(
        self,
        score_names: List[str],
        target_device: torch.device,
        table_id: int,
        thresholds: Optional[List[int]] = None,
        batch_size: int = 65536,
        return_index: bool = False,
    ):
        keys, scores, rows = self._s.index.export(table_id)
        if scores.dim() == 2:
            scores = scores[:, 0]  # the timestamp, the column an incremental dump thresholds
        if thresholds:
            keep = scores >= int(thresholds[0])
            keys, scores, rows = keys[keep], scores[keep], rows[keep]
        for start in range(0, int(keys.numel()), batch_size):
            sl = slice(start, start + batch_size)
            named = {name: scores[sl].to(SCORE_TYPE) for name in score_names}
            yield (
                keys[sl].view(self.key_type),
                named,
                rows[sl] if return_index else None,
            )

    def _slots_of(self, table_id: int, rows: torch.Tensor) -> torch.Tensor:
        """Flat slot of each row, from a map built when the table last changed."""
        index = self._s.index
        key = (table_id, index.version(table_id))
        if self._inverse is None or self._inverse[0] != key:
            cap = index.bucket_capacity
            occupied = (
                torch.arange(cap, device=index.device)[None, :]
                < index._bsize[table_id][:, None]
            ).reshape(-1)
            slots = occupied.nonzero().squeeze(1)
            held = index._index[table_id][slots].to(torch.int64) & 0xFFFFFFFF
            inverse = torch.full(
                (index.row_capacity_of(table_id),), -1, dtype=torch.int64, device=index.device
            )
            inverse[held] = slots
            self._inverse = (key, inverse)
        return self._inverse[1][rows.to(torch.int64)]

    def gather_score_blocks(self, table_id: int, rows: torch.Tensor) -> torch.Tensor:
        """The score words of the keys on these rows, ``[n, num_scores]``."""
        slots = self._slots_of(table_id, rows)
        return self._s.index._scores[table_id].view(-1, self.num_scores_)[slots]

    def scatter_score_blocks(
        self, table_id: int, rows: torch.Tensor, block: torch.Tensor
    ) -> None:
        slots = self._slots_of(table_id, rows)
        self._s.index._scores[table_id].view(-1, self.num_scores_)[slots] = block.to(
            torch.int64
        )

    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score,
        insert_results=None,
        score_out=None,
        collect_evicted: bool = False,
        guard=None,
    ) -> torch.Tensor:
        """Put keys in, returning the row of each: the existing row of a key already
        there, a new one otherwise. Whatever has to go to make room goes down a tier."""
        if collect_evicted:
            raise NotImplementedError("the chained index reports its evictions itself")
        s = self._s
        index = s.index
        if score.policy == ScorePolicy.CONST:
            values, policy = None, "none"
        elif self.num_scores_ == 2:
            # a (timestamp, frequency) pair: the clock, and the frequency it is given
            values = None if score.value is None else score.value.to(torch.int64)
            policy = "lru_lfu"
        elif score.policy == ScorePolicy.ACCUMULATE:
            values, policy = score.value.to(torch.int64), "accumulate"
        elif score.value is None:  # the device clock
            values = torch.full(
                (keys.numel(),), device_timestamp(), dtype=torch.int64, device=keys.device
            )
            policy = "assign"
        else:
            values, policy = score.value.to(torch.int64), "assign"
        rows = index.find_and_stage(keys, table_ids, values, policy)
        if index.num_staged == 0:
            return rows
        out = index.commit(guard=guard)
        s.sync_value_buffers()
        result = rows.clone()
        result[out.positions] = out.rows
        s.evict(out.victim_keys, out.victim_tables, out.victim_rows)
        index.release_rows(out.victim_rows, out.victim_tables)
        return result


class Missing(NamedTuple):
    """What the storage settles about the keys it does not have (all ``None`` if none)."""

    from_lower: Optional[torch.Tensor]  # bool per key: the tier below has it
    lower_values: Optional[torch.Tensor]  # its values, a row per key
    admit_mask: Optional[torch.Tensor]  # bool per key: stays; None: all stay


class ChainedStorage(Storage):
    def __init__(
        self,
        options,
        optimizer,
        lower: Optional[Storage] = None,
        spill_to: Optional["ChainedStorage"] = None,
        track_refs: Optional[bool] = None,
    ):
        """``lower`` is the tier below that a miss is asked of and an eviction goes to
        (a parameter server). ``spill_to`` only receives evictions: the tier of a hybrid
        storage next to this one, whose keys are disjoint from ours."""
        base = options[0]
        strategy = base.score_strategy
        lru_lfu = is_lru_lfu(strategy)
        if not lru_lfu and strategy not in _SUPPORTED:
            raise NotImplementedError(
                f"the chained index supports {[s.name for s in _SUPPORTED]} and the "
                f"compound (TIMESTAMP, LFU), not {strategy}"
            )
        if lru_lfu and base.score_function is not None:
            raise NotImplementedError(
                "the chained index ranks a (TIMESTAMP, LFU) table by frequency and then "
                "age; a custom score_function is not supported"
            )
        # The tables of a storage share one bucket capacity: it is part of what groups
        # tables.
        cap = int(base.bucket_capacity)
        num_buckets = [max(1, math.ceil(int(o.init_capacity) / cap)) for o in options]
        slots = [n * cap for n in num_buckets]
        rows = [s + max(s // 8, 1024) for s in slots]

        self._lower = None if lower is None else LowerTier(lower)
        self._admitter: Optional[MultiTableAdmitter] = None
        self._score = ScoreRules(strategy)
        self._state: DynamicEmbTableState = create_table_state(
            options,
            optimizer,
            # With nothing below it this is a last tier and honours evicted_item_mode.
            evicted_item_mode=(
                base.evicted_item_mode
                if lower is None and spill_to is None
                else EvictedItemMode.DISCARD
            ),
            build_index=False,
            row_capacities=rows,
        )
        # Without a cache in front, the rows in use have to be protected by a count.
        hbm_direct = (
            self._state.values.on_device if track_refs is None else track_refs
        )
        self._index = ChainedTables(
            num_buckets,
            cap,
            rows,
            track_refs=hbm_direct,
            no_eviction=[strategy == DynamicEmbScoreStrategy.NO_EVICTION] * len(options),
            device=self._state.device,
            num_scores=2 if lru_lfu else 1,
        )
        self._state.key_index_map = _ChainedIndexView(self)
        if spill_to is not None:
            self._sink: EvictionSink = ToTier(spill_to)
        elif lower is not None:
            self._sink = ToLower(self._lower)
        elif EvictedItemMode.RETAIN_KEY in base.evicted_item_mode:
            self._sink = Retain(self._state, Discard())
        else:
            self._sink = Discard()
        # An upper bound on the keys of each table, refreshed when it matters.
        self._estimated_size = [0] * len(options)

    # -- what prefetch uses ---------------------------------------------------

    @property
    def index(self) -> ChainedTables:
        return self._index

    def set_admitter(self, admitter: Optional[MultiTableAdmitter]) -> None:
        """The policy that decides which of the keys this storage lacks are let in."""
        self._admitter = admitter

    def resolve_missing(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequency: Optional[torch.Tensor],
    ) -> Missing:
        """Ask the tier below for ``keys`` the storage does not have, then the admitter.

        A key the tier below has is not new to the system, so it is not put to the
        admitter, and stays. ``frequency`` is what the step counted for each key.
        """
        if keys.numel() == 0:
            return Missing(None, None, None)
        from_lower = lower_values = None
        if self._lower is not None:
            fetched = self._lower.fetch(keys, table_ids, frequency)
            from_lower, lower_values = fetched.found, fetched.values
        admitter = self._admitter
        if admitter is None:
            return Missing(from_lower, lower_values, None)
        if from_lower is None:
            return Missing(
                from_lower, lower_values, admitter.admit(keys, table_ids, frequency)
            )
        new_to_system = ~from_lower
        mask = from_lower.clone()
        mask[new_to_system] = admitter.admit(
            keys[new_to_system],
            table_ids[new_to_system],
            frequency[new_to_system] if frequency is not None else None,
        )
        return Missing(from_lower, lower_values, mask)

    @property
    def lower(self) -> Optional[LowerTier]:
        return self._lower

    def score_args(
        self, n: int, device: torch.device, frequency: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], str]:
        """The score and policy that this step's accesses give the keys they touch."""
        return self._score.access(n, device, int(self._state.score or 0), frequency)

    def sync_value_buffers(self) -> None:
        """Give the value buffers as many rows as the index has handed out."""
        for t in range(self._state.num_tables):
            self._state.values.ensure_rows(t, self._index.row_capacity_of(t))

    def expand_if_need(self, unique_size_per_table: torch.Tensor) -> None:
        """Grow a table whose load would pass its maximum, up to its maximum capacity.

        A table that never evicts grows when a bucket cannot hold its keys, which is
        decided when they are placed.
        """
        for t, n in enumerate(unique_size_per_table.tolist()):
            if n:
                self.grow_for(t, int(n))

    def grow_for(self, table: int, extra_keys: int) -> None:
        """Make room, within the table's maximum capacity, for ``extra_keys`` more."""
        if self._score.strategy == DynamicEmbScoreStrategy.NO_EVICTION:
            return
        opt = self._state.options_list[table]
        max_lf = float(opt.max_load_factor)
        cap = self._index.bucket_capacity
        limit = max(1, int(opt.max_capacity) // cap)
        capacity = self._index.capacity_of(table)
        if max_lf <= 0 or capacity >= int(opt.max_capacity):
            return
        self._estimated_size[table] += extra_keys
        if self._estimated_size[table] / capacity <= max_lf:
            return
        self._estimated_size[table] = self._index.size(table)  # the real size, once
        grew = False
        while True:
            nb = self._index.num_buckets_of(table)
            if (self._estimated_size[table] + extra_keys) / (nb * cap) <= max_lf or nb >= limit:
                break
            self._index.expand(table, min(2 * nb, limit))
            grew = True
        if grew:
            self.sync_value_buffers()

    def increment_counter(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        keep = rows >= 0
        self._index.pin(rows[keep], table_ids[keep])

    def decrement_counter(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Give back the references a prefetch took on rows (-1: no row, no reference)."""
        keep = rows >= 0
        self._index.unpin(rows[keep], table_ids[keep])

    def evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        rows: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Hand what was evicted to the sink.

        ``rows`` still hold the victims' values; ``values`` overrides them (a key that
        is also cached has a newer one there). ``scores`` are the victims' own score
        words, which a tier of the same kind beside this one keeps them at.
        """
        if keys.numel() == 0:
            return
        if values is None and self._sink.wants_values:
            values = self._state.values.load(rows, table_ids, copy_mode=CopyMode.VALUE)
        self._sink.put(keys, table_ids, values, scores)

    # -- Storage interface ----------------------------------------------------

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ):
        if self._state.training and self._score.strategy != DynamicEmbScoreStrategy.NO_EVICTION:
            # a training step's access counts: it moves the score of what it finds
            scores, policy = self.score_args(
                unique_keys.numel(), unique_keys.device, lfu_accumulated_frequency
            )
            rows = self._index.touch(unique_keys, table_ids, scores, policy)
        else:
            rows = self._index.lookup(unique_keys, table_ids)
        founds = rows >= 0
        values = self._state.values.load(rows, table_ids, copy_mode=copy_mode)
        n_missing, miss_idx, (missing_keys, missing_tids) = flagged_compact(
            ~founds, [unique_keys, table_ids]
        )
        missing_scores = (
            lfu_accumulated_frequency[miss_idx]
            if lfu_accumulated_frequency is not None
            else None
        )
        scores = torch.zeros_like(rows)
        if self._lower is not None and n_missing > 0:
            (
                n_missing,
                missing_keys,
                miss_idx,
                missing_tids,
                missing_scores,
                founds,
                scores,
            ) = self._lower.fill(
                copy_mode,
                values,
                n_missing,
                missing_keys,
                miss_idx,
                missing_tids,
                missing_scores,
                founds,
                scores,
            )
        return (
            n_missing,
            missing_keys,
            miss_idx,
            missing_tids,
            missing_scores,
            founds,
            scores,
            values,
        )

    def update_existing(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
    ) -> torch.Tensor:
        """Write the rows of the keys that are here; the others are left alone. Returns
        which were here."""
        rows = self._index.lookup(unique_keys, table_ids)
        found = rows >= 0
        if bool(found.any()):
            self._state.values.store(rows[found], table_ids[found], unique_values[found]
            )
        return found

    def _write_score_blocks(
        self, rows: torch.Tensor, table_ids: torch.Tensor, block: torch.Tensor
    ) -> None:
        for t in range(self.num_tables):
            sel = table_ids == t
            if bool(sel.any()):
                self._state.key_index_map.scatter_score_blocks(t, rows[sel], block[sel])

    def insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        n = int(unique_keys.numel())
        if n == 0:
            return
        device = unique_keys.device
        sc, policy = self.score_args(n, device)
        block = None
        if scores is not None:
            if self._index.num_scores == 2 and scores.dim() == 2:
                block = scores.to(torch.int64)  # both words, written once the rows are known
                sc, policy = None, "none"
            elif policy != "lru_lfu":
                sc, policy = scores.reshape(-1).to(torch.int64), "assign"
        if preserve_existing:
            policy = "none"
        index = self._index
        rows = index.find_and_stage(unique_keys, table_ids, sc, policy)
        found = rows >= 0
        if bool(found.any()):
            self._state.values.store(rows[found], table_ids[found], unique_values[found]
            )
        if index.num_staged:
            out = index.commit()
            self.sync_value_buffers()
            self._state.values.store(out.rows,
                table_ids[out.positions],
                unique_values[out.positions],
            )
            rows[out.positions] = out.rows
            self.evict(
                out.victim_keys,
                out.victim_tables,
                out.victim_rows,
                scores=out.victim_scores,
            )
            index.release_rows(out.victim_rows, out.victim_tables)
        if block is not None:
            self._write_score_blocks(rows, table_ids, block)

    def flush_from(self, cache) -> None:
        """Write every row the cache holds back to this table's own row of the key."""
        batch_size = self._state.threads_in_wave
        for t in range(cache.num_tables):
            nodes, keys, _referenced = cache.resident(t)
            for start in range(0, nodes.numel(), batch_size):
                chunk_nodes = nodes[start : start + batch_size].to(torch.int64)
                chunk_keys = keys[start : start + batch_size]
                tid = torch.full(
                    (chunk_keys.numel(),), t, dtype=torch.int64, device=chunk_keys.device
                )
                values = cache.values.load(chunk_nodes, tid, copy_mode=CopyMode.VALUE
                )
                rows = self._index.lookup(chunk_keys, tid)
                if not bool((rows >= 0).all()):
                    raise RuntimeError(
                        "the cache holds a key that the backing table does not"
                    )
                self._state.values.store(rows, tid, values)

    def pop_evicted_keys(self, table_id: int) -> torch.Tensor:
        from dynamicemb.storage.table_state import _pop_state_evicted_keys

        return _pop_state_evicted_keys(self._state, table_id)

    def fill_tables(self, load_factor: float = 0.95, tolerance: float = 1e-5) -> None:
        """Put random keys in until every table holds about ``load_factor`` of its slots
        (at most 0.95). Only the index is written, not the rows; for benchmarks that want
        a table that is already full.
        """
        if load_factor < 0.0 or tolerance < 0.0:
            raise ValueError("load_factor and tolerance must not be negative")
        load_factor = min(load_factor, 0.95)
        device = self._state.device
        index = self._index
        for t in range(self.num_tables):
            capacity = index.capacity_of(t)
            target = min(capacity, int(load_factor * capacity))
            stalls = 0
            while index.size(t) < target:
                before = index.size(t)
                if tolerance > 0 and abs(before / capacity - load_factor) <= tolerance:
                    break
                # a bucket takes at most as many new keys at once as it has slots
                n = min(max(1, capacity // 8), target - before)
                keys = torch.unique(
                    torch.randint(0, 2**63 - 1, (n,), dtype=torch.int64, device=device)
                )
                tids = torch.full((keys.numel(),), t, dtype=torch.int64, device=device)
                scores, policy = self.score_args(keys.numel(), device)
                index.find_and_stage(keys, tids, scores, policy)
                if index.num_staged:
                    out = index.commit()
                    self.sync_value_buffers()
                    index.release_rows(out.victim_rows, out.victim_tables)
                if index.size(t) <= before:
                    stalls += 1
                    if stalls > 100:
                        raise RuntimeError(f"fill_tables: stalled on table {t}")

    def erase_keys(
        self,
        table_id: int,
        keys: torch.Tensor,
        mode: EvictedItemMode = EvictedItemMode.DISCARD,
    ) -> int:
        """Erase keys from a table; returns how many of them were there."""
        n = int(keys.numel())
        if n == 0:
            return 0
        device = self._state.device
        keys = keys.to(device=device, dtype=self._state.key_index_map.key_type)
        tids = torch.full((n,), table_id, dtype=torch.int64, device=device)
        removed = self._index.erase(keys, tids) >= 0
        if EvictedItemMode.RETAIN_KEY in mode:
            _append_erased(self._state, keys[removed], table_id)
        return int(removed.sum().item())

    def pop_erased_keys(self, table_id: int) -> torch.Tensor:
        return _pop_state_erased_keys(self._state, table_id)

    def replay_increment(
        self,
        table_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        optimizer_states: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
        slot_index: torch.Tensor,
        insert_score: int,
        content: ReplayContent = ReplayContent.ALL,
        timestamp: Optional[int] = None,
    ) -> ReplayStats:
        """Write an ``incremental_dump`` delta back into one table, key by key.

        A key that is there takes the delta's value (and score, when asked); one that
        is not is placed, in whatever slot and row the table hands out. ``slot_index``
        is the source's and says nothing here.
        """
        stats = ReplayStats()
        n = int(keys.numel())
        if n == 0:
            return stats
        state = self._state
        device = state.device
        timestamp = device_timestamp() if timestamp is None else timestamp
        keys = keys.to(device=device, dtype=state.key_index_map.key_type)
        embeddings = values.to(device=device, dtype=state.values.dtype)
        if ReplayContent.SCORE in content and scores is not None:
            score_vec = self._score.replay_block(scores.to(state.device), timestamp)
        else:
            score_vec = self._score.fresh_block(n, device, timestamp, insert_score)
        score_vec = score_vec.to(torch.int64)
        pair = self._index.num_scores == 2
        index = self._index
        # A bucket takes at most as many new keys at once as it has slots.
        step = max(
            1, min(int(state.threads_in_wave), index.capacity_of(table_id) // 4)
        )
        for start in range(0, n, step):
            end = min(start + step, n)
            k = keys[start:end]
            tids = torch.full((k.numel(),), table_id, dtype=torch.int64, device=device)
            self.grow_for(table_id, int(k.numel()))
            rows = index.find_and_stage(
                k,
                tids,
                None if pair else score_vec[start:end].reshape(-1),
                "none" if pair else "assign",
            )
            same_key = rows >= 0
            if index.num_staged:
                out = index.commit()
                self.sync_value_buffers()
                rows[out.positions] = out.rows
                self.evict(out.victim_keys, out.victim_tables, out.victim_rows)
                index.release_rows(out.victim_rows, out.victim_tables)
            if pair:  # both words, for the keys that were there as well as the new ones
                state.key_index_map.scatter_score_blocks(
                    table_id, rows, score_vec[start:end]
                )
            _replay_write_values(
                state,
                table_id,
                rows,
                embeddings[start:end],
                optimizer_states[start:end] if optimizer_states is not None else None,
                same_key,
                content,
            )
        stats.upserted = n
        return stats

    def dump(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool = True,
        include_meta: bool = True,
        current_score: Optional[int] = None,
        timestamp: int = 0,
        append: bool = False,
    ) -> None:
        _dump_table(
            self._state,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
            include_meta,
            timestamp=timestamp,
            current_score=current_score,
            append=append,
        )

    def load(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = True,
        timestamp: int = 0,
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> Optional[int]:
        params = _validate_load_meta(
            self._state,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
        )
        self.grow_for(table_id, params.num_keys)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        num_scores = self._state.key_index_map.num_scores_
        # The checkpoint stores multi-word scores in the configured (logical) order;
        # put each batch back into the physical device layout.
        for keys, embeddings, scores, opt_states in _iter_batches_from_files(
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path if params.include_optim else None,
            params.dim,
            params.file_optstate_dim,
            device,
            params.dist_type,
            num_scores=num_scores,
            emb_dtype=params.emb_dtype,
            opt_state_dtype=params.opt_state_dtype,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            if scores is not None and num_scores > 1:
                scores = self._score.to_physical(scores)
            if scores is not None and self._state.evict_strategy == EvictStrategy.KLru:
                scores = torch.clamp(timestamp - scores, min=0)
            _load_key_values(
                self._state, keys, embeddings, scores, opt_states, table_id=table_id
            )
        return params.meta_data.get("step_score", None)

    def export_keys_values(
        self, device: torch.device, batch_size: int = 65536, table_id: int = 0
    ) -> Iterator[RowChunk]:
        yield from export_keys_values_iter(self._state, device, batch_size, table_id)

    def incremental_dump(
        self,
        table_id: int,
        threshold: int,
        pg: Optional[dist.ProcessGroup],
        timestamp: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor
    ]:
        """Dump what scores at least ``threshold`` on the column an incremental dump
        thresholds on (the timestamp where there is one).

        Returns ``(keys, values, slot_index, optimizer_states, scores)`` on the host,
        column-aligned. ``values`` is embeddings only and ``optimizer_states`` the rest of
        the same row (``None`` when the optimizer keeps no per-row state). ``slot_index``
        is the row, for information: a replica places keys as it likes. ``scores`` is the
        ``[N, num_scores]`` block in logical column order, a timestamp held as an age
        relative to ``timestamp``. With a process group of several ranks the columns are
        gathered over it.
        """
        timestamp = device_timestamp() if timestamp is None else timestamp
        state = self._state
        if state.options_list[table_id].dist_type == "continuous":
            raise NotImplementedError(
                "incremental_dump does not support dist_type 'continuous' (replay "
                "cannot reconstruct the owning rank from a key)."
            )
        gather = (
            pg is not None and dist.is_initialized() and dist.get_world_size(group=pg) > 1
        )
        keys, named_scores, rows = state.key_index_map.incremental_dump(
            {state.incremental_score_name: threshold},
            pg=pg,
            return_index=True,
            table_id=table_id,
        )
        device = state.device
        rows = rows.to(device)
        scores_batch = named_scores[state.incremental_score_name]
        values = state.values.load_table(rows, table_id)
        value, opt = _split_value_row(state, table_id, values)
        key = keys.to(device)
        if self._score.num_words > 1:
            physical = state.key_index_map.gather_score_blocks(
                table_id, rows.to(torch.int64)
            )
        else:
            physical = scores_batch.to(device=device, dtype=SCORE_TYPE)
        score_block = self._score.dump_block(physical, timestamp)
        if not gather:
            key, value, rows, score_block = (
                key.cpu(), value.cpu(), rows.cpu(), score_block.cpu(),
            )  # fmt: skip
            opt = opt.cpu() if opt is not None else None
        else:
            columns = [key, value, rows, score_block]
            if opt is not None:
                columns.append(opt)
            gathered = _all_gather_dumped_columns(
                [c.to(device) for c in columns], pg
            )
            key, value, rows, score_block = gathered[:4]
            opt = gathered[4] if opt is not None else None
        return key, value, rows, opt, score_block

    # -- accessors ------------------------------------------------------------

    @property
    def values(self) -> MixedDimBuffers:
        return self._state.values

    def embedding_dtype(self) -> torch.dtype:
        return self._state.values.dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.values.emb_dim(table_id)

    def value_dim(self, table_id: int) -> int:
        return self._state.values.value_dim(table_id)

    def max_embedding_dim(self) -> int:
        return self._state.values.max_emb_dim

    def max_value_dim(self) -> int:
        return self._state.values.max_value_dim

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return (
            self._state.values.emb_dims if on_device else self._state.values.emb_dims_host
        )

    def all_dims_vec4(self) -> bool:
        return self._state.values.all_dims_vec4

    def init_optimizer_state(self) -> float:
        return self._state.optimizer.get_initial_optimizer_state()

    def set_score(self, score: int) -> None:
        self._state.score = score

    @property
    def training(self) -> bool:
        return self._state.training

    @training.setter
    def training(self, value: bool) -> None:
        self._state.training = value

    def evict_strategy(self):
        return self._state.evict_strategy

    @property
    def num_tables(self) -> int:
        return self._state.num_tables

    @property
    def tables(self) -> List[DynamicEmbTableState]:
        return [self._state]


class ChainedHybridStorage(Storage):
    """Two tiers, an HBM one and a host one, holding disjoint keys.

    A new key goes into the HBM tier; what that tier evicts goes down into the host
    tier with its score, and what the host tier evicts is dropped (or retained, when
    asked). A lookup asks the HBM tier and then the host tier; an update writes the row
    wherever the key is, and never moves it.
    """

    def __init__(self, hbm_options, host_options, optimizer):
        self._host_tier = ChainedStorage(host_options, optimizer, track_refs=False)
        self._hbm_tier = ChainedStorage(
            hbm_options, optimizer, spill_to=self._host_tier, track_refs=False
        )
        self.optimizer = optimizer
        # The tiers' states, as the existing hybrid storage exposes them.
        self._hbm = self._hbm_tier._state
        self._host = self._host_tier._state

    @property
    def _tiers(self) -> List[ChainedStorage]:
        return [self._hbm_tier, self._host_tier]

    @property
    def tables(self) -> List[DynamicEmbTableState]:
        return [self._hbm, self._host]

    @property
    def num_tables(self) -> int:
        return self._hbm.num_tables

    # -- score management and accessors, as the HBM tier has them -------------------

    @property
    def training(self) -> bool:
        return self._hbm.training

    @training.setter
    def training(self, value: bool) -> None:
        self._hbm.training = value
        self._host.training = value

    def set_score(self, score: int) -> None:
        self._hbm.score = score
        self._host.score = score

    def evict_strategy(self):
        return self._hbm.evict_strategy

    @property
    def values(self) -> MixedDimBuffers:
        """The HBM tier's; both tiers hold tables of the same dimensions."""
        return self._hbm.values

    def embedding_dtype(self) -> torch.dtype:
        return self._hbm.values.dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._hbm.values.emb_dim(table_id)

    def value_dim(self, table_id: int) -> int:
        return self._hbm.values.value_dim(table_id)

    def max_embedding_dim(self) -> int:
        return self._hbm.values.max_emb_dim

    def max_value_dim(self) -> int:
        return self._hbm.values.max_value_dim

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return self._hbm.values.emb_dims if on_device else self._hbm.values.emb_dims_host

    def all_dims_vec4(self) -> bool:
        return self._hbm.values.all_dims_vec4

    def init_optimizer_state(self) -> float:
        return self._hbm.optimizer.get_initial_optimizer_state()

    def expand_if_need(self, unique_size_per_table: torch.Tensor) -> None:
        for tier in self._tiers:
            tier.expand_if_need(unique_size_per_table)

    def pop_evicted_keys(self, table_id: int) -> torch.Tensor:
        # the HBM tier spills, so only the host tier ever lets a key go
        return self._host_tier.pop_evicted_keys(table_id)

    def pop_erased_keys(self, table_id: int) -> torch.Tensor:
        parts = [t.pop_erased_keys(table_id) for t in self._tiers]
        parts = [p for p in parts if p.numel()]
        if not parts:
            return self._hbm_tier.pop_erased_keys(table_id)[:0]
        return torch.unique(torch.cat(parts))

    # -- lookup and update ----------------------------------------------------------

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ):
        (n_missing, miss_keys, miss_idx, miss_tids, miss_scores, founds, scores, values) = (
            self._hbm_tier.find(unique_keys, table_ids, copy_mode, lfu_accumulated_frequency)
        )
        if n_missing == 0:
            return (0, miss_keys, miss_idx, miss_tids, miss_scores, founds, scores, values)
        (n_both, keys_both, idx_both, tids_both, scores_both, in_host, host_scores, host_values) = (
            self._host_tier.find(miss_keys, miss_tids, copy_mode, miss_scores)
        )
        if bool(in_host.any()):
            at = miss_idx[in_host]
            values[at] = host_values[in_host]
            founds = founds.clone()
            founds[at] = True
            scores = scores.clone()
            scores[at] = host_scores[in_host].to(scores.dtype)
        return (
            n_both,
            keys_both,
            miss_idx[idx_both],
            tids_both,
            scores_both,
            founds,
            scores,
            values,
        )

    def insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        """A new key goes into the HBM tier. With ``preserve_existing`` nothing is
        inserted: the rows of the keys that are in either tier are rewritten."""
        if preserve_existing:
            if unique_keys.numel() == 0:
                return
            in_hbm = self._hbm_tier.update_existing(unique_keys, table_ids, unique_values)
            rest = ~in_hbm
            if bool(rest.any()):
                self._host_tier.update_existing(
                    unique_keys[rest], table_ids[rest], unique_values[rest]
                )
            return
        self._hbm_tier.insert(unique_keys, table_ids, unique_values, scores)

    def erase_keys(
        self,
        table_id: int,
        keys: torch.Tensor,
        mode: EvictedItemMode = EvictedItemMode.DISCARD,
    ) -> int:
        """The tiers hold disjoint keys, so the counts add."""
        return sum(t.erase_keys(table_id, keys, mode) for t in self._tiers)

    # -- checkpoints ----------------------------------------------------------------

    def dump(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool = True,
        include_meta: bool = True,
        current_score: Optional[int] = None,
        timestamp: int = 0,
    ) -> None:
        """The host tier, then the HBM tier appended to the same files."""
        if self._host.key_index_map.num_scores_ > 1 or self._hbm.key_index_map.num_scores_ > 1:
            raise NotImplementedError(
                "dump of a hybrid storage with a (timestamp, frequency) score is not "
                "supported: the tiers share one score file"
            )
        self._host_tier.dump(
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
            include_meta,
            current_score,
            timestamp,
        )
        self._hbm_tier.dump(
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
            False,
            None,
            timestamp,
            append=True,
        )

    def load(self, *args, **kwargs) -> Optional[int]:
        """Through the HBM tier, whose evictions go down to the host tier."""
        return self._hbm_tier.load(*args, **kwargs)

    def export_keys_values(
        self, device: torch.device, batch_size: int = 65536, table_id: int = 0
    ) -> Iterator[RowChunk]:
        for tier in self._tiers:
            yield from tier.export_keys_values(device, batch_size, table_id)

    def incremental_dump(self, table_id, threshold, pg, timestamp=None):
        timestamp = device_timestamp() if timestamp is None else timestamp
        parts = [t.incremental_dump(table_id, threshold, pg, timestamp) for t in self._tiers]
        keys, values, slots, opts, scores = (list(col) for col in zip(*parts))
        opt = None if opts[0] is None else torch.cat(opts, dim=0)
        return (
            torch.cat(keys),
            torch.cat(values, dim=0),
            torch.cat(slots),
            opt,
            torch.cat(scores, dim=0),
        )

    def replay_increment(
        self,
        table_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        optimizer_states: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
        slot_index: torch.Tensor,
        insert_score: int,
        content: ReplayContent = ReplayContent.ALL,
        timestamp: Optional[int] = None,
    ) -> ReplayStats:
        """A key the host tier holds is updated there; every other key goes through the
        HBM tier, which sends what it evicts down."""
        stats = ReplayStats()
        if keys.numel() == 0:
            return stats
        device = self._hbm.device
        k = keys.to(device=device, dtype=self._hbm.key_index_map.key_type)
        tids = torch.full((k.numel(),), table_id, dtype=torch.int64, device=device)
        in_host = (self._host_tier.index.lookup(k, tids) >= 0).to(keys.device)

        def pick(t, mask):
            return None if t is None else t[mask]

        for mask, tier in ((in_host, self._host_tier), (~in_host, self._hbm_tier)):
            if not bool(mask.any()):
                continue
            stats.merge(
                tier.replay_increment(
                    table_id,
                    keys[mask],
                    values[mask],
                    pick(optimizer_states, mask),
                    pick(scores, mask),
                    slot_index[mask],
                    insert_score,
                    content,
                    timestamp,
                )
            )
        return stats
