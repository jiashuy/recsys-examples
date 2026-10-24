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

The same job as :class:`DynamicEmbStorage` for training -- the backing behind a cache,
or the HBM-direct table -- but the keys are in a :class:`ChainedTables`, so a key's row
is not where its hash slot is. What prefetch needs from it is the index
(``find_and_stage`` / ``commit``), the value buffers (``state``) and the tier below it
(``lower``). Checkpoints, exports and replay are not here yet.

Experimental: the module picks this when ``DEMB_CHAINED_INDEX`` is set.
"""

import math
from typing import Iterator, List, Optional, Tuple

import torch
from dynamicemb_extensions import EvictStrategy, ScorePolicy, device_timestamp, flagged_compact

from .chained_table import ChainedTables
from .dynamicemb_config import DynamicEmbScoreStrategy, EvictedItemMode, ReplayContent
from .key_value_table import (
    DynamicEmbTableState,
    _append_erased,
    _append_evicted,
    _dump_table,
    _fill_from_lower,
    _iter_batches_from_files,
    _fresh_score_block,
    _load_key_values,
    _pop_state_erased_keys,
    _replay_score_block,
    _replay_write_values,
    _validate_load_meta,
    create_table_state,
    export_keys_values_iter,
    get_table_ptrs,
    load_from_flat,
    store_to_flat,
)
from .types import SCORE_TYPE, CopyMode, ReplayStats, Storage

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
        s.retain_and_spill(out.victim_keys, out.victim_tables, out.victim_rows)
        index.release_rows(out.victim_rows, out.victim_tables)
        return result


def _is_lru_lfu(strategy) -> bool:
    return isinstance(strategy, tuple) and frozenset(strategy) == frozenset(
        {DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}
    )


class ChainedStorage(Storage):
    def __init__(self, options, optimizer, lower: Optional[Storage] = None):
        base = options[0]
        strategy = base.score_strategy
        lru_lfu = _is_lru_lfu(strategy)
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
        cap = int(base.bucket_capacity)
        if cap > 256:
            raise ValueError("the chained index takes buckets of at most 256 slots")
        num_buckets = [max(1, math.ceil(int(o.init_capacity) / cap)) for o in options]
        slots = [n * cap for n in num_buckets]
        rows = [s + max(s // 8, 1024) for s in slots]

        self._lower = lower
        self._strategy = strategy
        self._state: DynamicEmbTableState = create_table_state(
            options,
            optimizer,
            # With nothing below it this is a last tier and honours evicted_item_mode.
            evicted_item_mode=(
                base.evicted_item_mode if lower is None else EvictedItemMode.DISCARD
            ),
            build_index=False,
            row_capacities=rows,
        )
        # Without a cache in front, the rows in use have to be protected by a count.
        hbm_direct = self._state.tables[0].is_device_buffer()
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
        # An upper bound on the keys of each table, refreshed when it matters.
        self._estimated_size = [0] * len(options)

    # -- what prefetch uses ---------------------------------------------------

    @property
    def index(self) -> ChainedTables:
        return self._index

    @property
    def lower(self) -> Optional[Storage]:
        return self._lower

    def score_args(
        self, n: int, device: torch.device, frequency: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], str]:
        """The score and policy that this step's accesses give the keys they touch.

        Lower scores are evicted first, so a timestamp, a step, a user score and a
        frequency all rank the way they should.
        """
        s = self._strategy
        if s == DynamicEmbScoreStrategy.TIMESTAMP:
            # The device clock, as the existing index uses, so that a checkpoint's ages
            # (a timestamp minus a score) mean the same thing here.
            return (
                torch.full((n,), device_timestamp(), dtype=torch.int64, device=device),
                "assign",
            )
        if _is_lru_lfu(s):
            # the timestamp is the device clock at the access; the frequency grows by
            # what the step counted, or by one
            return (None if frequency is None else frequency.to(torch.int64)), "lru_lfu"
        if s in (DynamicEmbScoreStrategy.STEP, DynamicEmbScoreStrategy.CUSTOMIZED):
            value = int(self._state.score or 0)
            return torch.full((n,), value, dtype=torch.int64, device=device), "assign"
        if s == DynamicEmbScoreStrategy.LFU:
            return (None if frequency is None else frequency.to(torch.int64)), "accumulate"
        return None, "none"

    def sync_value_buffers(self) -> None:
        """Give the value buffers as many rows as the index has handed out."""
        state = self._state
        grew = False
        for t in range(state.num_tables):
            have = int(state.tables[t].tensor().size(0))
            want = self._index.row_capacity_of(t)
            if want > have:
                state.tables[t].extend((want - have, state.table_value_dims_cpu[t]))
                grew = True
        if grew:
            state.table_ptrs_dev.copy_(get_table_ptrs(state.tables, state.device))

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
        if self._strategy == DynamicEmbScoreStrategy.NO_EVICTION:
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

    def collect_table_sizes(self, non_blocking: bool = True) -> None:
        return None

    def increment_counter(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        keep = rows >= 0
        self._index.pin(rows[keep], table_ids[keep])

    def decrement_counter(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Give back the references a prefetch took on rows (-1: no row, no reference)."""
        keep = rows >= 0
        self._index.unpin(rows[keep], table_ids[keep])

    def retain_and_spill(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        rows: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> None:
        """What was evicted: kept for ``pop_evicted_keys``, and sent down a tier.

        ``rows`` still hold the victims' values; ``values`` overrides them (a key that
        is also cached has a newer one there).
        """
        n = int(keys.numel())
        if n == 0:
            return
        if self._lower is None:
            if EvictedItemMode.RETAIN_KEY in self._state.evicted_item_mode:
                _append_evicted(
                    self._state,
                    keys,
                    table_ids,
                    torch.tensor(n, dtype=torch.int64, device=keys.device),
                )
            return
        if values is None:
            values = load_from_flat(
                self._state, rows, table_ids, copy_mode=CopyMode.VALUE
            )
        self._lower.insert(keys, table_ids, values)

    # -- Storage interface ----------------------------------------------------

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ):
        rows = self._index.lookup(unique_keys, table_ids)
        founds = rows >= 0
        values = load_from_flat(self._state, rows, table_ids, copy_mode=copy_mode)
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
            ) = _fill_from_lower(
                self._lower,
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
        if scores is not None and policy != "lru_lfu":
            sc, policy = scores.to(torch.int64), "assign"
        if preserve_existing:
            policy = "none"
        index = self._index
        rows = index.find_and_stage(unique_keys, table_ids, sc, policy)
        found = rows >= 0
        if bool(found.any()):
            store_to_flat(
                self._state, rows[found], table_ids[found], unique_values[found]
            )
        if index.num_staged == 0:
            return
        out = index.commit()
        self.sync_value_buffers()
        store_to_flat(
            self._state,
            out.rows,
            table_ids[out.positions],
            unique_values[out.positions],
        )
        self.retain_and_spill(out.victim_keys, out.victim_tables, out.victim_rows)
        index.release_rows(out.victim_rows, out.victim_tables)

    def flush_from(self, cache) -> None:
        """Write every row the cache holds back to this table's own row of the key."""
        cstate = cache._state
        batch_size = cstate.threads_in_wave
        for t in range(cstate.num_tables):
            nodes, keys, _referenced = cache.resident(t)
            for start in range(0, nodes.numel(), batch_size):
                chunk_nodes = nodes[start : start + batch_size].to(torch.int64)
                chunk_keys = keys[start : start + batch_size]
                tid = torch.full(
                    (chunk_keys.numel(),), t, dtype=torch.int64, device=chunk_keys.device
                )
                values = load_from_flat(
                    cstate, chunk_nodes, tid, copy_mode=CopyMode.VALUE
                )
                rows = self._index.lookup(chunk_keys, tid)
                if not bool((rows >= 0).all()):
                    raise RuntimeError(
                        "the cache holds a key that the backing table does not"
                    )
                store_to_flat(self._state, rows, tid, values)

    def pop_evicted_keys(self, table_id: int) -> torch.Tensor:
        from .key_value_table import _pop_state_evicted_keys

        return _pop_state_evicted_keys(self._state, table_id)

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
        embeddings = values.to(device=device, dtype=state.emb_dtype)
        if ReplayContent.SCORE in content and scores is not None:
            score_vec = _replay_score_block(state, table_id, scores, timestamp)
        else:
            score_vec = _fresh_score_block(
                state, table_id, torch.zeros(n, dtype=torch.int64, device=device),
                timestamp, insert_score,
            )  # fmt: skip
        score_vec = score_vec.to(torch.int64)
        pair = index_pair = self._index.num_scores == 2
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
                self.retain_and_spill(out.victim_keys, out.victim_tables, out.victim_rows)
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
        for keys, embeddings, scores, opt_states in _iter_batches_from_files(
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path if params.include_optim else None,
            params.dim,
            params.file_optstate_dim,
            device,
            params.dist_type,
            num_scores=1,
            emb_dtype=params.emb_dtype,
            opt_state_dtype=params.opt_state_dtype,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            if scores is not None and self._state.evict_strategy == EvictStrategy.KLru:
                scores = torch.clamp(timestamp - scores, min=0)
            _load_key_values(
                self._state, keys, embeddings, scores, opt_states, table_id=table_id
            )
        return params.meta_data.get("step_score", None)

    def export_keys_values(
        self, device: torch.device, batch_size: int = 65536, table_id: int = 0
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        yield from export_keys_values_iter(self._state, device, batch_size, table_id)

    def incremental_dump(self, table_id, threshold, pg, timestamp=None):
        """What the existing storage's does: it reads only ``_state``, whose index view
        answers for this one. ``slot_index`` is the row, for information."""
        from .key_value_table import DynamicEmbStorage

        return DynamicEmbStorage.incremental_dump(
            self, table_id, threshold, pg, timestamp
        )

    # -- accessors ------------------------------------------------------------

    def embedding_dtype(self) -> torch.dtype:
        return self._state.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.emb_dim

    def max_value_dim(self) -> int:
        return self._state.value_dim

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return (
            self._state.table_emb_dims if on_device else self._state.table_emb_dims_host
        )

    def all_dims_vec4(self) -> bool:
        return self._state.all_dims_vec4

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
