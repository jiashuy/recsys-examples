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
from dynamicemb_extensions import flagged_compact

from .chained_table import ChainedTables
from .dynamicemb_config import DynamicEmbScoreStrategy, EvictedItemMode
from .key_value_table import (
    DynamicEmbTableState,
    _append_evicted,
    _fill_from_lower,
    create_table_state,
    get_table_ptrs,
    load_from_flat,
    store_to_flat,
)
from .types import CopyMode, Storage

_SUPPORTED = (
    DynamicEmbScoreStrategy.TIMESTAMP,
    DynamicEmbScoreStrategy.STEP,
    DynamicEmbScoreStrategy.CUSTOMIZED,
    DynamicEmbScoreStrategy.LFU,
    DynamicEmbScoreStrategy.NO_EVICTION,
)


class ChainedStorage(Storage):
    def __init__(self, options, optimizer, lower: Optional[Storage] = None):
        base = options[0]
        strategy = base.score_strategy
        if strategy not in _SUPPORTED:
            raise NotImplementedError(
                f"the chained index supports the single-score strategies "
                f"{[s.name for s in _SUPPORTED]}, not {strategy}"
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
        )
        self._tick = 0
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
            self._tick += 1
            return torch.full((n,), self._tick, dtype=torch.int64, device=device), "assign"
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
        if self._strategy == DynamicEmbScoreStrategy.NO_EVICTION:
            return
        state = self._state
        for t, n in enumerate(unique_size_per_table.tolist()):
            if n == 0:
                continue
            opt = state.options_list[t]
            capacity = self._index.capacity_of(t)
            max_lf = float(opt.max_load_factor)
            if max_lf <= 0 or capacity >= int(opt.max_capacity):
                continue
            self._estimated_size[t] += int(n)
            if self._estimated_size[t] / capacity <= max_lf:
                continue
            self._estimated_size[t] = self._index.size(t)  # the real size, once
            if (self._estimated_size[t] + int(n)) / capacity <= max_lf:
                continue
            old_nb = self._index.num_buckets_of(t)
            cap = self._index.bucket_capacity
            limit = max(1, int(opt.max_capacity) // cap)
            new_nb = min(2 * old_nb, max(limit, old_nb))
            if new_nb > old_nb:
                self._index.expand(t, new_nb)
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
        if scores is not None:
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

    def dump(self, *args, **kwargs) -> None:
        raise NotImplementedError("the chained index has no checkpoint format yet")

    def load(self, *args, **kwargs) -> None:
        raise NotImplementedError("the chained index has no checkpoint format yet")

    def export_keys_values(
        self, device: torch.device, batch_size: int = 65536, table_id: int = 0
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        keys, scores, rows = self._index.export(table_id)
        tid = torch.full((keys.numel(),), table_id, dtype=torch.int64, device=keys.device)
        for start in range(0, keys.numel(), batch_size):
            sl = slice(start, start + batch_size)
            yield (
                keys[sl],
                scores[sl],
                None,
                load_from_flat(
                    self._state, rows[sl], tid[sl], copy_mode=CopyMode.EMBEDDING
                ),
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
