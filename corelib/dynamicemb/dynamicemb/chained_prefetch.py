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

"""Prefetch over a :class:`ChainedStorage`: the cache path and the HBM-direct path.

The index answers in two calls, and the work between them -- the tier below, admission,
the cache -- is done with the table untouched:

    find_and_stage    the keys the table has are answered; the others are held back
    ...               tier below, admission, a node in the cache for every key
    commit            the held-back keys that stay are placed, and room is made

Eviction comes last. By then every key of the step has a node in the cache and so is
protected, which is why the cache's eviction guard is enough, and a key placed in the
step can never be the victim of another. The rows of the victims are untouched until
they are released, and the new keys have rows of their own, so a victim's value is
read where it is; nothing has to be copied out ahead of a write.
"""

from typing import Optional, Tuple

import torch
from dynamicemb.types import CopyMode, MultiTableAdmitter, counts_to_offsets
from dynamicemb_extensions import EvictStrategy, flagged_compact

from .batched_dynamicemb_function import _bool_item, _scalar_item
from .chained_storage import ChainedStorage


def _admit(
    admitter: Optional[MultiTableAdmitter],
    missing_keys: torch.Tensor,
    missing_tids: torch.Tensor,
    missing_positions: torch.Tensor,
    accumulated_frequency: Optional[torch.Tensor],
    from_lower: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Which of the missing keys stay (``None``: all). A key the tier below has is not
    new to the system, so it is not put to the admitter, and stays."""
    if admitter is None:
        return None
    freq = (
        accumulated_frequency[missing_positions]
        if accumulated_frequency is not None
        else None
    )
    if from_lower is None:
        return admitter.admit(missing_keys, missing_tids, freq)
    new_to_system = ~from_lower
    mask = from_lower.clone()
    mask[new_to_system] = admitter.admit(
        missing_keys[new_to_system],
        missing_tids[new_to_system],
        freq[new_to_system] if freq is not None else None,
    )
    return mask


def prefetch_cache_path(
    cache,
    storage: ChainedStorage,
    exchange,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    unique_offsets: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Cache prefetch where the storage is the authority on every key.

    1. pin the keys already cached
    2. find every key in the storage, updating its score; hold the misses back
    3. ask the tier below for the misses; decide admission for the rest
    4. give a node to every key that needs one, which also protects it
    5. move rows: write the cache's victims back to the storage rows they already
       have, load the rows with a history, initialise the new ones
    6. place the admitted keys in the storage, which asks the cache before it evicts
       anything, and drop from the cache whatever the storage evicted
    7. publish the nodes
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path_chained"):
        device = unique_keys.device
        values = cache.values
        sstate = storage._state
        index = storage.index
        h_num_total = unique_keys.numel()
        if h_num_total == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                None,
                torch.empty(0, dtype=torch.int32, device=device),
            )

        # 1.
        with torch.cuda.nvtx.range("op:cache_lookup_pin"):
            nodes = cache.lookup_pin(unique_keys, unique_offsets)

        # 2.
        with torch.cuda.nvtx.range("op:storage_find"):
            scores, policy = storage.score_args(
                h_num_total, device, accumulated_frequency
            )
            rows = index.find_and_stage(unique_keys, unique_table_ids, scores, policy)
        founds = rows >= 0
        missing_positions = index.staged_positions()
        missing_keys = unique_keys[missing_positions]
        missing_tids = unique_table_ids[missing_positions]
        h_num_missing = int(missing_positions.numel())

        # 3.
        lower = storage.lower
        from_lower: Optional[torch.Tensor] = None
        lower_values: Optional[torch.Tensor] = None
        if lower is not None and h_num_missing > 0:
            missing_freq = (
                accumulated_frequency[missing_positions]
                if accumulated_frequency is not None
                else None
            )
            out = lower.find(missing_keys, missing_tids, CopyMode.VALUE, missing_freq)
            from_lower, lower_values = out[5], out[7]
        admit_mask = (
            _admit(
                admitter,
                missing_keys,
                missing_tids,
                missing_positions,
                accumulated_frequency,
                from_lower,
            )
            if h_num_missing > 0
            else None
        )
        non_admitted_positions: Optional[torch.Tensor] = None
        admitted_positions = missing_positions
        if admit_mask is not None:
            admitted_positions = missing_positions[admit_mask]
            non_admit = ~admit_mask
            if _bool_item(non_admit.any()):
                non_admitted_positions = missing_positions[non_admit]

        # 4.
        need_node = (nodes < 0) & founds
        if admitted_positions.numel() > 0:
            need_node[admitted_positions] = True
        if not _bool_item(need_node.any()):
            index.discard_staged()
            return nodes.to(torch.int64), non_admitted_positions, nodes
        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, alloc_positions, (alloc_keys, alloc_tids) = flagged_compact(
                need_node, [unique_keys, unique_table_ids]
            )
        alloc_offsets = counts_to_offsets(
            torch.bincount(alloc_tids, minlength=cache.num_tables), device
        )
        with torch.cuda.nvtx.range("op:cache_allocate"):
            victim_keys, has_victim, new_nodes = cache.allocate(
                alloc_keys, alloc_positions, alloc_offsets, nodes
            )
        if _bool_item((new_nodes < 0).any()):
            raise RuntimeError(
                "The cache has no row to give: every row of some table is held by "
                "a step in flight. Reduce the prefetch pipeline depth or increase "
                "the cache size."
            )
        new_rows = new_nodes.to(torch.int64)

        # 5a. each node swaps with the storage: the victim's row goes back to the row
        # the victim already has there, and a key with a history brings its row in
        evict_rows = torch.full_like(new_rows, -1)
        if _bool_item(has_victim.any()):
            v_rows = index.lookup(victim_keys[has_victim], alloc_tids[has_victim])
            if not _bool_item((v_rows >= 0).all()):
                raise RuntimeError(
                    "a key the cache held is not in the backing table: the two "
                    "have drifted apart"
                )
            evict_rows[has_victim] = v_rows
        alloc_in_storage = founds[alloc_positions]
        with torch.cuda.nvtx.range("op:swap_rows"):
            exchange.swap(new_rows, alloc_tids, evict_rows, rows[alloc_positions])

        # 5c. new keys are initialised in their cache rows
        alloc_is_new = ~alloc_in_storage
        if _bool_item(alloc_is_new.any()):
            n_new = int(_scalar_item(alloc_is_new.sum()))
            init_vals = torch.empty(n_new, max_val_dim, dtype=emb_dtype, device=device)
            init_indices = torch.arange(n_new, dtype=torch.int64, device=device)
            new_keys = alloc_keys[alloc_is_new]
            new_tids = alloc_tids[alloc_is_new]
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(init_vals[:, :max_emb_dim], new_keys, new_tids, init_indices)
            sstate.optimizer.init_states(sstate.values.layout, init_vals, new_tids)
            if from_lower is not None and _bool_item(from_lower.any()):
                lower_row = torch.full((h_num_total,), -1, dtype=torch.int64, device=device)
                lower_row[missing_positions[from_lower]] = torch.arange(
                    int(_scalar_item(from_lower.sum())), dtype=torch.int64, device=device
                )
                row_of_new = lower_row[alloc_positions[alloc_is_new]]
                has_row = row_of_new >= 0
                init_vals[has_row] = lower_values[from_lower][row_of_new[has_row]]
            with torch.cuda.nvtx.range("op:store_to_flat"):
                values.store(new_rows[alloc_is_new], new_tids, init_vals)

        # 6. the admitted keys are placed; whatever the storage evicts leaves the cache
        if h_num_missing > 0 and admitted_positions.numel() > 0:
            with torch.cuda.nvtx.range("op:storage_commit"):
                out = index.commit(admit_mask, guard=cache.eviction_guard)
            storage.sync_value_buffers()
            if out.victim_keys.numel() > 0:
                ev_keys, ev_tids, ev_rows = (
                    out.victim_keys,
                    out.victim_tables,
                    out.victim_rows,
                )
                ev_nodes = cache.find_nodes(ev_keys, ev_tids)
                ev_values = None
                if lower is not None:
                    # Their rows still hold what the storage had; a key that is also
                    # cached has a newer value there.
                    ev_values = sstate.values.load(ev_rows, ev_tids, copy_mode=CopyMode.VALUE
                    )
                    cached = ev_nodes >= 0
                    if _bool_item(cached.any()):
                        ev_values[cached] = values.load(ev_nodes[cached].to(torch.int64),
                            ev_tids[cached],
                            copy_mode=CopyMode.VALUE,
                        )
                storage.evict(ev_keys, ev_tids, ev_rows, ev_values)
                cache.invalidate(ev_nodes, ev_tids)
                index.release_rows(ev_rows, ev_tids)
        else:
            index.discard_staged()

        # 7.
        cache.publish(new_nodes, alloc_offsets)
        return nodes.to(torch.int64), non_admitted_positions, nodes


def prefetch_hbm_direct_path(
    storage: ChainedStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """HBM-direct prefetch: the rows the step uses are pinned in the index.

    Returns ``(rows, non_admitted_positions, rows)``: the row of every key, -1 for a
    key that was not admitted, which gets its value at forward time.
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path_chained"):
        state = storage._state
        index = storage.index
        device = unique_keys.device
        h_num_total = unique_keys.numel()
        if h_num_total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None, empty

        with torch.cuda.nvtx.range("op:storage_find"):
            scores, policy = storage.score_args(
                h_num_total, device, accumulated_frequency
            )
            rows = index.find_and_stage(
                unique_keys, unique_table_ids, scores, policy, pin=True
            )
        missing_positions = index.staged_positions()
        h_num_missing = int(missing_positions.numel())
        if h_num_missing == 0:
            return rows, None, rows
        missing_keys = unique_keys[missing_positions]
        missing_tids = unique_table_ids[missing_positions]

        lower = storage.lower
        from_lower: Optional[torch.Tensor] = None
        lower_values: Optional[torch.Tensor] = None
        if lower is not None:
            missing_freq = (
                accumulated_frequency[missing_positions]
                if accumulated_frequency is not None
                else None
            )
            out = lower.find(missing_keys, missing_tids, CopyMode.VALUE, missing_freq)
            from_lower, lower_values = out[5], out[7]

        admit_mask = _admit(
            admitter,
            missing_keys,
            missing_tids,
            missing_positions,
            accumulated_frequency,
            from_lower,
        )
        non_admitted_positions: Optional[torch.Tensor] = None
        if admit_mask is None:
            admitted_keys, admitted_tids = missing_keys, missing_tids
            admitted_positions = missing_positions
        else:
            admitted_keys = missing_keys[admit_mask]
            admitted_tids = missing_tids[admit_mask]
            admitted_positions = missing_positions[admit_mask]
            non_admit = ~admit_mask
            if _bool_item(non_admit.any()):
                non_admitted_positions = missing_positions[non_admit]

        if admitted_keys.numel() > 0:
            n_admitted = admitted_keys.numel()
            init_values = torch.empty(
                n_admitted, max_val_dim, dtype=state.values.dtype, device=device
            )
            init_idx = torch.arange(n_admitted, dtype=torch.int64, device=device)
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(
                    init_values[:, :max_emb_dim], admitted_keys, admitted_tids, init_idx
                )
            state.optimizer.init_states(state.values.layout, init_values, admitted_tids)
            if from_lower is not None:
                if admit_mask is None:
                    in_lower, lower_rows = from_lower, lower_values
                else:
                    in_lower, lower_rows = from_lower[admit_mask], lower_values[admit_mask]
                init_values[in_lower] = lower_rows[in_lower]

            with torch.cuda.nvtx.range("op:storage_commit"):
                out = index.commit(admit_mask, pin=True)
            storage.sync_value_buffers()
            # The victims' rows are intact until released, and the new keys have rows of
            # their own: what goes down a tier is read from where it is.
            storage.evict(out.victim_keys, out.victim_tables, out.victim_rows)
            index.release_rows(out.victim_rows, out.victim_tables)
            with torch.cuda.nvtx.range("op:store_to_flat"):
                state.values.store(out.rows, admitted_tids, init_values)
            rows[admitted_positions] = out.rows
        else:
            index.discard_staged()

        return rows, non_admitted_positions, rows
