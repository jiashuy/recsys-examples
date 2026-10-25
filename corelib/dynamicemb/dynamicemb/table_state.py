# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The state of a table group that the storage and the cache share: its options, value rows, optimizer, and the keys it retained when they left.
"""

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch  # usort:skip
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EvictedItemMode,
)
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import (
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb_extensions import (
    EvictStrategy,
)


def get_score_policy(score_strategy):
    """Build the physical :class:`ScoreSpec` for a (validated) score strategy.

    *score_strategy* is either a single :class:`DynamicEmbScoreStrategy` or a
    supported compound tuple (currently only ``{TIMESTAMP, LFU}`` in either order).
    A compound strategy maps to the ``LruLfu`` policy whose physical layout is
    fixed at ``(timestamp, frequency)`` regardless of the user's tuple order; the
    user's ordering only affects checkpoint column order (see
    :func:`score_dump_permutation`).
    """
    if isinstance(score_strategy, tuple):
        # Only the {TIMESTAMP, LFU} compound is supported (validated at config
        # construction). Two adjacent AoS score words per key -- word 0 =
        # last-access timestamp (selects keys for a time-based incremental dump),
        # word 1 = frequency (drives LFU eviction). num_scores is derived from the
        # policy. The spec name maps to word 0 (score_index 0), i.e. the timestamp
        # column incremental_dump thresholds on -- hence "lru_lfu".
        if frozenset(score_strategy) == frozenset(
            {DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}
        ):
            return ScoreSpec(
                name="lru_lfu",
                policy=ScorePolicy.LRU_LFU,
                dtype=torch.uint64,
                is_reduction=True,
            )
        raise NotImplementedError(
            f"Unsupported compound score_strategy {score_strategy}."
        )
    if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
        return ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    elif score_strategy == DynamicEmbScoreStrategy.STEP:
        return ScoreSpec(name="step", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
        return ScoreSpec(name="customized", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.LFU:
        return ScoreSpec(name="frequency", policy=ScorePolicy.ACCUMULATE)
    elif score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        return ScoreSpec(name="index", policy=ScorePolicy.ASSIGN)
    else:
        raise RuntimeError("Not supported score strategy.")


# ---------------------------------------------------------------------------
# DynamicEmbTableState – shared state dataclass
# ---------------------------------------------------------------------------


@dataclass
class DynamicEmbTableState:
    options_list: List[DynamicEmbTableOptions]
    num_tables: int
    device: torch.device
    score_policy: ScoreSpec
    evict_strategy: EvictStrategy
    key_index_map: Any
    capacity: int
    # The value rows of the tables, which may differ in dimension.
    values: MixedDimBuffers
    optimizer: BaseDynamicEmbeddingOptimizer
    threads_in_wave: int
    score: Optional[int] = None
    training: bool = False
    # Name of the score column incremental_dump thresholds on. Equals
    # score_policy.name (for LruLfu this is the leading, timestamp, word).
    incremental_score_name: Optional[str] = None
    # Retained-key buffers (last tier only). A key leaves a table two ways, and
    # the two are kept apart because consumers want different things from them:
    # an eviction is reproduced by whoever overwrites its slot, while an erase
    # leaves the slot to nobody and has to be replayed as a removal.
    #
    #   evicted_*_chunks -- victims of an insert that had to make room
    #   erased_*_chunks  -- keys an explicit erase removed
    #
    # Both accumulate as (key, table_id) GPU-tensor chunks, one chunk per
    # operation that removed anything, and are only concatenated + de-duplicated
    # on pop (no compaction while appending). Which buffers are fed is
    # ``evicted_item_mode``'s call.
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD
    evicted_key_chunks: List[torch.Tensor] = field(default_factory=list)
    evicted_tid_chunks: List[torch.Tensor] = field(default_factory=list)
    erased_key_chunks: List[torch.Tensor] = field(default_factory=list)
    erased_tid_chunks: List[torch.Tensor] = field(default_factory=list)


def create_table_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD,
    build_index: bool = True,
    row_capacities: Optional[List[int]] = None,
) -> DynamicEmbTableState:
    """``build_index=False`` makes the value buffers only, ``row_capacities`` rows each:
    the key index lives elsewhere (see :mod:`dynamicemb.chained_table`)."""
    if not options:
        raise ValueError("options must be non-empty")
    if not build_index and row_capacities is None:
        raise ValueError("without an index, row_capacities says how big the buffers are")

    base_opt = options[0]
    num_tables = len(options)

    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")
    score_policy = get_score_policy(base_opt.score_strategy)
    evict_strategy = base_opt.evict_strategy.value

    # incremental_dump thresholds on the reduction score column (word 0 -- the
    # timestamp -- for the compound LruLfu policy).
    incremental_score_name = score_policy.name

    # LruLfu custom eviction: numba-compile + nvJitLink-link the user
    # score_function and route its inserts to that cubin via score_fn_key. Key 0
    # (no score_function) selects the default freq->ts Lex evictor (no numba).
    # The score_function is indexed in logical (tuple) order, so the compiler is
    # given score_strategy to remap logical->physical.
    score_fn_key = 0
    if (
        score_policy.policy == ScorePolicy.LRU_LFU
        and base_opt.score_function is not None
    ):
        from dynamicemb.jit.score_jit import register_score_function

        cc_major, cc_minor = torch.cuda.get_device_capability(device_idx)
        score_fn_key = register_score_function(
            base_opt.score_function, base_opt.score_strategy, cc_major, cc_minor
        )

    # NO_EVICTION: key_index_map uses max_load_factor=0.5 to avoid eviction; table uses init_capacity.
    bucket_capacity = base_opt.bucket_capacity
    if base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        no_eviction_max_lf = 0.5
        capacities = []
        for opt in options:
            if opt.init_capacity is not None:
                cap = math.ceil(opt.init_capacity / no_eviction_max_lf)
                aligned = (
                    (cap + bucket_capacity - 1) // bucket_capacity
                ) * bucket_capacity
                capacities.append(aligned)
            else:
                capacities.append(opt.init_capacity)
    else:
        capacities = [opt.init_capacity for opt in options]

    if build_index:
        key_index_map = get_scored_table(
            capacity=capacities,
            bucket_capacity=base_opt.bucket_capacity,
            key_type=base_opt.index_type,
            score_specs=[score_policy],
            device=device,
            score_fn_key=score_fn_key,
        )
        capacity = key_index_map.capacity()
    else:
        key_index_map = None
        capacity = int(sum(row_capacities))

    dims = [opt.dim for opt in options]
    value_dims = [d + optimizer.get_state_dim(d) for d in dims]

    key_index_map_caps = (
        key_index_map.per_table_capacity_ if build_index else list(row_capacities)
    )

    # Table (embedding) capacity may differ from key_index_map in NO_EVICTION:
    # key_index_map is larger (by max_load_factor); table uses init_capacity per table.
    if build_index and base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        table_caps = [
            (
                opt.init_capacity
                if opt.init_capacity is not None
                else key_index_map_caps[i]
            )
            for i, opt in enumerate(options)
        ]
    else:
        table_caps = list(key_index_map_caps)

    values = MixedDimBuffers(
        dims,
        value_dims,
        table_caps,
        base_opt.embedding_dtype,
        device,
        "host" if base_opt.local_hbm_for_values == 0 else "device",
    )

    props = torch.cuda.get_device_properties(device_idx)
    threads_in_wave = (
        props.multi_processor_count * props.max_threads_per_multi_processor
    )

    return DynamicEmbTableState(
        options_list=options,
        num_tables=num_tables,
        device=device,
        score_policy=score_policy,
        incremental_score_name=incremental_score_name,
        evict_strategy=evict_strategy,
        key_index_map=key_index_map,
        capacity=capacity,
        values=values,
        optimizer=optimizer,
        threads_in_wave=threads_in_wave,
        score=None,
        training=False,
        evicted_item_mode=evicted_item_mode,
    )


def _append_evicted(
    state: DynamicEmbTableState,
    evicted_keys: torch.Tensor,
    evicted_table_ids: torch.Tensor,
    num_evicted: torch.Tensor,
) -> None:
    """Append this insert's evicted (key, table_id) to the state's retain chunks.

    ``evicted_keys`` / ``evicted_table_ids`` are sized at the batch upper bound;
    only the first ``num_evicted`` entries are valid, so we slice + clone (freeing
    the large per-batch buffers and keeping only the compact evicted rows). No
    de-duplication here -- that happens on pop. One D2H sync per evicting insert
    to read the count, acceptable since retain is opt-in.
    """
    n = int(num_evicted.item())
    if n <= 0:
        return
    state.evicted_key_chunks.append(evicted_keys[:n].clone())
    state.evicted_tid_chunks.append(evicted_table_ids[:n].clone())


def _append_erased(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_id: int,
) -> None:
    """Append an explicit erase's keys to the state's erased-key buffer.

    Unlike :func:`_append_evicted` there is no count to read back: the caller has
    already masked the keys down to the ones the erase actually removed, so this
    costs no extra device sync.
    """
    if keys.numel() == 0:
        return
    state.erased_key_chunks.append(keys.clone())
    state.erased_tid_chunks.append(
        torch.full((keys.numel(),), table_id, dtype=torch.int64, device=keys.device)
    )


def _pop_chunked_keys(
    key_chunks: List[torch.Tensor],
    tid_chunks: List[torch.Tensor],
    table_id: int,
    state: DynamicEmbTableState,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Drain one table's rows out of a (key, table_id) chunk buffer.

    Concatenates the chunks, selects this table's rows, de-duplicates, and
    returns the rebuilt chunk lists holding only the OTHER tables' rows -- so
    this table's keys are cleared while the rest stay for their own pop.
    """
    if not key_chunks:
        # Match the non-empty path's dtype: chunks carry key_index_map.key_type
        # (== the table's index_type, which may be int32/uint32), so a hardcoded
        # int64 here would make callers that concat/compare across empty and
        # non-empty pops hit a dtype mismatch.
        empty = torch.empty(0, dtype=state.key_index_map.key_type, device=state.device)
        return empty, [], []
    keys = torch.cat(key_chunks)
    tids = torch.cat(tid_chunks)
    mask = tids == table_id
    out = torch.unique(keys[mask])
    keep = ~mask
    if bool(keep.any()):
        return out, [keys[keep]], [tids[keep]]
    return out, [], []


def _pop_state_evicted_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys ``table_id`` had evicted."""
    out, state.evicted_key_chunks, state.evicted_tid_chunks = _pop_chunked_keys(
        state.evicted_key_chunks, state.evicted_tid_chunks, table_id, state
    )
    return out


def _pop_state_erased_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys an explicit erase removed."""
    out, state.erased_key_chunks, state.erased_tid_chunks = _pop_chunked_keys(
        state.erased_key_chunks, state.erased_tid_chunks, table_id, state
    )
    return out
