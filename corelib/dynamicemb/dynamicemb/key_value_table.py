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

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EvictedItemMode,
    ReplayContent,
    ScoreStrategy,
    align_to_table_size,
    get_physical_score_order,
    score_dump_permutation,
    score_load_permutation,
)
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.key_ownership import owned_key_mask
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    Cache,
    CopyMode,
    Storage,
    torch_dtype_to_np_dtype,
)
from dynamicemb_extensions import (
    EmbeddingCacheCore,
    EvictStrategy,
    flagged_compact,
)
from torch import Tensor, nn  # usort:skip

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _timestamp_score_columns(score_strategy: ScoreStrategy) -> List[int]:
    """Logical score-column indices holding a device timestamp.

    ``%globaltimer`` is per-device and resets across boots, so a raw timestamp is
    meaningless anywhere but the rank that produced it. These columns are dumped
    as an *age* (``cur_ts - score``) -- the same convention the file checkpoint
    uses for single-column LRU tables. Every other column (LFU frequency, STEP,
    CUSTOMIZED, NO_EVICTION row) is carried verbatim.
    """
    if isinstance(score_strategy, tuple):
        return [
            i
            for i, s in enumerate(score_strategy)
            if s == DynamicEmbScoreStrategy.TIMESTAMP
        ]
    return [0] if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP else []


def _scores_to_age(
    score_strategy: ScoreStrategy, scores: Tensor, cur_ts: int
) -> Tensor:
    """Convert timestamp columns to ages, for dumping.

    *scores* is a ``[N, W]`` block in logical (configured) column order. When
    there is nothing to convert (no timestamp column, or an empty block) the
    input is returned unchanged rather than copied -- callers must not mutate the
    result in place. Otherwise a clone is returned.
    """
    ts_cols = _timestamp_score_columns(score_strategy)
    if not ts_cols or scores.numel() == 0:
        return scores
    out = scores.clone()
    for c in ts_cols:
        out[:, c] = torch.clamp(cur_ts - scores[:, c], min=0)
    return out


def _age_to_scores(
    score_strategy: ScoreStrategy, ages: Tensor, target_ts: int
) -> Tensor:
    """Inverse of :func:`_scores_to_age`, for replay.

    Same aliasing note: returns *ages* itself when there is nothing to convert.
    """
    ts_cols = _timestamp_score_columns(score_strategy)
    if not ts_cols or ages.numel() == 0:
        return ages
    out = ages.clone()
    for c in ts_cols:
        out[:, c] = torch.clamp(target_ts - ages[:, c], min=0)
    return out


# ---------------------------------------------------------------------------
# Utility helpers (continued)
# ---------------------------------------------------------------------------


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving data to JSON file: {e}")


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


# ---------------------------------------------------------------------------
# Checkpoint value layout
#
# Key / embedding / score / optimizer-state files are bare row-major blobs with
# no header, so the table's meta JSON is the only record of how to read them
# back. Embeddings and optimizer states are dumped at the table's own precision
# (``MixedDimBuffers.dtype``) rather than being upcast to fp32, which
# makes ``embedding_dtype`` and ``embedding_dim`` load-bearing rather than
# decorative. Checkpoints written before those keys existed are always fp32, and
# their dim is recoverable from the file size -- see
# :func:`resolve_checkpoint_value_layout`.
# ---------------------------------------------------------------------------

META_EMBEDDING_DTYPE = "embedding_dtype"
META_EMBEDDING_DIM = "embedding_dim"
META_OPT_STATE_DTYPE = "optim_state_dtype"


def dtype_to_meta_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``, the spelling checkpoint meta uses."""
    return str(dtype).split(".", 1)[-1]


def dtype_from_meta_name(name: Any, field_name: str) -> torch.dtype:
    """Inverse of :func:`dtype_to_meta_name`, rejecting anything not a dtype.

    Resolves against ``torch`` itself rather than a hand-kept table so any dtype
    a table can be built with round-trips, and a corrupt or hand-edited meta
    fails here instead of silently mis-striding the value file.
    """
    dtype = getattr(torch, name, None) if isinstance(name, str) else None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Checkpoint meta field {field_name!r} is {name!r}, which is not a "
            "torch dtype."
        )
    return dtype


def resolve_checkpoint_value_layout(
    meta_data: Dict[str, Any],
    embedding_file_path: str,
    num_keys: int,
    runtime_dim: int,
) -> Tuple[torch.dtype, torch.dtype, int]:
    """Decide how to read a value file: (embedding dtype, opt-state dtype, dim).

    Prefers what the dump recorded. For a checkpoint written before the meta
    carried a dtype, the historical answer is fp32 for both value files, and the
    row width is recovered from the value file's length -- exact, given the dtype
    and the key count. An empty shard is the one case the length cannot answer,
    so it defers to the runtime dim.

    Either way the result is cross-checked against the file size, which turns a
    wrong dtype assumption (or a truncated file) into an error here rather than
    a silently misparsed table.

    Raises:
        ValueError: the value file's size disagrees with the resolved layout, or
            a meta dtype field is not a torch dtype.
    """
    emb_dtype = (
        dtype_from_meta_name(meta_data[META_EMBEDDING_DTYPE], META_EMBEDDING_DTYPE)
        if META_EMBEDDING_DTYPE in meta_data
        else EMBEDDING_TYPE
    )
    opt_state_dtype = (
        dtype_from_meta_name(meta_data[META_OPT_STATE_DTYPE], META_OPT_STATE_DTYPE)
        if META_OPT_STATE_DTYPE in meta_data
        else OPT_STATE_TYPE
    )

    emb_bytes = os.path.getsize(embedding_file_path)
    ckpt_dim = meta_data.get(META_EMBEDDING_DIM, None)
    if ckpt_dim is None:
        if num_keys == 0:
            ckpt_dim = runtime_dim
        else:
            key_block = num_keys * emb_dtype.itemsize
            if emb_bytes % key_block != 0:
                raise ValueError(
                    f"Embedding file {embedding_file_path} size {emb_bytes} is not "
                    f"divisible by {key_block} (num_keys={num_keys}, dtype "
                    f"{emb_dtype}), so its row width cannot be recovered. This "
                    "checkpoint predates "
                    f"{META_EMBEDDING_DIM!r}/{META_EMBEDDING_DTYPE!r} in the meta "
                    "JSON, so it is assumed to be fp32."
                )
            ckpt_dim = emb_bytes // key_block
    ckpt_dim = int(ckpt_dim)

    expected_bytes = num_keys * ckpt_dim * emb_dtype.itemsize
    if emb_bytes != expected_bytes:
        raise ValueError(
            f"Embedding file {embedding_file_path} holds {emb_bytes} bytes but "
            f"{expected_bytes} were expected for {num_keys} keys of {ckpt_dim} "
            f"{emb_dtype} elements."
        )
    return emb_dtype, opt_state_dtype, ckpt_dim


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    """Little-endian raw bytes of *tensor*, for any dtype it can hold.

    Views as uint8 instead of going through ``numpy().tobytes()`` because numpy
    has no bfloat16, and value files now carry the table's own precision.
    """
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(
    buf: bytes, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Decode *buf* as a 1-D tensor of *dtype* on *device*.

    ``torch.frombuffer`` rather than ``np.frombuffer`` for the same bfloat16
    reason as :func:`_raw_bytes`. The ``bytearray`` copy makes the buffer writable
    so torch does not warn about aliasing read-only memory; the ``to(device)``
    would copy regardless.
    """
    if len(buf) == 0:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.frombuffer(bytearray(buf), dtype=dtype).to(device)


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


def create_cache_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    capacities: List[int],
) -> DynamicEmbTableState:
    """The value side of a cache: one fixed row pool per table, addressed by node.

    The same state the kernels that gather and update rows read (value buffers and
    their pointers, per-table dims, dtype, optimizer), without a key index: the
    cache keeps its own, in :class:`EmbeddingCacheCore`.
    """
    if not options:
        raise ValueError("options must be non-empty")
    assert len(capacities) == len(options)

    base_opt = options[0]
    num_tables = len(options)
    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")
    score_policy = get_score_policy(base_opt.score_strategy)

    dims = [opt.dim for opt in options]
    value_dims = [d + optimizer.get_state_dim(d) for d in dims]
    values = MixedDimBuffers(
        dims, value_dims, capacities, base_opt.embedding_dtype, device, "fixed"
    )

    props = torch.cuda.get_device_properties(device_idx)
    threads_in_wave = props.multi_processor_count * props.max_threads_per_multi_processor

    return DynamicEmbTableState(
        options_list=options,
        num_tables=num_tables,
        device=device,
        score_policy=score_policy,
        incremental_score_name=score_policy.name,
        evict_strategy=base_opt.evict_strategy.value,
        key_index_map=None,
        capacity=sum(capacities),
        values=values,
        optimizer=optimizer,
        threads_in_wave=threads_in_wave,
        score=None,
        training=False,
    )


def _all_gather_dumped_columns(
    columns: List[Tensor],
    pg: dist.ProcessGroup,
) -> List[Tensor]:
    """Gather a set of row-aligned dump columns from all ranks into CPU tensors.

    Every tensor in *columns* must live on the same device and share ``size(0)``
    (the per-rank dumped-key count); trailing dimensions may differ (e.g. keys
    ``(N,)``, values ``(N, D)``, scores ``(N, W)``). Returns the same columns with
    all ranks' rows concatenated in rank order, on host and still row-aligned.

    NCCL has no variable-length gather, so this is one size all_gather followed by
    a padded all_gather per column.
    """
    assert columns, "_all_gather_dumped_columns needs at least one column"
    device = columns[0].device
    world_size = dist.get_world_size(group=pg)
    n = columns[0].size(0)
    d_count = torch.tensor([n], dtype=torch.long, device=device)
    gathered_counts = [torch.empty_like(d_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, d_count, group=pg)
    counts = [int(c.item()) for c in gathered_counts]
    max_n = max(counts)

    out: List[Tensor] = []
    for col in columns:
        padded = torch.zeros(
            (max_n,) + tuple(col.shape[1:]), dtype=col.dtype, device=device
        )
        if n > 0:
            padded[:n] = col
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded, group=pg)
        out.append(
            torch.cat(
                [gathered[i][: counts[i]] for i in range(world_size)], dim=0
            ).cpu()
        )
    return out


def _split_value_row(
    state: DynamicEmbTableState, table_id: int, values: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a loaded ``[N, value_dim]`` row into (embedding, optimizer state).

    ``MixedDimBuffers.load_table`` returns the table's own compact row, so the
    optimizer state is the trailing ``value_dim - emb_dim`` columns. That runtime
    width is not all payload: rowwise Adagrad reserves a fixed 16 bytes per row
    in the fused FBGEMM layout but only ever fills one accumulator scalar. The
    dumped block is therefore narrowed to the width the file checkpoint uses --
    the same ``optimizer.states_for_checkpoint`` ``_dump_table``
    applies -- so a delta and a checkpoint describe a row identically, and the
    padding is not paid for on every dump. Replay expands it back with
    ``optimizer.states_from_checkpoint``.

    Tables whose optimizer keeps no per-row state (e.g. plain SGD) get ``None``
    rather than a zero-width tensor, matching :func:`export_keys_values_iter`.
    """
    emb_dim = state.values.emb_dims_list[table_id]
    optim_state_dim = state.values.value_dims_list[table_id] - emb_dim
    emb = values[:, :emb_dim].to(dtype=state.values.dtype)
    if optim_state_dim <= 0:
        return emb, None
    opt = values[:, -optim_state_dim:]
    return (
        emb,
        state.optimizer.states_for_checkpoint(opt, emb_dim).contiguous(),
    )


def _flat_row_indices_from_slots_and_scores(
    state: DynamicEmbTableState,
    slot_indices: torch.Tensor,
    stored_scores: torch.Tensor,
) -> torch.Tensor:
    """Flat-buffer row indices for load/store after export or incremental_dump.

    The index hands out rows, so what an export or a dump calls a slot index is a row.
    """
    return slot_indices.to(device=state.device, dtype=torch.int64)


def _fresh_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    value_rows: torch.Tensor,
    timestamp: int,
    insert_score: int,
) -> torch.Tensor:
    """``[N, num_scores]`` physical score block for keys written by replay.

    Used when the caller did not ask for ``ReplayContent.SCORE``. The delta does
    carry the source's scores, but they were not loaded, so a replayed key is
    scored as if it had just been inserted here instead: recency words get
    *timestamp*, every other word gets *insert_score*. The replica then ranks its
    restored keys by when it received them rather than by how the source ranked
    them -- acceptable because the score only orders future evictions, and a
    replica that evicts on its own schedule still holds correct embeddings.
    Ask for ``SCORE`` to inherit the source's ranking instead.

    *insert_score* is passed in rather than read off ``state.score``: that field
    is populated by the forward pass, and replay is the one write path that can
    run before a target has ever done a forward. The caller owns the table's
    score bookkeeping and always has a value.

    """
    device = state.device
    n = value_rows.numel()
    num_scores = state.key_index_map.num_scores_
    # Zeros, not ``empty``: the loop below fills one column per configured
    # strategy, while the width comes from the score policy. The two agree for
    # every strategy there is, but they are derived from different places, so a
    # future divergence should leave a quiet zero rather than whatever the
    # allocator handed back.
    block = torch.zeros((n, num_scores), device=device, dtype=SCORE_TYPE)
    physical = get_physical_score_order(state.options_list[table_id].score_strategy)
    for word, strategy in enumerate(physical):
        block[:, word] = (
            timestamp if strategy == DynamicEmbScoreStrategy.TIMESTAMP else insert_score
        )
    return block


def _dump_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    slot_indices: torch.Tensor,
    primary_scores: torch.Tensor,
    cur_ts: int,
) -> torch.Tensor:
    """``[N, num_scores]`` per-key score block for a dumped table.

    Columns are in the user's configured (logical) order -- the same order the
    file checkpoint uses -- and timestamp columns are converted to an age
    relative to *cur_ts* so they survive the trip to another device or host.
    Single-score tables reuse the score column the dump already thresholded on;
    multi-word layouts gather every word from the slots.
    """
    device = state.device
    num_scores = state.key_index_map.num_scores_
    score_strategy = state.options_list[table_id].score_strategy
    if num_scores > 1:
        block = state.key_index_map.gather_score_blocks(
            table_id, slot_indices.to(device=device, dtype=torch.int64)
        )
        perm = score_dump_permutation(score_strategy)
        if perm != list(range(block.size(1))):
            block = block[:, perm].contiguous()
    else:
        block = primary_scores.to(device=device, dtype=SCORE_TYPE).view(-1, 1)
    return _scores_to_age(score_strategy, block, cur_ts)


def _replay_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    scores: torch.Tensor,
    target_ts: int,
) -> torch.Tensor:
    """Inverse of :func:`_dump_score_block`: logical ages -> physical scores.

    Takes the ``[N, num_scores]`` logical block carried in the delta, restores
    timestamp columns against this device's *target_ts*, and permutes back into
    the physical device layout.

    Both steps index by *column*, so a block that is not exactly as wide as this
    table's score layout would silently transform or permute the wrong words --
    e.g. writing a frequency into the timestamp column. The width is validated
    here rather than trusted.
    """
    device = state.device
    score_strategy = state.options_list[table_id].score_strategy
    num_scores = state.key_index_map.num_scores_
    block = scores.to(device=device, dtype=SCORE_TYPE)
    if block.dim() == 1:
        block = block.view(-1, 1)
    if block.dim() != 2 or block.size(1) != num_scores:
        raise ValueError(
            f"replay_increment: table {table_id} has {num_scores} score word(s) "
            f"per key, but the delta carries a score block of shape "
            f"{tuple(scores.shape)}."
        )
    block = _age_to_scores(score_strategy, block, target_ts)
    perm = score_load_permutation(score_strategy)
    if perm != list(range(num_scores)):
        block = block[:, perm]
    return block.contiguous()


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


def export_keys_values_iter(
    state: DynamicEmbTableState,
    device: torch.device,
    batch_size: int = 65536,
    table_id: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    """Export keys, embeddings, opt_states, scores for a logical table.

    Embeddings and optimizer states come out at the table's own precision
    (``state.values.dtype``), matching what ``incremental_dump`` yields, so a
    checkpoint stores what the table actually holds instead of an fp32 widening
    of it. Callers that want fp32 cast on their own.

    NO_EVICTION tables load flat values by stored score (logical row index), not by
    hash slot ``indices`` from export.
    """
    emb_dim_t = state.values.emb_dims_list[table_id]
    vdim = state.values.value_dims_list[table_id]
    optim_state_dim = vdim - emb_dim_t

    for (
        keys,
        named_scores,
        indices,
    ) in state.key_index_map._batched_export_keys_scores(
        [state.score_policy.name],
        state.device,
        batch_size=batch_size,
        return_index=True,
        table_id=table_id,
    ):
        scores = named_scores[state.score_policy.name]
        flat_rows = _flat_row_indices_from_slots_and_scores(state, indices, scores)
        values = state.values.load_table(flat_rows, table_id)
        embeddings = values[:, :emb_dim_t].contiguous()
        if optim_state_dim != 0:
            opt_states = values[:, -optim_state_dim:].contiguous().to(device)
        else:
            opt_states = None
        # Multi-word score layouts (e.g. LruLfu: timestamp + frequency) must
        # persist ALL words, not just the exported column, so the checkpoint can
        # restore them exactly. Gather the full [N, num_scores] block at the
        # exported slots; single-score tables keep the [N] shape unchanged.
        if state.key_index_map.num_scores_ > 1:
            out_scores = state.key_index_map.gather_score_blocks(table_id, indices)
            # Reorder the physical device layout into the user's configured
            # (logical) score-column order so both the checkpoint and callers of
            # export_keys_values see scores in the order they configured. Identity
            # when the logical order matches the physical layout.
            dump_perm = score_dump_permutation(
                state.options_list[table_id].score_strategy
            )
            if dump_perm != list(range(out_scores.size(1))):
                out_scores = out_scores[:, dump_perm].contiguous()
        else:
            out_scores = scores.to(SCORE_TYPE)
        yield (
            keys.to(device),
            embeddings.to(device),
            opt_states,
            out_scores.to(device),
        )


def _dump_table(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: str,
    opt_file_path: str,
    include_optim: bool,
    include_meta: bool,
    timestamp: int,
    current_score: Optional[int] = None,
    append: bool = False,
) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if not append and include_meta:
        meta_data = {}
        meta_data.update(state.optimizer.get_opt_args())
        meta_data["evict_strategy"] = str(state.evict_strategy)
        meta_data["dist_type"] = state.options_list[table_id].dist_type
        # How to read the value files back. They carry no header, so without
        # these a loader can only assume the historical fp32 and guess the row
        # width from the file length. Note this dict is rebuilt and rewritten
        # whole on every dump (by rank 0 only, and not by the appending second
        # tier of a hybrid dump): anything a loader needs has to be contributed
        # here, because there is no later pass that merges into the file.
        meta_data[META_EMBEDDING_DTYPE] = dtype_to_meta_name(state.values.dtype)
        meta_data[META_EMBEDDING_DIM] = int(state.values.emb_dims_list[table_id])
        # Recorded separately from the embedding dtype because the two can
        # differ: rowwise Adagrad keeps its accumulator in fp32 in any table.
        meta_data[META_OPT_STATE_DTYPE] = dtype_to_meta_name(
            state.optimizer.get_state_dtype(state.values.dtype)
        )

        if current_score is not None:
            meta_data["step_score"] = current_score

        save_to_json(meta_data, meta_json_file_path)

    mode = "ab" if append else "wb"
    fkey = open(emb_key_path, mode)
    fembedding = open(embedding_file_path, mode)
    fscore = open(score_file_path, mode)
    fopt_states = open(opt_file_path, mode) if include_optim else None

    for keys, embeddings, opt_states_batch, scores in export_keys_values_iter(
        state, device=device, table_id=table_id
    ):
        fkey.write(keys.cpu().numpy().tobytes())
        fembedding.write(_raw_bytes(embeddings))
        if state.evict_strategy == EvictStrategy.KLru:
            scores = timestamp - scores
        fscore.write(scores.cpu().numpy().tobytes())
        if fopt_states and opt_states_batch is not None:
            to_write = state.optimizer.states_for_checkpoint(
                opt_states_batch,
                state.values.emb_dims_list[table_id],
            )
            fopt_states.write(_raw_bytes(to_write))

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


def _iter_batches_from_files(
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    dim: int,
    optstate_dim: int,
    device: torch.device,
    dist_type: str,
    batch_size: int = 65536,
    num_scores: int = 1,
    emb_dtype: torch.dtype = EMBEDDING_TYPE,
    opt_state_dtype: torch.dtype = OPT_STATE_TYPE,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Yield (keys, embeddings, scores, opt_states) batches from checkpoint files.

    *shard_index* / *num_shards* are the table's own shard geometry (its position
    among the ranks that hold it, and how many there are): the world for a
    row-wise table, one node for a table-row-wise one. Left ``None`` they fall
    back to the global rank and world size, which is the row-wise answer.

    ``num_scores`` is the number of score words per key in the score file. When
    > 1 (e.g. LruLfu's timestamp+frequency) the yielded ``scores`` is
    [n, num_scores]; otherwise it is [n].

    *emb_dtype* / *opt_state_dtype* are the precisions the files were written at,
    which is how far apart the batches step through them -- get them wrong and
    the files misparse rather than fail. Callers holding the table's meta should
    pass what :func:`resolve_checkpoint_value_layout` resolved; the fp32 defaults
    are what every checkpoint written before the meta recorded a dtype used.
    The yielded tensors keep the file's dtype; the insert path casts to the
    table's.

    Handles file I/O, deserialization, and keeping only the keys this rank owns.

    *dist_type* selects the ownership rule, which is the table's own -- the load
    path checks it against the checkpoint's before getting here
    (``_validate_load_meta``). It matters in two ways. When the checkpoint's
    shard count matches the world size each rank reads only its own file, every
    key in it already belongs to that rank, and the filter is a no-op -- but
    only if the rule is the right one. When the counts differ the files are
    re-split across ranks and the filter is what does the resharding.
    Pass *score_file_path* / *opt_file_path* as ``None`` to skip those files.
    """
    fkey = open(emb_key_path, "rb")
    fembedding = open(embedding_file_path, "rb")
    fscore = (
        open(score_file_path, "rb")
        if score_file_path and os.path.exists(score_file_path)
        else None
    )
    fopt = open(opt_file_path, "rb") if opt_file_path else None
    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

    if num_shards is None or shard_index is None:
        num_shards = dist.get_world_size() if dist.is_initialized() else 1
        shard_index = dist.get_rank() if dist.is_initialized() else 0

    try:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(KEY_TYPE.itemsize * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            emb_bytes = fembedding.read(emb_dtype.itemsize * dim * n)
            embeddings = _tensor_from_bytes(emb_bytes, emb_dtype, device).view(-1, dim)

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * n * num_scores)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )
                if num_scores > 1:
                    scores = scores.view(-1, num_scores)

            opt_states = None
            if fopt:
                opt_bytes = fopt.read(opt_state_dtype.itemsize * optstate_dim * n)
                opt_states = _tensor_from_bytes(
                    opt_bytes, opt_state_dtype, device
                ).view(-1, optstate_dim)

            masks = owned_key_mask(keys, shard_index, num_shards, dist_type)
            if masks is not None:
                keys = keys[masks]
                embeddings = embeddings[masks]
                if scores is not None:
                    scores = scores[masks]
                if opt_states is not None:
                    opt_states = opt_states[masks]

            yield keys, embeddings, scores, opt_states
    finally:
        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt:
            fopt.close()


@dataclass
class _LoadParams:
    meta_data: Dict[str, Any]
    dim: int
    runtime_optstate_dim: int
    file_optstate_dim: int
    include_optim: bool
    num_keys: int
    # The table's key -> rank rule, already checked against the checkpoint's.
    dist_type: str
    # Precisions the value files were written at, resolved from meta (fp32 for
    # checkpoints predating it). Pass both to ``_iter_batches_from_files``.
    emb_dtype: torch.dtype = EMBEDDING_TYPE
    opt_state_dtype: torch.dtype = OPT_STATE_TYPE


def _validate_load_meta(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    include_optim: bool,
) -> _LoadParams:
    """Shared validation for checkpoint loading.

    Reads meta JSON, validates opt_type / evict_strategy / dist_type, resolves
    the value files' precision and row width, resolves include_optim, and checks
    file-size consistency.
    """
    meta_data = load_from_json(meta_json_file_path)
    opt_type = meta_data.get("opt_type", None)
    # The optimizer checks only matter when the caller asked for its states.
    if (
        include_optim
        and opt_type
        and state.optimizer.get_opt_args().get("opt_type", None) != opt_type
    ):
        include_optim = False
        print(
            f"Optimizer type mismatch: {opt_type} != {state.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
        )

    evict_strategy = meta_data.get("evict_strategy", None)
    if evict_strategy and str(state.evict_strategy) != evict_strategy:
        raise ValueError(
            f"Evict strategy mismatch: {evict_strategy} != {state.evict_strategy}"
        )
    ckpt_dist_type = meta_data.get("dist_type", "roundrobin")
    runtime_dist_type = state.options_list[table_id].dist_type
    if runtime_dist_type != ckpt_dist_type:
        raise ValueError(
            "Input dist_type mismatch: checkpoint was dumped with "
            f"{ckpt_dist_type!r}, but runtime table is configured with "
            f"{runtime_dist_type!r}. Please load with a matching dist_type."
        )

    if score_file_path is None:
        print(
            f"Score file {score_file_path} does not exist. Will not load score states."
        )

    if include_optim and (not opt_file_path or not os.path.exists(opt_file_path)):
        include_optim = False
        print(
            f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
        )

    dim = state.values.emb_dims_list[table_id]
    runtime_optstate_dim = state.optimizer.get_state_dim(dim)

    if runtime_optstate_dim == 0:
        include_optim = False

    if include_optim:
        state.optimizer.set_opt_args(meta_data)

    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize
    emb_dtype, opt_state_dtype, ckpt_emb_dim = resolve_checkpoint_value_layout(
        meta_data, embedding_file_path, num_keys, dim
    )
    if ckpt_emb_dim != dim:
        raise ValueError(
            f"Embedding dim mismatch: {embedding_file_path} holds rows of "
            f"{ckpt_emb_dim} but the runtime table is configured with dim {dim}."
        )
    if emb_dtype != state.values.dtype:
        # Not fatal: the insert path casts, so this is a deliberate precision
        # migration as often as it is a mistake. Say which way it goes, since
        # narrowing loses bits that the checkpoint still has.
        warnings.warn(
            f"Checkpoint {embedding_file_path} stores {emb_dtype} embeddings but "
            f"the runtime table is {state.values.dtype}; values will be converted on "
            "load.",
            UserWarning,
            stacklevel=2,
        )
    if score_file_path and os.path.exists(score_file_path):
        # The score file holds num_scores words per key (e.g. LruLfu = 2:
        # timestamp + frequency), so compare against num_keys * num_scores.
        per_key = state.key_index_map.num_scores_
        num_score_words = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
        if num_score_words != num_keys * per_key:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
            )

    file_optstate_dim = 0
    if include_optim and opt_file_path and os.path.exists(opt_file_path):
        file_bytes = os.path.getsize(opt_file_path)
        if num_keys == 0:
            if file_bytes != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} is non-empty but key file has no keys."
                )
        else:
            row_block = num_keys * opt_state_dtype.itemsize
            if file_bytes % row_block != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} size {file_bytes} is not divisible by "
                    f"{row_block} (num_keys={num_keys}, dtype itemsize={opt_state_dtype.itemsize})."
                )
            file_optstate_dim = file_bytes // row_block
            ckpt_dim = state.optimizer.get_ckpt_state_dim(dim)
            if file_optstate_dim != ckpt_dim:
                raise ValueError(
                    f"Optimizer state width in checkpoint is {file_optstate_dim}; expected "
                    f"{ckpt_dim}."
                )

    return _LoadParams(
        meta_data=meta_data,
        dim=dim,
        runtime_optstate_dim=runtime_optstate_dim,
        file_optstate_dim=file_optstate_dim,
        include_optim=include_optim,
        num_keys=num_keys,
        dist_type=runtime_dist_type,
        emb_dtype=emb_dtype,
        opt_state_dtype=opt_state_dtype,
    )


def _load_key_values(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
    table_id: int = 0,
) -> None:
    dim = embeddings.size(1)
    emb_dim_cfg = state.values.emb_dims_list[table_id]
    runtime_optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    # Normalize the embeddings to the table's precision up front. A checkpoint
    # may carry a different precision than the table (load converts -- see
    # _validate_load_meta). The optimizer state is deliberately left alone:
    # optimizer.states_from_checkpoint owns its conversion, because it need not
    # be the table's dtype (rowwise Adagrad keeps an fp32 accumulator), and an
    # elementwise cast here would round that accumulator away. The branch below
    # only calls it when the state is non-empty, which is the case it casts.
    embeddings = embeddings.to(state.values.dtype)

    if opt_states is None and runtime_optstate_dim > 0:
        opt_states = torch.empty(
            keys.numel(),
            runtime_optstate_dim,
            dtype=state.values.dtype,
            device=embeddings.device,
        )
        state.optimizer.reset_optimizer_states(opt_states, emb_dims=emb_dim_cfg)
    elif opt_states is not None and runtime_optstate_dim > 0:
        opt_states = state.optimizer.states_from_checkpoint(
            opt_states,
            emb_dim_cfg,
            state.values.dtype,
            embeddings.device,
        )

    # Both halves are already state.values.dtype (normalized above), so this cat
    # cannot silently widen the row via type promotion.
    values = (
        torch.cat(
            [embeddings.view(-1, dim), opt_states.view(-1, runtime_optstate_dim)],
            dim=-1,
        )
        if opt_states is not None
        else embeddings
    )

    # Multi-word score layouts (LruLfu: timestamp + frequency) cannot be restored
    # by a single-value insert (which only writes word 0). Place the keys without
    # touching scores (CONST), then scatter the full [N, num_scores] block.
    num_scores = state.key_index_map.num_scores_
    if num_scores > 1:
        # Validate the score-block shape before scatter_score_blocks writes it
        # into device memory: a malformed checkpoint (e.g. a 1-D score tensor)
        # would otherwise produce out-of-bounds / wrong writes. Use ValueError
        # (not assert) so the check survives `python -O`.
        if (
            scores is None
            or scores.dim() != 2
            or scores.size(1) != num_scores
            or scores.size(0) != keys.numel()
        ):
            raise ValueError(
                f"multi-word load expects [{keys.numel()}, {num_scores}] scores, "
                f"got {None if scores is None else tuple(scores.shape)}"
            )
        tid_tensor = torch.full(
            (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
        )
        place_arg = ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )
        indices = state.key_index_map.insert(keys, tid_tensor, place_arg)
        state.key_index_map.scatter_score_blocks(
            table_id, indices, scores.to(SCORE_TYPE).contiguous()
        )
        state.values.store_table(indices, table_id, values)
        return

    policy = ScorePolicy.ASSIGN
    tid_tensor = torch.full(
        (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
    )

    if scores is None:
        assert (
            state.evict_strategy == EvictStrategy.KLru
        ), "scores is None for KLru evict strategy is allowed but will be deprecated in future."
        policy = ScorePolicy.GLOBAL_TIMER
    else:
        scores = scores.to(SCORE_TYPE)

    score_arg_insert = ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=policy,
    )

    indices = state.key_index_map.insert(keys, tid_tensor, score_arg_insert)
    state.values.store_table(indices, table_id, values)


# ---------------------------------------------------------------------------
# Replay – write an incremental_dump delta back into a table
# ---------------------------------------------------------------------------


def _replay_write_values(
    state: DynamicEmbTableState,
    table_id: int,
    rows: torch.Tensor,
    embeddings: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    keep_optimizer: torch.Tensor,
    content: ReplayContent,
) -> None:
    """Write a replayed row's value columns, per :class:`ReplayContent`.

    A value row is an embedding followed by the optimizer state, and the two are
    written together or not at all -- ``MixedDimBuffers.store_table`` copies from
    the row base, so there is no way to write the tail without the head. The
    embedding is therefore always written, and the only choice is what the tail
    gets:

    - **With ``OPTIMIZER_STATE``**, the delta carries the whole row; write it as
      it was dumped. Nothing is inferred.
    - **Without it**, rows that already belonged to this same key keep their
      state, because writing just the embedding columns leaves the tail
      untouched. Every other row is new to this key, so its tail still holds the
      previous occupant's moments and has to be reset by the optimizer.
    """
    if rows.numel() == 0:
        return
    emb_dim_cfg = state.values.emb_dims_list[table_id]
    optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    embeddings = embeddings.to(dtype=state.values.dtype)
    want_opt = ReplayContent.OPTIMIZER_STATE in content and optstate_dim > 0

    if optstate_dim == 0:
        state.values.store_table(rows, table_id, embeddings)
        return

    if want_opt:
        if optimizer_states is None:
            raise ValueError(
                f"replay_increment: table {table_id} keeps optimizer state per "
                "row, but the delta carries none. Drop "
                "ReplayContent.OPTIMIZER_STATE, or replay a delta dumped from a "
                "table with the same optimizer."
            )
        ckpt_dim = state.optimizer.get_ckpt_state_dim(emb_dim_cfg)
        if optimizer_states.dim() != 2 or optimizer_states.size(1) != ckpt_dim:
            raise ValueError(
                f"replay_increment: table {table_id} dumps {ckpt_dim} "
                f"optimizer-state column(s) per row, but the delta carries a "
                f"block of shape {tuple(optimizer_states.shape)}."
            )
        # Back to the runtime width the fused value row expects.
        opt = state.optimizer.states_from_checkpoint(
            optimizer_states.to(device=embeddings.device),
            emb_dim_cfg,
            state.values.dtype,
            embeddings.device,
        )
        state.values.store_table(rows, table_id, torch.cat([embeddings, opt], dim=-1)
        )
        return

    keep = keep_optimizer
    if bool(keep.any()):
        state.values.store_table(rows[keep], table_id, embeddings[keep])
    fresh = torch.logical_not(keep)
    if bool(fresh.any()):
        fresh_emb = embeddings[fresh]
        opt_states = torch.empty(
            (fresh_emb.size(0), optstate_dim),
            dtype=state.values.dtype,
            device=fresh_emb.device,
        )
        state.optimizer.reset_optimizer_states(opt_states, emb_dims=emb_dim_cfg)
        state.values.store_table(rows[fresh],
            table_id,
            torch.cat([fresh_emb, opt_states], dim=-1),
        )


class DynamicEmbCache(Cache):
    """The GPU cache: placement and reference protection, from ``EmbeddingCacheCore``.

    It holds no score and decides nothing about which keys stay in the system.
    The rows live in ``_state.values``, one fixed pool per table addressed by node,
    and everything that reads or writes them goes through the same
    ``values.ptrs`` the backing's rows do. See docs/lightweight_cache_design.md.
    """

    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        # Rounded up to a multiple of the bucket capacity, as the cache's index
        # always was. A cache has to hold every key of a step at once, so a
        # configuration smaller than a step would otherwise fail where it used to
        # work.
        capacities = [
            align_to_table_size(int(opt.init_capacity), int(opt.bucket_capacity))
            for opt in options
        ]
        self._state = create_cache_state(options, optimizer, capacities)
        self._core = EmbeddingCacheCore(capacities)
        self._cache_metrics = torch.zeros(10, dtype=torch.long, device="cpu")
        self._record_cache_metrics = False

    # -- Cache interface --

    def lookup_pin(
        self, unique_keys: torch.Tensor, offsets: List[int]
    ) -> torch.Tensor:
        nodes = self._core.lookup_pin(unique_keys.to(KEY_TYPE), offsets)
        if self._record_cache_metrics:
            self._cache_metrics[0] = unique_keys.size(0)
            self._cache_metrics[1] = int((nodes >= 0).sum().item())
        return nodes

    def allocate(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        offsets: List[int],
        nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        victim_keys, has_victim, new_nodes = self._core.allocate(
            keys.to(KEY_TYPE), positions, offsets, nodes
        )
        if self._record_cache_metrics:
            self._cache_metrics[2] = keys.numel()
            self._cache_metrics[3] = int(has_victim.sum().item())
        return victim_keys, has_victim.to(torch.bool), new_nodes

    def publish(self, new_nodes: torch.Tensor, offsets: List[int]) -> None:
        self._core.publish(new_nodes, offsets)

    def release(self, nodes: torch.Tensor, offsets: List[int]) -> None:
        self._core.release(nodes, offsets)

    def find_nodes(
        self, keys: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return self._core.find_nodes(keys.to(KEY_TYPE), table_ids)

    def invalidate(self, nodes: torch.Tensor, table_ids: torch.Tensor) -> None:
        self._core.invalidate(nodes, table_ids)

    def resident(self, table_id: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Every resident node of one table, its key, and how many nodes still
        hold a reference."""
        return self._core.resident(table_id)

    def num_resident(self, table_id: int) -> int:
        return self._core.num_resident(table_id)

    def invalidate_keys(self, keys: torch.Tensor, table_id: int) -> None:
        """Drop these keys from the cache if they are in it, whatever they hold."""
        if keys.numel() == 0:
            return
        tids = torch.full(
            (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
        )
        self.invalidate(self.find_nodes(keys, tids), tids)

    def reset(self) -> None:
        self._core.reset()

    @property
    def eviction_guard(self) -> Tuple[int, int]:
        """What the backing's insert asks before evicting a key: a device pointer
        to this cache's per-table maps, and how many tables."""
        return self._core.guard()

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record

    # -- Convenience accessors --

    @property
    def num_tables(self) -> int:
        return self._state.num_tables

    def embedding_dtype(self) -> torch.dtype:
        return self._state.values.dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.values.emb_dims_list[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.values.value_dims_list[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.values.max_emb_dim

    def max_value_dim(self) -> int:
        return self._state.values.max_value_dim

    def init_optimizer_state(self) -> float:
        return self._state.optimizer.get_initial_optimizer_state()

    def evict_strategy(self) -> EvictStrategy:
        return self._state.evict_strategy

    def capacity(self, table_id: int) -> int:
        """How many rows table ``table_id`` has."""
        return self._core.capacity(table_id)

    def size(self) -> int:
        return sum(self.num_resident(t) for t in range(self.num_tables))


# ---------------------------------------------------------------------------
# Storage interface helpers shared by the tiers
# ---------------------------------------------------------------------------


def _fill_from_lower(
    lower: Storage,
    copy_mode: CopyMode,
    values: torch.Tensor,
    num_missing: int,
    missing_keys: torch.Tensor,
    missing_positions: torch.Tensor,
    missing_table_ids: torch.Tensor,
    missing_scores: Optional[torch.Tensor],
    founds: torch.Tensor,
    scores: torch.Tensor,
):
    """Ask ``lower`` for the keys a table does not have.

    What it finds is written into ``values`` (at the keys' positions) and marked
    found; the rest stays missing. Returns the missing description and ``founds``
    and ``scores`` as they are afterwards. Only the found mask, the scores and
    the values of ``lower.find`` are used, so ``lower`` need not describe its
    misses the way a table does.
    """
    out = lower.find(missing_keys, missing_table_ids, copy_mode, missing_scores)
    l_founds, l_scores, l_values = out[5], out[6], out[7]
    if not bool(l_founds.any()):
        return (
            num_missing,
            missing_keys,
            missing_positions,
            missing_table_ids,
            missing_scores,
            founds,
            scores,
        )
    hit = missing_positions[l_founds]
    values[hit] = l_values[l_founds]
    founds = founds.clone()
    founds[hit] = True
    scores = scores.clone()
    scores[hit] = l_scores[l_founds].to(scores.dtype)
    keep = ~l_founds
    return (
        int(keep.sum().item()),
        missing_keys[keep],
        missing_positions[keep],
        missing_table_ids[keep],
        missing_scores[keep] if missing_scores is not None else None,
        founds,
        scores,
    )


def flush_cache(cache: DynamicEmbCache, storage) -> None:
    """Write every resident row back to the storage.

    Every resident row is written whatever its references: a key that has never
    left the cache is the one whose storage copy is furthest behind. The storage
    already has a row for every cached key (a key enters it when it is admitted),
    so the rows are written in place and nothing is inserted.
    """
    storage.flush_from(cache)


# ---------------------------------------------------------------------------
# eval_lookup – unified eval path for storage-only and cache+storage
# ---------------------------------------------------------------------------


def _eval_lookup_storage(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
    (
        h_num_missing,
        _,
        missing_indices,
        _,
        _,
        _,
        _,
        embs,
    ) = storage.find(
        keys,
        table_ids,
        copy_mode=CopyMode.EMBEDDING,
    )

    if h_num_missing > 0:
        initializer(embs, keys, table_ids, missing_indices)

    return embs


def _eval_lookup_cached(
    cache: Cache,
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
    nodes = cache.find_nodes(keys, table_ids)
    founds = nodes >= 0

    embs = cache._state.values.load(nodes.to(torch.int64), table_ids, copy_mode=CopyMode.EMBEDDING
    )

    missing_mask = ~founds
    h_num_miss, miss_compact_idx, (missing_keys, missing_table_ids) = flagged_compact(
        missing_mask, [keys, table_ids]
    )

    if h_num_miss == 0:
        return embs

    (
        h_num_missing_in_storage,
        _,
        missing_indices_in_storage,
        _,
        _,
        _,
        _,
        storage_embs,
    ) = storage.find(
        missing_keys,
        missing_table_ids,
        copy_mode=CopyMode.EMBEDDING,
    )

    if h_num_missing_in_storage > 0:
        initializer(
            storage_embs, missing_keys, missing_table_ids, missing_indices_in_storage
        )

    embs[miss_compact_idx, :] = storage_embs

    return embs


def eval_lookup(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
    cache: Optional[Cache] = None,
) -> torch.Tensor:
    """Eval-only lookup (no insertion, no admission, no backward).

    When *cache* is ``None``, looks up directly from *storage*.
    When *cache* is provided, looks up from cache first, then falls back to
    *storage* for cache misses.  Only embedding columns are copied
    (``CopyMode.EMBEDDING``); optimizer states are never touched.

    Returns the embedding tensor of shape ``[len(keys), emb_dim]``.
    """
    assert keys.dim() == 1
    if keys.numel() == 0:
        return torch.empty(
            0,
            storage.max_embedding_dim(),
            dtype=storage.embedding_dtype(),
            device=keys.device,
        )

    if cache is None:
        return _eval_lookup_storage(storage, keys, table_ids, initializer)

    return _eval_lookup_cached(
        cache,
        storage,
        keys,
        table_ids,
        initializer,
    )
