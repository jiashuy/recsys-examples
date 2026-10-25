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

from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import List, Optional, Tuple

import torch
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    Cache,
    Storage,
    eval_lookup,
)
from dynamicemb.chained_storage import ChainedStorage
from dynamicemb.lookup_layout import EmbeddingLayout
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode, MultiTableAdmitter
from dynamicemb_extensions import (
    EvictStrategy,
    expand_table_ids_cuda,
    flagged_compact,
    gather_embedding,
    gather_embedding_pooled,
    get_table_range,
    reduce_grads,
    segmented_unique_cuda,
)


def _copy_cuda_tensor_to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        return tensor.detach().cpu()
    cpu = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=torch.device("cpu"),
        pin_memory=True,
    )
    cpu.copy_(tensor.detach(), non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return cpu


def _scalar_item(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.numel() != 1:
        raise ValueError(f"expected scalar tensor, got shape {tuple(tensor.shape)}")
    return _copy_cuda_tensor_to_pinned_cpu(tensor.reshape(1))[0].item()


def _bool_item(tensor: torch.Tensor) -> bool:
    return bool(_scalar_item(tensor))


def segmented_unique(
    keys: torch.Tensor,
    segment_range: torch.Tensor,
    evict_strategy: Optional[EvictStrategy] = None,
    frequency_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform segmented unique operation on keys with segment_range.

    This function deduplicates keys within each table segment, using the
    GPU-accelerated segmented_unique_cuda kernel.

    Args:
        keys: Input key tensor (int64 or uint64)
        segment_range: Table boundary offsets where segment_range[i] is the
                       start index for table i (int64)
        evict_strategy: Optional eviction strategy (for LFU mode)
        frequency_counts: Optional input frequency counts per key

    Returns:
        Tuple of (unique_keys, reverse_indices, unique_keys_table_range,
                  output_scores, unique_size_per_table).
        unique_size_per_table is a CPU tensor of shape (num_tables,) with
        per-table unique counts; total unique count is its sum or
        unique_keys_table_range[-1].
    """
    with torch.cuda.nvtx.range("op:segmented_unique"):
        num_keys = keys.size(0)
        num_tables = segment_range.size(0) - 1
        device = keys.device

        if num_keys == 0:
            empty_keys = torch.empty(0, dtype=keys.dtype, device=device)
            empty_reverse_indices = torch.empty(0, dtype=torch.int64, device=device)
            d_table_range = torch.zeros(
                num_tables + 1, dtype=torch.int64, device=device
            )
            unique_size_per_table = torch.zeros(
                num_tables, dtype=torch.int64, device="cpu"
            )
            return (
                empty_keys,
                empty_reverse_indices,
                d_table_range,
                None,
                unique_size_per_table,
            )

        is_lfu_enabled = (
            evict_strategy == EvictStrategy.KLfu if evict_strategy else False
        )
        need_frequency_output = is_lfu_enabled or frequency_counts is not None

        input_frequencies = None
        if frequency_counts is not None:
            input_frequencies = frequency_counts
        elif need_frequency_output:
            input_frequencies = torch.empty(0, dtype=torch.int64, device=device)

        (
            _num_uniques,
            unique_keys,
            reverse_indices,
            table_offsets,
            freq_counters,
        ) = segmented_unique_cuda(keys, segment_range, num_tables, input_frequencies)

        table_offsets_cpu = _copy_cuda_tensor_to_pinned_cpu(table_offsets)
        total_unique = table_offsets_cpu[num_tables].item()
        unique_size_per_table = (
            table_offsets_cpu[1 : num_tables + 1] - table_offsets_cpu[0:num_tables]
        )

        unique_keys_out = unique_keys[:total_unique]

        output_scores = None
        if need_frequency_output and total_unique > 0:
            output_scores = freq_counters[:total_unique]

        return (
            unique_keys_out,
            reverse_indices,
            table_offsets,
            output_scores,
            unique_size_per_table,
        )


class StorageMode(Enum):
    DEFAULT = auto()
    CACHE = auto()
    HBM_DIRECT = auto()


@dataclass
class PrefetchState:
    unique_keys: torch.Tensor
    reverse_indices: torch.Tensor
    unique_table_ids: torch.Tensor
    lfu_accumulated_frequency: torch.Tensor
    table_num: int
    emb_dtype: torch.dtype
    storage_mode: StorageMode
    slot_indices: Optional[torch.Tensor]
    # Value-buffer row per key -- what forward and backward both address values
    # with. It differs from ``slot_indices`` only under NO_EVICTION, whose rows
    # come from a per-table auto-increment counter unrelated to where the key
    # hashes, and whose key map is deliberately larger than the value buffer, so
    # indexing values by slot reads past the end of it. Every other strategy
    # points this at the slot tensor itself rather than leaving it unset, so a
    # consumer never has to ask which of the two it is holding. Anything indexed
    # by hash slot (the ref counter) uses ``slot_indices``.
    value_row_indices: Optional[torch.Tensor] = None
    non_admitted_positions: Optional[torch.Tensor] = None
    num_prefetched_keys: int = 0
    outstanding_keys_ref: Optional[torch.Tensor] = None
    # Cache mode: where each table starts in the unique arrays, and the node each
    # key holds (int32, -1 for none), which is what the reference to give back in
    # the backward is indexed by. ``unique_size_per_table`` is the per-table count
    # the outstanding-key tally is kept in.
    unique_offsets: Optional[List[int]] = None
    cache_nodes: Optional[torch.Tensor] = None
    unique_size_per_table: Optional[torch.Tensor] = None


def _record_stream_of_prefetch_state(state: PrefetchState) -> None:
    """Tell the allocator the current stream reads what the prefetch allocated.

    The prefetch may run on its own stream, while forward and backward (which
    runs on the stream of its forward) read these tensors on another. Without the
    record, freeing the context returns their memory to the prefetch stream's pool,
    and the next prefetch can overwrite it under a kernel still reading it.
    ``outstanding_keys_ref`` belongs to the module and is never freed.
    """
    stream = torch.cuda.current_stream()
    for f in fields(state):
        t = getattr(state, f.name)
        if isinstance(t, torch.Tensor) and t.is_cuda and f.name != "outstanding_keys_ref":
            t.record_stream(stream)


def _apply_admission(
    missing_keys: torch.Tensor,
    missing_indices: torch.Tensor,
    missing_table_ids: torch.Tensor,
    missing_scores: Optional[torch.Tensor],
    values: torch.Tensor,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    freq_for_admission: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply admission filtering for missing keys.

    If an admission strategy is active, runs ``non_admitted_initializer`` over
    the rows it rejects when the strategy asked for one, then filters
    keys/scores/table_ids/indices to only the admitted subset.

    Returns (keys_to_insert, scores_to_insert, table_ids_to_insert,
             positions_in_unique, indices_to_init) where indices_to_init are
    the positions in values that the caller should initialize with its
    embeddings initializer. Every missing row is covered exactly once: the
    non-admitted ones drop out of indices_to_init exactly when this function
    has already run the strategy's own initializer over them.
    """
    with torch.cuda.nvtx.range("_apply_admission"):
        if admitter is None or missing_keys.numel() == 0:
            return (
                missing_keys,
                missing_scores,
                missing_table_ids,
                missing_indices,
                missing_indices,
            )

        admit_mask = admitter.admit(missing_keys, missing_table_ids, freq_for_admission)
        non_admitted_initializer = admitter.non_admitted_initializer

        non_admitted_mask = ~admit_mask
        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, _, (non_admitted_indices,) = flagged_compact(
                non_admitted_mask, [missing_indices]
            )
        if non_admitted_initializer is not None and non_admitted_indices.numel() > 0:
            with torch.cuda.nvtx.range("op:initializer"):
                non_admitted_initializer(
                    values[:, :max_emb_dim],
                    # Alongside ``values``, which is why these are the whole
                    # unique vectors and not the missing slice; the rows to
                    # write come last.
                    unique_keys,
                    unique_table_ids,
                    non_admitted_indices,
                )

        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                _,
                _,
                (
                    keys_to_insert,
                    positions_in_unique,
                    table_ids_to_insert,
                    scores_to_insert,
                ),
            ) = flagged_compact(
                admit_mask,
                [missing_keys, missing_indices, missing_table_ids, missing_scores],
            )
        # Whoever owns the non-admitted rows was settled at construction. When
        # the strategy asked for its own initializer the table's only has to
        # cover the admitted rows; otherwise it has to cover every miss, or the
        # non-admitted rows reach the forward holding whatever
        # ``storage.find`` left in its ``torch.empty`` value buffer.
        indices_to_init = (
            positions_in_unique
            if non_admitted_initializer is not None
            else missing_indices
        )

        return (
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            indices_to_init,
        )


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    initializer: MultiTableInitializer,
    forward_stream: Optional[torch.cuda.Stream] = None,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    admitter: Optional[MultiTableAdmitter] = None,
    outstanding_keys_ref: Optional[torch.Tensor] = None,
) -> PrefetchState:
    """Unified prefetch for all storage types (cache, HBM-direct, generic).

    Returns a PrefetchState containing unique embeddings + optimizer states
    and the metadata needed by forward/backward.
    """
    with torch.cuda.nvtx.range("dynamicemb_prefetch"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        emb_dtype = storage.embedding_dtype()
        max_emb_dim = storage.max_embedding_dim()
        max_val_dim = storage.max_value_dim()
        caching = cache is not None

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None
        # todo: double check
        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        indices_table_range = get_table_range(offsets, feature_offsets)
        (
            unique_keys,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
            unique_size_per_table,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        num_prefetched_keys = unique_keys.numel()

        if hasattr(storage, "expand_if_need"):
            storage.expand_if_need(unique_size_per_table)

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            unique_keys.numel(),
        )

        slot_indices = None
        non_admitted_positions = None
        value_row_indices = None
        if caching:
            storage_mode = StorageMode.CACHE
        elif (
            isinstance(storage, ChainedStorage)
            and storage.values.on_device
        ):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT

        unique_offsets: Optional[List[int]] = None
        cache_nodes: Optional[torch.Tensor] = None
        if storage_mode == StorageMode.CACHE:
            # Where each table starts in the unique arrays: the cache is driven
            # table by table, and the per-table counts are already on the host.
            unique_offsets = [0]
            for n in unique_size_per_table.tolist():
                unique_offsets.append(unique_offsets[-1] + int(n))

            if outstanding_keys_ref is not None:
                # A table's rows are all taken once the keys of the steps in flight
                # outnumber them; that is a capacity error, and clearer here than
                # a failed allocate.
                outstanding_keys_ref += unique_size_per_table
                for t in range(table_num):
                    cache_capacity = cache.capacity(t)
                    outstanding_keys = int(outstanding_keys_ref[t])
                    if outstanding_keys > cache_capacity:
                        raise RuntimeError(
                            f"Outstanding prefetched keys of table {t} "
                            f"({outstanding_keys}) "
                            f"exceed its cache capacity ({cache_capacity}). "
                            "Reduce prefetch pipeline depth or increase "
                            "cache size."
                        )
            from dynamicemb.chained_prefetch import prefetch_cache_path

            (
                slot_indices,
                non_admitted_positions,
                cache_nodes,
            ) = prefetch_cache_path(
                cache,
                storage,
                unique_keys,
                unique_table_ids,
                unique_offsets,
                max_emb_dim,
                max_val_dim,
                emb_dtype,
                initializer,
                evict_strat,
                lfu_accumulated_frequency,
                admitter,
            )
            # The cache addresses its own value buffer by node, which is the row.
            # There is no separate hash slot to tell apart from it: the cache keeps
            # no ref counter beside a hash table, so the two index spaces coincide.
            value_row_indices = slot_indices
        elif storage_mode == StorageMode.HBM_DIRECT:
            from dynamicemb.chained_prefetch import prefetch_hbm_direct_path

            (
                slot_indices,
                non_admitted_positions,
                value_row_indices,
            ) = prefetch_hbm_direct_path(
                storage,
                unique_keys,
                unique_table_ids,
                max_emb_dim,
                max_val_dim,
                initializer,
                evict_strat,
                lfu_accumulated_frequency,
                admitter,
            )

        return PrefetchState(
            unique_keys=unique_keys,
            reverse_indices=reverse_indices,
            unique_table_ids=unique_table_ids,
            lfu_accumulated_frequency=lfu_accumulated_frequency,
            table_num=table_num,
            emb_dtype=emb_dtype,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            non_admitted_positions=non_admitted_positions,
            num_prefetched_keys=num_prefetched_keys,
            outstanding_keys_ref=outstanding_keys_ref,
            value_row_indices=value_row_indices,
            unique_offsets=unique_offsets,
            cache_nodes=cache_nodes,
            unique_size_per_table=unique_size_per_table,
        )


def dynamicemb_eval_forward(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    output_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    layout: EmbeddingLayout,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    pooling_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eval-only forward for all storage configurations (no autograd)."""
    with torch.cuda.nvtx.range("dynamicemb_eval_forward"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        emb_dtype = storage.embedding_dtype()

        is_pooling = layout.is_pooling

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None

        indices_table_range = get_table_range(offsets, feature_offsets)

        if not is_pooling:
            table_ids = expand_table_ids_cuda(
                indices_table_range,
                indices.numel(),
            )
            frequency_counts_int64 = (
                frequency_counters.long() if frequency_counters is not None else None
            )
            output_embs = eval_lookup(
                storage,
                indices,
                table_ids,
                initializer,
                cache=cache,
            )
            if output_dtype != emb_dtype:
                output_embs = output_embs.to(output_dtype)
            return output_embs

        frequency_counts_int64 = (
            frequency_counters.long() if frequency_counters is not None else None
        )

        (
            unique_indices,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
            unique_size_per_table,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            unique_indices.numel(),
        )

        unique_embs = eval_lookup(
            storage,
            unique_indices,
            unique_table_ids,
            initializer,
            cache=cache,
        )

        output_embs = torch.empty(
            *layout.pooled_shape(),
            dtype=output_dtype,
            device=indices.device,
        )
        gather_embedding_pooled(
            unique_embs,
            output_embs,
            reverse_indices,
            offsets,
            layout.pooling_mode,
            layout.total_D,
            layout.batch_size,
            layout.D_offsets,
            layout.max_D,
            pooling_weights,
            layout.feature_dims_vec4,
        )
        return output_embs


def _generic_forward_path(
    storage: Storage,
    optimizer: BaseDynamicEmbeddingOptimizer,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone forward for generic (DEFAULT) storage mode.

    Does storage.find + admission + init + insert
    in a single pass with no PrefetchState.

    Returns (unique_values, persisted_unique_indices). The latter is an int64
    vector of row indices into the unique-key tensors for keys that are stored
    in ``storage`` after this forward (lookup hit or admitted insert). Used in
    backward to ``storage.insert(..., preserve_existing=True)`` only for those
    rows, without compacting via a boolean mask again.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        device = unique_keys.device
        unique_keys.numel()

        with torch.cuda.nvtx.range("op:storage_find"):
            (
                h_num_missing,
                missing_keys,
                missing_indices,
                missing_table_ids,
                missing_scores,
                founds,
                _,
                unique_values,
            ) = storage.find(
                unique_keys,
                unique_table_ids,
                copy_mode=CopyMode.VALUE,
                lfu_accumulated_frequency=accumulated_frequency,
            )

        key_persisted = founds.clone()

        if h_num_missing == 0:
            n = unique_keys.numel()
            if n == 0:
                persisted_unique_indices = torch.empty(
                    0, dtype=torch.int64, device=device
                )
            else:
                persisted_unique_indices = torch.arange(
                    n, device=device, dtype=torch.int64
                )
            return unique_values, persisted_unique_indices

        freq_for_admission = (
            accumulated_frequency[missing_indices]
            if admitter is not None and accumulated_frequency is not None
            else None
        )
        (
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            indices_to_init,
        ) = _apply_admission(
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            unique_values,
            unique_keys,
            unique_table_ids,
            max_emb_dim,
            freq_for_admission,
            admitter,
        )

        if indices_to_init.numel() > 0:
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(
                    unique_values[:, :max_emb_dim],
                    unique_keys,
                    unique_table_ids,
                    indices_to_init,
                )

        if max_val_dim != max_emb_dim:
            optimizer.reset_optimizer_states(
                unique_values[:, max_emb_dim:],
                indices=missing_indices,
                emb_dims=storage.embedding_dims(on_device=True)[missing_table_ids],
            )

        values_to_insert = unique_values[positions_in_unique]

        with torch.cuda.nvtx.range("op:storage_insert"):
            storage.insert(
                keys_to_insert,
                table_ids_to_insert,
                values_to_insert,
                scores_to_insert,
            )
        key_persisted[positions_in_unique] = True
        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, persisted_unique_indices, _ = flagged_compact(
                key_persisted, [unique_keys]
            )
        return unique_values, persisted_unique_indices


class DynamicEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prefetch_state: PrefetchState,
        offsets: torch.Tensor,
        cache: Optional[Cache],
        storage: Storage,
        output_dtype: torch.dtype,
        initializer: MultiTableInitializer,
        optimizer: BaseDynamicEmbeddingOptimizer,
        layout: EmbeddingLayout,
        admitter=None,
        evict_strategy=None,
        pooling_weights: Optional[torch.Tensor] = None,
        *args,
    ):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.forward"):
            _record_stream_of_prefetch_state(prefetch_state)
            max_emb_dim = storage.max_embedding_dim()
            max_val_dim = storage.max_value_dim()
            emb_dtype = storage.embedding_dtype()

            # pooling_weights is validated by the sole caller,
            # BatchedDynamicEmbeddingTablesV2.forward (pooling mode, dtype,
            # numel, and exclusivity with frequency_counters), and again by
            # TORCH_CHECK in gather_embedding_pooled / reduce_grads.
            is_pooling = layout.is_pooling

            use_counter = prefetch_state.slot_indices is not None
            if use_counter:
                state = (
                    cache._state
                    if prefetch_state.storage_mode == StorageMode.CACHE
                    else storage._state
                )
                with torch.cuda.nvtx.range("op:load_from_flat"):
                    unique_embs = state.values.load(# Value buffer: address by row, which is the slot for
                        # every strategy except NO_EVICTION.
                        prefetch_state.value_row_indices,
                        prefetch_state.unique_table_ids,
                        copy_mode=CopyMode.EMBEDDING,
                    )
                if prefetch_state.non_admitted_positions is not None:
                    na = prefetch_state.non_admitted_positions
                    non_admitted_initializer = (
                        admitter.non_admitted_initializer
                        if admitter is not None
                        else None
                    )
                    with torch.cuda.nvtx.range("op:initializer"):
                        # Whichever initializer owns rejected rows was settled
                        # at construction, so this is one call either way.
                        (non_admitted_initializer or initializer)(
                            unique_embs[:, :max_emb_dim],
                            prefetch_state.unique_keys,
                            prefetch_state.unique_table_ids,
                            na,
                        )
                unique_values = None
                persisted_unique_indices = None
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                evict_strat = (
                    EvictStrategy(evict_strategy.value) if evict_strategy else None
                )
                unique_values, persisted_unique_indices = _generic_forward_path(
                    storage,
                    optimizer,
                    prefetch_state.unique_keys,
                    prefetch_state.unique_table_ids,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    evict_strat,
                    prefetch_state.lfu_accumulated_frequency,
                    admitter,
                )
                # find() was asked for CopyMode.VALUE, so a row here is
                # embedding ++ optimizer state; keep only the embedding.
                unique_embs = unique_values[:, :max_emb_dim]

            device = prefetch_state.unique_keys.device
            if is_pooling:
                output_embs = torch.empty(
                    *layout.pooled_shape(),
                    dtype=output_dtype,
                    device=device,
                )
                with torch.cuda.nvtx.range("op:gather_embedding"):
                    gather_embedding_pooled(
                        unique_embs,
                        output_embs,
                        prefetch_state.reverse_indices,
                        offsets,
                        layout.pooling_mode,
                        layout.total_D,
                        layout.batch_size,
                        layout.D_offsets,
                        layout.max_D,
                        pooling_weights,
                        layout.feature_dims_vec4,
                    )
            else:
                output_embs = torch.empty(
                    *layout.sequence_shape(),
                    dtype=output_dtype,
                    device=device,
                )
                with torch.cuda.nvtx.range("op:gather_embedding"):
                    gather_embedding(
                        unique_embs, output_embs, prefetch_state.reverse_indices
                    )

            ctx.unique_keys = prefetch_state.unique_keys
            ctx.reverse_indices = prefetch_state.reverse_indices
            ctx.unique_table_ids = prefetch_state.unique_table_ids
            ctx.cache = cache
            ctx.storage = storage
            ctx.slot_indices = prefetch_state.slot_indices
            # Same split as the forward: the fused optimizer writes into the
            # value buffer (rows), the ref counter is indexed by hash slot.
            ctx.update_value_rows = prefetch_state.value_row_indices
            ctx.storage_mode = prefetch_state.storage_mode
            ctx.optimizer = optimizer
            ctx.layout = layout
            ctx.offsets = offsets
            ctx.max_emb_dim = max_emb_dim
            ctx.max_val_dim = max_val_dim
            ctx.emb_dtype = emb_dtype
            ctx.pooling_weights = pooling_weights
            ctx.use_counter = use_counter
            ctx.unique_values = unique_values
            ctx.persisted_unique_indices = persisted_unique_indices
            ctx.outstanding_keys_ref = prefetch_state.outstanding_keys_ref
            ctx.num_prefetched_keys = prefetch_state.num_prefetched_keys
            ctx.unique_offsets = prefetch_state.unique_offsets
            ctx.cache_nodes = prefetch_state.cache_nodes

            # Recover outstanding count at end of forward so it is decremented
            # even when backward is not run, avoiding overflow.
            if prefetch_state.outstanding_keys_ref is not None:
                prefetch_state.outstanding_keys_ref -= (
                    prefetch_state.unique_size_per_table
                )

            return output_embs

    @staticmethod
    def backward(ctx, grads):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.backward"):
            cache = ctx.cache
            storage = ctx.storage
            optimizer = ctx.optimizer
            grads = grads.contiguous()

            if optimizer.need_gradient_clipping():
                optimizer.clip_gradient(grads)

            layout = ctx.layout
            if layout.is_pooling:
                with torch.cuda.nvtx.range("op:reduce_grads"):
                    unique_grads = reduce_grads(
                        ctx.reverse_indices,
                        grads,
                        ctx.unique_keys.numel(),
                        layout.batch_size,
                        ctx.max_emb_dim,
                        ctx.offsets,
                        layout.D_offsets,
                        layout.pooling_mode,
                        layout.total_D,
                        ctx.pooling_weights,
                        layout.feature_dims_vec4,
                    )
            else:
                with torch.cuda.nvtx.range("op:reduce_grads"):
                    unique_grads = reduce_grads(
                        ctx.reverse_indices,
                        grads,
                        ctx.unique_keys.numel(),
                        layout.batch_size,
                        ctx.max_emb_dim,
                    )

            optimizer.step()

            unique_table_ids = ctx.unique_table_ids

            with torch.cuda.nvtx.range("DynamicEmbeddingFunction.update"):
                if ctx.use_counter:
                    state = (
                        cache._state
                        if ctx.storage_mode == StorageMode.CACHE
                        else storage._state
                    )
                    with torch.cuda.nvtx.range("op:optimizer_update_fused"):
                        # Writes into the value buffer -> rows.
                        state.values.update_rows(
                            optimizer,
                            unique_grads,
                            ctx.update_value_rows,
                            unique_table_ids,
                        )

                    if ctx.storage_mode == StorageMode.CACHE:
                        # Give back the reference the prefetch took on each node.
                        cache.release(ctx.cache_nodes, ctx.unique_offsets)
                    else:
                        # Ref counter sits beside the hash table -> slots.
                        storage.decrement_counter(
                            ctx.slot_indices, ctx.unique_table_ids
                        )

                if not ctx.use_counter and ctx.unique_values is not None:
                    with torch.cuda.nvtx.range("op:optimizer_update_padded"):
                        storage.values.update_padded(
                            optimizer,
                            unique_grads,
                            ctx.unique_values,
                            unique_table_ids,
                        )
                    pui = ctx.persisted_unique_indices
                    assert pui is not None
                    if pui.numel() > 0:
                        keys_c = ctx.unique_keys[pui]
                        tids_c = unique_table_ids[pui]
                        vals_c = ctx.unique_values[pui]
                        # preserve_existing=True → CONST score policy on insert:
                        # refresh embedding/optimizer buffers only; do not overwrite
                        # table scores (LFU / timestamp / etc.). For the hybrid storage,
                        # the HBM tier uses this on backward; existing scores stay.
                        with torch.cuda.nvtx.range("op:storage_insert"):
                            storage.insert(
                                keys_c,
                                tids_c,
                                vals_c,
                                preserve_existing=True,
                            )

            return (None,) * 18
