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

"""Rows in and out of a table: the export of its rows in batches, the write of a
checkpoint's or a delta's rows into it, and the dump and load that connect them to files.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.dynamicemb_config import (
    ReplayContent,
)
from dynamicemb.checkpoint.files import (
    META_EMBEDDING_DIM,
    META_EMBEDDING_DTYPE,
    META_OPT_STATE_DTYPE,
    _raw_bytes,
    dtype_to_meta_name,
    load_from_json,
    resolve_checkpoint_value_layout,
    save_to_json,
)
from dynamicemb.checkpoint.row_chunk import RowChunk
from dynamicemb.storage.score import ScoreRules
from dynamicemb.storage.table_state import DynamicEmbTableState
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
)
from dynamicemb.types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
)
from dynamicemb_extensions import (
    EvictStrategy,
)
from torch import Tensor  # usort:skip


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


def export_keys_values_iter(
    state: DynamicEmbTableState,
    device: torch.device,
    batch_size: int = 65536,
    table_id: int = 0,
) -> Iterator[RowChunk]:
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
            out_scores = ScoreRules(
                state.options_list[table_id].score_strategy
            ).to_logical(out_scores)
        else:
            out_scores = scores.to(SCORE_TYPE)
        yield RowChunk(
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

    for chunk in export_keys_values_iter(state, device=device, table_id=table_id):
        fkey.write(chunk.keys.cpu().numpy().tobytes())
        fembedding.write(_raw_bytes(chunk.embeddings))
        scores = chunk.scores
        if state.evict_strategy == EvictStrategy.KLru:
            scores = timestamp - scores
        fscore.write(scores.cpu().numpy().tobytes())
        if fopt_states and chunk.opt_states is not None:
            to_write = state.optimizer.states_for_checkpoint(
                chunk.opt_states,
                state.values.emb_dims_list[table_id],
            )
            fopt_states.write(_raw_bytes(to_write))

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


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
