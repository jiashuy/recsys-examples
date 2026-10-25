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

"""The files of a table's checkpoint: the meta JSON, the value layout it describes, and
reading the key, embedding, score and optimizer-state files back in batches.
"""

import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.key_ownership import owned_key_mask
from dynamicemb.types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    torch_dtype_to_np_dtype,
)
from torch import Tensor  # usort:skip


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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
