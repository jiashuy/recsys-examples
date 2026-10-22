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

"""Builders shared by the BatchedDynamicEmbeddingTablesV2 module tests.

An FBGEMM SplitTableBatchedEmbeddingBagsCodegen reference with matching
optimizer and initial weights, a dict-backed external storage (PyDictStorage),
and the small multi-table module that the dump/load tests build and train.
"""

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import torch
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import DynamicEmbStorage, Storage
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

POOLING_MODE: Dict[DynamicEmbPoolingMode, PoolingMode] = {
    DynamicEmbPoolingMode.NONE: PoolingMode.NONE,
    DynamicEmbPoolingMode.MEAN: PoolingMode.MEAN,
    DynamicEmbPoolingMode.SUM: PoolingMode.SUM,
}


def _split_table_batched_optimizer_for_compare(opt_type: EmbOptimType) -> EmbOptimType:
    """Optimizer passed to SplitTableBatchedEmbeddingBagsCodegen for TorchREC comparison.

    DynamicEmb may use ``SGD``; STBE uses ``EXACT_SGD`` so updates match for the test.
    All other optimizers match ``opt_type``."""
    return EmbOptimType.EXACT_SGD if opt_type == EmbOptimType.SGD else opt_type


class PyDictStorage(Storage[DynamicEmbTableOptions, BaseDynamicEmbeddingOptimizer]):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self.options = options
        self.dict: Dict[Tuple[int, int], torch.Tensor] = {}
        self.scores: Dict[Tuple[int, int], int] = {}
        self.capacity = max(o.max_capacity for o in options)
        self.optimizer = optimizer

        self._emb_dims = [o.dim for o in options]
        self._emb_dtype = options[0].embedding_dtype
        self._value_dims = [o.dim + optimizer.get_state_dim(o.dim) for o in options]
        self._optstate_dims = [optimizer.get_state_dim(o.dim) for o in options]
        self._max_emb_dim = max(self._emb_dims)
        self._max_value_dim = max(self._value_dims)
        self._max_optstate_dim = max(self._optstate_dims)

        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")

        # Persistent per-table emb-dim tensors (device + host) so embedding_dims()
        # returns a long-lived tensor regardless of on_device.
        self._emb_dims_dev = torch.tensor(
            self._emb_dims, dtype=torch.int64, device=self.device
        )
        self._emb_dims_host = torch.tensor(self._emb_dims, dtype=torch.int64)

    def size(self) -> int:
        return len(self.dict)

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_embs: torch.Tensor,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h_unique_keys = unique_keys.cpu()
        h_table_ids = table_ids.cpu()
        lookup_dim = unique_embs.size(1)
        results = []
        missing_keys = []
        missing_indices = []
        missing_scores_list = []
        founds_ = []
        for i in range(h_unique_keys.size(0)):
            key = h_unique_keys[i].item()
            tid = h_table_ids[i].item()
            composite_key = (tid, key)
            if composite_key in self.dict:
                results.append(self.dict[composite_key][0:lookup_dim])
                founds_.append(True)
            else:
                missing_keys.append(key)
                missing_indices.append(i)
                if input_scores is not None:
                    missing_scores_list.append(input_scores[i].item())
                founds_.append(False)
        founds_ = torch.tensor(founds_, dtype=torch.bool, device=self.device)
        if len(results) > 0:
            unique_embs[founds_, :] = torch.cat(
                [t.unsqueeze(0) for t in results], dim=0
            )

        num_missing = torch.tensor(
            [len(missing_keys)], dtype=torch.long, device=self.device
        )
        missing_keys = torch.tensor(
            missing_keys, dtype=unique_keys.dtype, device=self.device
        )
        missing_indices = torch.tensor(
            missing_indices, dtype=torch.long, device=self.device
        )

        if input_scores is not None and len(missing_scores_list) > 0:
            missing_scores = torch.tensor(
                missing_scores_list, dtype=input_scores.dtype, device=self.device
            )
        else:
            missing_scores = torch.empty(0, dtype=torch.uint64, device=self.device)

        # output_scores: scores for all keys (0 for missing, input_scores for found)
        output_scores = torch.zeros(
            unique_keys.size(0), dtype=torch.int64, device=self.device
        )
        if input_scores is not None:
            output_scores[founds_] = input_scores[founds_].to(torch.int64)

        return (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds_,
            output_scores,
        )

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_embs: torch.Tensor,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return self.find_impl(unique_keys, table_ids, unique_embs, input_scores)

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        *,
        find_debug_context: Optional[str] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if copy_mode == CopyMode.EMBEDDING:
            out_dim = self._max_emb_dim
        else:
            out_dim = self._max_value_dim
        unique_vals = torch.zeros(
            unique_keys.size(0), out_dim, dtype=self._emb_dtype, device=self.device
        )
        (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds,
            output_scores,
        ) = self.find_impl(
            unique_keys, table_ids, unique_vals, lfu_accumulated_frequency
        )

        h_table_ids = table_ids.cpu()
        missing_table_ids_list = []
        for idx in missing_indices.cpu().tolist():
            missing_table_ids_list.append(h_table_ids[idx].item())
        missing_table_ids = torch.tensor(
            missing_table_ids_list, dtype=torch.int64, device=self.device
        )

        return (
            num_missing.item()
            if isinstance(num_missing, torch.Tensor)
            else num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores if missing_scores.numel() > 0 else None,
            founds,
            output_scores,
            unique_vals,
        )

    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        h_keys = keys.cpu()
        h_table_ids = table_ids.cpu()
        h_scores = scores.cpu() if scores is not None and scores.numel() > 0 else None
        for i in range(h_keys.size(0)):
            key = h_keys[i].item()
            tid = h_table_ids[i].item()
            composite_key = (tid, key)
            self.dict[composite_key] = values[i, :].clone()
            if h_scores is not None:
                self.scores[composite_key] = h_scores[i].item()

    def update(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise ValueError("Can't call update of PyDictStorage")

    def enable_update(self) -> bool:
        return False

    def dump(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = False,
        include_meta: bool = False,
        timestamp: int = 0,
    ) -> None:
        if include_meta:
            meta_data = {}
            meta_data.update(self.optimizer.get_opt_args())
            meta_data["dist_type"] = self.options[table_id].dist_type
            with open(meta_file_path, "w") as f:
                json.dump(meta_data, f)

        fkey = open(emb_key_path, "wb")
        fembedding = open(embedding_file_path, "wb")
        fscore = open(score_file_path, "wb") if score_file_path else None
        fopt_states = open(opt_file_path, "wb") if include_optim else None

        for keys, embeddings, opt_states, scores_out in self.export_keys_values(
            table_id, self.device
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            if fscore is not None:
                fscore.write(scores_out.cpu().numpy().tobytes())
            fembedding.write(embeddings.cpu().numpy().tobytes())
            if fopt_states is not None and opt_states is not None:
                to_write = self.optimizer.states_for_checkpoint(
                    opt_states,
                    self._emb_dims[table_id],
                )
                fopt_states.write(to_write.cpu().numpy().tobytes())

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def load(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = False,
        timestamp: int = 0,
        # The table's own shard geometry, which this demo accepts and ignores:
        # it reads every key in the file it is handed. A store that re-splits a
        # checkpoint -- loaded at a different shard count than it was dumped at
        # -- filters by these instead of by the global rank and world size,
        # which a table-row-wise table does not share.
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> None:
        if meta_file_path and os.path.exists(meta_file_path):
            with open(meta_file_path, "r") as f:
                meta_data = json.load(f)
            opt_type = meta_data.get("opt_type", None)
            if (
                opt_type
                and self.optimizer.get_opt_args().get("opt_type", None) != opt_type
            ):
                include_optim = False
            ckpt_dist_type = meta_data.get("dist_type", "roundrobin")
            runtime_dist_type = self.options[table_id].dist_type
            if runtime_dist_type != ckpt_dist_type:
                raise ValueError(
                    "Input dist_type mismatch: checkpoint was dumped with "
                    f"{ckpt_dist_type!r}, but runtime table is configured with "
                    f"{runtime_dist_type!r}. Please load with a matching dist_type."
                )
            if include_optim:
                self.optimizer.set_opt_args(meta_data)

        if not opt_file_path or not os.path.exists(opt_file_path):
            include_optim = False

        dim = self._emb_dims[table_id]
        runtime_optstate_dim = self._optstate_dims[table_id]
        value_dim = self._value_dims[table_id]
        ckpt_dim = self.optimizer.get_ckpt_state_dim(dim)

        num_keys = os.path.getsize(emb_key_path) // 8  # int64

        file_opt_dim = 0
        if (
            include_optim
            and opt_file_path
            and os.path.exists(opt_file_path)
            and runtime_optstate_dim > 0
        ):
            if num_keys == 0:
                if os.path.getsize(opt_file_path) != 0:
                    raise ValueError("Non-empty optimizer file but zero keys.")
            else:
                opt_sz = os.path.getsize(opt_file_path)
                row_block = 4 * num_keys
                if opt_sz % row_block != 0:
                    raise ValueError(
                        f"Optimizer file size {opt_sz} not divisible by {row_block}."
                    )
                file_opt_dim = opt_sz // row_block
                if file_opt_dim != ckpt_dim:
                    raise ValueError(
                        f"Checkpoint optimizer width {file_opt_dim}; expected {ckpt_dim}."
                    )

        fkey = open(emb_key_path, "rb")
        fembedding = open(embedding_file_path, "rb")
        fscore = (
            open(score_file_path, "rb")
            if score_file_path and os.path.exists(score_file_path)
            else None
        )
        fopt_states = open(opt_file_path, "rb") if include_optim else None

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(8 * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=np.int64).copy(),
                dtype=torch.int64,
                device=self.device,
            )

            emb_bytes = fembedding.read(4 * dim * n)
            embeddings = torch.tensor(
                np.frombuffer(emb_bytes, dtype=np.float32).copy(),
                dtype=torch.float32,
                device=self.device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states is not None and file_opt_dim > 0 and n > 0:
                opt_bytes = fopt_states.read(4 * file_opt_dim * n)
                opt_states = torch.tensor(
                    np.frombuffer(opt_bytes, dtype=np.float32).copy(),
                    dtype=torch.float32,
                    device=self.device,
                ).view(-1, file_opt_dim)
                opt_states = self.optimizer.states_from_checkpoint(
                    opt_states,
                    dim,
                    torch.float32,
                    self.device,
                )

            scores = None
            if fscore:
                score_bytes = fscore.read(8 * n)
                scores = torch.tensor(
                    np.frombuffer(score_bytes, dtype=np.int64).copy(),
                    dtype=torch.int64,
                    device=self.device,
                )

            # Build full values tensor [N, value_dim]
            if opt_states is not None:
                values = torch.cat([embeddings, opt_states], dim=1)
            else:
                if value_dim > dim:
                    values = torch.empty(
                        n, value_dim, dtype=torch.float32, device=self.device
                    )
                    values[:, :dim] = embeddings
                    self.optimizer.reset_optimizer_states(values[:, dim:], emb_dims=dim)
                else:
                    values = embeddings

            tid = torch.full(
                (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
            )
            self.insert(keys, tid, values, scores)

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def export_keys_values(
        self,
        table_id: int,
        device: torch.device,
        batch_size: int = 65536,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        """Yield (keys, embeddings, opt_states, scores) batches for a given table."""
        emb_dim = self._emb_dims[table_id]
        optstate_dim = self._optstate_dims[table_id]
        all_keys = [ck for ck in self.dict.keys() if ck[0] == table_id]
        for start in range(0, len(all_keys), batch_size):
            batch_keys = all_keys[start : start + batch_size]

            keys_t = torch.tensor(
                [ck[1] for ck in batch_keys], dtype=torch.int64, device=device
            )
            values_t = torch.stack(
                [self.dict[ck].to(device) for ck in batch_keys], dim=0
            )
            embeddings = values_t[:, :emb_dim].contiguous()

            opt_states = None
            if optstate_dim > 0:
                opt_states = values_t[:, emb_dim : emb_dim + optstate_dim].contiguous()

            scores_list = [self.scores.get(ck, 0) for ck in batch_keys]
            scores_t = torch.tensor(scores_list, dtype=torch.int64, device=device)

            yield keys_t, embeddings, opt_states, scores_t

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._emb_dims[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._value_dims[table_id]

    def max_embedding_dim(self) -> int:
        return self._max_emb_dim

    def max_value_dim(self) -> int:
        return self._max_value_dim

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return self._emb_dims_dev if on_device else self._emb_dims_host

    def all_dims_vec4(self) -> bool:
        return all(d % 4 == 0 for d in self._emb_dims) and all(
            v % 4 == 0 for v in self._value_dims
        )

    def init_optimizer_state(
        self,
    ) -> float:
        return self.optimizer.get_initial_optimizer_state()


def create_split_table_batched_embedding(
    table_names,
    feature_table_map,
    optimizer_type,
    opt_params,
    dims,
    num_embs,
    pooling_mode,
    device,
):
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                e,
                d,
                EmbeddingLocation.DEVICE,
                ComputeDevice.CUDA,
            )
            for (e, d) in zip(num_embs, dims)
        ],
        optimizer=optimizer_type,
        weights_precision=SparseType.FP32,
        stochastic_rounding=False,
        pooling_mode=pooling_mode,
        output_dtype=SparseType.FP32,
        device=device,
        table_names=table_names,
        feature_table_map=feature_table_map,
        **opt_params,
        bounds_check_mode=BoundsCheckMode.FATAL,
    ).cuda()
    return emb


def init_embedding_tables(stbe, bdet):
    stbe.init_embedding_weights_uniform(0, 1)
    storage = bdet.tables
    optimizer = bdet.optimizer
    for table_idx, split in enumerate(stbe.split_embedding_weights()):
        num_emb = split.size(0)
        emb_dim = split.size(1)
        opt_state_dim = optimizer.get_state_dim(emb_dim)
        val_dim = emb_dim + opt_state_dim
        indices = torch.arange(num_emb, device=split.device, dtype=torch.long)
        table_ids = torch.full(
            (num_emb,), table_idx, dtype=torch.int64, device=split.device
        )
        if isinstance(storage, DynamicEmbStorage):
            max_emb_dim = storage.max_embedding_dim()
            max_value_dim = storage.max_value_dim()
            values = torch.zeros(
                num_emb, max_value_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            if opt_state_dim > 0:
                optimizer.reset_optimizer_states(
                    values[:, max_emb_dim : max_emb_dim + opt_state_dim],
                    emb_dims=emb_dim,
                )
            storage.set_score(1)
            storage.insert(indices, table_ids, values)
        elif isinstance(storage, PyDictStorage):
            pydict = cast(PyDictStorage, storage)
            values = torch.empty(
                num_emb, val_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            if val_dim > emb_dim:
                optimizer.reset_optimizer_states(values[:, emb_dim:], emb_dims=emb_dim)
            pydict.insert(indices, table_ids, values)
        else:
            raise ValueError("Not support table type")


def _make_multi_table_bdeb_for_dump_load(
    opt_type: EmbOptimType,
    opt_params: Dict[str, Any],
    dims: List[int],
    table_names: List[str],
    feature_table_map: List[int],
    max_capacity: int,
    key_type: torch.dtype,
    value_type: torch.dtype,
    device_id: int,
    dist_type: str = "roundrobin",
    optimizer_state_dtype: Optional[torch.dtype] = None,
) -> BatchedDynamicEmbeddingTablesV2:
    opts = []
    for dim in dims:
        opts.append(
            DynamicEmbTableOptions(
                dim=dim,
                init_capacity=max_capacity,
                max_capacity=max_capacity,
                index_type=key_type,
                embedding_dtype=value_type,
                device_id=device_id,
                score_strategy=DynamicEmbScoreStrategy.STEP,
                caching=False,
                local_hbm_for_values=1024**3,
                dist_type=dist_type,
                optimizer_state_dtype=optimizer_state_dtype,
            )
        )
    return BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=opts,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )


def _train_multi_table_bdeb_once(
    bdeb: BatchedDynamicEmbeddingTablesV2,
    key_type: torch.dtype,
    device: torch.device,
) -> None:
    indices = torch.tensor([0, 1, 2, 3, 10, 11, 12, 13], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=key_type, device=device)

    for _ in range(3):
        embs = bdeb(indices, offsets)
        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()
