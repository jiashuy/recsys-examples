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

"""Backward of ``BatchedDynamicEmbeddingTablesV2`` on one GPU, pooled and sequence.

Level: unit.   Needs: 1 GPU.
Covers: the module alone (no sharding, no TorchRec model) trained with SGD
(lr 1.0) on a gradient of ones, over embedding dim (aligned or not), batch size
(odd or not) and per-feature bag-size profile. Pooled (SUM): output shape only.
Sequence (pooling NONE), with and without index dedup: the update each key
received equals its occurrence count in the batch (checked by re-reading the
rows in eval mode and comparing with the pre-update output).
Does not cover (see test_batched_dynamic_embedding_tables_v2.py): other optimizers,
caching, deterministic mode; MEAN pooling backward.
"""

import gc
from typing import Dict, List

import pytest
import torch
from common.features import generate_sparse_feature, table_idx_to_name
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

NUM_EMBEDDINGS = [196608, 131072, 9437184, 851968]
DEFAULT_MULTI_HOT = [20, 17, 101, 49]
NUM_ITERATIONS = 32


def _make_tables(dim, pooling_mode, use_index_dedup=None):
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=dim,
            max_capacity=num_emb,
            local_hbm_for_values=1023**3,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG,
            ),
        )
        for num_emb in NUM_EMBEDDINGS
    ]
    kwargs = {}
    if use_index_dedup is not None:
        kwargs["use_index_dedup"] = use_index_dedup
    return BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        output_dtype=torch.float32,
        table_names=[table_idx_to_name(i) for i in range(len(options))],
        pooling_mode=pooling_mode,
        optimizer=EmbOptimType.SGD,
        learning_rate=1.0,
        **kwargs,
    )


def _make_batch(batch_size, multi_hot_sizes):
    return generate_sparse_feature(
        feature_num=len(NUM_EMBEDDINGS),
        num_embeddings_list=NUM_EMBEDDINGS,
        multi_hot_sizes=multi_hot_sizes,
        local_batch_size=batch_size,
    )


def _count_occurrences(keys: torch.Tensor) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for key in keys.to("cpu").tolist():
        counts[key] = counts.get(key, 0) + 1
    return counts


def _release(*objs):
    del objs
    gc.collect()
    torch.cuda.empty_cache()


# (embedding_dim, batch_size, multi_hot_sizes); odd sizes hit the unaligned paths.
SHAPES = [
    pytest.param(128, 1024, DEFAULT_MULTI_HOT, id="dim128-batch1024"),
    pytest.param(129, 1024, DEFAULT_MULTI_HOT, id="dim129-batch1024"),
    pytest.param(15, 1023, [20, 1, 101, 49], id="dim15-batch1023-hot-20-1-101-49"),
    pytest.param(15, 1023, [20, 49, 101, 1], id="dim15-batch1023-hot-20-49-101-1"),
]


@pytest.mark.parametrize("dim,batch_size,multi_hot_sizes", SHAPES)
def test_pooled_backward_output_shape(
    backend_session, dim, batch_size, multi_hot_sizes
):
    """Pooled SUM forward+backward over 32 batches gives [batch, tables * dim]."""
    _run_pooled(dim, batch_size, multi_hot_sizes)


def _run_pooled(dim, batch_size, multi_hot_sizes):
    table_num = len(NUM_EMBEDDINGS)
    var = _make_tables(dim, DynamicEmbPoolingMode.SUM)
    for _ in range(NUM_ITERATIONS):
        sparse_feature = _make_batch(batch_size, multi_hot_sizes)
        res = var(sparse_feature.values(), sparse_feature.offsets())
        res.backward(torch.ones_like(res))
        with torch.no_grad():
            assert res.dim() == 2
            assert res.size()[0] == batch_size
            assert res.size()[1] == table_num * dim
    _release(var)


@pytest.mark.parametrize("use_index_dedup", [True, False], ids=["dedup", "no-dedup"])
@pytest.mark.parametrize("dim,batch_size,multi_hot_sizes", SHAPES)
def test_sequence_backward_updates_each_key_by_its_count(
    backend_session, use_index_dedup, dim, batch_size, multi_hot_sizes
):
    """After SGD(lr=1, grad=1) a key's value dropped by its occurrence count."""
    table_num = len(NUM_EMBEDDINGS)
    var = _make_tables(dim, DynamicEmbPoolingMode.NONE, use_index_dedup)

    for iteration in range(NUM_ITERATIONS):
        sparse_feature = _make_batch(batch_size, multi_hot_sizes)
        indices = sparse_feature.values()
        res = var(indices, sparse_feature.offsets())
        res.backward(torch.ones_like(res))

        with torch.no_grad():
            per_table = [x.values() for x in sparse_feature.split([1] * table_num)]
            lengths: List[int] = [keys.size()[0] for keys in per_table]
            counts = [_count_occurrences(keys) for keys in per_table]
            assert res.dim() == 2
            sample_res = res[:, 0].to("cpu")

            var.eval()
            updated_res = var(indices, sparse_feature.offsets())
            var.train()
            diffs = sample_res - updated_res[:, 0].to("cpu")

            offset = 0
            wrong = 0
            for length, keys, count in zip(lengths, per_table, counts):
                diffs_cur_table = diffs[offset : offset + length]
                offset += length
                for j, key in enumerate(keys.tolist()):
                    expected_diff = float(count[key] % 100000)
                    if diffs_cur_table[j].item() != expected_diff:
                        wrong += 1
            assert wrong == 0, (
                f"iteration {iteration}: {wrong} of {diffs.numel()} keys got a "
                "different update than their occurrence count"
            )
    _release(var)
