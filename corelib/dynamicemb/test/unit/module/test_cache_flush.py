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

"""Cache prefetch and flush of BatchedDynamicEmbeddingTablesV2.

Level: unit.   Needs: 1 GPU.
Covers: prefetch and flush when the cache sits in front of the storage (also
with an external PyDictStorage), and flushing every table of a multi-table module.
Does not cover (see test_forward_backward.py): plain forward / backward.
"""

import os

import pytest
import torch
from common.bdet_builders import (
    POOLING_MODE,
    _split_table_batched_optimizer_for_compare,
    PyDictStorage,
    create_split_table_batched_embedding,
    init_embedding_tables,
)
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_prefetch_flush_in_cache(opt_type, opt_params, deterministic, PS):
    print(
        f"step in test_prefetch_flush , opt_type = {opt_type} opt_params = {opt_params}"
    )
    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048
    dims = [8, 8, 8]

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            caching=True,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        enable_prefetch=False,
        **opt_params,
    )
    bdeb.enable_prefetch = True
    bdeb.set_record_cache_metrics(True)

    num_embs = [max_capacity // 2 for d in dims]
    stbe = create_split_table_batched_embedding(
        table_names,
        feature_table_map,
        _split_table_batched_optimizer_for_compare(opt_type),
        opt_params,
        dims,
        num_embs,
        POOLING_MODE[DynamicEmbPoolingMode.NONE],
        device,
    )
    init_embedding_tables(stbe, bdeb)

    forward_stream = torch.cuda.Stream()
    pretch_stream = torch.cuda.Stream()

    # 1. Prepare input
    # Input A
    """
    feature number = 4, batch size = 2

    f0  [0, 1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indicesA = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(
        key_type
    )
    offsetsA = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)

    # Input B
    # A intersection B is not none
    """
    feature number = 4, batch size = 2

    f0  [4, 12],        [55],
    f1  [2, 17],        [1],
    f2  [],             [5, 13, 105],
    f3  [0, 23],        [42]
    """
    indicesB = torch.tensor(
        [4, 12, 55, 2, 17, 1, 5, 13, 105, 0, 23, 42], device=device
    ).to(key_type)
    offsetsB = torch.tensor([0, 2, 3, 5, 6, 6, 9, 11, 12], device=device).to(key_type)

    # stream capture will bring a cudaMalloc.
    with torch.cuda.stream(forward_stream):
        indicesB + 1
    with torch.cuda.stream(pretch_stream):
        indicesB + 1

    # 2. Test prefetch works when Cache empty
    with torch.cuda.stream(pretch_stream):
        assert list(bdeb.get_score().values()) == [1] * len(dims)
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        assert list(bdeb.get_score().values()) == [2] * len(dims)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(pretch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()

    embs_stbe_A = stbe(indicesA, offsetsA)
    loss_stbe_A = embs_stbe_A.mean()
    loss_stbe_A.backward()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_stbe_A, rtol=1e-06, atol=1e-06)

        cache = bdeb.cache
        if cache is not None:
            metrics = cache.cache_metrics
            assert metrics[0].item() == metrics[1].item()

    with torch.no_grad():
        bdeb.flush()
        bdeb.reset_cache_states()
        # bdeb.set_score({table_name:1 for table_name in table_names})

    # 3. Test prefetch works when Cache not empty
    with torch.cuda.stream(pretch_stream):
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        assert list(bdeb.get_score().values()) == [3] * len(dims)
        bdeb.prefetch(indicesB, offsetsB, forward_stream)
        assert list(bdeb.get_score().values()) == [4] * len(dims)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(pretch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()
        embs_bdeb_B = bdeb(indicesB, offsetsB)
        loss_bdet_B = embs_bdeb_B.mean()
        loss_bdet_B.backward()

    embs_stbe_A = stbe(indicesA, offsetsA)
    loss_stbe_A = embs_stbe_A.mean()
    loss_stbe_A.backward()
    embs_stbe_B = stbe(indicesB, offsetsB)
    loss_stbe_B = embs_stbe_B.mean()
    loss_stbe_B.backward()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(embs_bdeb_B, embs_stbe_B, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_B, loss_stbe_B, rtol=1e-06, atol=1e-06)

        cache = bdeb.cache
        if cache is not None:
            metrics = cache.cache_metrics
            assert metrics[0].item() == metrics[1].item()

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
def test_multi_table_caching_flush(opt_type, opt_params):
    """Multi-table caching mode: prefetch, forward, flush, verify cache metrics.

    ``DynamicEmbCache`` updates ``cache_metrics[0]`` (last lookup unique count) and
    ``[1]`` (last lookup hit count) only inside ``lookup()``, overwriting prior values.
    Prefetch calls ``lookup``; training forward with cache loads from flat via
    ``slot_indices`` and does not call ``lookup`` again. The first prefetch is thus
    cold (all misses); a second prefetch on the same keys should record all hits.
    """
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 2048

    # Caching tier + backing storage is only created when total value bytes exceed
    # the summed per-table local_hbm_for_values (see _create_cache_storage).
    # Keep HBM budget below table footprint so DynamicEmbCache is instantiated.
    local_hbm_per_table = 64 * 1024  # 64 KiB each -> 128 KiB total < ~192 KiB data

    dyn_emb_table_options_list = []
    for dim in dims:
        opt = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            caching=True,
            local_hbm_for_values=local_hbm_per_table,
        )
        dyn_emb_table_options_list.append(opt)

    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    bdeb.enable_prefetch = True
    bdeb.set_record_cache_metrics(True)

    indices = torch.tensor([0, 1, 10, 11], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4], dtype=key_type, device=device)

    forward_stream = torch.cuda.Stream()
    prefetch_stream = torch.cuda.Stream()

    with torch.cuda.stream(forward_stream):
        indices + 1
    with torch.cuda.stream(prefetch_stream):
        indices + 1

    with torch.cuda.stream(prefetch_stream):
        bdeb.prefetch(indices, offsets, forward_stream)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        embs = bdeb(indices, offsets)
        loss = embs.mean()
        loss.backward()

    torch.cuda.synchronize()

    cache = bdeb.cache
    assert cache is not None
    metrics = cache.cache_metrics
    assert metrics[0].item() == 4
    assert metrics[1].item() == 0

    with torch.cuda.stream(prefetch_stream):
        bdeb.prefetch(indices, offsets, forward_stream)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        embs2 = bdeb(indices, offsets)
        _ = embs2.mean()

    torch.cuda.synchronize()
    assert metrics[0].item() == metrics[1].item() == 4

    bdeb.flush()
    bdeb.reset_cache_states()

    print("test_multi_table_caching_flush passed")
