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

"""BatchedDynamicEmbeddingTablesV2 forward and backward against an FBGEMM reference.

Level: unit.   Needs: 1 GPU.
Covers: forward in train/eval mode, backward updates per optimizer, deterministic
insert, empty batches, tables of mixed dims in one module, tables with admission.
Does not cover (see test_cache_flush.py): prefetch / flush of the cache;
(see test_dump_load.py): checkpoints.
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
from common.features import random_indices
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import DynamicEmbStorage, HybridStorage


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
    ],
)
@pytest.mark.parametrize(
    "deterministic",
    [True, False],
)
@pytest.mark.parametrize(
    "caching, PS, local_hbm_for_values",
    [
        (True, None, 1024),  # Caching + DynamicEmbStorage backend
        (True, PyDictStorage, 1024),  # Caching + PS backend
        (False, None, 1024**3),  # HBM-only
        (False, None, 1024),  # HybridStorage
        (False, None, 0),  # No HBM budget, no cache -> DynamicEmbStorage (host-only)
    ],
)
@pytest.mark.parametrize(
    "pooling_mode, dims",
    [
        (DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (DynamicEmbPoolingMode.NONE, [7, 7, 7]),
        (DynamicEmbPoolingMode.SUM, [8, 8, 8]),
        (DynamicEmbPoolingMode.MEAN, [8, 8, 8]),
        (DynamicEmbPoolingMode.SUM, [8, 16, 32]),
        (DynamicEmbPoolingMode.MEAN, [8, 16, 32]),
        (DynamicEmbPoolingMode.SUM, [7, 11, 13]),
        (DynamicEmbPoolingMode.MEAN, [7, 11, 13]),
    ],
)
def test_forward_train_eval(
    opt_type,
    opt_params,
    caching,
    deterministic,
    PS,
    local_hbm_for_values,
    pooling_mode,
    dims,
):
    print(
        f"step in test_forward_train_eval , opt_type = {opt_type} opt_params = {opt_params}"
        f" caching = {caching} PS = {PS} local_hbm_for_values = {local_hbm_for_values}"
        f" pooling_mode = {pooling_mode} dims = {dims}"
    )

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    feature_table_map = [0, 0, 1, 2]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=local_hbm_for_values,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    if caching:
        assert bdebt._cache is not None, "Caching mode should create a cache"
    elif local_hbm_for_values == 1024**3:
        assert bdebt._cache is None, "HBM-only mode should have no cache"
        assert isinstance(
            bdebt._storage, DynamicEmbStorage
        ), f"Expected DynamicEmbStorage, got {type(bdebt._storage)}"
    elif local_hbm_for_values == 0:
        assert bdebt._cache is None, "Host-only mode should have no cache"
        assert isinstance(
            bdebt._storage, DynamicEmbStorage
        ), f"Expected DynamicEmbStorage when local_hbm=0 without cache, got {type(bdebt._storage)}"
    else:
        assert bdebt._cache is None, "HybridStorage mode should have no cache"
        assert isinstance(
            bdebt._storage, HybridStorage
        ), f"Expected HybridStorage, got {type(bdebt._storage)}"

    """
    feature number = 4, batch size = 2

    f0  [0,1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indices = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )
    offsets = torch.tensor(
        [0, 2, 3, 5, 6, 8, 10, 10, 11], dtype=key_type, device=device
    )
    batch_size = 2

    embs_train = bdebt(indices, offsets)
    torch.cuda.synchronize()

    # Verify output shape
    if pooling_mode == DynamicEmbPoolingMode.NONE:
        assert embs_train.shape == (indices.numel(), dims[0])
    else:
        total_D = sum(dims[feature_table_map[f]] for f in range(len(feature_table_map)))
        assert embs_train.shape == (batch_size, total_D)

    with torch.no_grad():
        bdebt.eval()
        embs_eval = bdebt(indices, offsets)

    # corner case when all keys missed in eval.
    indices_ne_all = indices + 1024
    bdebt(indices_ne_all, offsets)

    torch.cuda.synchronize()

    # Train and eval should produce identical results for the same keys
    torch.testing.assert_close(embs_train, embs_eval)

    # non-exist key: replace index 0 (key=0) with key=777
    indices_ne = torch.tensor(
        [777, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
    ).to(key_type)
    offsets_ne = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(
        key_type
    )
    embs_non_exist = bdebt(indices_ne, offsets_ne)
    torch.cuda.synchronize()

    # train
    bdebt.train()
    embs_train_non_exist = bdebt(indices_ne, offsets_ne)
    torch.cuda.synchronize()

    if pooling_mode == DynamicEmbPoolingMode.NONE:
        # Sequence mode: row 0 is the embedding for key 777
        # In eval, non-exist key -> zero embedding
        torch.testing.assert_close(embs_train[1:, :], embs_non_exist[1:, :])
        assert torch.all(embs_non_exist[0, :] == 0)
        # In train, non-exist key gets initialized -> non-zero
        assert torch.all(embs_train_non_exist[0, :] != 0)
        torch.testing.assert_close(embs_train_non_exist[1:, :], embs_non_exist[1:, :])
    else:
        # Pooled mode: sample 1 is unaffected by the non-exist key (key 777
        # only appears in sample 0's f0 bag).
        torch.testing.assert_close(embs_non_exist[1, :], embs_train[1, :])
        torch.testing.assert_close(embs_train_non_exist[1, :], embs_non_exist[1, :])
        # Sample 0 should differ from the original because key 777 replaced
        # key 0 in f0's bag.  In eval the missing key contributes zero, so
        # the pooled result for sample 0 changes compared to embs_train.
        assert not torch.allclose(embs_non_exist[0, :], embs_train[0, :])

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]

    print("all check passed")


"""
For torchrec's adam optimizer, it will increment the optimizer_step in every forward,
    which will affect the weights update, pay attention to it or try to use `set_optimizer_step()`
    to control(not verified) it.
"""


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
@pytest.mark.parametrize(
    "caching, pooling_mode, dims",
    [
        (True, DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (False, DynamicEmbPoolingMode.NONE, [16, 16, 16]),
        (False, DynamicEmbPoolingMode.NONE, [17, 17, 17]),
        (False, DynamicEmbPoolingMode.SUM, [128, 32, 16]),
        (False, DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_backward(opt_type, opt_params, caching, pooling_mode, dims, deterministic, PS):
    print(f"step in test_backward , opt_type = {opt_type} opt_params = {opt_params}")

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    cmp_with_torchrec = True
    for dim in dims:
        if dim % 4 != 0:
            cmp_with_torchrec = False
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        **opt_params,
    )
    num_embs = [max_capacity // 2 for d in dims]

    if cmp_with_torchrec:
        stbe = create_split_table_batched_embedding(
            table_names,
            feature_table_map,
            _split_table_batched_optimizer_for_compare(opt_type),
            opt_params,
            dims,
            num_embs,
            POOLING_MODE[pooling_mode],
            device,
        )
        init_embedding_tables(stbe, bdeb)
        """
        feature number = 4, batch size = 2

        f0  [0,1],      [12],
        f1  [64,8],     [12],
        f2  [15, 2, 7], [105],
        f3  [],         [0]
        """
        for i in range(10):
            indices = torch.tensor(
                [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
            ).to(key_type)
            offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(
                key_type
            )

            embs_bdeb = bdeb(indices, offsets)
            embs_stbe = stbe(indices, offsets)

            torch.cuda.synchronize()
            with torch.no_grad():
                torch.testing.assert_close(embs_bdeb, embs_stbe, rtol=1e-06, atol=1e-06)

            loss = embs_bdeb.mean()
            loss.backward()
            loss_stbe = embs_stbe.mean()
            loss_stbe.backward()

            torch.cuda.synchronize()
            torch.testing.assert_close(loss, loss_stbe)

            print(f"Passed iteration {i}")
    else:
        # This scenario will not test correctness, but rather test whether it functions correctly.
        for i in range(10):
            indices = torch.tensor(
                [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
            ).to(key_type)
            offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(
                key_type
            )

            embs_bdeb = bdeb(indices, offsets)
            loss = embs_bdeb.mean()
            loss.backward()

            torch.cuda.synchronize()

            print(f"Passed iteration {i}")

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("PS", [None])
@pytest.mark.parametrize("iteration", [16])
@pytest.mark.parametrize("batch_size", [2048, 65536])  # ,[])
def test_deterministic_insert(opt_type, opt_params, caching, PS, iteration, batch_size):
    print(
        f"step in test_deterministic_insert , opt_type = {opt_type} opt_params = {opt_params}"
    )

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8]
    table_names = ["table0"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = iteration * batch_size
    max_capacity = init_capacity

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=init_capacity * dim * 4,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt_x = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    bdebt_y = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    print(
        f"Test deterministic insert with batch={batch_size}, iteration={iteration}, capacity={init_capacity}"
    )
    os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    for i in range(iteration):
        indices = torch.tensor(
            list(random_indices(batch_size, 0, 2**63 - 1)),
            dtype=key_type,
            device=device,
        )
        offsets = torch.arange(0, batch_size + 1, dtype=key_type, device=device)

        bdebt_x(indices, offsets)
        bdebt_y(indices, offsets)

        torch.cuda.synchronize()

        storage_x = bdebt_x.tables
        storage_y = bdebt_y.tables
        map_x = storage_x.key_index_map
        map_y = storage_y.key_index_map

        assert torch.equal(map_x.keys_, map_y.keys_)

        print(
            f"Iteration {i} passed for deterministic insertion with table_x's size({map_x.size()}), table_y's size({map_y.size()}), totoal({map_x.capacity()})"
        )
        cache_x = bdebt_x.cache
        cache_y = bdebt_y.cache
        if cache_x is not None:
            map_x = cache_x.key_index_map
            map_y = cache_y.key_index_map

            assert torch.equal(map_x.keys_, map_y.keys_)

            print(
                f"Iteration {i} passed for deterministic insertion with cache_x's size({map_x.size()}), cache_y's size({map_y.size()}), totoal({map_x.capacity()})"
            )

    del os.environ["DEMB_DETERMINISM_MODE"]
    print("all check passed")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("dim", [7, 8])
@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None])
def test_empty_batch(opt_type, opt_params, dim, caching, deterministic, PS):
    print(
        f"step in test_forward_train_eval_empty_batch , opt_type = {opt_type} opt_params = {opt_params}"
    )

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [dim, dim, dim]
    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = 1024
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0, 0, 1, 2],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    bdebt.enable_prefetch = True
    """
    feature number = 4, batch size = 1

    f0  [],     
    f1  [],  
    f2  [],  
    f3  [],       
    """
    indices = torch.tensor([], dtype=key_type, device=device)
    offsets = torch.tensor([0, 0, 0, 0, 0], dtype=key_type, device=device)

    pretch_stream = torch.cuda.Stream()
    forward_stream = torch.cuda.Stream()

    if caching:
        with torch.cuda.stream(pretch_stream):
            bdebt.prefetch(indices, offsets, forward_stream)
            torch.cuda.synchronize()

    with torch.cuda.stream(forward_stream):
        res = bdebt(indices, offsets)
        torch.cuda.synchronize()

        res.mean().backward()

        with torch.no_grad():
            bdebt.eval()
            bdebt(indices, offsets)
        torch.cuda.synchronize()

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]

    print("all check passed")


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
    ],
)
@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize(
    "pooling_mode, dims",
    [
        (DynamicEmbPoolingMode.SUM, [8, 16, 32]),
        (DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
def test_multi_table_mixed_dims_forward_backward(
    opt_type, opt_params, caching, pooling_mode, dims
):
    """Multi-table forward/backward with mixed embedding dimensions."""
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    feature_table_map = [0, 1, 2]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        opt = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
        )
        dyn_emb_table_options_list.append(opt)

    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    assert isinstance(bdeb.tables, DynamicEmbStorage)
    assert bdeb.tables.num_tables == 3

    batch_size = 4
    indices = torch.tensor(
        [0, 1, 2, 3, 10, 11, 12, 13, 100, 101, 102, 103],
        dtype=key_type,
        device=device,
    )
    offsets = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=key_type,
        device=device,
    )

    for i in range(5):
        embs = bdeb(indices, offsets)
        torch.cuda.synchronize()

        total_D = sum(dims[feature_table_map[f]] for f in range(len(feature_table_map)))
        assert embs.shape == (batch_size, total_D)

        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    print("test_multi_table_mixed_dims_forward_backward passed")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
@pytest.mark.parametrize("caching", [True, False])
def test_multi_table_with_admission(opt_type, opt_params, caching):
    """Multi-table with frequency-based admission counter."""
    from dynamicemb.embedding_admission import FrequencyAdmissionStrategy, KVCounter

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 2048

    admit_strategy = FrequencyAdmissionStrategy(
        threshold=2, counter=KVCounter(capacity=max_capacity)
    )

    dyn_emb_table_options_list = []
    for dim in dims:
        opt = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            admit_strategy=admit_strategy,
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

    assert bdeb._admission_counter is not None

    batch_size = 2
    indices = torch.tensor([0, 1, 10, 11], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4], dtype=key_type, device=device)

    for i in range(5):
        embs = bdeb(indices, offsets)
        torch.cuda.synchronize()

        total_D = sum(dims[f] for f in feature_table_map)
        assert embs.shape == (batch_size, total_D)

        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    print("test_multi_table_with_admission passed")
