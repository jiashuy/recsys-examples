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

"""Capacity growth of BatchedDynamicEmbeddingTablesV2 as keys are inserted.

Level: unit.   Needs: 1 GPU.
Covers: a table that expands (rehash) until every slot is used still returns the
embeddings an FBGEMM reference holds, for single and multi table modules, with and
without the cache, across storage backends; capacity and occupancy grow as expected.
Does not cover (see unit/table/test_expansion_value_capacity.py): value-buffer
expansion of a single table.
"""

from typing import Any, Dict, List, Optional

import pytest
import torch
from common.bdet_expansion import (
    _CACHE_RATIO,
    _WORLD_SIZE,
    _EMB_DIM_EXPANSION,
    _BUCKET_CAPACITY_EXP,
    _STRICT_MAX_LOAD_FACTOR,
    _init_capacity_strict,
    _embedding_configs_for_strict_expansion_test,
    _EXPANSION_STBE_TRAIN_BATCH,
    _init_stbe_debug_embedding_weights,
    _bdebt_forward_maybe_lfu,
    _iter_compare_batches_single,
    _iter_compare_batches_multi_three,
    _kim_occupied_sizes_per_table,
    _kim_capacities_per_table,
    _print_test_expansion_storage_cache_sizes_after_train_step,
    _stbe_size_capacity_for_table,
    _assert_bdebt_storage_cache_merged_export_matches_lookup,
    _compare_bdebt_export_to_stbe_weights,
    _assert_expected_storage_backend,
)
from common.bdet_builders import (
    _split_table_batched_optimizer_for_compare,
    create_split_table_batched_embedding,
)
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    get_sharded_table_capacity,
    get_table_value_bytes,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import ScoreStrategy, _sharded_table_bucket_layout
from dynamicemb.key_value_table import DynamicEmbCache, DynamicEmbStorage, HybridStorage
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.LFU,
        DynamicEmbScoreStrategy.STEP,
        DynamicEmbScoreStrategy.NO_EVICTION,
    ],
)
@pytest.mark.parametrize(
    "caching,hbm_budget_ratio",
    [
        (True, None),
        (False, 0.25),
        (False, 1.0),
    ],
)
@pytest.mark.parametrize("multi_table", [False, True])
@pytest.mark.parametrize(
    "optimizer_type,opt_params",
    [
        (EmbOptimType.SGD, {}),
        (
            EmbOptimType.ADAM,
            {
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
def test_table_expansion_capacity_growth(
    score_strategy: ScoreStrategy,
    caching: bool,
    hbm_budget_ratio: Optional[float],
    multi_table: bool,
    optimizer_type: EmbOptimType,
    opt_params: Dict[str, Any],
) -> None:
    """Train keys ``0..CMP_CAPACITY-1`` twice vs STBE (half max rows), then export/compare embeddings.

    After each forward pair, runs backward with ``learning_rate=0`` and ``weight_decay=0`` so
    fused backward/optimizer steps do not move embeddings while still exercising the training path
    (DEBUG ``key % mod`` pattern preserved for export/compare).
    """
    pytest.importorskip("torchrec")

    # HybridStorage (non-caching, partial-HBM: hbm_budget_ratio < 1.0) no longer
    # supports NO_EVICTION -- it raises at construction. Skip that combination.
    if (
        score_strategy == DynamicEmbScoreStrategy.NO_EVICTION
        and not caching
        and hbm_budget_ratio < 1.0
    ):
        pytest.skip(
            "HybridStorage (non-caching partial-HBM) does not support NO_EVICTION"
        )

    assert torch.cuda.is_available()
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")
    key_type = torch.int64

    (
        emb_cfgs,
        table_names,
        feature_table_map,
    ) = _embedding_configs_for_strict_expansion_test(multi_table)
    memory_ratio = _CACHE_RATIO if caching else float(hbm_budget_ratio)

    table_options: List[DynamicEmbTableOptions] = []
    user_max_by_table: List[int] = []
    dims_list: List[int] = []
    for ec in emb_cfgs:
        _, bucket_cap = _sharded_table_bucket_layout(
            ec, _WORLD_SIZE, _BUCKET_CAPACITY_EXP
        )
        max_capacity = get_sharded_table_capacity(ec, _WORLD_SIZE, _BUCKET_CAPACITY_EXP)
        user_max_by_table.append(max_capacity)
        dims_list.append(_EMB_DIM_EXPANSION)
        init_capacity = _init_capacity_strict(max_capacity, bucket_cap)
        value_bytes = get_table_value_bytes(
            ec, optimizer_type, _WORLD_SIZE, _BUCKET_CAPACITY_EXP
        )
        local_hbm = max(1, int(value_bytes * memory_ratio))
        table_options.append(
            DynamicEmbTableOptions(
                dim=_EMB_DIM_EXPANSION,
                max_capacity=max_capacity,
                init_capacity=init_capacity,
                max_load_factor=_STRICT_MAX_LOAD_FACTOR,
                bucket_capacity=bucket_cap,
                index_type=key_type,
                embedding_dtype=torch.float32,
                device_id=device_id,
                score_strategy=score_strategy,
                caching=caching,
                local_hbm_for_values=local_hbm,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG,
                ),
                eval_initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT,
                    value=0.0,
                ),
            )
        )

    train_opt_params = dict(opt_params)
    train_opt_params["learning_rate"] = 0.0
    train_opt_params["weight_decay"] = 0.0

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=table_options,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=optimizer_type,
        use_index_dedup=True,
        **train_opt_params,
    )
    _assert_expected_storage_backend(
        bdebt, caching=caching, hbm_budget_ratio=hbm_budget_ratio
    )

    storage = bdebt.tables
    if isinstance(storage, DynamicEmbStorage):
        km = storage.key_index_map
        nt = storage._state.num_tables
        storage_sz = _kim_occupied_sizes_per_table(km, nt)
        storage_caps = _kim_capacities_per_table(km, nt)
        print(
            f"[test_table_expansion_capacity_growth] storage KIM size (per table): "
            f"{storage_sz}, capacity (per table): {storage_caps}",
            flush=True,
        )
    elif isinstance(storage, HybridStorage):
        hbm, host = storage.tables
        hkm, okm = hbm.key_index_map, host.key_index_map
        nt_h, nt_o = hbm.num_tables, host.num_tables
        h_sz = _kim_occupied_sizes_per_table(hkm, nt_h)
        o_sz = _kim_occupied_sizes_per_table(okm, nt_o)
        h_caps = _kim_capacities_per_table(hkm, nt_h)
        o_caps = _kim_capacities_per_table(okm, nt_o)
        print(
            f"[test_table_expansion_capacity_growth] storage HBM KIM size: {h_sz}, "
            f"capacity: {h_caps}; host KIM size: {o_sz}, capacity: {o_caps}",
            flush=True,
        )
    else:
        print(
            f"[test_table_expansion_capacity_growth] storage type: {type(storage).__name__}",
            flush=True,
        )
    cache = bdebt.cache
    if isinstance(cache, DynamicEmbCache):
        ckm = cache.key_index_map
        nt_c = cache._state.num_tables
        cache_sz = _kim_occupied_sizes_per_table(ckm, nt_c)
        cache_caps = _kim_capacities_per_table(ckm, nt_c)
        print(
            f"[test_table_expansion_capacity_growth] cache KIM size (per table): "
            f"{cache_sz}, capacity (per table): {cache_caps}",
            flush=True,
        )
    else:
        print(
            "[test_table_expansion_capacity_growth] cache: disabled (no DynamicEmbCache)",
            flush=True,
        )

    cmp_caps = [max(1, m // 2) for m in user_max_by_table]
    stbe_opt = _split_table_batched_optimizer_for_compare(optimizer_type)
    stbe = create_split_table_batched_embedding(
        table_names,
        feature_table_map,
        stbe_opt,
        train_opt_params,
        dims_list,
        cmp_caps,
        PoolingMode.NONE,
        device,
    )
    _init_stbe_debug_embedding_weights(stbe)

    stbe_parts: List[str] = []
    for tid, name in enumerate(table_names):
        w_rows, rpt = _stbe_size_capacity_for_table(stbe, tid)
        stbe_parts.append(f"{name!r}:weight_rows={w_rows},rows_per_table={rpt}")
    print(
        f"[test_table_expansion_capacity_growth] STBE table size (per logical table): "
        + "; ".join(stbe_parts),
        flush=True,
    )

    bdebt.train()
    stbe.train()
    # we need to do backward to unlock the keys in the cache, otherwise overflow buffer will be also full.
    for i in range(2):
        if not multi_table:
            for j, (idx, off) in enumerate(
                _iter_compare_batches_single(
                    cmp_caps[0], _EXPANSION_STBE_TRAIN_BATCH, device, key_type
                )
            ):
                res_b = _bdebt_forward_maybe_lfu(
                    bdebt, score_strategy, idx, off, device
                )
                res_s = stbe(idx, off)
                res_b.mean().backward()
                res_s.mean().backward()
                torch.cuda.synchronize()
                _print_test_expansion_storage_cache_sizes_after_train_step(
                    bdebt, outer_iter=i, batch_index=j
                )
        else:
            for j, (idx, off) in enumerate(
                _iter_compare_batches_multi_three(
                    cmp_caps, _EXPANSION_STBE_TRAIN_BATCH, device, key_type
                )
            ):
                res_b = _bdebt_forward_maybe_lfu(
                    bdebt, score_strategy, idx, off, device
                )
                # Merged storage+cache export (no flush) runs once after the train loop;
                # see ``_assert_bdebt_storage_cache_merged_export_matches_lookup`` below.
                res_s = stbe(idx, off)
                res_b.mean().backward()
                res_s.mean().backward()
                torch.cuda.synchronize()
                _print_test_expansion_storage_cache_sizes_after_train_step(
                    bdebt, outer_iter=i, batch_index=j
                )
        torch.cuda.synchronize()

    if caching:
        _assert_bdebt_storage_cache_merged_export_matches_lookup(bdebt, device)

    _compare_bdebt_export_to_stbe_weights(
        bdebt,
        stbe,
        table_names,
        cmp_caps,
        device,
        dump_sample_embeddings=(not multi_table and caching),
    )
