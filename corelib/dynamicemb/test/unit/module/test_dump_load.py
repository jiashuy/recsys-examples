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

"""Dump and load of a BatchedDynamicEmbeddingTablesV2 (module-level checkpoint).

Level: unit.   Needs: 1 GPU (a one-rank NCCL group is created if none exists).
Covers: exporting an empty table, multi-table dump then load round trip per
optimizer, dist_type (roundrobin / hash_roundrobin) preserved, mismatch rejected,
legacy metadata without dist_type defaulting to roundrobin.
Does not cover (see test_dump_value_layout.py): checkpoint value layout / dtypes;
(see unit/table/test_table_dump_load.py): the table-level dump/load.
"""

import json

import pytest
import torch
from common.bdet_builders import (
    _make_multi_table_bdeb_for_dump_load,
    _train_multi_table_bdeb_once,
)
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    encode_meta_json_file_path,
)


def test_export_keys_values_empty_table():
    """export_keys_values() on a never-used table must return empty tensors
    (not crash on torch.cat([])) -- covers the empty keys_list guard."""
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    opt = DynamicEmbTableOptions(
        dim=8,
        init_capacity=1024,
        max_capacity=1024,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=0,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=False,
        local_hbm_for_values=1024**3,
    )
    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=["t0"],
        table_options=[opt],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.1,
    )

    keys, values = bdebt.export_keys_values("t0", device)

    assert keys.shape == (0,)
    assert keys.dtype == torch.int64
    assert values.shape[0] == 0


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
def test_multi_table_dump_load(opt_type, opt_params, tmp_path):
    """Multi-table dump and load using fused storage."""
    import torch.distributed as dist

    assert torch.cuda.is_available()
    device_id = 0
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    device = torch.device(f"cuda:{device_id}")

    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 1024

    def make_bdeb():
        opts = []
        for dim in dims:
            opt = DynamicEmbTableOptions(
                dim=dim,
                init_capacity=max_capacity,
                max_capacity=max_capacity,
                index_type=key_type,
                embedding_dtype=value_type,
                device_id=device_id,
                score_strategy=DynamicEmbScoreStrategy.STEP,
                caching=False,
                local_hbm_for_values=1024**3,
            )
            opts.append(opt)
        return BatchedDynamicEmbeddingTablesV2(
            table_names=table_names,
            table_options=opts,
            feature_table_map=feature_table_map,
            pooling_mode=DynamicEmbPoolingMode.SUM,
            optimizer=opt_type,
            use_index_dedup=True,
            **opt_params,
        )

    bdeb_src = make_bdeb()

    indices = torch.tensor([0, 1, 2, 3, 10, 11, 12, 13], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=key_type, device=device)

    for _ in range(3):
        embs = bdeb_src(indices, offsets)
        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    save_dir = str(tmp_path)
    bdeb_src.dump(save_dir)

    keys0_src, vals0_src = bdeb_src.export_keys_values("table0", device)
    keys1_src, vals1_src = bdeb_src.export_keys_values("table1", device)

    bdeb_dst = make_bdeb()
    bdeb_dst.load(save_dir)

    keys0_dst, vals0_dst = bdeb_dst.export_keys_values("table0", device)
    keys1_dst, vals1_dst = bdeb_dst.export_keys_values("table1", device)

    idx0_src = keys0_src.argsort()
    idx0_dst = keys0_dst.argsort()
    torch.testing.assert_close(keys0_src[idx0_src], keys0_dst[idx0_dst])
    torch.testing.assert_close(vals0_src[idx0_src], vals0_dst[idx0_dst])

    idx1_src = keys1_src.argsort()
    idx1_dst = keys1_dst.argsort()
    torch.testing.assert_close(keys1_src[idx1_src], keys1_dst[idx1_dst])
    torch.testing.assert_close(vals1_src[idx1_src], vals1_dst[idx1_dst])

    print("test_multi_table_dump_load passed")


@pytest.mark.parametrize("dist_type", ["roundrobin", "hash_roundrobin"])
def test_multi_table_dump_load_preserves_dist_type(dist_type, tmp_path):
    import torch.distributed as dist

    assert torch.cuda.is_available()
    device_id = 0
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    device = torch.device(f"cuda:{device_id}")
    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 1024
    opt_type = EmbOptimType.SGD
    opt_params = {"learning_rate": 0.3}

    bdeb_src = _make_multi_table_bdeb_for_dump_load(
        opt_type,
        opt_params,
        dims,
        table_names,
        feature_table_map,
        max_capacity,
        key_type,
        value_type,
        device_id,
        dist_type=dist_type,
    )
    _train_multi_table_bdeb_once(bdeb_src, key_type, device)

    save_dir = str(tmp_path)
    bdeb_src.dump(save_dir)
    meta_path = encode_meta_json_file_path(save_dir, table_names[0])
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    assert meta_data["dist_type"] == dist_type

    bdeb_dst = _make_multi_table_bdeb_for_dump_load(
        opt_type,
        opt_params,
        dims,
        table_names,
        feature_table_map,
        max_capacity,
        key_type,
        value_type,
        device_id,
        dist_type=dist_type,
    )
    bdeb_dst.load(save_dir)

    keys0_src, vals0_src = bdeb_src.export_keys_values("table0", device)
    keys0_dst, vals0_dst = bdeb_dst.export_keys_values("table0", device)
    idx0_src = keys0_src.argsort()
    idx0_dst = keys0_dst.argsort()
    torch.testing.assert_close(keys0_src[idx0_src], keys0_dst[idx0_dst])
    torch.testing.assert_close(vals0_src[idx0_src], vals0_dst[idx0_dst])


def test_multi_table_load_rejects_dist_type_mismatch(tmp_path):
    import torch.distributed as dist

    assert torch.cuda.is_available()
    device_id = 0
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    device = torch.device(f"cuda:{device_id}")
    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 1024
    opt_type = EmbOptimType.SGD
    opt_params = {"learning_rate": 0.3}

    bdeb_src = _make_multi_table_bdeb_for_dump_load(
        opt_type,
        opt_params,
        dims,
        table_names,
        feature_table_map,
        max_capacity,
        key_type,
        value_type,
        device_id,
        dist_type="roundrobin",
    )
    _train_multi_table_bdeb_once(bdeb_src, key_type, device)

    save_dir = str(tmp_path)
    bdeb_src.dump(save_dir)

    bdeb_dst = _make_multi_table_bdeb_for_dump_load(
        opt_type,
        opt_params,
        dims,
        table_names,
        feature_table_map,
        max_capacity,
        key_type,
        value_type,
        device_id,
        dist_type="hash_roundrobin",
    )
    with pytest.raises(ValueError, match="Input dist_type mismatch"):
        bdeb_dst.load(save_dir)


def test_multi_table_load_legacy_metadata_defaults_to_roundrobin(tmp_path):
    import torch.distributed as dist

    assert torch.cuda.is_available()
    device_id = 0
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    device = torch.device(f"cuda:{device_id}")
    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 1024
    opt_type = EmbOptimType.SGD
    opt_params = {"learning_rate": 0.3}

    bdeb_src = _make_multi_table_bdeb_for_dump_load(
        opt_type,
        opt_params,
        dims,
        table_names,
        feature_table_map,
        max_capacity,
        key_type,
        value_type,
        device_id,
        dist_type="roundrobin",
    )
    _train_multi_table_bdeb_once(bdeb_src, key_type, device)

    save_dir = str(tmp_path)
    bdeb_src.dump(save_dir)
    for table_name in table_names:
        meta_path = encode_meta_json_file_path(save_dir, table_name)
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        meta_data.pop("dist_type", None)
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)

    bdeb_dst = _make_multi_table_bdeb_for_dump_load(
        opt_type,
        opt_params,
        dims,
        table_names,
        feature_table_map,
        max_capacity,
        key_type,
        value_type,
        device_id,
        dist_type="roundrobin",
    )
    bdeb_dst.load(save_dir)

    keys1_src, vals1_src = bdeb_src.export_keys_values("table1", device)
    keys1_dst, vals1_dst = bdeb_dst.export_keys_values("table1", device)
    idx1_src = keys1_src.argsort()
    idx1_dst = keys1_dst.argsort()
    torch.testing.assert_close(keys1_src[idx1_src], keys1_dst[idx1_dst])
    torch.testing.assert_close(vals1_src[idx1_src], vals1_dst[idx1_dst])
