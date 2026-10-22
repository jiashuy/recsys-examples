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

"""Checkpoint value layout of BatchedDynamicEmbeddingTablesV2: table precision and optimizer state dtype.

Level: unit.   Needs: 1 GPU (a one-rank NCCL group is created if none exists).
Covers: a dump stores the table precision and the configured optimizer_state_dtype
in its metadata, optimizer_state_dtype is a grouping key and is validated at
construction, a legacy checkpoint without value-layout metadata still loads, and
the optimizer-state width of the dumped rows per optimizer / dim / dtype.
Does not cover (see test_dump_load.py): dist_type metadata and the basic round trip;
(see unit/optimizer/test_rowwise_adagrad_state_precision.py): the kernel-level
accumulator precision.
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.distributed as dist
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
    encode_checkpoint_file_path,
    encode_meta_json_file_path,
)
from dynamicemb.optimizer import DynamicEmbOptimType, get_optimizer_ckpt_state_dim
from dynamicemb.types import KEY_TYPE

_VALUE_LAYOUT_META_KEYS = ("embedding_dtype", "embedding_dim", "optim_state_dtype")

# Rowwise Adagrad rather than SGD so the stored row is not embeddings alone.
# Its checkpoint state is one accumulator per row while its runtime state is
# ``16 // itemsize`` elements (4 at fp32, 8 at fp16/bf16), so a load has to
# widen and re-type the block rather than copy it straight through -- the part
# of _load_key_values that a stateless optimizer never reaches.
_DUMP_LOAD_OPT_TYPE = EmbOptimType.EXACT_ROWWISE_ADAGRAD

_DUMP_LOAD_OPT_PARAMS = {
    "learning_rate": 0.01,
    "eps": 1e-8,
    "initial_accumulator_value": 0.0,
}

pytestmark = pytest.mark.usefixtures("backend_session")


def _init_single_rank_pg(device_id: int = 0) -> torch.device:
    assert torch.cuda.is_available()
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def _shard_file(save_dir: str, table_name: str, kind: str) -> str:
    return encode_checkpoint_file_path(
        save_dir, table_name, dist.get_rank(), dist.get_world_size(), kind
    )


def _read_meta(save_dir: str, table_name: str) -> Dict[str, Any]:
    with open(encode_meta_json_file_path(save_dir, table_name), "r") as f:
        return json.load(f)


def _write_meta(save_dir: str, table_name: str, meta: Dict[str, Any]) -> None:
    with open(encode_meta_json_file_path(save_dir, table_name), "w") as f:
        json.dump(meta, f)


def _dumped_key_count(save_dir: str, table_name: str) -> int:
    return os.path.getsize(_shard_file(save_dir, table_name, "keys")) // (
        _dtype_element_size(KEY_TYPE)
    )


def _make_dump_load_tables(
    dims: List[int],
    table_names: List[str],
    value_type: torch.dtype,
    device_id: int = 0,
    optimizer_state_dtype: Optional[torch.dtype] = None,
) -> BatchedDynamicEmbeddingTablesV2:
    return _make_multi_table_bdeb_for_dump_load(
        _DUMP_LOAD_OPT_TYPE,
        _DUMP_LOAD_OPT_PARAMS,
        dims,
        table_names,
        list(range(len(table_names))),
        1024,
        torch.int64,
        value_type,
        device_id,
        optimizer_state_dtype=optimizer_state_dtype,
    )


def _dump_fp32_source(tmp_path, dims, table_names, device) -> str:
    """Train an fp32 table and dump it; returns the save dir.

    Every precision case starts from the same fp32 rows, so what a later dump
    writes is the only thing under test.
    """
    src = _make_dump_load_tables(dims, table_names, torch.float32)
    _train_multi_table_bdeb_once(src, torch.int64, device)
    save_dir = os.path.join(str(tmp_path), "fp32")
    os.makedirs(save_dir, exist_ok=True)
    src.dump(save_dir, optim=True)
    return save_dir


@pytest.mark.parametrize(
    "value_type",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_dump_stores_table_precision(value_type, tmp_path):
    """A dump writes the table's own precision, and records how to read it back."""
    device = _init_single_rank_pg()
    dims = [8, 16]
    table_names = ["table0", "table1"]

    fp32_dir = _dump_fp32_source(tmp_path, dims, table_names, device)

    mid = _make_dump_load_tables(dims, table_names, value_type)
    with warnings.catch_warnings():
        # Reading fp32 rows into a narrower table warns by design -- that is the
        # conversion path, not the thing under test here.
        warnings.simplefilter("ignore", UserWarning)
        mid.load(fp32_dir, optim=True)

    native_dir = os.path.join(str(tmp_path), "native")
    os.makedirs(native_dir, exist_ok=True)
    mid.dump(native_dir, optim=True)

    dtype_name = str(value_type).split(".")[-1]
    for table_id, name in enumerate(table_names):
        meta = _read_meta(native_dir, name)
        assert meta["embedding_dtype"] == dtype_name
        # Rowwise Adagrad keeps its accumulator in fp32 whatever the table dtype.
        assert meta["optim_state_dtype"] == "float32"
        assert meta["embedding_dim"] == dims[table_id]

        num_keys = _dumped_key_count(native_dir, name)
        assert num_keys > 0, f"{name}: nothing was dumped, the test proves nothing"
        elem = _dtype_element_size(value_type)
        assert (
            os.path.getsize(_shard_file(native_dir, name, "values"))
            == num_keys * dims[table_id] * elem
        )
        # The accumulator is written as fp32 whatever the table's precision --
        # check the file rather than trusting the meta string.
        ckpt_opt_dim = get_optimizer_ckpt_state_dim(
            _DUMP_LOAD_OPT_TYPE, dims[table_id], value_type
        )
        assert os.path.getsize(
            _shard_file(native_dir, name, "opt_values")
        ) == num_keys * ckpt_opt_dim * _dtype_element_size(torch.float32)

    dst = _make_dump_load_tables(dims, table_names, value_type)
    dst.load(native_dir, optim=True)
    for name in table_names:
        keys_mid, vals_mid = mid.export_keys_values(name, device)
        keys_dst, vals_dst = dst.export_keys_values(name, device)
        assert vals_mid.dtype == value_type
        assert vals_dst.dtype == value_type
        order_mid = keys_mid.argsort()
        order_dst = keys_dst.argsort()
        torch.testing.assert_close(keys_mid[order_mid], keys_dst[order_dst])
        # Same precision on both sides, so the round trip is exact.
        torch.testing.assert_close(
            vals_mid[order_mid], vals_dst[order_dst], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "value_type", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_dump_records_configured_optimizer_state_dtype(value_type, tmp_path):
    """A table keeping its accumulator at table precision dumps it that way, and loads back."""
    device = _init_single_rank_pg()
    dims = [8, 16]
    table_names = ["table0", "table1"]

    fp32_dir = _dump_fp32_source(tmp_path, dims, table_names, device)
    mid = _make_dump_load_tables(
        dims, table_names, value_type, optimizer_state_dtype=value_type
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mid.load(fp32_dir, optim=True)

    native_dir = os.path.join(str(tmp_path), "native")
    os.makedirs(native_dir, exist_ok=True)
    mid.dump(native_dir, optim=True)

    dtype_name = str(value_type).split(".")[-1]
    for table_id, name in enumerate(table_names):
        assert _read_meta(native_dir, name)["optim_state_dtype"] == dtype_name
        num_keys = _dumped_key_count(native_dir, name)
        assert num_keys > 0, f"{name}: nothing was dumped, the test proves nothing"
        ckpt_opt_dim = get_optimizer_ckpt_state_dim(
            _DUMP_LOAD_OPT_TYPE, dims[table_id], value_type
        )
        assert os.path.getsize(
            _shard_file(native_dir, name, "opt_values")
        ) == num_keys * ckpt_opt_dim * _dtype_element_size(value_type)

    # The default table keeps an fp32 accumulator and must still read this one.
    dst = _make_dump_load_tables(dims, table_names, value_type)
    dst.load(native_dir, optim=True)
    for name in table_names:
        keys_mid, vals_mid = mid.export_keys_values(name, device)
        keys_dst, vals_dst = dst.export_keys_values(name, device)
        order_mid = keys_mid.argsort()
        order_dst = keys_dst.argsort()
        torch.testing.assert_close(keys_mid[order_mid], keys_dst[order_dst])
        torch.testing.assert_close(
            vals_mid[order_mid], vals_dst[order_dst], rtol=0, atol=0
        )


def test_optimizer_state_dtype_is_a_grouping_key():
    default = DynamicEmbTableOptions(dim=8)
    assert default == DynamicEmbTableOptions(dim=8)
    assert default != DynamicEmbTableOptions(dim=8, optimizer_state_dtype=torch.float16)


def test_unsupported_optimizer_state_dtype_is_rejected_at_construction():
    _init_single_rank_pg()
    with pytest.raises(ValueError, match="only supported by row-wise Adagrad"):
        _make_multi_table_bdeb_for_dump_load(
            EmbOptimType.ADAM,
            {"learning_rate": 0.1},
            [8],
            ["table0"],
            [0],
            1024,
            torch.int64,
            torch.float16,
            0,
            optimizer_state_dtype=torch.float32,
        )


def test_load_reads_legacy_checkpoint_without_value_layout_meta(tmp_path):
    """A checkpoint predating the layout keys loads as fp32, dim from file size.

    The two tables have different dims, so a dim recovered from one file cannot
    accidentally stand in for the other.
    """
    device = _init_single_rank_pg()
    dims = [8, 16]
    table_names = ["table0", "table1"]

    src = _make_dump_load_tables(dims, table_names, torch.float32)
    _train_multi_table_bdeb_once(src, torch.int64, device)
    save_dir = str(tmp_path)
    src.dump(save_dir, optim=True)

    for name in table_names:
        meta = _read_meta(save_dir, name)
        for key in _VALUE_LAYOUT_META_KEYS:
            meta.pop(key, None)
        _write_meta(save_dir, name, meta)

    dst = _make_dump_load_tables(dims, table_names, torch.float32)
    dst.load(save_dir, optim=True)

    for name in table_names:
        keys_src, vals_src = src.export_keys_values(name, device)
        keys_dst, vals_dst = dst.export_keys_values(name, device)
        order_src = keys_src.argsort()
        order_dst = keys_dst.argsort()
        torch.testing.assert_close(keys_src[order_src], keys_dst[order_dst])
        torch.testing.assert_close(vals_src[order_src], vals_dst[order_dst])


def _dtype_element_size(dtype: torch.dtype) -> int:
    return int(torch.empty(0, dtype=dtype).element_size())


def _assert_opt_values_ckpt_row_width(
    save_dir: str,
    table_name: str,
    emb_dim: int,
    value_type: torch.dtype,
    opt_type: EmbOptimType,
) -> None:
    """``opt_values`` checkpoint uses ``get_ckpt_state_dim`` elements per key row."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    keys_path = encode_checkpoint_file_path(
        save_dir, table_name, rank, world_size, "keys"
    )
    opt_path = encode_checkpoint_file_path(
        save_dir, table_name, rank, world_size, "opt_values"
    )
    key_el = _dtype_element_size(KEY_TYPE)
    # Optimizer states share the value row, so they are checkpointed at the
    # table's precision, not at a fixed fp32.
    opt_el = _dtype_element_size(value_type)
    num_keys = os.path.getsize(keys_path) // key_el
    opt_sz = os.path.getsize(opt_path)
    ckpt_elems = get_optimizer_ckpt_state_dim(opt_type, emb_dim, value_type)

    if opt_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        assert ckpt_elems == 1
    elif opt_type == EmbOptimType.EXACT_ADAGRAD:
        assert ckpt_elems == emb_dim
    elif opt_type == EmbOptimType.ADAM:
        assert ckpt_elems == emb_dim * 2
    elif opt_type == DynamicEmbOptimType.FTRL:
        assert ckpt_elems == emb_dim * 2
    elif opt_type == EmbOptimType.SGD:
        assert ckpt_elems == 0
    else:
        raise AssertionError(f"unexpected optimizer type: {opt_type}")

    if ckpt_elems == 0:
        assert opt_sz == 0, (
            f"{table_name}: SGD has no optimizer state in checkpoint; "
            f"expected empty opt_values, got {opt_sz} bytes"
        )
    else:
        assert opt_sz == num_keys * opt_el * ckpt_elems, (
            f"{table_name}: opt_values size={opt_sz}, num_keys={num_keys}, "
            f"expected {num_keys * opt_el * ckpt_elems} (ckpt_elems={ckpt_elems})"
        )


@pytest.mark.parametrize(
    "multi_table", [False, True], ids=["single_table", "multi_table"]
)
@pytest.mark.parametrize("emb_dim", [1, 16, 1023], ids=lambda d: f"dim_{d}")
@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {"learning_rate": 0.01, "eps": 1e-8, "initial_accumulator_value": 0.0},
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {"learning_rate": 0.01, "eps": 1e-8, "initial_accumulator_value": 0.0},
        ),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        ),
        (
            DynamicEmbOptimType.FTRL,
            {
                "learning_rate": 0.01,
                "learning_rate_power": -0.5,
                # FTRL re-solves the weight from its state, so both halves have
                # to start at zero or the first update discards the initializer.
                "initial_accumulator_value": 0.0,
                "l1_reg": 0.0,
                "l2_reg": 0.0,
            },
        ),
        (EmbOptimType.SGD, {"learning_rate": 0.01}),
    ],
    ids=["rowwise_adagrad", "adagrad", "adam", "ftrl", "sgd"],
)
def test_dump_optimizer_states_ckpt_width(
    tmp_path,
    multi_table: bool,
    emb_dim: int,
    opt_type: EmbOptimType,
    opt_params: Dict[str, Any],
) -> None:
    """After train + dump(optim=True), ``opt_values`` row width matches checkpoint layout."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device_id = 0
    torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{device_id}")
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 8192

    if multi_table:
        table_names = ["table0", "table1"]
        feature_table_map = [0, 1]
        dims = [emb_dim, emb_dim]
        indices = torch.tensor(
            [0, 1, 2, 3, 10, 11, 12, 13], dtype=key_type, device=device
        )
        offsets = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=key_type, device=device
        )
    else:
        table_names = ["table0"]
        feature_table_map = [0, 0]
        dims = [emb_dim]
        indices = torch.tensor(
            [0, 1, 12, 64, 8, 12, 15, 2], dtype=key_type, device=device
        )
        offsets = torch.tensor([0, 2, 4, 6, 8], dtype=key_type, device=device)

    opts: List[DynamicEmbTableOptions] = []
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
            )
        )

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=opts,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    torch.manual_seed(emb_dim + (1001 if multi_table else 0))
    for _ in range(4):
        embs = bdebt(indices, offsets)
        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    save_dir = str(tmp_path)
    bdebt.dump(save_dir, optim=True)

    for tid, name in enumerate(table_names):
        _assert_opt_values_ckpt_row_width(
            save_dir, name, dims[tid], value_type, opt_type
        )

    print(
        f"test_dump_optimizer_states_ckpt_width passed "
        f"(multi_table={multi_table}, emb_dim={emb_dim}, opt={opt_type})"
    )
