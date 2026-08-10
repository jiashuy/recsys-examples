# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module-level tests for ``BatchedDynamicEmbeddingTablesV2.pop_evicted_keys``.

The important coverage is the *training forward+backward path*. Non-caching
training inserts keys through ``_prefetch_hbm_direct_path`` (a last tier). Because
prefetched keys are ref-counter protected (counter > 0), an overflow within a
single step evicts nothing (the newcomers just fail to insert / Busy) -- a real
eviction only happens ACROSS steps, once backward has decremented the previous
step's keys back to counter 0. So these tests run full forward+backward steps,
then assert the retained set. This is a different insert path than the direct
``storage.insert`` the storage-level tests exercise, and it must also retain.
Plus the module API's per-table filtering.
"""

import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EvictedItemMode,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

LRU_LFU = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _build_model(
    device_id,
    retain,
    caching=False,
    table_num=1,
    max_capacity=256,
    dim=8,
    index_type=torch.int64,
):
    """A BatchedDynamicEmbeddingTablesV2 with one feature per table. Small
    max_capacity so training overflows the last tier and evicts across steps."""
    retain = [retain] * table_num if isinstance(retain, bool) else retain
    options_list = [
        DynamicEmbTableOptions(
            index_type=index_type,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=dim,
            max_capacity=max_capacity,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=LRU_LFU,
            caching=caching,
            evicted_item_mode=(
                EvictedItemMode.RETAIN_KEY if retain[i] else EvictedItemMode.DISCARD
            ),
        )
        for i in range(table_num)
    ]
    return BatchedDynamicEmbeddingTablesV2(
        table_options=options_list,
        output_dtype=torch.float32,
        table_names=[f"t_{i}" for i in range(table_num)],
        feature_table_map=list(range(table_num)),  # one feature per table
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
    )


def _train_step(model, per_table_keys, device):
    """One full training step (forward + backward) over a KJT with one feature per
    table, one key per bag. backward is what decrements the ref-counter so the
    NEXT step can actually evict (and retain) this step's keys."""
    indices, offsets = [], [0]
    for keys in per_table_keys:
        for k in keys:
            indices.append(k)
            offsets.append(offsets[-1] + 1)
    out = model(
        torch.tensor(indices, dtype=torch.int64, device=device),
        torch.tensor(offsets, dtype=torch.int64, device=device),
    )
    out.sum().backward()
    torch.cuda.synchronize()


def _train_fresh(model, device, steps=20, per=200, base=0, tables=1):
    """`steps` steps of `per` brand-new keys per table (disjoint across steps)."""
    for step in range(steps):
        lo = base + 1 + step * per
        keys = list(range(lo, lo + per))
        per_table = [[k + t * 10_000_000 for k in keys] for t in range(tables)]
        _train_step(model, per_table, device)


def test_module_pop_hbm_direct_training_collects(current_device):
    """Non-caching training (HBM-direct path) that overflows the last tier across
    steps must retain its evicted keys -- the gap the direct-storage.insert path
    did not cover. pop returns unique keys and is incremental (second pop empty)."""
    device = torch.device(f"cuda:{current_device}")
    model = _build_model(current_device, retain=True, caching=False, max_capacity=256)

    _train_fresh(model, device, steps=20, per=200)  # 4000 keys, cap 256

    result = model.pop_evicted_keys()
    assert "t_0" in result, "retain-enabled table must appear in the result"
    evk = result["t_0"]
    assert evk.numel() > 0, "HBM-direct training eviction must be retained (gap fix)"
    assert evk.numel() == len(set(evk.tolist())), "retained keys must be unique"

    # Incremental: a second pop with no new training is empty.
    result2 = model.pop_evicted_keys()
    assert result2.get("t_0", torch.empty(0)).numel() == 0


def test_module_pop_retain_disabled_omitted(current_device):
    """evicted_item_mode=DISCARD: eviction still happens across steps but nothing
    is retained -> the table is omitted from the result entirely."""
    device = torch.device(f"cuda:{current_device}")
    model = _build_model(current_device, retain=False, caching=False, max_capacity=256)
    _train_fresh(model, device, steps=20, per=200)
    assert model.pop_evicted_keys() == {}, "retain-disabled table must be omitted"


def test_module_pop_table_names_filter(current_device):
    """Two retain-enabled tables both evict; pop(table_names=['t_0']) returns and
    clears only t_0, leaving t_1's retained keys for a later pop."""
    device = torch.device(f"cuda:{current_device}")
    model = _build_model(
        current_device, retain=True, caching=False, table_num=2, max_capacity=256
    )
    _train_fresh(model, device, steps=20, per=200, tables=2)

    only0 = model.pop_evicted_keys(table_names=["t_0"])
    assert set(only0.keys()) == {"t_0"}, "table_names filter must restrict to t_0"
    assert only0["t_0"].numel() > 0, "t_0 must have retained evictions"

    # t_1 was untouched by the t_0-only pop and still has its retained keys; t_0
    # is now drained (present but empty -- retain-enabled tables are always keyed).
    rest = model.pop_evicted_keys()
    assert "t_1" in rest and rest["t_1"].numel() > 0, "t_1 survives a t_0-only pop"
    assert rest.get("t_0", torch.empty(0)).numel() == 0, "t_0 was already drained"


@pytest.mark.parametrize("index_type", [torch.int64, torch.uint64])
def test_module_pop_empty_dtype_matches_key_type(current_device, index_type):
    """The empty pop_evicted_keys path must return the table's key dtype
    (``key_index_map.key_type`` == ``index_type``; the key map accepts only 64-bit
    integer keys, int64/uint64), NOT a hardcoded int64. Otherwise a caller
    concatenating an empty pop with a later non-empty pop would hit a dtype
    mismatch. Regression: the empty guard used to hardcode int64."""
    model = _build_model(current_device, retain=True, index_type=index_type)
    # Retain-enabled table is present but has no evictions yet -> an empty tensor
    # whose dtype must already be index_type (not a hardcoded int64).
    empty = model.pop_evicted_keys()["t_0"]
    assert empty.numel() == 0
    assert (
        empty.dtype == index_type
    ), f"empty pop dtype {empty.dtype} != index_type {index_type}"
