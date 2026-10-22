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

"""Filling BatchedDynamicEmbeddingTablesV2 tables to a load factor.

Level: unit.   Needs: 1 GPU.
Covers: test_fill_tables, parametrized over load factor and table configuration.
"""

import time

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import ScoreStrategy
from dynamicemb.key_value_table import DynamicEmbStorage


@pytest.mark.parametrize("load_factor", [0.1, 1.0])
@pytest.mark.parametrize(
    "max_capacity",
    [
        pytest.param(8_192, id="cap_8k"),
        pytest.param(1_048_576, id="cap_1M"),
        pytest.param(64 * 1024 * 1024, id="cap_64M"),
    ],
)
@pytest.mark.parametrize(
    "score_strategy",
    [DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU],
    ids=lambda s: s.name,
)
def test_fill_tables(
    score_strategy: ScoreStrategy,
    load_factor: float,
    max_capacity: int,
) -> None:
    """
    ``fill_tables`` timing and correctness across table scales (up to 64M slots
    requested), load factors, and score strategies (``TIMESTAMP`` / ``LFU``).
    Physical ``capacity`` may be bucket-aligned above ``max_capacity``.

    Uses ``tolerance=0`` (integer entry target only; no load-factor early exit).
    ``effective_lf = min(requested_load_factor, 0.95)`` (``fill_tables`` caps at
    ``0.95``). For ``effective_lf < 0.95`` asserts exact
    ``size == min(cap, int(effective_lf * cap))``.
    For ``effective_lf == 0.95`` (e.g. requested ``1.0`` clamped), inserts can evict
    while ``remaining`` counts batch inserts; assert ``size`` in a loose band vs target.

    At 64M scale, ``dim=1`` keeps HBM footprint modest; smaller capacities use
    ``dim=8``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    dim = 1 if max_capacity >= 16 * 1024 * 1024 else 8
    local_hbm = max(
        512 * 1024 * 1024, int(max_capacity * dim * 4 * 1.25) + 64 * 1024 * 1024
    )

    opt = DynamicEmbTableOptions(
        dim=dim,
        max_capacity=max_capacity,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device_id,
        score_strategy=score_strategy,
        caching=False,
        local_hbm_for_values=local_hbm,
        bucket_capacity=128,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.NORMAL,
        ),
    )
    m = BatchedDynamicEmbeddingTablesV2(
        table_options=[opt],
        table_names=["t0"],
        device=device,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.01,
    )
    assert isinstance(m._storage, DynamicEmbStorage)
    km = m._storage.key_index_map

    fill_tol = 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m.fill_tables(load_factor, tolerance=fill_tol)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cap0 = km.capacity(0)
    size0 = km.size(0)
    size_i = int(size0.item()) if torch.is_tensor(size0) else int(size0)
    effective_lf = min(load_factor, 0.95)
    actual_lf = float(size_i) / float(cap0)
    expected = min(cap0, int(effective_lf * cap0))
    print(
        f"\nfill_tables score_strategy={score_strategy.name} load_factor={load_factor} "
        f"max_capacity(requested)={max_capacity} dim={dim} cap0={cap0} size0={size_i} "
        f"effective_lf={effective_lf} actual_lf={actual_lf:.6f} expected={expected} "
        f"tol={fill_tol} time_sec={elapsed:.4f}"
    )
    if effective_lf >= 0.95:
        assert size_i <= cap0
        assert size_i >= int(0.92 * expected), (
            f"fill_tables({score_strategy.name}, {load_factor} -> effective {effective_lf}): "
            f"size={size_i} below 0.92 * int_target={expected} capacity={cap0}"
        )
    else:
        assert size_i == expected, (
            f"fill_tables({score_strategy.name}, {load_factor} -> effective {effective_lf}): "
            f"size={size_i} expected={expected} capacity={cap0}"
        )
