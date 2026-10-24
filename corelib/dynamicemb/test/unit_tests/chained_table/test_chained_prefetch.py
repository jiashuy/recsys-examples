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

"""Training over the chained index, end to end.

The module is built with ``DEMB_CHAINED_INDEX`` set, so its storage is a
ChainedStorage: HBM-direct with nothing in front of it, or the backing behind an HBM
cache. The check is the one the tier-below tests use: every key starts at 1.0, the
loss is the sum of the outputs, so each occurrence of a key has gradient 1 and plain
SGD leaves a key at ``1 - lr * occurrences`` however often it was evicted, sent to
the tier below and brought back on the way.
"""

import os
import sys

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    StorageLayout,
)
from dynamicemb.chained_storage import ChainedStorage
from fbgemm_gpu.split_embedding_configs import EmbOptimType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from test_batched_dynamic_embedding_tables_v2 import PyDictStorage  # noqa: E402
from test_ps_tier import (  # noqa: E402
    BATCH,
    DIM,
    KEYS,
    LR,
    STEPS,
    DroppingStorage,
    _mismatched_keys,
    _train,
)

LAYOUT = {"hbm": StorageLayout.HBM_ONLY, "caching": StorageLayout.CACHING}


@pytest.fixture(autouse=True)
def chained_index(monkeypatch):
    monkeypatch.setenv("DEMB_CHAINED_INDEX", "1")


def _module(layout, ps, capacity, strategy=DynamicEmbScoreStrategy.TIMESTAMP):
    value_bytes = capacity * DIM * 4
    caching = layout == "caching"
    local_hbm = value_bytes // 2 if caching else 2 * value_bytes
    options = DynamicEmbTableOptions(
        dim=DIM,
        init_capacity=capacity,
        max_capacity=capacity,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        score_strategy=strategy,
        caching=caching,
        local_hbm_for_values=local_hbm,
        external_storage=ps,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT, value=1.0
        ),
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_names=["t"],
        table_options=[options],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=LR,
        use_index_dedup=True,
    )


@pytest.mark.parametrize("layout", ["hbm", "caching"])
@pytest.mark.parametrize(
    "strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.LFU,
        DynamicEmbScoreStrategy.STEP,
    ],
)
def test_what_the_table_evicts_comes_back_from_the_storage_below(layout, strategy):
    module = _module(layout, PyDictStorage, 2048, strategy)
    assert module._storage_layout == LAYOUT[layout]
    storage = module._storage
    assert isinstance(storage, ChainedStorage)
    assert storage.lower is not None

    uses = _train(module)
    assert int((uses > 0).sum()) > 2 * 2048, "the run must overflow the table"
    assert storage.lower.size() > 0, "nothing was evicted into the storage below"

    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys lost an update on the way down or back"
    storage.index.check_invariants(
        [0] * storage.index.num_tables
    )  # no row lost or used twice


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_control_a_storage_that_drops_what_it_is_given_fails_the_check(layout):
    module = _module(layout, DroppingStorage, 2048)
    assert module._storage_layout == LAYOUT[layout]
    uses = _train(module)
    bad, total = _mismatched_keys(module, uses)
    assert bad > 0, "with a lower tier that forgets everything the check still passes"


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_a_table_that_holds_every_key_needs_no_tier_below(layout):
    module = _module(layout, None, 16384)
    assert isinstance(module._storage, ChainedStorage)
    uses = _train(module)
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys are wrong with room for every key"
    module._storage.index.check_invariants([0])


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_a_table_that_never_evicts_grows_with_the_keys(layout):
    module = _module(layout, None, 512, DynamicEmbScoreStrategy.NO_EVICTION)
    storage = module._storage
    assert isinstance(storage, ChainedStorage)
    before = storage.index.capacity_of(0)
    uses = _train(module)
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys were lost while the table grew"
    assert storage.index.capacity_of(0) > before, "the table did not grow"
    assert storage.index.size(0) == int((uses > 0).sum())
    storage.index.check_invariants([0])
