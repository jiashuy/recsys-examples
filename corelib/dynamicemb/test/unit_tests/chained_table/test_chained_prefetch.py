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

The module's storage is a
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

LAYOUT = {
    "hbm": StorageLayout.HBM_ONLY,
    "caching": StorageLayout.CACHING,
    "host": StorageLayout.HOST_ONLY,
    "hybrid": StorageLayout.HYBRID,
}


def _module(layout, ps, capacity, strategy=DynamicEmbScoreStrategy.TIMESTAMP, **extra):
    value_bytes = capacity * DIM * 4
    caching = layout == "caching"
    # hbm: room for everything on the device; caching: half; hybrid: an eighth, the rest
    # on the host; host: none
    local_hbm = {
        "hbm": 2 * value_bytes,
        "caching": value_bytes // 2,
        "hybrid": value_bytes // 8,
        "host": 0,
    }[layout]
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
        **extra,
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


# -- a prefetch stream and a backward that runs late ---------------------------


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_a_pipelined_prefetch_does_not_disturb_a_step_in_flight(layout):
    """Step i+1 is prefetched, on a stream of its own, while step i's backward is still
    queued behind a long kernel: its victims must not be rows step i is using, and its
    tensors must outlive the stream they were made on. Without the stream bookkeeping
    of the prefetch state the cache layout fails here with an illegal memory access;
    the HBM-direct layout shows nothing either way, which is why it is not a control."""
    module = _module(layout, PyDictStorage, 2048)
    assert isinstance(module._storage, ChainedStorage)
    device = torch.device("cuda", torch.cuda.current_device())
    fs, ps = torch.cuda.Stream(), torch.cuda.Stream()
    g = torch.Generator().manual_seed(7)
    uses = torch.zeros(KEYS, dtype=torch.int64)
    offsets = torch.arange(BATCH + 1, dtype=torch.int64, device=device)

    def batch():
        k = torch.randint(0, KEYS, (BATCH,), generator=g, dtype=torch.int64)
        uses.add_(torch.bincount(k, minlength=KEYS))
        return k.to(device)

    def prefetch(idx):
        with torch.cuda.stream(ps):
            module.prefetch(idx, offsets, fs)

    steps = 80
    cur = batch()
    prefetch(cur)
    for i in range(steps):
        with torch.cuda.stream(fs):
            torch.cuda.current_stream().wait_stream(ps)
            out = module(cur, offsets)
            torch.cuda._sleep(20_000_000)  # the backward queues behind this
            out.backward(torch.ones_like(out))
            del out
        if i + 1 < steps:
            cur = batch()
            prefetch(cur)  # no wait on fs: step i is still running
    torch.cuda.synchronize()
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys are wrong"
    module._storage.index.check_invariants([0])


# -- growing up to the maximum capacity ----------------------------------------


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_a_table_grows_toward_its_maximum_and_keeps_its_keys(layout):
    caching = layout == "caching"
    options = DynamicEmbTableOptions(
        dim=DIM,
        init_capacity=1024,
        max_capacity=16384,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=caching,
        local_hbm_for_values=16384 * DIM * 4 // 2 if caching else 1024**3,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT, value=1.0
        ),
    )
    module = BatchedDynamicEmbeddingTablesV2(
        table_names=["t"],
        table_options=[options],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=LR,
        use_index_dedup=True,
    )
    storage = module._storage
    assert isinstance(storage, ChainedStorage)
    before = storage.index.capacity_of(0)
    uses = _train(module)
    assert storage.index.capacity_of(0) > before, "the table did not grow"
    assert storage.index.capacity_of(0) <= 16384
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys were lost while the table grew"
    storage.index.check_invariants([0])


# -- what is evicted can be retained -------------------------------------------


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_every_key_that_is_not_held_was_retained_as_evicted(layout):
    from dynamicemb.dynamicemb_config import EvictedItemMode

    module = _module(layout, None, 2048, evicted_item_mode=EvictedItemMode.RETAIN_KEY)
    assert isinstance(module._storage, ChainedStorage)
    uses = _train(module)
    module.flush()
    popped = set(module.pop_evicted_keys()["t"].tolist())
    held = set(module._storage.index.export(0)[0].tolist())
    used = set((uses > 0).nonzero().squeeze(1).tolist())
    assert popped, "the run must evict"
    assert used <= held | popped, "a key was lost without being reported"
    assert popped <= used, "a key that was never used was reported"
    assert module.pop_evicted_keys()["t"].numel() == 0, "the keys are drained by a pop"


# -- the compound (TIMESTAMP, LFU) strategy -------------------------------------

LRU_LFU = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)


@pytest.mark.parametrize("layout", ["hbm", "caching"])
def test_a_timestamp_and_frequency_table_evicts_and_loses_nothing(layout):
    module = _module(layout, PyDictStorage, 2048, LRU_LFU)
    storage = module._storage
    assert isinstance(storage, ChainedStorage)
    uses = _train(module)
    assert int((uses > 0).sum()) > 2 * 2048
    assert storage.lower.size() > 0, "nothing was evicted into the storage below"
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys lost an update"
    storage.index.check_invariants([0])


# -- the hybrid layout: an HBM tier, a host tier, disjoint -------------------------


def test_hybrid_the_hbm_tier_spills_into_the_host_tier_and_nothing_is_lost():
    from dynamicemb.chained_storage import ChainedHybridStorage

    module = _module("hybrid", None, 16384)
    storage = module._storage
    assert isinstance(storage, ChainedHybridStorage)
    assert module._storage_layout == LAYOUT["hybrid"]
    uses = _train(module)
    hbm, host = storage._hbm_tier.index, storage._host_tier.index
    assert hbm.size(0) > 0 and host.size(0) > 0, "both tiers must hold keys"
    used = int((uses > 0).sum())
    assert hbm.size(0) + host.size(0) == used, "a key is in exactly one tier"
    hbm_keys, host_keys = set(hbm.export(0)[0].tolist()), set(host.export(0)[0].tolist())
    assert not (hbm_keys & host_keys), "the tiers must hold disjoint keys"
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys are wrong"
    hbm.check_invariants([0])
    host.check_invariants([0])


@pytest.mark.parametrize(
    "strategy",
    [DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU, LRU_LFU],
)
def test_hybrid_keeps_every_key_whatever_the_strategy(strategy):
    module = _module("hybrid", None, 16384, strategy)
    uses = _train(module)
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys are wrong"


def test_hybrid_what_the_host_tier_evicts_is_dropped_and_what_it_keeps_is_right():
    # more keys than the two tiers hold: some are lost, but none of the rest is wrong
    from dynamicemb.dynamicemb_config import EvictedItemMode

    module = _module("hybrid", None, 4096, evicted_item_mode=EvictedItemMode.RETAIN_KEY)
    uses = _train(module)
    storage = module._storage
    held = set()
    for tier in (storage._hbm_tier, storage._host_tier):
        held |= set(tier.index.export(0)[0].tolist())
    popped = set(module.pop_evicted_keys()["t"].tolist())
    used = set((uses > 0).nonzero().squeeze(1).tolist())
    assert popped, "the run must overflow both tiers"
    assert used <= held | popped
    # a key that was never evicted is exactly as trained (one that was, and came back,
    # started again from its initial value)
    kept = torch.tensor(sorted(held - popped), dtype=torch.int64)
    assert kept.numel() > 0
    device = torch.device("cuda")
    module.flush()
    module.eval()
    with torch.no_grad():
        out = module(kept.to(device), torch.arange(kept.numel() + 1, dtype=torch.int64, device=device))
    expected = (1.0 - LR * uses[kept].float()).to(device)
    assert bool(torch.isclose(out[:, 0], expected, atol=1e-4).all())


def test_host_only_scores_are_updated_by_a_training_find():
    """A table of rows on the host, nothing in front of it: the keys a step touches are
    the ones that stay when the table is full."""
    module = _module("host", PyDictStorage, 2048)
    assert isinstance(module._storage, ChainedStorage)
    uses = _train(module)
    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys lost an update"
