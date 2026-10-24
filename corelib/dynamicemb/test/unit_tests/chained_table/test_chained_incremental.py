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

"""Incremental dump and replay over the chained index.

The existing replay writes every key at the slot it held in the source, which only
means something between two tables of the same layout. Here a replica places each key
as it likes, so what has to hold is the end state: every key the source holds is in
the replica with the value, optimizer state and score it had there, and what the
source erased or evicted is not.
"""

import os
import sys

import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy
from dynamicemb.chained_storage import ChainedStorage
from dynamicemb.dynamicemb_config import EvictedItemMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from test_chained_prefetch import _module  # noqa: E402
from test_ps_tier import KEYS, _train  # noqa: E402


def _answers(module, keys):
    device = torch.device("cuda", torch.cuda.current_device())
    module.eval()
    k = keys.to(device)
    offsets = torch.arange(k.numel() + 1, dtype=torch.int64, device=device)
    with torch.no_grad():
        out = module(k, offsets)
    return out.detach().clone()


def _held(module):
    storage = module._storage
    tiers = (
        [storage._hbm_tier, storage._host_tier]
        if hasattr(storage, "_hbm_tier")
        else [storage]
    )
    held = set()
    for tier in tiers:
        held |= set(tier.index.export(0)[0].tolist())
    return held


@pytest.mark.parametrize("layout", ["hbm", "caching", "hybrid", "host"])
@pytest.mark.parametrize(
    "strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.LFU,
        DynamicEmbScoreStrategy.STEP,
    ],
)
def test_a_replica_holds_what_the_source_held(layout, strategy):
    source = _module(layout, None, 16384, strategy)
    uses = _train(source)
    used = (uses > 0).nonzero().squeeze(1)
    assert source._storage is not None

    delta = source.incremental_dump({"t": 0})
    assert delta.keys[0].numel() == used.numel()

    replica = _module(layout, None, 16384, strategy)
    stats = replica.replay_increment(delta)
    assert stats["t"].upserted == used.numel()

    assert torch.equal(_answers(source, used), _answers(replica, used))
    replica.flush()
    assert _held(replica) == set(used.tolist())


def test_what_the_source_evicted_is_not_in_the_replica():
    """A table too small for its keys evicts; the replica, which has the same
    capacity, ends up with the keys the source kept and none that it lost."""
    source = _module("hbm", None, 2048)
    uses = _train(source)
    assert int((uses > 0).sum()) > 2048, "the run must evict"
    kept = _held(source)

    delta = source.incremental_dump({"t": 0})
    replica = _module("hbm", None, 2048)
    replica.replay_increment(delta)
    assert _held(replica) == kept
    kept_keys = torch.tensor(sorted(kept), dtype=torch.int64)
    assert torch.equal(_answers(source, kept_keys), _answers(replica, kept_keys))


def test_a_replica_drops_what_the_source_erased():
    source = _module("hbm", None, 16384)
    uses = _train(source)
    used = (uses > 0).nonzero().squeeze(1)

    replica = _module("hbm", None, 16384)
    replica.replay_increment(source.incremental_dump({"t": 0}))
    assert _held(replica) == set(used.tolist())

    gone = used[::7]
    removed = source._storage.erase_keys(0, gone.cuda(), EvictedItemMode.RETAIN_KEY)
    assert removed == gone.numel()

    # nothing was touched since, so the window holds no keys, only the erasures
    big = 2**62
    delta = source.incremental_dump({"t": big})
    assert delta.keys[0].numel() == 0
    assert set(delta.erased_keys[0].tolist()) == set(gone.tolist())

    stats = replica.replay_increment(delta)
    assert stats["t"].erased == gone.numel()
    assert _held(replica) == set(used.tolist()) - set(gone.tolist())
    replica._storage.index.check_invariants([0])


def test_a_key_erased_and_written_again_in_one_window_survives():
    source = _module("hbm", None, 16384)
    uses = _train(source)
    used = (uses > 0).nonzero().squeeze(1)
    replica = _module("hbm", None, 16384)
    replica.replay_increment(source.incremental_dump({"t": 0}))

    victim = used[:1]
    source._storage.erase_keys(0, victim.cuda(), EvictedItemMode.RETAIN_KEY)
    # train once more on a batch that brings the key back
    device = torch.device("cuda")
    keys = torch.cat([victim, used[1:50]]).to(device)
    offsets = torch.arange(keys.numel() + 1, dtype=torch.int64, device=device)
    source.train()
    out = source(keys, offsets)
    out.backward(torch.ones_like(out))
    torch.cuda.synchronize()

    delta = source.incremental_dump({"t": 0})
    replica.replay_increment(delta)
    assert int(victim.item()) in _held(replica)
