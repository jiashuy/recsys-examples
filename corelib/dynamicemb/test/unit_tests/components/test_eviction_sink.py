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

from types import SimpleNamespace

import torch
from dynamicemb.storage.eviction_sink import Discard, Retain, ToLower, ToTier


class _Recorder:
    def __init__(self):
        self.calls = []

    def insert(self, *args):
        self.calls.append(args)


def _victims():
    keys = torch.tensor([5, 6, 7])
    tids = torch.tensor([0, 1, 0])
    values = torch.randn(3, 4)
    scores = torch.tensor([[1], [2], [3]])
    return keys, tids, values, scores


def test_discard_takes_nothing_and_wants_no_values():
    sink = Discard()
    assert not sink.wants_values
    sink.put(*_victims())


def test_to_lower_gets_the_values_and_no_scores():
    lower = _Recorder()
    keys, tids, values, scores = _victims()
    sink = ToLower(lower)
    assert sink.wants_values
    sink.put(keys, tids, values, scores)
    (got,) = lower.calls
    assert len(got) == 3 and got[2] is values


def test_to_tier_keeps_the_scores():
    tier = _Recorder()
    keys, tids, values, scores = _victims()
    ToTier(tier).put(keys, tids, values, scores)
    (got,) = tier.calls
    assert got[3] is scores


def test_retain_records_the_keys_and_passes_the_victims_on():
    state = SimpleNamespace(evicted_key_chunks=[], evicted_tid_chunks=[])
    tier = _Recorder()
    keys, tids, values, scores = _victims()
    sink = Retain(state, ToTier(tier))
    assert sink.wants_values  # it follows what it wraps
    sink.put(keys, tids, values, scores)
    assert torch.equal(torch.cat(state.evicted_key_chunks), keys)
    assert torch.equal(torch.cat(state.evicted_tid_chunks), tids)
    assert len(tier.calls) == 1
    assert not Retain(state, Discard()).wants_values
