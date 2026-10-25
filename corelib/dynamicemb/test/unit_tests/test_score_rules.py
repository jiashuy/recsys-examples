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

import pytest
import torch
from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy as S
from dynamicemb.score import ScoreRules
from dynamicemb.types import SCORE_TYPE

LRU_LFU = (S.TIMESTAMP, S.LFU)
LFU_LRU = (S.LFU, S.TIMESTAMP)


@pytest.fixture
def device():
    return torch.device("cuda", torch.cuda.current_device())


def test_words_and_timestamps():
    assert ScoreRules(S.LFU).num_words == 1
    assert not ScoreRules(S.LFU).has_timestamp
    assert ScoreRules(S.TIMESTAMP).has_timestamp
    for pair in (LRU_LFU, LFU_LRU):
        r = ScoreRules(pair)
        assert r.num_words == 2 and r.is_lru_lfu and r.has_timestamp


def test_access_declares_the_update_mode(device):
    v, mode = ScoreRules(S.STEP).access(3, device, step_score=7)
    assert mode == "assign" and v.tolist() == [7, 7, 7]
    freq = torch.tensor([1, 2], device=device)
    v, mode = ScoreRules(S.LFU).access(2, device, frequency=freq)
    assert mode == "accumulate" and v.tolist() == [1, 2]
    v, mode = ScoreRules(LFU_LRU).access(2, device)
    assert mode == "lru_lfu" and v is None
    v, mode = ScoreRules(S.NO_EVICTION).access(2, device)
    assert mode == "none" and v is None
    v, mode = ScoreRules(S.TIMESTAMP).access(2, device)
    assert mode == "assign" and v.numel() == 2


@pytest.mark.parametrize("pair", [LRU_LFU, LFU_LRU])
def test_a_dump_and_a_replay_give_the_block_back(device, pair):
    r = ScoreRules(pair)
    physical = torch.stack(
        [torch.tensor([1000, 2000, 3000]), torch.tensor([5, 6, 7])], dim=1
    ).to(device=device, dtype=SCORE_TYPE)  # (timestamp, frequency)
    logical = r.dump_block(physical, now=4000)
    ts_col = list(pair).index(S.TIMESTAMP)
    assert logical[:, ts_col].tolist() == [3000, 2000, 1000]  # ages
    assert logical[:, 1 - ts_col].tolist() == [5, 6, 7]
    back = r.replay_block(logical, now=4000)
    assert torch.equal(back, physical)


def test_a_single_column_is_dumped_as_it_is(device):
    r = ScoreRules(S.LFU)
    block = torch.tensor([3, 4], device=device, dtype=SCORE_TYPE)
    assert r.dump_block(block, now=100).tolist() == [[3], [4]]


def test_replay_checks_the_width(device):
    with pytest.raises(ValueError):
        ScoreRules(LRU_LFU).replay_block(torch.zeros(4, 3, device=device), now=1)


def test_fresh_block_scores_recency_words_with_the_clock(device):
    block = ScoreRules(LRU_LFU).fresh_block(2, device, timestamp=99, insert_score=1)
    assert block.tolist() == [[99, 1], [99, 1]]
    block = ScoreRules(S.LFU).fresh_block(2, device, timestamp=99, insert_score=1)
    assert block.tolist() == [[1], [1]]
