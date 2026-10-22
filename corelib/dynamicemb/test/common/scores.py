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
# limitations under the License.

"""Score inputs and a step counter shared by the table-level tests.

``get_scores`` hands out one score tensor per call, tagged with a global step
so a test can predict what an ASSIGN table must hold afterwards.
``CustomizedScore`` does the same per table name for the module-level
``DynamicEmbScoreStrategy.CUSTOMIZED`` tests.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from dynamicemb_extensions import ScorePolicy

_score_step = 0


def current_score_step() -> int:
    """The step stamped on the scores of the most recent ``get_scores`` call."""
    return _score_step


def get_scores(score_policy, keys: torch.Tensor) -> torch.Tensor:
    """Scores for ``keys`` under ``score_policy`` (advances the global step).

    ASSIGN -> every key gets the new step; ACCUMULATE -> ones; anything else
    (CONST, GLOBAL_TIMER, ...) -> zeros, the table fills the value itself.
    """
    global _score_step
    _score_step += 1
    batch = keys.numel()
    device = keys.device
    if score_policy == ScorePolicy.ASSIGN:
        return torch.empty(batch, dtype=torch.uint64, device=device).fill_(_score_step)
    elif score_policy == ScorePolicy.ACCUMULATE:
        return torch.ones(batch, dtype=torch.uint64, device=device)
    return torch.zeros(batch, dtype=torch.uint64, device=device)


class CustomizedScore:
    """A monotonically increasing per-table score for CUSTOMIZED strategies."""

    def __init__(self, table_names: List[str]):
        self.table_names_ = table_names
        self.step_ = 1

    def get(self) -> Dict[str, int]:
        ret: Dict[str, int] = {name: self.step_ for name in self.table_names_}
        self.step_ += 1
        return ret


class PerTableCustomizedScore:
    """Per-table step counters: ``get(table_name)`` returns that table's next score."""

    def __init__(self, table_names: List[str]):
        self.table_names_ = table_names
        self.steps_: Dict[str, int] = {table_name: 1 for table_name in table_names}

    def get(self, table_name: str):
        assert table_name in self.table_names_
        ret = self.steps_[table_name]
        self.steps_[table_name] += 1
        return ret


