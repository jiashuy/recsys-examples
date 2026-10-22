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

"""Small helpers for tests of ``incremental_dump`` / ``replay_increment``."""

from __future__ import annotations

from typing import Dict, List

import torch


def dumped_keys(res, table_name: str) -> torch.Tensor:
    """Per-table dumped keys from a module-level DeltaDumpResult."""
    return res.keys[res.table_names.index(table_name)]


def current_scores(res) -> Dict[str, int]:
    """{table_name: current_score} from a DeltaDumpResult (next dump threshold)."""
    return {tn: res.meta[j]["current_score"] for j, tn in enumerate(res.table_names)}


def feature_from_keys(keys: List[int], device: torch.device):
    """(indices, offsets) of one SUM-pooled feature, one key per bag."""
    indices = torch.tensor(keys, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(keys) + 1, dtype=torch.int64, device=device)
    return indices, offsets
