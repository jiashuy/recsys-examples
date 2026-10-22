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

"""Builders for bare scored hash tables used by the table / eviction tests."""

from __future__ import annotations

import torch
from dynamicemb.scored_hashtable import ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy


def _capacities(capacity):
    return [capacity] if isinstance(capacity, int) else list(capacity)


def lru_lfu_table(capacity, bucket_capacity=128, key_type=torch.int64):
    """A compound (timestamp, frequency) table; ``capacity`` is an int or a list."""
    return get_scored_table(
        capacity=_capacities(capacity),
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
    )


def assign_table(capacity, bucket_capacity=128, key_type=torch.int64):
    """A single-score ASSIGN table (the AOT, non-compound eviction path)."""
    return get_scored_table(
        capacity=_capacities(capacity),
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=ScorePolicy.ASSIGN)],
    )
