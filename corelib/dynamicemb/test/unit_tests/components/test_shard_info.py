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
from dynamicemb.checkpoint.checkpointer import ShardInfo


def test_without_a_process_group_the_rank_holds_the_only_shard():
    shard = ShardInfo(None, ["t0"])
    assert (shard.index, shard.num_shards, shard.ranks) == (0, 1, [0])
    assert shard.sharding_type == "row_wise"
    assert shard.pg is None


def test_a_table_row_wise_table_has_its_own_shard_count():
    shard = ShardInfo([0, 1], ["t0"])
    assert shard.num_shards == 2 and shard.index == 0


def test_a_rank_outside_the_shards_is_rejected():
    with pytest.raises(ValueError, match="holds no shard"):
        ShardInfo([3, 4], ["t0"])


def test_the_barrier_is_a_no_op_without_a_process_group():
    ShardInfo(None, ["t0"]).barrier()
