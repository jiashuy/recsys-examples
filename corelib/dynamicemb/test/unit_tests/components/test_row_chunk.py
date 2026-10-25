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

import torch
from dynamicemb.checkpoint.row_chunk import RowChunk


def test_a_chunk_unpacks_as_the_four_columns():
    keys, emb = torch.arange(3), torch.randn(3, 4)
    chunk = RowChunk(keys, emb, None, torch.ones(3))
    k, e, o, s = chunk
    assert k is keys and e is emb and o is None
    assert chunk.scores is s and chunk.opt_states is None
