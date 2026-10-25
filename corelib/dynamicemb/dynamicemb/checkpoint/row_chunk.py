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

"""A batch of rows as a checkpoint carries them.

The unit that storage exports and imports and that the checkpoint files hold: the keys
of the rows, their embeddings, the optimizer state (``None`` for an optimizer without
any), and the scores in the user's configured column order (``[N]`` for a one-word score,
``[N, W]`` otherwise). Embeddings and states are at the table's own precision.
"""

from typing import NamedTuple, Optional

import torch


class RowChunk(NamedTuple):
    keys: torch.Tensor
    embeddings: torch.Tensor
    opt_states: Optional[torch.Tensor]
    scores: torch.Tensor
