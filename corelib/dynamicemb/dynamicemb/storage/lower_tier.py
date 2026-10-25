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

"""The tier below the tables: what a miss is asked of and what an eviction is given to.

It is any :class:`~dynamicemb.types.Storage` a user passes as ``external_storage`` (a
parameter server); :class:`LowerTier` is the narrow face the tables use of it, so that
nothing else has to know the positions in the tuple ``Storage.find`` returns.
"""

from typing import NamedTuple, Optional

import torch
from dynamicemb.types import CopyMode, Storage


class Fetched(NamedTuple):
    found: torch.Tensor  # bool, per key asked
    scores: torch.Tensor
    values: torch.Tensor  # a row per key asked; those not found are meaningless


class LowerTier:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def fetch(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequency: Optional[torch.Tensor] = None,
        copy_mode: CopyMode = CopyMode.VALUE,
    ) -> Fetched:
        """Ask for ``keys``; ``frequency`` is what the step counted for each (LFU)."""
        out = self.storage.find(keys, table_ids, copy_mode, frequency)
        return Fetched(out[5], out[6], out[7])

    def insert(
        self, keys: torch.Tensor, table_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        self.storage.insert(keys, table_ids, values)

    def fill(
        self,
        copy_mode: CopyMode,
        values: torch.Tensor,
        num_missing: int,
        missing_keys: torch.Tensor,
        missing_positions: torch.Tensor,
        missing_table_ids: torch.Tensor,
        missing_scores: Optional[torch.Tensor],
        founds: torch.Tensor,
        scores: torch.Tensor,
    ):
        """Ask for the keys a table does not have.

        What is found is written into ``values`` (at the keys' positions) and marked
        found; the rest stays missing. Returns the missing description and ``founds``
        and ``scores`` as they are afterwards.
        """
        got = self.fetch(missing_keys, missing_table_ids, missing_scores, copy_mode)
        if not bool(got.found.any()):
            return (
                num_missing,
                missing_keys,
                missing_positions,
                missing_table_ids,
                missing_scores,
                founds,
                scores,
            )
        hit = missing_positions[got.found]
        values[hit] = got.values[got.found]
        founds = founds.clone()
        founds[hit] = True
        scores = scores.clone()
        scores[hit] = got.scores[got.found].to(scores.dtype)
        keep = ~got.found
        return (
            int(keep.sum().item()),
            missing_keys[keep],
            missing_positions[keep],
            missing_table_ids[keep],
            missing_scores[keep] if missing_scores is not None else None,
            founds,
            scores,
        )
