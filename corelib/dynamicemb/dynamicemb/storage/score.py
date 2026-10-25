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

"""What a score strategy means, in one place.

A key's score ranks it for eviction (the lowest goes first) and, for a timestamp, selects
the keys an incremental dump takes. Everything that depends on the strategy is answered
here: what an access gives the keys it touches, how many words a score takes in the index,
how the words are ordered on the device and in a checkpoint, which of them are timestamps
(dumped as ages, since the device clock means nothing on another device), and what a key
written by replay or a load is scored.

The eviction order itself is executed by the index's kernel, which cannot call back into
Python; :meth:`ScoreRules.access` only declares the update mode (``"assign"``,
``"accumulate"``, ``"lru_lfu"`` or ``"none"``) that kernel interprets.
"""

from typing import List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    ScoreStrategy,
    get_physical_score_order,
    score_dump_permutation,
    score_load_permutation,
)
from dynamicemb.types import SCORE_TYPE
from dynamicemb_extensions import device_timestamp


def is_lru_lfu(strategy: ScoreStrategy) -> bool:
    """The compound (TIMESTAMP, LFU) in either order: two words, frequency first."""
    return isinstance(strategy, tuple) and frozenset(strategy) == frozenset(
        {DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}
    )


class ScoreRules:
    """The answers for one score strategy. Columns of a *logical* block follow the order
    the user configured (the checkpoint's); those of a *physical* one the device's."""

    def __init__(self, strategy: ScoreStrategy) -> None:
        self.strategy = strategy
        self.physical: Tuple[DynamicEmbScoreStrategy, ...] = get_physical_score_order(
            strategy
        )
        self.num_words = len(self.physical)
        self._to_logical: List[int] = score_dump_permutation(strategy)
        self._to_physical: List[int] = score_load_permutation(strategy)
        logical = strategy if isinstance(strategy, tuple) else (strategy,)
        self._timestamp_columns = [
            i for i, s in enumerate(logical) if s == DynamicEmbScoreStrategy.TIMESTAMP
        ]

    @property
    def is_lru_lfu(self) -> bool:
        return is_lru_lfu(self.strategy)

    @property
    def has_timestamp(self) -> bool:
        return bool(self._timestamp_columns)

    # -- what an access gives -------------------------------------------------

    def access(
        self,
        n: int,
        device: torch.device,
        step_score: int = 0,
        frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """The score values and update mode that this step's accesses give ``n`` keys.

        Lower scores are evicted first, so a timestamp, a step, a user score and a
        frequency all rank the way they should. ``step_score`` is the current STEP /
        CUSTOMIZED value; ``frequency`` what the step counted for LFU (``None``: one).
        """
        s = self.strategy
        if s == DynamicEmbScoreStrategy.TIMESTAMP:
            # The device clock, as the existing index uses, so that a checkpoint's ages
            # (a timestamp minus a score) mean the same thing here.
            return (
                torch.full((n,), device_timestamp(), dtype=torch.int64, device=device),
                "assign",
            )
        if self.is_lru_lfu:
            # the timestamp is the device clock at the access; the frequency grows by
            # what the step counted, or by one
            return (None if frequency is None else frequency.to(torch.int64)), "lru_lfu"
        if s in (DynamicEmbScoreStrategy.STEP, DynamicEmbScoreStrategy.CUSTOMIZED):
            return torch.full((n,), step_score, dtype=torch.int64, device=device), "assign"
        if s == DynamicEmbScoreStrategy.LFU:
            return (None if frequency is None else frequency.to(torch.int64)), "accumulate"
        return None, "none"

    # -- dump and load --------------------------------------------------------

    def to_logical(self, physical: torch.Tensor) -> torch.Tensor:
        """Reorder a ``[N, W]`` physical block into the user's configured column order."""
        if self._to_logical == list(range(self.num_words)):
            return physical
        return physical[:, self._to_logical].contiguous()

    def to_physical(self, logical: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to_logical`."""
        if self._to_physical == list(range(self.num_words)):
            return logical
        return logical[:, self._to_physical].contiguous()

    def to_age(self, logical: torch.Tensor, now: int) -> torch.Tensor:
        """Timestamp columns of a logical block as ages (``now - score``), for dumping.

        The input is returned itself when there is nothing to convert, so callers must
        not mutate the result in place; otherwise a clone is returned."""
        if not self._timestamp_columns or logical.numel() == 0:
            return logical
        out = logical.clone()
        for c in self._timestamp_columns:
            out[:, c] = torch.clamp(now - logical[:, c], min=0)
        return out

    def from_age(self, ages: torch.Tensor, now: int) -> torch.Tensor:
        """Inverse of :meth:`to_age`, for replay and load; same aliasing note."""
        if not self._timestamp_columns or ages.numel() == 0:
            return ages
        out = ages.clone()
        for c in self._timestamp_columns:
            out[:, c] = torch.clamp(now - ages[:, c], min=0)
        return out

    def dump_block(self, physical: torch.Tensor, now: int) -> torch.Tensor:
        """``[N, W]`` scores as a checkpoint or an incremental dump carries them: configured
        column order, timestamps as ages relative to ``now``."""
        return self.to_age(self.to_logical(physical.view(-1, self.num_words)), now)

    def replay_block(self, logical: torch.Tensor, now: int) -> torch.Tensor:
        """Inverse of :meth:`dump_block`: logical ages to physical scores on this device.

        Both steps index by column, so a block of the wrong width would silently
        transform or permute the wrong words (a frequency written into the timestamp
        column, say); the width is checked rather than trusted."""
        block = logical.to(SCORE_TYPE)
        if block.dim() == 1:
            block = block.view(-1, 1)
        if block.dim() != 2 or block.size(1) != self.num_words:
            raise ValueError(
                f"a score block has {self.num_words} word(s) per key, got shape "
                f"{tuple(logical.shape)}"
            )
        return self.to_physical(self.from_age(block, now)).contiguous()

    def fresh_block(
        self, n: int, device: torch.device, timestamp: int, insert_score: int
    ) -> torch.Tensor:
        """``[N, W]`` physical scores for ``n`` keys scored as if just inserted here: recency
        words get ``timestamp``, every other word ``insert_score``.

        For keys written by replay when the caller did not ask for the source's scores:
        the replica then ranks them by when it received them. ``insert_score`` is passed
        in because replay can run before a target has ever done a forward pass."""
        block = torch.zeros((n, self.num_words), device=device, dtype=SCORE_TYPE)
        for word, strategy in enumerate(self.physical):
            block[:, word] = (
                timestamp if strategy == DynamicEmbScoreStrategy.TIMESTAMP else insert_score
            )
        return block
