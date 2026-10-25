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

"""Where the rows an index evicts go.

An index that has no room takes victims out and hands them to an :class:`EvictionSink`:

- :class:`Discard`: they are dropped.
- :class:`ToLower`: they go to the tier below (a parameter server), as values.
- :class:`ToTier`: they go to a tier of the same kind beside this one (a hybrid
  storage's host tier, whose keys are disjoint), with their own score words.
- :class:`Retain`: wraps any of these and also records the evicted keys, for
  ``pop_evicted_keys``.
"""

import abc
from typing import Optional

import torch
from dynamicemb.key_value_table import DynamicEmbTableState, _append_evicted
from dynamicemb.types import Storage


class EvictionSink(abc.ABC):
    # Whether :meth:`put` needs the victims' values; if not, the caller does not load them.
    wants_values = False

    @abc.abstractmethod
    def put(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
    ) -> None:
        ...


class Discard(EvictionSink):
    def put(self, keys, table_ids, values, scores) -> None:
        pass


class ToLower(EvictionSink):
    wants_values = True

    def __init__(self, lower: Storage) -> None:
        self._lower = lower

    def put(self, keys, table_ids, values, scores) -> None:
        self._lower.insert(keys, table_ids, values)


class ToTier(EvictionSink):
    """``tier`` keeps the victims at the scores they had here."""

    wants_values = True

    def __init__(self, tier: Storage) -> None:
        self._tier = tier

    def put(self, keys, table_ids, values, scores) -> None:
        self._tier.insert(keys, table_ids, values, scores)


class Retain(EvictionSink):
    """Records the evicted keys in ``state`` and then passes the victims on to ``inner``."""

    def __init__(self, state: DynamicEmbTableState, inner: EvictionSink) -> None:
        self._state = state
        self._inner = inner
        self.wants_values = inner.wants_values

    def put(self, keys, table_ids, values, scores) -> None:
        _append_evicted(
            self._state,
            keys,
            table_ids,
            torch.tensor(keys.numel(), dtype=torch.int64, device=keys.device),
        )
        self._inner.put(keys, table_ids, values, scores)
