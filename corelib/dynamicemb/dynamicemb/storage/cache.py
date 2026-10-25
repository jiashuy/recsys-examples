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

"""The GPU cache of rows: placement and reference protection by node.
"""

from typing import List, Optional, Tuple

import torch  # usort:skip
from dynamicemb.storage.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.types import (
    KEY_TYPE,
    Cache,
    CacheMetrics,
)
from dynamicemb_extensions import (
    CacheCore,
)


class DynamicEmbCache(Cache):
    """The GPU cache: placement and reference protection, from ``CacheCore``.

    It holds no score and decides nothing about which keys stay in the system.
    The rows live in ``values``, one fixed pool per table addressed by node, and
    everything that reads or writes them goes through the same
    ``values.ptrs`` the backing's rows do. The cache knows nothing of the optimizer:
    it is given the buffers, and what is done to the values is up to its user. See docs/lightweight_cache_design.md.
    """

    def __init__(self, capacities: List[int], values: MixedDimBuffers):
        """``capacities`` is how many nodes each table has; ``values`` holds that many rows
        of each."""
        assert len(capacities) == values.num_tables
        assert all(values.rows(t) == c for t, c in enumerate(capacities))
        self.values = values
        self._core = CacheCore(list(capacities))
        self._cache_metrics = CacheMetrics()
        self._record_cache_metrics = False

    # -- Cache interface --

    def lookup_pin(
        self, unique_keys: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        nodes = self._core.lookup_pin(unique_keys.to(KEY_TYPE), offsets)
        if self._record_cache_metrics:
            self._cache_metrics.unique_keys = unique_keys.size(0)
            self._cache_metrics.hits = int((nodes >= 0).sum().item())
        return nodes

    def allocate(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        offsets: torch.Tensor,
        nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        victim_keys, has_victim, new_nodes = self._core.allocate(
            keys.to(KEY_TYPE), positions, offsets, nodes
        )
        if self._record_cache_metrics:
            self._cache_metrics.allocated = keys.numel()
            self._cache_metrics.evicted = int(has_victim.sum().item())
        return victim_keys, has_victim.to(torch.bool), new_nodes

    def publish(self, new_nodes: torch.Tensor, offsets: torch.Tensor) -> None:
        self._core.publish(new_nodes, offsets)

    def release(self, nodes: torch.Tensor, offsets: torch.Tensor) -> None:
        self._core.release(nodes, offsets)

    def find_nodes(
        self, keys: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return self._core.find_nodes(keys.to(KEY_TYPE), table_ids)

    def invalidate(self, nodes: torch.Tensor, table_ids: torch.Tensor) -> None:
        self._core.invalidate(nodes, table_ids)

    def resident(self, table_id: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Every resident node of one table, its key, and how many nodes still
        hold a reference."""
        return self._core.resident(table_id)

    def num_resident(self, table_id: int) -> int:
        return self._core.num_resident(table_id)

    def invalidate_keys(self, keys: torch.Tensor, table_id: int) -> None:
        """Drop these keys from the cache if they are in it, whatever they hold."""
        if keys.numel() == 0:
            return
        tids = torch.full(
            (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
        )
        self.invalidate(self.find_nodes(keys, tids), tids)

    def reset(self) -> None:
        self._core.reset()

    @property
    def eviction_guard(self) -> Tuple[int, int]:
        """What the backing's insert asks before evicting a key: a device pointer
        to this cache's per-table maps, and how many tables."""
        return self._core.guard()

    @property
    def cache_metrics(self) -> Optional[CacheMetrics]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record

    # -- Convenience accessors --

    @property
    def num_tables(self) -> int:
        return self.values.num_tables

    def capacity(self, table_id: int) -> int:
        """How many rows table ``table_id`` has."""
        return self._core.capacity(table_id)

    def size(self) -> int:
        return sum(self.num_resident(t) for t in range(self.num_tables))


# ---------------------------------------------------------------------------
# Storage interface helpers shared by the tiers
# ---------------------------------------------------------------------------


def flush_cache(cache: DynamicEmbCache, storage) -> None:
    """Write every resident row back to the storage.

    Every resident row is written whatever its references: a key that has never
    left the cache is the one whose storage copy is furthest behind. The storage
    already has a row for every cached key (a key enters it when it is admitted),
    so the rows are written in place and nothing is inserted.
    """
    storage.flush_from(cache)
