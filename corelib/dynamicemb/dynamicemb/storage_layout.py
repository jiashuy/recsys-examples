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

"""Which storage a set of tables gets, and the building of it.

The tables of a module are given a layout by how their values compare with the HBM they
may use:

  HBM_ONLY   everything fits: one storage on the GPU
  HOST_ONLY  nothing may go on the GPU: one storage in host memory
  HYBRID     part fits: an HBM storage and a host one, with disjoint keys
  CACHING    part fits and the tables ask for a cache: a GPU cache in front of a host
             storage that holds everything

:func:`build_storage` makes it, from the tables' options and the optimizer, and returns
the parts as a :class:`StorageBundle`. What a module does with them is its own business.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import warnings

import torch
from dynamicemb.chained_storage import ChainedHybridStorage, ChainedStorage
from dynamicemb.dynamicemb_config import (
    DynamicEmbEvictStrategy,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    align_to_table_size,
)
from dynamicemb.cache import DynamicEmbCache
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.row_exchange import RowExchange
from dynamicemb.types import Cache, Storage
from dynamicemb.utils import DTYPE_NUM_BYTES


class StorageLayout(Enum):
    HBM_ONLY = "HBM-only (single tier)"
    HOST_ONLY = "Host-only (single tier, CPU)"
    HYBRID = "Hybrid (HBM + Host, disjoint partitions; total = HBM + Host)"
    CACHING = "Caching (HBM cache + Host full backing; HBM rows duplicate Host)"


@dataclass
class StorageBundle:
    layout: StorageLayout
    storage: Storage
    cache: Optional[Cache] = None
    exchange: Optional[RowExchange] = None  # between the cache and the storage


def _external_storage(
    options: List[DynamicEmbTableOptions], optimizer: BaseDynamicEmbeddingOptimizer
) -> Optional[Storage]:
    """The tier below the tables, if ``external_storage`` asks for one.

    It holds what the tables evict, and is asked for what they do not have.
    """
    PS = options[0].external_storage
    if PS is None:
        return None
    ps_options = deepcopy(options)
    for ps_option in ps_options:
        ps_option.local_hbm_for_values = 0
    return PS(ps_options, optimizer)


def _tables_in(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    host: bool,
) -> ChainedStorage:
    """The storage of the rows themselves, with whatever tier is below it; in host memory
    when ``host`` (an HBM budget of zero makes the buffers host ones)."""
    options = deepcopy(options)
    if host:
        for option in options:
            option.local_hbm_for_values = 0
    return ChainedStorage(
        options, optimizer, lower=_external_storage(options, optimizer)
    )


def _cache_and_backing(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    device_id: int,
    cap_scale: float,
) -> StorageBundle:
    cache_options = deepcopy(options)
    for cache_option in cache_options:
        cache_option.bucket_capacity = 1024
        cap = max(1, int(cache_option.max_capacity * cap_scale))
        cache_option.max_capacity = min(cache_option.max_capacity, cap)
        cache_option.init_capacity = min(cache_option.max_capacity, cap)

    # The cache holds no score and does no eviction of its own, so NO_EVICTION means
    # nothing to it: its options carry TIMESTAMP/LRU, as the hybrid's HBM tier does,
    # and the backing keeps NO_EVICTION.
    if options[0].score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        for cache_option in cache_options:
            cache_option.score_strategy = DynamicEmbScoreStrategy.TIMESTAMP
            cache_option.evict_strategy = DynamicEmbEvictStrategy.LRU

    # Rounded up to a multiple of the bucket capacity, as the cache's index always was.
    # A cache has to hold every key of a step at once, so a configuration smaller than a
    # step would otherwise fail where it used to work.
    capacities = [
        align_to_table_size(int(o.init_capacity), int(o.bucket_capacity))
        for o in cache_options
    ]
    cache = DynamicEmbCache(
        capacities,
        MixedDimBuffers(
            [o.dim for o in cache_options],
            [o.dim + optimizer.get_state_dim(o.dim) for o in cache_options],
            capacities,
            cache_options[0].embedding_dtype,
            torch.device(f"cuda:{device_id}"),
            "fixed",
        ),
    )
    storage = _tables_in(options, optimizer, host=True)
    return StorageBundle(
        StorageLayout.CACHING, storage, cache, RowExchange(cache.values, storage.values)
    )


def _hybrid(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    cap_scale: float,
) -> StorageBundle:
    if any(opt.external_storage is not None for opt in options):
        warnings.warn(
            "external_storage is ignored in hybrid mode. "
            "Set caching=True to use external storage.",
            UserWarning,
        )
    # Its tiers hold disjoint keys and spill by eviction.
    if options[0].score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        raise ValueError(
            "NO_EVICTION is not supported with the hybrid storage "
            "(non-caching partial-HBM storage). Use caching=True "
            "(the backing store keeps NO_EVICTION) or a single-tier "
            "configuration."
        )

    hbm_options = deepcopy(options)
    for hbm_option in hbm_options:
        cap = max(1, int(hbm_option.max_capacity * cap_scale))
        hbm_option.max_capacity = min(hbm_option.max_capacity, cap)
        hbm_option.init_capacity = hbm_option.max_capacity

    host_options = deepcopy(options)
    for host_option in host_options:
        host_option.local_hbm_for_values = 0
        cap = max(1, int(host_option.max_capacity * (1.0 - cap_scale)))
        host_option.max_capacity = min(host_option.max_capacity, cap)
        host_option.init_capacity = min(host_option.init_capacity, cap)

    # A table that is one bucket (its bucket is as large as the table) is one bucket in
    # each tier, of the tier's size.
    for tier in (hbm_options, host_options):
        for opt, whole in zip(tier, options):
            if opt.bucket_capacity >= whole.max_capacity:
                opt.bucket_capacity = opt.max_capacity

    return StorageBundle(
        StorageLayout.HYBRID, ChainedHybridStorage(hbm_options, host_options, optimizer)
    )


def build_storage(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    device_id: int,
) -> StorageBundle:
    """The layout the tables' values and HBM budget call for, built."""
    value_dims = [o.dim + optimizer.get_state_dim(o.dim) for o in options]
    total_memory = sum(
        o.max_capacity * DTYPE_NUM_BYTES[o.embedding_dtype] * v
        for o, v in zip(options, value_dims)
    )
    local_hbm = sum(o.local_hbm_for_values for o in options)

    if total_memory <= local_hbm:
        return StorageBundle(
            StorageLayout.HBM_ONLY, _tables_in(options, optimizer, host=False)
        )

    # The values do not all fit in HBM.
    cap_scale = local_hbm / total_memory if total_memory > 0 else 1.0
    if any(o.caching for o in options):
        if local_hbm <= 0:
            raise ValueError(
                "Can't use caching mode as the reserved HBM size is too small."
            )
        return _cache_and_backing(options, optimizer, device_id, cap_scale)
    if local_hbm <= 0:
        return StorageBundle(
            StorageLayout.HOST_ONLY, _tables_in(options, optimizer, host=True)
        )
    return _hybrid(options, optimizer, cap_scale)
