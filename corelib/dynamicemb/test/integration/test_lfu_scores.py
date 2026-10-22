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

"""LFU score correctness of a sharded dynamicemb model, end to end.

Level: integration.   Needs: 1 GPU (cases at 1 rank) or 8 GPUs (cases at 8).
Covers: after training through TorchRec (DistributedModelParallel, forward and
    backward), the score each key carries in the table equals the number of
    times it was looked up, for every storage path (HBM-only, hybrid, cache +
    storage), every optimizer, index dedup on and off, and the compound
    (TIMESTAMP, LFU) score. Also asserts the storage class the planner chose.
Does not cover (see unit/table/test_hybrid_storage_export.py,
    unit/table/test_lru_lfu.py): the same score semantics on a table without
    the model stack.
The work is done by ``workers/lfu_scores.py`` under torchrun.
"""

from pathlib import Path

import pytest
from common.cli_args import as_flags
from common.launch import run_worker

WORKER = Path(__file__).parent / "workers" / "lfu_scores.py"

GPUS = (1, 8)
OPTIMIZERS = ("sgd", "adam", "rowwise_adagrad")
DEDUP = {"dedup": "--use-index-dedup", "no-dedup": "--no-use-index-dedup"}

_FOUR_TABLES = {
    "num_embedding_collections": 2,
    "num_embeddings": "10000,10000,10000,10000",
    "multi_hot_sizes": "5,5,5,5",
    "embedding_dim": 16,
    "batch_size": 32,
    "num_iterations": 10,
    "tolerance": 0.0,
}

# How a lookup reaches the table. Hybrid is HybridStorage + StorageMode DEFAULT
# (no cache): a 0.25 HBM budget forces part of the table out of HBM. The cache
# ratio of 0.3 is small enough to trigger evictions between cache and storage.
STORAGE_MODES = {
    "storage-only": {},
    "hybrid": {"global_hbm_budget_scale": 0.25},
    "cache": {"caching": True, "cache_capacity_ratio": 0.3},
}

# Compound (TIMESTAMP, LFU) score: LFU tables use the compound LruLfu policy (2
# score words: timestamp + frequency). The frequency word must still equal the
# true access count, proving the extra timestamp word does not corrupt LFU. Only
# the storage-only (HBM_DIRECT) path supports multi-word scores; caching/Hybrid
# is a follow-up, so the compound score is storage-only.
COMPOUND = {"lru-lfu": {"lru_lfu": True}}

_ONE_TABLE = {
    "num_embedding_collections": 1,
    "embedding_dim": 16,
    "optimizer_type": "sgd",
    "tolerance": 0.0,
}


def _matrix():
    cases = []
    for dedup, dedup_flag in DEDUP.items():
        for modes in (STORAGE_MODES, COMPOUND):
            for mode, mode_options in modes.items():
                for nproc in GPUS:
                    for optimizer in OPTIMIZERS:
                        options = {
                            **_FOUR_TABLES,
                            **mode_options,
                            "optimizer_type": optimizer,
                        }
                        cases.append(
                            pytest.param(
                                nproc,
                                as_flags(options) + [dedup_flag],
                                id=f"{nproc}gpu-{mode}-{optimizer}-{dedup}",
                            )
                        )
    return cases


# High-frequency test: more iterations to test frequency accumulation.
_HIGH_FREQUENCY = {
    **_ONE_TABLE,
    "num_embeddings": 5000,
    "multi_hot_sizes": 3,
    "batch_size": 16,
    "num_iterations": 50,
}

# A cache far too small for the keys (8%), many iterations and a large batch, so
# scores have to survive heavy eviction between cache and storage.
_TINY_CACHE = {
    **_ONE_TABLE,
    "num_embeddings": 10000,
    "multi_hot_sizes": 5,
    "batch_size": 64,
    "num_iterations": 25,
    "caching": True,
    "cache_capacity_ratio": 0.08,
}

SINGLE_RANK_CASES = [
    pytest.param(as_flags(_HIGH_FREQUENCY), id="high-frequency-storage-only"),
    pytest.param(
        as_flags({**_HIGH_FREQUENCY, "caching": True, "cache_capacity_ratio": 0.4}),
        id="high-frequency-cache",
    ),
] + [
    pytest.param(
        as_flags({**_TINY_CACHE, "optimizer_type": optimizer}),
        id=f"tiny-cache-{optimizer}",
    )
    for optimizer in ("sgd", "adam")
]


@pytest.mark.parametrize("nproc,flags", _matrix())
def test_lfu_score_equals_access_count(nproc, flags):
    """Each stored key's score equals its true lookup count (storage x optimizer)."""
    run_worker(WORKER, nproc, flags)


@pytest.mark.parametrize("flags", SINGLE_RANK_CASES)
def test_lfu_score_accumulates_single_rank(flags):
    """Frequency accumulation over many iterations, and under a cache far too small."""
    run_worker(WORKER, 1, flags)
