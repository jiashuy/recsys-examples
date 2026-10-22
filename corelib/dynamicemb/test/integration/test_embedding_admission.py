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

"""Frequency admission of a sharded dynamicemb model, end to end.

Level: integration.   Needs: 1 GPU (cases at 1 rank) or 8 GPUs (cases at 8).
Covers: through TorchRec (DistributedModelParallel, forward and backward) only
    keys looked up at least ``threshold`` times are stored, over every storage
    path (HBM-only, hybrid, cache + storage), optimizer and score strategy; the
    cases with long runs and with a tiny cache (admission against eviction);
    and a threshold no key can reach, where every forward value must come from
    the initializer that owns non-admitted rows (the strategy's, or the
    table's when the strategy has none).
Does not cover (see unit/admission/test_probabilistic_admission.py,
    unit/admission/test_admission_counter_deprecation.py): the strategies on
    their own, without the model stack.
The work is done by ``workers/embedding_admission.py`` under torchrun.
"""

from pathlib import Path

import pytest
from common.cli_args import as_flags
from common.launch import run_worker

pytestmark = pytest.mark.load_dump

WORKER = Path(__file__).parent / "workers" / "embedding_admission.py"

# A case names only the options that differ from the worker's defaults.
STORAGE_MODES = {
    # The three ways a lookup reaches admission.
    "hbm-direct": {},
    "hybrid": {"global_hbm_budget_scale": 0.25},
    "cache": {"caching": True, "cache_capacity_ratio": 0.3},
}
OPTIMIZERS = ("sgd", "adam", "rowwise_adagrad")
SCORE_STRATEGIES = ("timestamp", "lfu", "step")

_FOUR_TABLES = {
    "num_embedding_collections": 2,
    "num_embeddings": "10000,10000,10000,10000",
    "multi_hot_sizes": "5,5,5,5",
    "embedding_dim": 16,
    "batch_size": 32,
    "num_iterations": 10,
    "threshold": 4,
}
_ONE_TABLE = {
    "num_embedding_collections": 1,
    "embedding_dim": 16,
    "optimizer_type": "sgd",
    "threshold": 4,
}

# Process counts a case is worth running at: the multi-rank ones shard the keys
# across ranks, the long single-table ones need no more than one.
MULTI_RANK = (1, 8)

# Admission over every storage path, every optimizer and every score strategy a
# table can be built with.
MATRIX = [
    pytest.param(
        nproc,
        {
            **_FOUR_TABLES,
            **storage_options,
            "optimizer_type": optimizer,
            "score_strategy": score_strategy,
        },
        id=f"{nproc}gpu-{storage}-{optimizer}-{score_strategy}",
    )
    for nproc in MULTI_RANK
    for storage, storage_options in STORAGE_MODES.items()
    for optimizer in OPTIMIZERS
    for score_strategy in SCORE_STRATEGIES
]

# Long enough for a key to accumulate its way past the threshold.
HIGH_FREQUENCY = [
    pytest.param(
        {
            **_ONE_TABLE,
            **storage_options,
            "num_embeddings": "5000",
            "multi_hot_sizes": "3",
            "batch_size": 16,
            "num_iterations": 50,
        },
        id=f"high-frequency-{storage}",
    )
    for storage, storage_options in (
        ("hbm-direct", {}),
        ("cache", {"caching": True, "cache_capacity_ratio": 0.4}),
    )
]

# A cache far too small for the keys, so admission runs against eviction.
TINY_CACHE = [
    pytest.param(
        {
            **_ONE_TABLE,
            "optimizer_type": optimizer,
            "num_embeddings": "10000",
            "multi_hot_sizes": "5",
            "batch_size": 64,
            "num_iterations": 25,
            "caching": True,
            "cache_capacity_ratio": 0.08,
        },
        id=f"tiny-cache-{optimizer}",
    )
    for optimizer in ("sgd", "adam")
]

# A threshold no key can reach, so every value the forward returns comes from
# whichever initializer owns the rows admission rejects.
REJECT_ALL = [
    pytest.param(
        nproc,
        {
            **_FOUR_TABLES,
            **storage_options,
            **owner_options,
            "optimizer_type": "sgd",
            "threshold": 10**9,
            "expect_all_rejected": True,
        },
        id=f"{nproc}gpu-reject-all-{owner}-{storage}",
    )
    for nproc in MULTI_RANK
    for owner, owner_options in (
        ("strategy", {"non_admitted_init_value": 0.5}),
        ("table", {"no_strategy_initializer": True}),
    )
    for storage, storage_options in STORAGE_MODES.items()
]


@pytest.mark.parametrize("nproc,options", MATRIX)
def test_only_keys_past_threshold_are_admitted(nproc, options):
    """No key looked up fewer than threshold times is stored (storage x opt x score)."""
    run_worker(WORKER, nproc, as_flags(options))


@pytest.mark.parametrize("options", HIGH_FREQUENCY + TINY_CACHE)
def test_admission_single_rank_long_and_evicting(options):
    """Admission after many iterations, and when the cache is too small for the keys."""
    run_worker(WORKER, 1, as_flags(options))


@pytest.mark.parametrize("nproc,options", REJECT_ALL)
def test_rejected_keys_read_the_owning_initializer(nproc, options):
    """Unreachable threshold: nothing is stored, every value is the owner's constant."""
    run_worker(WORKER, nproc, as_flags(options))
