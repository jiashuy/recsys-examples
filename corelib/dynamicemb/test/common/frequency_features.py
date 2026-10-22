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

"""Sparse features whose per-key access counts are known, for score checks.

The expected LFU score of a key, and whether admission lets it in, are both
functions of how many times the key was looked up; this generator returns the
per-rank KJTs together with that count for every key of every table.
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from common.dynamicemb_model import idx_to_name
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def generate_deterministic_sparse_features_with_frequency_tracking(
    num_embedding_collections: int,
    num_embeddings: List[int],
    multi_hot_sizes: List[int],
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    seed: int = 42,
    caching: bool = False,
) -> Tuple[List[KeyedJaggedTensor], Dict[str, Dict[int, int]]]:
    """
    Generate deterministic sparse features and track frequency for each embedding table.

    Args:
        caching: If True, generate more unique keys to trigger cache eviction

    Returns:
        kjts: List of KeyedJaggedTensor for each iteration
        table_frequency_counters: Dict mapping table_name -> {key: frequency}
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size_per_rank = batch_size // world_size
    kjts = []
    table_frequency_counters = {}

    # Initialize frequency counters for each table
    for embedding_collection_id in range(num_embedding_collections):
        for embedding_id, num_embedding in enumerate(num_embeddings):
            _, embedding_name = idx_to_name(embedding_collection_id, embedding_id)
            table_frequency_counters[embedding_name] = defaultdict(int)

    for iteration in range(num_iterations):
        cur_indices = []
        cur_lengths = []
        keys = []

        for embedding_collection_id in range(num_embedding_collections):
            for embedding_id, num_embedding in enumerate(num_embeddings):
                feature_name, embedding_name = idx_to_name(
                    embedding_collection_id, embedding_id
                )

                for sample_id in range(batch_size):
                    hotness = random.randint(
                        1, multi_hot_sizes[embedding_collection_id]
                    )
                    if caching:
                        # In caching mode, use wider range to trigger eviction
                        # Use 80% of num_embedding to generate enough unique keys
                        max_key = int(num_embedding * 0.8) - 1
                    else:
                        # In storage-only mode, limit to 100 keys for more duplicates
                        max_key = min(num_embedding - 1, 100)
                    indices = [random.randint(0, max_key) for _ in range(hotness)]
                    # Track frequency for all generated indices
                    for idx in indices:
                        table_frequency_counters[embedding_name][idx] += 1

                    if sample_id // batch_size_per_rank == rank:
                        cur_indices.extend(indices)
                        cur_lengths.append(hotness)

                keys.append(feature_name)

        kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(cur_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(cur_lengths, dtype=torch.int64).cuda(),
            )
        )

    # Convert defaultdicts to regular dicts
    for table_name in table_frequency_counters:
        table_frequency_counters[table_name] = dict(
            table_frequency_counters[table_name]
        )

    return kjts, table_frequency_counters
