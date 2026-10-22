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

"""Sparse-feature generators and naming conventions shared by the tests.

A table ``i`` is called ``t_i`` and its single feature ``cate_i``; every test that
builds a TorchRec model from ``num_embeddings_per_feature`` uses these names so
the generated ``KeyedJaggedTensor`` lines up with the model's tables.
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
import torchrec


def table_idx_to_name(i: int) -> str:
    return f"t_{i}"


def feature_idx_to_name(i: int) -> str:
    return f"cate_{i}"


def init_fn(x: torch.Tensor) -> None:
    """Initializer that fills a table with the constant 2.0."""
    with torch.no_grad():
        x.fill_(2.0)


def random_indices(batch: int, min_index: int, max_index: int) -> set:
    """``batch`` distinct integers drawn uniformly from ``[min_index, max_index]``."""
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


def generate_sparse_feature(
    feature_num: int,
    num_embeddings_list: List[int],
    multi_hot_sizes: List[int],
    local_batch_size: int = 50,
    offset: int = 0,
    feature_names: Optional[List[str]] = None,
    weighted: bool = False,
) -> torchrec.KeyedJaggedTensor:
    """A feature-major ``KeyedJaggedTensor`` on the current CUDA device.

    Feature ``f`` gets ``local_batch_size`` bags; each bag holds
    ``randint(0, multi_hot_sizes[f])`` distinct keys in ``[0, num_embeddings_list[f])``
    (so lengths vary and empty bags occur). Keys are named ``cate_{f + offset}``
    unless ``feature_names`` is given. ``weighted`` adds per-key weights in
    ``[0.5, 2.0)``.
    """
    feature_batch = feature_num * local_batch_size

    indices = []
    lengths = []

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    weights = None
    if weighted:
        weights = torch.rand(len(indices), dtype=torch.float32).cuda() * 1.5 + 0.5

    if feature_names is None:
        feature_names = [
            feature_idx_to_name(feature_idx + offset)
            for feature_idx in range(feature_num)
        ]
    return torchrec.KeyedJaggedTensor(
        keys=feature_names,
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
        weights=weights,
    )
