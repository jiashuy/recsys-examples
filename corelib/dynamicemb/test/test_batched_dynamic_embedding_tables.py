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

import torch
from dynamicemb import (
    DynamicEmbEvictStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbStorageConfig,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables


def test_embedding_optimizer(opt_type, opt_params):
    print(
        f"step in test_embedding_optimizer , opt_type = {opt_type} opt_params = {opt_params}"
    )
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [128, 31, 16]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = 1024
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbStorageConfig(
            dim=dim, init_capacity=init_capacity, max_capacity=max_capacity
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdeb = BatchedDynamicEmbeddingTables(
        table_options=dyn_emb_table_options_list,
        index_type=key_type,
        embedding_dtype=value_type,
        device_id=device_id,
        evict_strategy=DynamicEmbEvictStrategy.LRU,
        feature_table_map=[0, 0, 1, 2],
        pooling_mode=DynamicEmbPoolingMode.MEAN,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.UNIFORM,
        ),
        optimizer=opt_type,
        **opt_params,
    )
    """
    feature number = 4, batch size = 2

    f0  [0,1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indices = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(
        key_type
    )
    offsets = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)

    embs = bdeb(indices, offsets)

    torch.cuda.synchronize()

    print(f"forward embs.shape : {embs.shape}")
    print(f"forward Embedding : {embs}")
    loss = embs.mean()

    loss.backward()


if __name__ == "__main__":
    optimizer_params = [
        {
            "learning_rate": 0.3,
        },
        {
            "learning_rate": 0.3,
            "weight_decay": 0.06,
            "eps": 3e-5,
            "beta1": 0.8,
            "beta2": 0.888,
        },
    ]

    opt_types = [EmbOptimType.SGD, EmbOptimType.ADAM]
    for i in range(len(opt_types)):
        test_embedding_optimizer(opt_types[i], optimizer_params[i])
