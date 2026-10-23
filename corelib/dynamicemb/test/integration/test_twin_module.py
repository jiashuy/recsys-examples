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

"""A dynamicemb-sharded model trains identically to the same model in plain TorchRec.

Level: integration.   Needs: 1 GPU, and 8 GPUs (collected as one item per count).
Covers: ``ConstructTwinModule`` builds a sharded dynamicemb model and its sharded
TorchRec twin with the same initial rows; both see the same batches for 10
iterations of forward + backward + optimizer step and must produce equal outputs.
Axes: 1 or 4 tables; pooled SUM (bags of up to 8 keys) or sequence (1 key per bag);
batch 32 or 2048; optimizer (sgd, exact_adagrad, exact_row_wise_adagrad); index
dedup on/off; per-key weights (pooled SUM only, one optimizer).
Does not cover (see unit/module/test_weighted_pooled_embedding_v2.py): the weighted
pooling kernels against an unsharded reference (alignment, eval path, argument
errors); (see unit/module/test_embedding_collection_context.py): the context object.
"""

import copy
import os
from typing import List

import pytest
import torch
from common.features import generate_sparse_feature
from dynamicemb.construct_twin_module import ConstructTwinModule
from torchrec.modules.embedding_configs import PoolingType

DISTRIBUTED_NPROCS = (1, 8)


def embedding_lookup_backward(ret, is_pooled):
    """
    Perform backward pass on embeddings for both pooled and non-pooled cases
    """
    if is_pooled:
        # For pooled embeddings (EmbeddingBagCollection output)
        # The output is a KeyedTensor - directly sum values
        reduced_tensor = ret.values().sum()
        reduced_tensor.backward()
    else:
        # For non-pooled embeddings (EmbeddingCollection output)
        # The output is a dict of KeyedJaggedTensors
        jagged_tensors = []
        for k, v in ret.items():
            jagged_tensors.append(v.values())

        if jagged_tensors:  # Check if list is not empty
            concatenated_tensor = torch.cat(jagged_tensors, dim=0)
            reduced_tensor = concatenated_tensor.sum()
            reduced_tensor.backward()


optimizer_dict = {
    "sgd": {
        "optimizer": "sgd",
        "learning_rate": 0.01,
    },
    "exact_adagrad": {
        "optimizer": "exact_adagrad",
        "learning_rate": 0.01,
        "eps": 1e-10,
    },
    "exact_row_wise_adagrad": {
        "optimizer": "exact_row_wise_adagrad",
        "learning_rate": 0.01,
        "eps": 1e-10,
    },
    "adam": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-12,
        "weight_decay": 0,
    },
    "adamw": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-12,
        "weight_decay": 0.1,
    },
}


@pytest.fixture
def seed():
    return 1234


@pytest.mark.parametrize(
    "table_num, num_embeddings",
    [
        pytest.param(1, [128 * 1024]),
        pytest.param(4, [i * 128 * 1024 for i in [1, 2, 3, 4]]),
    ],
)
# The bag size belongs with the pooling mode: a sequence lookup has one
# embedding per key by definition, while SUM has to actually accumulate several
# of them -- with a bag of at most one key the reduction being tested never runs.
@pytest.mark.parametrize(
    "is_pooled, pooling_mode, max_bag_size",
    [(True, PoolingType.SUM, 8), (False, None, 1)],
)
@pytest.mark.parametrize("batch_size", [32, 2048])
@pytest.mark.parametrize("num_iteration", [10])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize(
    "optimizer_name",
    ["sgd", "exact_row_wise_adagrad", "exact_adagrad", "adam", "adamw"],
)
@pytest.mark.parametrize("use_index_dedup", [True, False])
@pytest.mark.parametrize("is_weighted", [False, True])
def test_twin_module(
    table_num: int,
    dim: int,
    num_embeddings: List[int],
    is_pooled: bool,
    pooling_mode,
    max_bag_size: int,
    batch_size,
    num_iteration,
    seed,
    optimizer_name,
    backend_session,
    use_index_dedup,
    is_weighted,
):
    """Outputs of the sharded dynamicemb model and its TorchRec twin stay equal."""
    if is_weighted:
        if not (is_pooled and pooling_mode == PoolingType.SUM):
            pytest.skip("weighted pooling only exists for pooled SUM collections")
        if optimizer_name != "exact_row_wise_adagrad":
            # The weights only scale what reaches the gradient reduction; which
            # optimizer consumes the result afterwards is orthogonal. Covering
            # every optimizer again would triple this axis for no extra signal.
            pytest.skip("weighted path is covered once, on exact_row_wise_adagrad")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    [f"t_{t}" for t in range(table_num)]
    feature_names = [f"f_{t}" for t in range(table_num)]

    optimizer_kwargs = copy.deepcopy(optimizer_dict[optimizer_name])

    dims = [dim] * table_num
    # randint(0, max_bag_size) per bag, so lengths still vary and empty bags
    # still occur; this only sets the upper end.
    multi_hot_sizes = [max_bag_size] * table_num

    construct = ConstructTwinModule(
        table_num,
        dims,
        num_embeddings,
        pooling_mode,
        is_pooled,
        multi_hot_sizes,
        optimizer_kwargs=optimizer_kwargs,
        use_index_dedup=use_index_dedup,
        is_weighted=is_weighted,
        rank=local_rank,
        world_size=world_size,
    )
    construct.init_twin_embedding_model()
    dynamicemb_model = construct.dynamicemb_model
    torchrec_model = construct.torchrec_model

    for i in range(num_iteration):
        if i % 2 == 0:
            sparse_feature = generate_sparse_feature(
                len(feature_names),
                num_embeddings,
                multi_hot_sizes,
                batch_size,
                feature_names=feature_names,
                weighted=is_weighted,
            )
        ret = dynamicemb_model(sparse_feature)
        ret_compare = torchrec_model(sparse_feature)

        embedding_lookup_backward(ret, is_pooled)
        embedding_lookup_backward(ret_compare, is_pooled)

        if is_pooled:
            # Check for equivalence with relative tolerance
            assert ret_compare.keys() == ret.keys()

            # Get the values
            torchrec_values = ret_compare.values()
            dynamicemb_values = ret.values()
            torch.testing.assert_close(torchrec_values, dynamicemb_values)
        else:
            for key, value in ret_compare.items():
                torchrec_values = value.values()
                dynamicemb_values = ret[key].values()
                torch.testing.assert_close(torchrec_values, dynamicemb_values)
