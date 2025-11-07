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

import os
import shutil
from typing import Dict, List

import pytest
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbDump,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.dump_load import load_to_cput_tensor
from dynamicemb.incremental_dump import get_score, incremental_dump
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.comm import intra_and_cross_node_pg
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


class Platform:
    def __init__(self, device):
        device_id = device.index
        gpu_name = torch.cuda.get_device_name(device_id)
        if "A100" in gpu_name:
            self.platform = "a100"
            self.intra_host_bw = 300e9
            self.inter_host_bw = 25e9
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H100" in gpu_name:
            self.platform = "h100"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 25e9  # TODO: need check
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H200" in gpu_name:
            self.platform = "h200"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 450e9
            self.hbm_cap = 140 * 1024 * 1024 * 1024
        else:
            raise RuntimeError(f"Not plan for {gpu_name}")


def get_planner(
    table_names: List[str],
    eb_configs: List[BaseEmbeddingConfig],
    use_dynamicembs: List[bool],
    score_strategies: List[DynamicEmbScoreStrategy],
    batch_size: int,
    multi_hot_sizes: List[int],
    device,
):
    dict_const = {}
    for i in range(len(table_names)):
        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
            ],
            pooling_factors=[multi_hot_sizes[i]],
            num_poolings=[1],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=use_dynamicembs[i],
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                score_strategy=score_strategies[i],
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG,
                ),
            ),
        )
        dict_const[table_names[i]] = const

    platform = Platform(device)
    topology = Topology(
        local_world_size=torchrec.distributed.comm.get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=platform.hbm_cap,
        ddr_cap=1024 * 1024 * 1024 * 1024,
        intra_host_bw=platform.intra_host_bw,
        inter_host_bw=platform.inter_host_bw,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def generate_sequence_sparse_feature(
    feature_names: List[str], local_batch_size: int, device
):
    feature_num = len(feature_names)
    feature_batch = feature_num * local_batch_size

    indices = torch.randint(
        0, (1 << 63) - 1, (feature_batch,), device=device, dtype=torch.int64
    )
    lengths = torch.ones(feature_batch, device=device, dtype=torch.int64)

    return torchrec.KeyedJaggedTensor(
        keys=feature_names,
        values=indices,
        lengths=lengths,
    )


initial_accumulator_value = 3.14159


@pytest.fixture
def optimizer_kwargs():
    optimizer_kwargs_ = {
        "optimizer": EmbOptimType.ADAM,
        "learning_rate": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "eps": 0.001,
        "initial_accumulator_value": initial_accumulator_value,
    }
    return optimizer_kwargs_


@pytest.fixture(scope="session")
def backend_session():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    yield
    # dist.barrier()
    dist.destroy_process_group()


def train_with_random_input(model, feature_names, local_batch, num_iteration, device):
    for i in range(num_iteration):
        sparse_feature = generate_sequence_sparse_feature(
            feature_names,
            local_batch,
            device=device,
        )
        ret = model(sparse_feature)  # => this is awaitable
        kt = ret.values()  # wait


def check_emb_and_opt(input, table_names, dim):
    for table_name in table_names:
        key, emb, opt = input[table_name]
        dump_keys = key % 100000
        dump_embs = emb.to(dump_keys.dtype)

        print(dump_keys.size(), dump_embs.size())
        assert torch.all(
            dump_keys.unsqueeze(1).expand(-1, dim).reshape(-1) == dump_embs.view(-1)
        )
        torch.testing.assert_close(
            opt, torch.full_like(opt, initial_accumulator_value), atol=1e-5, rtol=1e-8
        )


def check_key_equal(a, b, table_names, world_size, rank):
    for table_name in table_names:
        key_a, _, _ = a[table_name]
        key_b, _, _ = b[table_name]

        mask_a = key_a % world_size == rank
        mask_b = key_b % world_size == rank

        masked_key_a = key_a[mask_a]
        masked_key_b = key_b[mask_b]

        assert masked_key_a.numel() == torch.unique(masked_key_a).numel()
        assert masked_key_b.numel() == torch.unique(masked_key_b).numel()

        a_unique_sorted = torch.sort(torch.unique(masked_key_a)).values
        b_unique_sorted = torch.sort(torch.unique(masked_key_b)).values

        assert torch.equal(a_unique_sorted, b_unique_sorted)


def clean_file(file):
    dist.barrier()
    if dist.get_rank() == 0:
        try:
            shutil.rmtree(file)
        except Exception as e:
            print(f"Warning: Failed to remove {file}: {e}")
    dist.barrier()


def merge_ckpt(a, b, table_names):
    res = {}
    for table_name in table_names:
        key_a, emb_a, opt_a = a[table_name]
        key_b, emb_b, opt_b = b[table_name]
        key = torch.cat((key_a, key_b), dim=0)
        emb = torch.cat((emb_a.view(-1), emb_b.view(-1)), dim=0)
        opt = torch.cat((opt_a.view(-1), opt_b.view(-1)), dim=0)
        res[table_name] = (key, emb, opt)
    return res


@pytest.mark.parametrize(
    "table_num, num_embeddings, use_dynamicembs, score_strategies, multi_hot_sizes",
    [
        pytest.param(
            1,
            [100 * 1024 * 1024],
            [True],
            [DynamicEmbScoreStrategy.TIMESTAMP],
            [10],
        ),
    ],
)
@pytest.mark.parametrize(
    "is_pooled, pooling_mode",
    [
        (False, None),
    ],
)
@pytest.mark.parametrize("local_batch", [262144])
@pytest.mark.parametrize(
    "num_iteration", [12]
)  # [24, 128] is too slow: move to GPU may OOM.
@pytest.mark.parametrize("dump_interval", [3])
@pytest.mark.parametrize("dim", [8])
def test_incremental_dump_api(
    request,
    table_num,
    num_embeddings,
    use_dynamicembs,
    score_strategies,
    multi_hot_sizes,
    is_pooled,
    pooling_mode,
    local_batch,
    num_iteration,
    dump_interval,
    dim,
    optimizer_kwargs,
    backend_session,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    table_names = [f"t_{t}" for t in range(table_num)]
    feature_names = [f"f_{t}" for t in range(table_num)]

    if is_pooled:
        eb_configs = [
            torchrec.EmbeddingBagConfig(
                name=table_names[feature_idx],
                embedding_dim=dim,
                num_embeddings=num_embeddings[feature_idx],
                feature_names=[feature_names[feature_idx]],
                pooling=pooling_mode,
            )
            for feature_idx in range(table_num)
        ]
        ebc = torchrec.EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=eb_configs,
        )
    else:
        eb_configs = [
            torchrec.EmbeddingConfig(
                name=table_names[feature_idx],
                embedding_dim=dim,
                num_embeddings=num_embeddings[feature_idx],
                feature_names=[feature_names[feature_idx]],
            )
            for feature_idx in range(table_num)
        ]
        ebc = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=eb_configs,
        )

    print("EmbeddingCollection:", ebc)
    planner = get_planner(
        table_names,
        eb_configs,
        use_dynamicembs,
        score_strategies,
        local_batch,
        multi_hot_sizes,
        device,
    )

    if is_pooled:
        sharder = DynamicEmbeddingBagCollectionSharder(fused_params=optimizer_kwargs)
    else:
        sharder = DynamicEmbeddingCollectionSharder(
            fused_params=optimizer_kwargs, use_index_dedup=False
        )

    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    print("Plan:", plan)

    model = DistributedModelParallel(
        module=ebc,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )

    ret: Dict[str, Dict[str, int]] = get_score(model)
    prefix_path = "model"
    undump_score: Dict[str, int] = ret[prefix_path]

    # at present, the model is empty, so will dump nothing
    ret_tensors, _ = incremental_dump(
        model, {prefix_path: undump_score}, intra_and_cross_node_pg()[0]
    )
    for table_name in table_names:
        dump_keys = ret_tensors[prefix_path][table_name][0]
        dump_embs = ret_tensors[prefix_path][table_name][1]
        dump_opts = ret_tensors[prefix_path][table_name][2]
        assert dump_keys.numel() == 0
        assert dump_embs.numel() == 0
        assert dump_opts.numel() == 0

    clean_file("full_ckpt_1")
    clean_file("full_ckpt_2")

    train_with_random_input(model, feature_names, local_batch, num_iteration, device)
    DynamicEmbDump("full_ckpt_1", model, optim=True)

    ret = get_score(model)
    intermediate_score: Dict[str, int] = ret[prefix_path]

    train_with_random_input(model, feature_names, local_batch, num_iteration, device)

    ret_tensors, _ = incremental_dump(
        model, {prefix_path: intermediate_score}, intra_and_cross_node_pg()[0]
    )
    partial_ckpt = {}
    for table_name in table_names:
        dump_keys = ret_tensors[prefix_path][table_name][0]
        dump_embs = ret_tensors[prefix_path][table_name][1]
        dump_opts = ret_tensors[prefix_path][table_name][2]
        partial_ckpt[table_name] = (dump_keys, dump_embs, dump_opts)

    zero_scores = {prefix_path: {table_name: 0 for table_name in table_names}}
    ret_tensors, _ = incremental_dump(model, zero_scores, intra_and_cross_node_pg()[0])
    inc_ckpt = {}
    for table_name in table_names:
        dump_keys = ret_tensors[prefix_path][table_name][0]
        dump_embs = ret_tensors[prefix_path][table_name][1]
        dump_opts = ret_tensors[prefix_path][table_name][2]
        inc_ckpt[table_name] = (dump_keys, dump_embs, dump_opts)

    DynamicEmbDump("full_ckpt_2", model, optim=True)

    full_ckpt_1 = load_to_cput_tensor(
        "full_ckpt_1", model, table_names={prefix_path: table_names}, optim=True
    )
    full_ckpt_2 = load_to_cput_tensor(
        "full_ckpt_2", model, table_names={prefix_path: table_names}, optim=True
    )

    check_emb_and_opt(inc_ckpt, table_names, dim)
    check_emb_and_opt(partial_ckpt, table_names, dim)
    check_emb_and_opt(full_ckpt_1, table_names, dim)
    check_emb_and_opt(full_ckpt_2, table_names, dim)

    ############### inc_ckpt == full_ckpt_2
    check_key_equal(inc_ckpt, full_ckpt_2, table_names, world_size, local_rank)

    ############### full_ckpt_1 + partial_ckpt == full_ckpt_2
    merged_ckpt = merge_ckpt(full_ckpt_1, partial_ckpt, table_names)
    check_key_equal(merged_ckpt, full_ckpt_2, table_names, world_size, local_rank)

    clean_file("full_ckpt_1")
    clean_file("full_ckpt_2")
