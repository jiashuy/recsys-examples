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

"""Planner / optimizer / platform helpers for tests that shard a TorchRec model.

``get_planner`` reads a flat ``args`` namespace (the one the worker scripts parse)
with: ``num_embedding_table``, ``data_parallel_embeddings`` (list or None),
``multi_hot_sizes``, ``dynamicemb_num``, ``hbm_cap``, ``intra_host_bw``,
``inter_host_bw``, ``batch_size``. ``apply_platform`` fills in the three machine
fields from a platform name.
"""

from typing import List, Optional

import torch.distributed as dist
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.dynamicemb_config import ScoreStrategy
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.fbgemm_qcomm_codec import CommType
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig

from common.features import table_idx_to_name
from common.gpu_platform import Platform

_COMM_TYPES = {
    "fp32": CommType.FP32,
    "fp16": CommType.FP16,
    "bf16": CommType.BF16,
    "fp8": CommType.FP8,
}

_FUSED_OPTIMIZERS = {
    "sgd": EmbOptimType.EXACT_SGD,
    "exact_sgd": EmbOptimType.EXACT_SGD,
    "adam": EmbOptimType.ADAM,
    "exact_adagrad": EmbOptimType.EXACT_ADAGRAD,
    "exact_row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}


def get_comm_precission(precision_str):
    if precision_str not in _COMM_TYPES:
        raise ValueError("unknown comm precision type")
    return _COMM_TYPES[precision_str]


def fused_optimizer_kwargs(args):
    """``fused_params`` entries that select the in-kernel (TorchRec) optimizer."""
    if args.optimizer_type not in _FUSED_OPTIMIZERS:
        raise ValueError("unknown optimizer type")
    return {
        "learning_rate": args.learning_rate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "optimizer": _FUSED_OPTIMIZERS[args.optimizer_type],
    }


def apply_platform(args):
    """Set ``intra_host_bw`` / ``inter_host_bw`` / ``hbm_cap`` from ``args.platform``."""
    if args.platform == "a100":
        args.intra_host_bw = 300e9
        args.inter_host_bw = 25e9
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h100":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 25e9  # TODO: need check
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h200":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 450e9
        args.hbm_cap = 140 * 1024 * 1024 * 1024


def get_planner(args, device):
    """Row-wise plan for every table; the first ``dynamicemb_num`` use dynamicemb."""
    dict_const = {}
    for i in range(args.num_embedding_table):
        if (
            args.data_parallel_embeddings is not None
            and i in args.data_parallel_embeddings
        ):
            # DynamicEmbeddingShardingPlanner reads ``use_dynamicemb`` from every
            # constraint, so a TorchRec-owned table still gets this class.
            const = DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.DATA_PARALLEL.value],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
            )
        else:
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                ],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=i < args.dynamicemb_num,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=1024**3,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.DEBUG
                    ),
                    safe_check_mode=DynamicEmbCheckMode.WARNING,
                ),
            )
        dict_const[table_idx_to_name(i)] = const

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=args.hbm_cap,
        # One node's host memory, not one rank's: Topology replicates
        # whatever it is given to every device instead of dividing it.
        ddr_cap=1024 * 1024 * 1024 * 1024 // get_local_size(),
        intra_host_bw=args.intra_host_bw,
        inter_host_bw=args.inter_host_bw,
    )

    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints=dict_const,
        batch_size=args.batch_size,
        debug=True,
    )


def get_per_table_planner(
    table_names: List[str],
    eb_configs: List[BaseEmbeddingConfig],
    use_dynamicembs: List[bool],
    score_strategies: List[Optional[ScoreStrategy]],
    batch_size: int,
    multi_hot_sizes: List[int],
    device,
    training: bool = True,
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
                training=training,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG,
                ),
            ),
        )
        dict_const[table_names[i]] = const

    platform = Platform(device)
    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=platform.hbm_cap,
        # One node's host memory, not one rank's: Topology replicates
        # whatever it is given to every device instead of dividing it.
        ddr_cap=1024 * 1024 * 1024 * 1024 // get_local_size(),
        intra_host_bw=platform.intra_host_bw,
        inter_host_bw=platform.inter_host_bw,
    )
    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        debug=True,
    )


