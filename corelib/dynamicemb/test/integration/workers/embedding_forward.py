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

"""Worker: shard a TorchRec embedding model with dynamicemb and run forward passes.

Launched under torchrun by ``test_embedding_forward_backward.py``. Every iteration
feeds random sparse features through a ``DistributedModelParallel`` model and the
``Debugger`` compares what comes out of the all-to-all with a reference computed
from the inputs (it only checks the first ``PYHKV_DEBUG_ITER`` iterations, so the
launcher must set that variable).

``--kind`` picks the model:

* ``pooled``         one ``EmbeddingBagCollection`` (pooled output, sum).
* ``sequence``       one ``EmbeddingCollection`` (one vector per key); the last
                     iteration has an empty batch on rank 0 and data elsewhere.
* ``multi_sequence`` two ``EmbeddingCollection`` s (first / second half of the
                     tables) in one model, each fed its own features.

Flags left out keep the default of the kind (see ``KIND_DEFAULTS``).
"""

import argparse
import os
import sys
from typing import List

import torch
import torch.distributed as dist
import torchrec
from common.cli import str2bool
from common.debug import Debugger
from common.features import (
    feature_idx_to_name,
    generate_sparse_feature,
    table_idx_to_name,
)
from common.sharding import (
    apply_platform,
    fused_optimizer_kwargs,
    get_comm_precission,
    get_planner,
)
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from fbgemm_gpu.split_embedding_configs import SparseType
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.fbgemm_qcomm_codec import (
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import (
    DefaultDataParallelWrapper,
    DistributedModelParallel,
)

KIND_DEFAULTS = {
    "pooled": dict(
        batch_size=32,
        multi_hot_sizes="15,100,20,1",
        dynamicemb_num=2,
        use_torch_opt=False,
    ),
    "sequence": dict(
        batch_size=16,
        multi_hot_sizes="16,8,20,1",
        dynamicemb_num=2,
        use_torch_opt=False,
    ),
    "multi_sequence": dict(
        batch_size=16,
        multi_hot_sizes="16,8,20,1",
        dynamicemb_num=4,
        use_torch_opt=True,
    ),
}


class TwoCollections(torch.nn.Module):
    """Two ``EmbeddingCollection`` s over the first and second half of the tables."""

    def __init__(self, eb_configs: List[torchrec.EmbeddingConfig]):
        super().__init__()
        mid = len(eb_configs) // 2
        self.ebc1 = torchrec.EmbeddingCollection(
            device=torch.device("meta"), tables=eb_configs[:mid]
        )
        self.ebc2 = torchrec.EmbeddingCollection(
            device=torch.device("meta"), tables=eb_configs[mid:]
        )

    def forward(self, features1, features2):
        return [self.ebc1(features1), self.ebc2(features2)]


def build_module(args):
    """The unsharded (meta-device) module, and the list of its collections."""
    names = range(args.num_embedding_table)
    table_kwargs = lambda i: dict(  # noqa: E731
        name=table_idx_to_name(i),
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings_per_feature[i],
        feature_names=[feature_idx_to_name(i)],
    )
    if args.kind == "pooled":
        module = torchrec.EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[torchrec.EmbeddingBagConfig(**table_kwargs(i)) for i in names],
        )
        return module, [module]
    configs = [torchrec.EmbeddingConfig(**table_kwargs(i)) for i in names]
    if args.kind == "sequence":
        module = torchrec.EmbeddingCollection(
            device=torch.device("meta"), tables=configs
        )
        return module, [module]
    module = TwoCollections(configs)
    return module, [module.ebc1, module.ebc2]


def torch_optimizer(args):
    if args.optimizer_type == "sgd":
        return torch.optim.SGD
    if args.optimizer_type == "adam":
        return torch.optim.Adam
    if args.optimizer_type == "exact_adagrad":
        return torch.optim.Adagrad
    raise ValueError("unknown optimizer type")


def build_model(args, device):
    module, collections = build_module(args)

    if args.use_torch_opt:
        optimizer_kwargs = {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.weight_decay,
            "eps": args.eps,
        }
        embedding_optimizer = torch_optimizer(args)
        fused_params = {"output_dtype": SparseType.FP32}
    else:
        # TODO: parse args.output_dtype
        fused_params = {"output_dtype": SparseType.FP32}
        fused_params.update(fused_optimizer_kwargs(args))

    planner = get_planner(args, device)

    precision = get_comm_precission(args.fwd_a2a_precision)
    qcomm_codecs_registry = get_qcomm_codecs_registry(
        qcomms_config=QCommsConfig(
            # pyre-ignore
            forward_precision=precision,
            # pyre-ignore
            backward_precision=precision,
        )
    )
    if args.kind == "pooled":
        sharder = DynamicEmbeddingBagCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry, fused_params=fused_params
        )
    else:
        sharder = DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
            use_index_dedup=args.use_index_dedup,
        )

    plan = planner.collective_plan(module, [sharder], dist.GroupMember.WORLD)
    if args.use_torch_opt:
        for collection in collections:
            apply_optimizer_in_backward(
                embedding_optimizer, collection.parameters(), optimizer_kwargs
            )

    model = DistributedModelParallel(
        module=module,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
        data_parallel_wrapper=DefaultDataParallelWrapper(
            allreduce_comm_precision=args.allreduce_precision
        ),
    )

    if dist.get_rank() == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, table_plan in plans.items():
                print(table_name, "\n", table_plan, "\n")
    return model


def make_features(args, local_batch_size):
    return generate_sparse_feature(
        feature_num=args.num_embedding_table,
        num_embeddings_list=args.num_embeddings_per_feature,
        multi_hot_sizes=args.multi_hot_sizes,
        local_batch_size=local_batch_size,
    )


def check_pooled(args, model, debugger, local_batch_size):
    for i in range(args.num_iterations):
        sparse_feature = make_features(args, local_batch_size)
        ret = model(sparse_feature)  # => this is awaitable
        kt = ret.values()
        feature_num = len(ret.keys())
        dims = [args.embedding_dim for _ in range(feature_num)]
        dyn_emb_features = [i for i in range(args.dynamicemb_num)]

        debugger.feature_before_all2all(sparse_feature)
        debugger.pooled_embds_after_all2all(kt, dims, feature_num, dyn_emb_features)
        print(f"DynamicEmb iteration {i+1} Passed")

    # The sharded state dict must load back into the model it came from.
    model.load_state_dict(model.state_dict())


def check_sequence_output(args, debugger, features, ret):
    feature_names = []
    jagged_tensors = []
    for k, v in ret.items():
        feature_names.append(k)
        jagged_tensors.append(v)
    dyn_emb_features = [feature_idx_to_name(i) for i in range(args.dynamicemb_num)]

    debugger.feature_before_all2all(features)
    debugger.sequence_embds_after_all2all(
        jagged_tensors, feature_names, dyn_emb_features, args.embedding_dim
    )


def check_sequence(args, model, debugger, local_batch_size, device):
    for i in range(args.num_iterations + 1):
        if i == args.num_iterations and dist.get_rank() == 0:
            # Extra iteration: rank 0 empty indices; other ranks (if any) non-empty.
            sparse_feature = torchrec.KeyedJaggedTensor(
                keys=[
                    feature_idx_to_name(f) for f in range(args.num_embedding_table)
                ],
                values=torch.tensor([], dtype=torch.int64, device=device),
                lengths=torch.zeros(
                    args.num_embedding_table * local_batch_size,
                    dtype=torch.int64,
                    device=device,
                ),
            )
        else:
            sparse_feature = make_features(args, local_batch_size)
        ret = model(sparse_feature)  # => this is awaitable
        check_sequence_output(args, debugger, sparse_feature, ret)
        print(f"DynamicEmb iteration {i+1} Passed")


def check_multi_sequence(args, model, debugger, local_batch_size):
    mid = args.num_embedding_table // 2
    for i in range(args.num_iterations):
        feature1 = generate_sparse_feature(
            feature_num=mid,
            num_embeddings_list=args.num_embeddings_per_feature[:mid],
            multi_hot_sizes=args.multi_hot_sizes[:mid],
            local_batch_size=local_batch_size,
        )
        # The second collection's keys are drawn from the first half's row
        # ranges (kept as the test always had it): they are valid keys for
        # dynamicemb tables, which hash them, so this also covers out-of-range
        # ids.
        feature2 = generate_sparse_feature(
            feature_num=args.num_embedding_table - mid,
            num_embeddings_list=args.num_embeddings_per_feature[:mid],
            multi_hot_sizes=args.multi_hot_sizes[:mid],
            local_batch_size=local_batch_size,
            offset=mid,
        )
        ret1, ret2 = model(feature1, feature2)
        check_sequence_output(args, debugger, feature1, ret1)
        check_sequence_output(args, debugger, feature2, ret2)
        print(f"DynamicEmb iteration {i+1} Passed")

    model.load_state_dict(model.state_dict())


def run(args):
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = build_model(args, device)
    debugger = Debugger()
    local_batch_size = args.batch_size // world_size

    if args.kind == "pooled":
        check_pooled(args, model, debugger, local_batch_size)
    elif args.kind == "sequence":
        check_sequence(args, model, debugger, local_batch_size, device)
    else:
        assert args.num_embedding_table == len(args.multi_hot_sizes)
        check_multi_sequence(args, model, debugger, local_batch_size)

    dist.barrier()
    dist.destroy_process_group()


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Forward of a dynamicemb-sharded TorchRec model"
    )
    parser.add_argument(
        "--kind", required=True, choices=sorted(KIND_DEFAULTS), help="model to build"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="65536,32768,4096,8192",
        help="Comma separated number of embeddings (rows) of each table.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated maximum bag size of each feature.",
    )
    parser.add_argument("--print_sharding_plan", action="store_true")
    parser.add_argument(
        "--fwd_a2a_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--allreduce_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        choices=["sgd", "adam", "exact_adagrad", "row_wise_adagrad"],
    )
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--use_torch_opt",
        type=str2bool,
        default=None,
        help="Register a torch optimizer in backward instead of the fused one.",
    )
    parser.add_argument(
        "--data_parallel_embeddings",
        type=str,
        default=None,
        help="Comma separated data parallel embedding table ids.",
    )
    parser.add_argument(
        "--platform", type=str, default="a100", choices=["a100", "h100", "h200"]
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
        help="Accepted but not applied yet (the model always outputs fp32).",
    )
    parser.add_argument(
        "--dynamicemb_num",
        type=int,
        default=None,
        help="Number of tables (from the first) that are dynamicemb tables.",
    )
    parser.add_argument(
        "--use_index_dedup",
        type=str2bool,
        default=True,
        help="Use index deduplication (ignored by the pooled kind).",
    )
    args = parser.parse_args(argv)

    for name, default in KIND_DEFAULTS[args.kind].items():
        if getattr(args, name) is None:
            setattr(args, name, default)

    args.num_embeddings_per_feature = [
        int(v) for v in args.num_embeddings_per_feature.split(",")
    ]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]
    args.data_parallel_embeddings = (
        None
        if args.data_parallel_embeddings is None
        else [int(v) for v in args.data_parallel_embeddings.split(",")]
    )
    args.num_embedding_table = len(args.num_embeddings_per_feature)

    if args.embedding_dim % 4 != 0:
        print(
            f"INFO: args.embedding_dim = {args.embedding_dim} is not aligned with 4, "
            "which can't use TorchREC raw embedding table, so all embedding tables "
            "are dynamic embedding tables"
        )
        args.dynamicemb_num = args.num_embedding_table

    apply_platform(args)
    return args


@record
def main(argv: List[str]) -> None:
    args = parse_args(argv)
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
