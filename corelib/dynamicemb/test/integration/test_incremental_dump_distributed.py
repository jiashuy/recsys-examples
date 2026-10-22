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

"""``incremental_dump`` / ``replay_increment`` on sharded models, 1 and 8 ranks.

Level: integration.   Needs: 1 GPU (the ``[1-gpu]`` item) or 8 GPUs (``[8-gpu]``).
Covers: through the planner and DistributedModelParallel (row-wise sharding),
    ``get_score`` / ``set_score`` / ``incremental_dump`` over EmbeddingBagCollection
    and EmbeddingCollection with dynamic and non-dynamic tables and the STEP,
    TIMESTAMP and CUSTOMIZED score strategies: every key touched since the
    last dump is in the delta, with its value; and ``replay_increment`` of a
    globally gathered delta into a serving replica, whose ranks together hold the
    whole delta and whose forward equals the source's.
Does not cover (see unit/incremental_dump/): the dump and replay on a bare
    table, without the model stack.
"""

DISTRIBUTED_NPROCS = (1, 8)

import os
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torchrec
from common.features import generate_unbounded_sparse_feature
from common.scores import PerTableCustomizedScore
from common.sharding import get_per_table_planner
from dynamicemb import (
    BATCH_SIZE_PER_DUMP,
    DynamicEmbScoreStrategy,
)
from dynamicemb.incremental_dump import (
    get_score,
    incremental_dump,
    replay_increment,
    set_score,
)
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.comm import intra_and_cross_node_pg
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import PoolingType

pytestmark = pytest.mark.load_dump


@pytest.fixture
def optimizer_kwargs():
    optimizer_kwargs_ = {
        "optimizer": EmbOptimType.ADAM,
        "learning_rate": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "eps": 0.001,
    }
    return optimizer_kwargs_


@pytest.mark.parametrize(
    "table_num, num_embeddings, use_dynamicembs, score_strategies, multi_hot_sizes",
    [
        pytest.param(
            1,
            [BATCH_SIZE_PER_DUMP * 8],
            [True],
            [DynamicEmbScoreStrategy.TIMESTAMP],
            [10],
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [True] * 4,
            [
                DynamicEmbScoreStrategy.STEP,
                DynamicEmbScoreStrategy.TIMESTAMP,
                DynamicEmbScoreStrategy.CUSTOMIZED,
                DynamicEmbScoreStrategy.STEP,
            ],
            [10] * 4,
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [False, False, False, False],
            [None, None, None, None],
            [10] * 4,
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [True, False, True, False],
            [
                DynamicEmbScoreStrategy.STEP,
                None,
                DynamicEmbScoreStrategy.CUSTOMIZED,
                None,
            ],
            [10] * 4,
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [True] * 4,
            [DynamicEmbScoreStrategy.CUSTOMIZED] * 4,
            [10] * 4,
        ),
    ],
)
@pytest.mark.parametrize(
    "is_pooled, pooling_mode",
    [
        (True, PoolingType.SUM),
        (False, None),
    ],
)
@pytest.mark.parametrize("local_batch", [128])
@pytest.mark.parametrize("num_iteration", [12])
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
    """Every key touched between dumps is dumped with its value (EBC and EC, all scores)."""
    local_rank = int(os.environ["LOCAL_RANK"])
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
    planner = get_per_table_planner(
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

    customized_scores = PerTableCustomizedScore(table_names)
    ret: Dict[str, Dict[str, int]] = get_score(model)
    prefix_path = "model"

    if ret is None:
        return
    else:
        assert len(ret) == 1 and prefix_path in ret
    undump_score: Dict[str, int] = ret[prefix_path]

    scores_to_set: Dict[str, int] = {}
    for i in range(table_num):
        if score_strategies[i] == DynamicEmbScoreStrategy.CUSTOMIZED:
            scores_to_set[table_names[i]] = customized_scores.get(table_names[i])
    all_customized = (
        True
        if score_strategies == [DynamicEmbScoreStrategy.CUSTOMIZED] * table_num
        else False
    )
    param_scores = (
        scores_to_set[table_names[0]]
        if all_customized
        else {prefix_path: scores_to_set}
    )
    set_score(model, param_scores)
    undump_score.update(scores_to_set)

    for i in range(0, num_iteration):
        unique_indices = [set({}) for _ in table_names]
        for j in range(dump_interval):
            scores_to_set: Dict[str, int] = {}
            for i in range(table_num):
                if score_strategies[i] == DynamicEmbScoreStrategy.CUSTOMIZED:
                    scores_to_set[table_names[i]] = customized_scores.get(
                        table_names[i]
                    )
            set_score(model, {prefix_path: scores_to_set})
            sparse_feature = generate_unbounded_sparse_feature(
                feature_names,
                multi_hot_sizes,
                local_batch,
                unique_indices,
                use_dynamicembs,
                num_embeddings,
            )
            ret = model(sparse_feature)  # => this is awaitable
            kt = ret.values()  # wait

        print("Dump score=", undump_score)
        param_scores = (
            undump_score[table_names[0]]
            if all_customized
            else {prefix_path: undump_score}
        )
        res = incremental_dump(model, param_scores, intra_and_cross_node_pg()[0])
        dr = res[prefix_path]  # DeltaDumpResult for this collection
        undump_score = {
            tn: dr.meta[j]["current_score"] for j, tn in enumerate(dr.table_names)
        }
        for i, (table_name, indices) in enumerate(zip(table_names, unique_indices)):
            if use_dynamicembs[i]:
                col = dr.table_names.index(table_name)
                dump_keys = dr.keys[col]
                dump_vals = dr.values[col]
                dumped_indices = set(dump_keys.tolist())
                assert indices.issubset(dumped_indices)

                ############### Test the dumped embeddings ###############
                dump_keys = dump_keys % 100000
                dump_vals = dump_vals.to(dump_keys.dtype)
                assert torch.all(dump_keys.unsqueeze(1).expand(-1, dim) == dump_vals)


def _build_sharded_model(
    table_names, eb_configs, score_strategies, batch, dim, device, training=True
):
    """One row-wise sharded EmbeddingCollection over dynamic embedding tables.

    ``training=False`` builds a serving replica: the table option (not
    ``nn.Module.training``) is what drops the optimizer, so the value row is the
    embedding alone and there is no per-row state for a replay to restore.
    """
    ebc = torchrec.EmbeddingCollection(device=torch.device("meta"), tables=eb_configs)
    planner = get_per_table_planner(
        table_names,
        eb_configs,
        [True] * len(table_names),
        score_strategies,
        batch,
        [1] * len(table_names),
        device,
        training=training,
    )
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params={"optimizer": EmbOptimType.SGD, "learning_rate": 0.1},
        use_index_dedup=False,
    )
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    return DistributedModelParallel(
        module=ebc, device=device, sharders=[sharder], plan=plan
    )


@pytest.mark.parametrize("table_num, dim, local_batch", [(2, 8, 64)])
def test_replay_increment_distributed(
    request, table_num, dim, local_batch, backend_session
):
    """A globally gathered delta replayed into a second sharded model: every rank
    keeps only the keys it owns, and together they restore the whole delta."""
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    prefix_path = "model"

    table_names = [f"t_{t}" for t in range(table_num)]
    eb_configs = [
        torchrec.EmbeddingConfig(
            name=table_names[t],
            embedding_dim=dim,
            num_embeddings=BATCH_SIZE_PER_DUMP * 8,
            feature_names=[f"f_{t}"],
        )
        for t in range(table_num)
    ]
    strategies = [DynamicEmbScoreStrategy.TIMESTAMP] * table_num

    train_model = _build_sharded_model(
        table_names, eb_configs, strategies, local_batch, dim, device
    )
    # A serving replica: no optimizer state in the value row, and eval forwards
    # that neither insert nor admit, so everything it holds came from the delta.
    serve_model = _build_sharded_model(
        table_names, eb_configs, strategies, local_batch, dim, device, training=False
    )
    serve_model.eval()

    unique_indices = [set({}) for _ in table_names]
    sparse_feature = generate_unbounded_sparse_feature(
        [f"f_{t}" for t in range(table_num)],
        [1] * table_num,
        local_batch,
        unique_indices,
        [True] * table_num,
        [BATCH_SIZE_PER_DUMP * 8] * table_num,
    )
    trained = {k: v.values().clone() for k, v in train_model(sparse_feature).items()}

    pg = intra_and_cross_node_pg()[0]
    deltas = incremental_dump(train_model, 0, pg)
    stats = replay_increment(serve_model, deltas)

    delta = deltas[prefix_path]
    for j, table_name in enumerate(delta.table_names):
        s = stats[prefix_path][table_name]
        total = delta.keys[j].numel()
        assert s.upserted + s.skipped == total

        # Row-wise sharding is a partition: the ranks' shares must sum to the
        # whole delta, with no key landing on two ranks.
        counted = torch.tensor([s.upserted], dtype=torch.int64, device=device)
        dist.all_reduce(counted, group=pg)
        assert int(counted.item()) == total, (int(counted.item()), total)

    # What the replica is for: serving the same batch and returning what the
    # source would. Comparing forward outputs rather than a dump of the replica
    # exercises the path a replica is actually used through -- input dist,
    # lookup, pooling -- instead of re-reading the table the replay just wrote.
    served = {k: v.values() for k, v in serve_model(sparse_feature).items()}
    assert set(served.keys()) == set(trained.keys())
    for feature in trained:
        torch.testing.assert_close(served[feature], trained[feature])
