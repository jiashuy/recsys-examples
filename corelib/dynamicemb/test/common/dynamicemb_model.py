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

"""Models and checks shared by the DynamicEmb end-to-end (torchrun) workers.

Builds a TorchRec EmbeddingCollection model sharded by dynamicemb with one
call (:func:`create_model`), and holds the assertions about the result that
the dump/load, LFU-score and admission workers all make: which storage class
backs the tables, that the dist_type reached the sharding and input-dist,
which rank owns a key, that the module finder sees every table.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamicemb.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    ScoreStrategy,
    get_table_value_bytes,
)
from dynamicemb.key_ownership import murmur3_fmix64
from dynamicemb.key_value_table import DynamicEmbStorage, HybridStorage
from dynamicemb.planner import get_planner
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb.shard.input_dist import dist_type_codes
from dynamicemb.types import MAX_BUCKET_CAPACITY, AdmissionStrategy
from dynamicemb.utils import TORCHREC_TYPES
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec import DataType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Constant value every table in these tests initializes its embeddings with.
# Tests that assert on embedding values compare against it.
TABLE_INITIALIZER_VALUE = 1e-1

# Per-rank room for keys an admission strategy is still counting. They are
# erased on admission, so this only has to hold the ones still waiting.
ADMISSION_COUNTER_CAPACITY = 1024 * 1024


def idx_to_name(embedding_collection_idx: int, embedding_idx: int) -> Tuple[str, str]:
    return (
        f"feature_{embedding_collection_idx}_{embedding_idx}",
        f"embedding_collection_{embedding_collection_idx}_{embedding_idx}",
    )


def get_optimizer_kwargs(optimizer_type: str) -> Dict[str, Any]:
    if optimizer_type == "sgd":
        return {"optimizer": EmbOptimType.SGD, "lr": 1e-3}
    elif optimizer_type == "adam":
        return {"optimizer": EmbOptimType.ADAM, "lr": 1e-3}
    elif optimizer_type == "adagrad":
        return {"optimizer": EmbOptimType.EXACT_ADAGRAD, "lr": 1e-3}
    elif optimizer_type == "rowwise_adagrad":
        return {"optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD, "lr": 1e-3}
    else:
        raise ValueError("unknown optimizer type")


def get_score_strategy(score_strategy_str: str) -> DynamicEmbScoreStrategy:
    if score_strategy_str == "timestamp":
        return DynamicEmbScoreStrategy.TIMESTAMP
    elif score_strategy_str == "step":
        return DynamicEmbScoreStrategy.STEP
    elif score_strategy_str == "lfu":
        return DynamicEmbScoreStrategy.LFU
    else:
        raise ValueError(f"Invalid score strategy: {score_strategy_str}")


def assert_batched_dynamicemb_storage_class(
    model,
    *,
    caching: bool,
    global_hbm_budget_scale: float = 1.0,
) -> None:
    """Check ``BatchedDynamicEmbeddingTablesV2`` backing storage type.

    - Must be :class:`DynamicEmbStorage` or :class:`HybridStorage`.
    - With ``--caching``: expect ``DynamicEmbStorage`` (backing store under cache).
    - No cache and ``global_hbm_budget_scale < 1``: expect ``HybridStorage``
      (StorageMode DEFAULT / two-tier).
    - No cache and full budget: expect single-tier ``DynamicEmbStorage``.
    """
    for _, _, sharded_module in find_sharded_modules(model, ""):
        for emb in get_dynamic_emb_module(sharded_module):
            storage = emb.tables
            assert isinstance(storage, (DynamicEmbStorage, HybridStorage)), (
                "BatchedDynamicEmbedding storage must be DynamicEmbStorage or "
                f"HybridStorage, got {type(storage)}"
            )
            if caching:
                assert isinstance(
                    storage, DynamicEmbStorage
                ), f"With --caching, expected DynamicEmbStorage, got {type(storage)}"
            elif global_hbm_budget_scale < 1.0:
                assert isinstance(storage, HybridStorage), (
                    f"With global_hbm_budget_scale={global_hbm_budget_scale} (no cache), "
                    f"expected HybridStorage, got {type(storage)}"
                )
            else:
                assert isinstance(storage, DynamicEmbStorage), (
                    "Full HBM budget without cache: expected DynamicEmbStorage, "
                    f"got {type(storage)}"
                )


def assert_get_dynamic_emb_module_finds_submodules(model) -> None:
    """Verify get_dynamic_emb_module discovers BatchedDynamicEmbeddingTablesV2.

    Tests two paths:
      1. Via find_sharded_modules + get_dynamic_emb_module (existing usage)
      2. Via get_dynamic_emb_module directly on the DMP model (requires
         children() traversal through wrapper modules, the fix for #353)

    Both must return the same set of modules.
    """
    # Path 1: existing approach - find sharded modules first, then search each
    via_sharded = []
    for _, _, sharded_module in find_sharded_modules(model, ""):
        via_sharded.extend(get_dynamic_emb_module(sharded_module))

    # Path 2: search directly on the DMP wrapper (requires children() traversal)
    via_dmp = get_dynamic_emb_module(model)

    assert (
        len(via_sharded) > 0
    ), "find_sharded_modules + get_dynamic_emb_module found no modules"
    assert (
        len(via_dmp) > 0
    ), "get_dynamic_emb_module on DMP model found no modules (children() traversal broken)"

    # Every module found via either path must be BatchedDynamicEmbeddingTablesV2
    for m in via_sharded:
        assert isinstance(
            m, BatchedDynamicEmbeddingTablesV2
        ), f"Expected BatchedDynamicEmbeddingTablesV2, got {type(m)}"
    for m in via_dmp:
        assert isinstance(
            m, BatchedDynamicEmbeddingTablesV2
        ), f"Expected BatchedDynamicEmbeddingTablesV2, got {type(m)}"

    # Both paths must discover the exact same set of modules (by identity)
    ids_sharded = set(id(m) for m in via_sharded)
    ids_dmp = set(id(m) for m in via_dmp)
    assert ids_sharded == ids_dmp, (
        f"Module sets differ: via_sharded has {len(ids_sharded)} modules, "
        f"via_dmp has {len(ids_dmp)} modules"
    )

    # No duplicates in either result
    assert len(via_sharded) == len(ids_sharded), "Duplicates in via_sharded path"
    assert len(via_dmp) == len(ids_dmp), "Duplicates in via_dmp path"

class CollectionsModel(nn.Module):
    def __init__(
        self,
        embedding_modules: List[EmbeddingCollection],
    ):
        super().__init__()
        self.embedding_modules = nn.ModuleList(embedding_modules)

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings_dict = [
            embedding_module(kjt).wait() for embedding_module in self.embedding_modules
        ]
        embeddings = []
        for embedding_dict in embeddings_dict:
            for embedding in embedding_dict.values():
                embeddings.append(embedding.values())
        return torch.cat(embeddings, dim=0)


def apply_dmp(
    model: torch.nn.Module,
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
    score_strategy: ScoreStrategy = DynamicEmbScoreStrategy.LFU,
    dist_type: str = "roundrobin",
    use_index_dedup: bool = False,
    caching: bool = False,
    cache_capacity_ratio: float = 0.5,
    admit_strategy: AdmissionStrategy = None,
    global_hbm_budget_scale: float = 1.0,
):
    eb_configs: List[EmbeddingConfig] = []
    for _, m in model.named_modules():
        if type(m) in TORCHREC_TYPES:
            eb_configs.extend(m.embedding_configs())

    world_size = dist.get_world_size()
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions] = {}
    for eb_config in eb_configs:
        emb_opt_type = (
            optimizer_kwargs.get("optimizer") if optimizer_kwargs else None
        ) or EmbOptimType.SGD

        value_bytes = get_table_value_bytes(
            eb_config,
            emb_opt_type,
            world_size,
            MAX_BUCKET_CAPACITY,
        )
        if caching:
            global_hbm = int(value_bytes * cache_capacity_ratio)
        else:
            global_hbm = int(value_bytes * global_hbm_budget_scale)

        dynamicemb_options_dict[eb_config.name] = DynamicEmbTableOptions(
            global_hbm_for_values=global_hbm,
            score_strategy=score_strategy,
            dist_type=dist_type,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT,
                value=TABLE_INITIALIZER_VALUE,
            ),
            bucket_capacity=MAX_BUCKET_CAPACITY,  # keep same to the bucket capacity from get_table_value_bytes
            caching=caching,
            admit_strategy=admit_strategy,
        )
    planner = get_planner(
        eb_configs,
        {},
        dynamicemb_options_dict,
        device,
    )

    fused_params = {}
    fused_params["output_dtype"] = SparseType.FP32
    fused_params.update(optimizer_kwargs)

    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=use_index_dedup,
    )
    plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)

    # Same usage of TorchREC
    dmp = DistributedModelParallel(
        module=model,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )
    return dmp


def create_model(
    num_embedding_collections: int,
    num_embeddings: List[int],
    embedding_dim: int,
    optimizer_kwargs: Dict[str, Any],
    score_strategy: ScoreStrategy = DynamicEmbScoreStrategy.LFU,
    dist_type: str = "roundrobin",
    use_index_dedup: bool = False,
    caching: bool = False,
    cache_capacity_ratio: float = 0.5,
    admit_strategy: AdmissionStrategy = None,
    global_hbm_budget_scale: float = 1.0,
):
    ebc_list = []
    for embedding_collection_id in range(num_embedding_collections):
        eb_configs = []
        for embedding_id, num_embedding in enumerate(num_embeddings):
            feature_name, embedding_name = idx_to_name(
                embedding_collection_id, embedding_id
            )
            eb_configs.append(
                EmbeddingConfig(
                    name=embedding_name,
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embedding,
                    feature_names=[feature_name],
                    data_type=DataType.FP32,
                )
            )
        ebc_list.append(
            EmbeddingCollection(
                device=torch.device("meta"),
                tables=eb_configs,
            )
        )
    model = CollectionsModel(
        embedding_modules=ebc_list,
    )

    model = apply_dmp(
        model,
        optimizer_kwargs,
        torch.device(f"cuda:{torch.cuda.current_device()}"),
        score_strategy=score_strategy,
        dist_type=dist_type,
        use_index_dedup=use_index_dedup,
        caching=caching,
        cache_capacity_ratio=cache_capacity_ratio,
        admit_strategy=admit_strategy,
        global_hbm_budget_scale=global_hbm_budget_scale,
    )
    return model


def check_counter_table_checkpoint(x, y):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tables_x = get_dynamic_emb_module(x)
    tables_y = get_dynamic_emb_module(y)
    assert len(tables_x) == len(tables_y)

    for table_x, table_y in zip(tables_x, tables_y):
        cnt_x = table_x._admission_counter
        cnt_y = table_y._admission_counter
        if cnt_x is None:
            assert cnt_y is None
            continue
        assert cnt_x.table_.size() == cnt_y.table_.size()

        freq_name = cnt_x.score_name_
        for table_id in range(len(table_x._table_names)):
            for keys, named_scores, _ in cnt_x.table_._batched_export_keys_scores(
                [freq_name], device, table_id
            ):
                if keys.numel() == 0:
                    continue
                frequencies = named_scores[freq_name]

                lookup_table_ids = torch.full(
                    (keys.numel(),), table_id, dtype=torch.int64, device=device
                )
                score_arg_lookup = ScoreArg(
                    name=freq_name,
                    value=torch.zeros_like(frequencies),
                    policy=ScorePolicy.CONST,
                )
                score_out, founds, _ = cnt_y.table_.lookup(
                    keys, lookup_table_ids, score_arg_lookup
                )
                assert founds.all(), (
                    f"counter keys missing from loaded table_id={table_id}: "
                    f"{keys[~founds].tolist()}"
                )
                assert torch.equal(
                    frequencies, score_out
                ), f"counter frequency mismatch for table_id={table_id}"


def owns_key(key: int, rank: int, world_size: int, dist_type: str) -> bool:
    """Whether *rank* is the one holding *key*, under the rule the model uses.

    The question `dynamicemb.key_ownership` answers for tensors, asked one
    scalar at a time because the checks below walk the input KJTs in Python.

    This used to be spelled `key % world_size == rank` inline, which is the
    roundrobin rule written out by hand -- right for the default, wrong for
    hash_roundrobin, and invisible either way at world_size 1, where the only
    rank owns everything. The hash_roundrobin coverage ran at one rank.
    """
    if world_size <= 1:
        return True
    if dist_type == "hash_roundrobin":
        return int(murmur3_fmix64(key)) % world_size == rank
    return (int(key) & 0xFFFFFFFFFFFFFFFF) % world_size == rank


def assert_dist_type_path(model: nn.Module, expected_dist_type: str) -> None:
    seen_sharding = False
    seen_input_dist = False

    for _, _, sharded_module in find_sharded_modules(model):
        for sharding in sharded_module._sharding_type_to_sharding.values():
            if hasattr(sharding, "_dist_type_per_feature"):
                assert set(sharding._dist_type_per_feature.values()) == {
                    expected_dist_type
                }
                seen_sharding = True

        for input_dist in getattr(sharded_module, "_input_dists", []):
            codes = getattr(input_dist, "_dist_type_tensor", None)
            if codes is not None:
                # The distributor holds the rule as the kernel's int32 codes,
                # resolved once at construction; go through the same function
                # rather than restating the mapping here.
                expected_code = int(dist_type_codes([expected_dist_type])[0])
                assert set(codes.tolist()) == {expected_code}
                seen_input_dist = True

    assert seen_sharding, "Did not find any DynamicEmb sharding carrying dist_type."
    assert seen_input_dist, "Did not find any input_dist carrying dist_type."
