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


"""Worker: LFU score correctness through a sharded dynamicemb model.

Launched by ``integration/test_lfu_scores.py`` under ``torchrun``. Trains a model
on features whose per-key access counts are known, exports the scores of the
tables and checks each key's score equals its true access count.
"""

import os

import click
import torch
import torch.distributed as dist
from common.dynamicemb_model import (
    assert_batched_dynamicemb_storage_class,
    create_model,
    get_optimizer_kwargs,
)
from common.frequency_features import (
    generate_deterministic_sparse_features_with_frequency_tracking,
)
from common.lfu_checks import export_table_scores, validate_lfu_scores
from dynamicemb import DynamicEmbScoreStrategy
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module


@click.command()
@click.option("--num-embedding-collections", type=int, default=1)
@click.option("--num-embeddings", type=str, default="1000")
@click.option("--multi-hot-sizes", type=str, default="3")
@click.option("--embedding-dim", type=int, default=16)
@click.option(
    "--optimizer-type",
    type=click.Choice(["sgd", "adam", "adagrad", "rowwise_adagrad"]),
    default="sgd",
)
@click.option("--batch-size", type=int, default=16)
@click.option("--num-iterations", type=int, default=3)
@click.option("--tolerance", type=float, default=0.0)
@click.option("--caching", is_flag=True, help="Enable cache + storage architecture")
@click.option(
    "--cache-capacity-ratio",
    type=float,
    default=0.5,
    help="Cache capacity as ratio of storage capacity (only used when --caching is enabled)",
)
@click.option(
    "--use-index-dedup/--no-use-index-dedup",
    default=True,
    help="Enable/disable index deduplication",
)
@click.option(
    "--global-hbm-budget-scale",
    type=float,
    default=1.0,
    help=(
        "Scale DynamicEmb HBM byte budget vs full table. Values < 1.0 force "
        "HybridStorage and prefetch StorageMode DEFAULT (_generic_forward_path)."
    ),
)
@click.option(
    "--lru-lfu",
    is_flag=True,
    help=(
        "Use the compound (TIMESTAMP, LFU) score strategy (the LruLfu policy): LFU "
        "tables also carry a timestamp column (physical word 0 = timestamp, word 1 = "
        "frequency). The frequency (word 1) must still equal the true access count -- "
        "this reuses the same validation to prove the extra timestamp word does not "
        "corrupt LFU semantics. Not supported with caching/HybridStorage (multi-word "
        "checkpoints are a follow-up)."
    ),
)
def main(
    num_embedding_collections: int,
    num_embeddings: str,
    multi_hot_sizes: str,
    embedding_dim: int,
    optimizer_type: str,
    batch_size: int,
    num_iterations: int,
    tolerance: float,
    caching: bool,
    cache_capacity_ratio: float,
    use_index_dedup: bool,
    global_hbm_budget_scale: float,
    lru_lfu: bool,
):
    """Test LFU score correctness by comparing with naive frequency counting.

    This test supports:
    - Storage-only (default): HBM-only storage when budget fits (HBM_DIRECT path)
    - StorageMode DEFAULT: --global-hbm-budget-scale < 1 → HybridStorage
    - Cache + Storage (--caching): LFU through cache to storage
    """

    num_embeddings = [int(v) for v in num_embeddings.split(",")]
    multi_hot_sizes = [int(v) for v in multi_hot_sizes.split(",")]

    if lru_lfu and (caching or global_hbm_budget_scale < 1.0):
        raise ValueError(
            "--lru-lfu is only supported for storage-only (HBM_DIRECT) LFU; "
            "caching / HybridStorage multi-word score layouts are a follow-up."
        )

    if not caching:
        for num_embedding, multi_hot_size in zip(num_embeddings, multi_hot_sizes):
            if batch_size * num_iterations * multi_hot_size > num_embedding:
                raise ValueError(
                    "batch_size * num_iterations * multi_hot_size > num_embedding, "
                    "this may lead to eviction of dynamicemb and cause test fail"
                )

    print(f"Configuration:")
    print(f"  - Embedding collections: {num_embedding_collections}")
    print(f"  - Num embeddings: {num_embeddings}")
    print(f"  - Multi-hot sizes: {multi_hot_sizes}")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Tolerance: {tolerance}")
    print(f"  - Use index dedup: {use_index_dedup}")
    if caching:
        print(f"  - Caching: ENABLED ✓")
        print(f"  - Cache capacity ratio: {cache_capacity_ratio}")
    else:
        print(f"  - Caching: DISABLED")
    print(f"  - Global HBM budget scale: {global_hbm_budget_scale}")
    print(f"  - compound (TIMESTAMP, LFU) score: {lru_lfu} (LruLfu 2-word score)")
    if not caching and global_hbm_budget_scale < 1.0:
        print(f"  - Storage path: expect HybridStorage (StorageMode DEFAULT)")

    # Create model. When --lru-lfu is set, use the compound (TIMESTAMP, LFU) score
    # strategy so LFU tables also carry a timestamp column; eviction is still driven
    # by the LFU frequency, so the same validation applies.
    optimizer_kwargs = get_optimizer_kwargs(optimizer_type)
    score_strategy = (
        (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)
        if lru_lfu
        else DynamicEmbScoreStrategy.LFU
    )
    model = create_model(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        optimizer_kwargs=optimizer_kwargs,
        score_strategy=score_strategy,
        use_index_dedup=use_index_dedup,
        caching=caching,
        cache_capacity_ratio=cache_capacity_ratio if caching else 0.1,
        global_hbm_budget_scale=global_hbm_budget_scale,
    )
    assert_batched_dynamicemb_storage_class(
        model,
        caching=caching,
        global_hbm_budget_scale=global_hbm_budget_scale,
    )
    if dist.get_rank() == 0:
        for _, _, sm in find_sharded_modules(model, ""):
            for emb in get_dynamic_emb_module(sm):
                print(f"  - Storage class: {type(emb.tables).__name__}")

    # Generate features with frequency tracking
    (
        kjts,
        expected_frequencies,
    ) = generate_deterministic_sparse_features_with_frequency_tracking(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        multi_hot_sizes=multi_hot_sizes,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=batch_size,
        num_iterations=num_iterations,
        caching=caching,
    )

    # Run forward passes to populate frequency information
    if caching:
        print(f"\nRunning {num_iterations} iterations with cache enabled...")
    else:
        print(f"\nRunning {num_iterations} iterations...")

    for iteration, kjt in enumerate(kjts):
        ret = model(kjt)
        torch.cuda.synchronize()
        loss = ret.sum() * dist.get_world_size()
        loss.backward()
        torch.cuda.synchronize()

    # Extract actual scores
    actual_scores = export_table_scores(model, optim=True)

    torch.cuda.synchronize()

    # Validate scores
    validate_lfu_scores(expected_frequencies, actual_scores, tolerance)


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")
    main()
    dist.barrier()
    dist.destroy_process_group()
