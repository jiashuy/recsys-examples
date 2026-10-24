# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A training step over the existing key index and over the chained one.

One table, plain SGD, keys drawn uniformly from a universe several times the table, so
every step brings new keys and evicts old ones. Time is per step, prefetch, forward
and backward together, on a table that is full.

Usage: python bench_chained_prefetch.py [--rows 4194304] [--batch 65536] [--dim 64]
"""

import argparse
import gc
import os

import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from fbgemm_gpu.split_embedding_configs import EmbOptimType


def build(layout, rows, dim):
    value_bytes = rows * dim * 4
    caching = layout == "caching"
    options = DynamicEmbTableOptions(
        dim=dim,
        init_capacity=rows,
        max_capacity=rows,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=caching,
        local_hbm_for_values=value_bytes // 4 if caching else 2 * value_bytes,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT, value=1.0
        ),
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_names=["t"],
        table_options=[options],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.1,
        use_index_dedup=True,
    )


def run(layout, chained, rows, batch, dim, steps, warmup):
    if chained:
        os.environ["DEMB_CHAINED_INDEX"] = "1"
    else:
        os.environ.pop("DEMB_CHAINED_INDEX", None)
    module = build(layout, rows, dim)
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(1)
    universe = 4 * rows
    offsets = torch.arange(batch + 1, dtype=torch.int64, device=device)

    def step():
        keys = torch.randint(0, universe, (batch,), generator=g, dtype=torch.int64)
        out = module(keys.to(device), offsets)
        out.backward(torch.ones_like(out))

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        step()
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / steps
    del module
    gc.collect()
    torch.cuda.empty_cache()
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1 << 22)
    parser.add_argument("--batch", type=int, default=1 << 16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=60)
    args = parser.parse_args()
    print(
        f"rows {args.rows}  batch {args.batch}  dim {args.dim}  "
        f"{torch.cuda.get_device_name()}"
    )
    print(f"{'layout':>9} {'existing ms':>12} {'chained ms':>12} {'ratio':>7}")
    for layout in ("hbm", "caching"):
        old = run(layout, False, args.rows, args.batch, args.dim, args.steps, args.warmup)
        new = run(layout, True, args.rows, args.batch, args.dim, args.steps, args.warmup)
        print(f"{layout:>9} {old:>12.3f} {new:>12.3f} {old / new:>7.2f}")


if __name__ == "__main__":
    main()
