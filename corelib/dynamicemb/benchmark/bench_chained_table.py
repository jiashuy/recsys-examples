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

"""The chained bucket table against the existing key index, one table of one device.

Three numbers per batch size, on a table that is full:

  hit      lookup of keys that are all there
  miss     lookup of keys that are all absent
  step     the cost of a step of all-new keys: for the existing index one insert that
           evicts as it goes; for the chained table find_and_stage + commit + the
           release of the victims' rows, each of which is more than one kernel and
           synchronises with the host, so this one measures the python glue too

Usage: python bench_chained_table.py [--slots 16777216] [--batches 65536,524288]
"""

import argparse

import torch
from dynamicemb.storage.chained_table import ChainedTable
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy

BUCKET = 128


def _time(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=1 << 24)
    parser.add_argument("--batches", type=str, default="65536,524288")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda")
    slots = (args.slots // BUCKET) * BUCKET
    nb = slots // BUCKET
    batches = [int(b) for b in args.batches.split(",")]

    # -- fill both with the same keys, in batches the bucket can take ------------
    chained = ChainedTable(num_buckets=nb, bucket_capacity=BUCKET, device=device)
    spec = ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    old = get_scored_table(
        capacity=[slots], bucket_capacity=BUCKET, key_type=torch.int64, score_specs=[spec]
    )
    score_name = old.score_names_[0]
    lru = ScoreArg(name=score_name, value=None, policy=ScorePolicy.GLOBAL_TIMER)
    const = ScoreArg(name=score_name, value=None, policy=ScorePolicy.CONST)

    def old_insert(keys):
        n = keys.numel()
        tids = torch.zeros(n, dtype=torch.int64, device=device)
        results = torch.empty(n, dtype=old.result_type, device=device).fill_(
            InsertResult.INIT.value
        )
        return old.insert(keys, tids, lru, results)

    def old_lookup(keys):
        tids = torch.zeros(keys.numel(), dtype=torch.int64, device=device)
        return old.lookup(keys, tids, const)

    base = 1
    fill_batch = max(1, slots // 8)
    stored = []
    filled = 0
    while filled < int(slots * 0.9):
        keys = torch.arange(base, base + fill_batch, dtype=torch.int64, device=device)
        base += fill_batch
        chained.find_and_stage(keys, torch.full_like(keys, base))
        chained.release_rows(chained.commit().victim_rows)
        old_insert(keys)
        filled += fill_batch
        stored.append(keys)
    stored = torch.cat(stored)
    print(
        f"slots {slots}  buckets {nb}  stored {chained.size()} (chained)  "
        f"device {torch.cuda.get_device_name()}"
    )
    print(f"{'batch':>9} {'':>8} {'existing ms':>12} {'chained ms':>12} {'ratio':>7}")

    for batch in batches:
        hit_keys = stored[torch.randperm(stored.numel(), device=device)[:batch]]
        miss_keys = torch.arange(
            base + (1 << 40), base + (1 << 40) + batch, dtype=torch.int64, device=device
        )
        t_old = _time(lambda: old_lookup(hit_keys), args.iters)
        t_new = _time(lambda: chained.lookup(hit_keys), args.iters)
        print(f"{batch:>9} {'hit':>8} {t_old:>12.3f} {t_new:>12.3f} {t_old / t_new:>7.2f}")
        t_old = _time(lambda: old_lookup(miss_keys), args.iters)
        t_new = _time(lambda: chained.lookup(miss_keys), args.iters)
        print(f"{batch:>9} {'miss':>8} {t_old:>12.3f} {t_new:>12.3f} {t_old / t_new:>7.2f}")

    # The steps evict, so they come after every lookup of keys that are meant to be there.
    for batch in batches:
        counter = [base + (1 << 41)]

        def fresh():
            k = torch.arange(
                counter[0], counter[0] + batch, dtype=torch.int64, device=device
            )
            counter[0] += batch
            return k

        def old_step():
            old_insert(fresh())

        def new_step():
            k = fresh()
            chained.find_and_stage(k, torch.full_like(k, counter[0]))
            chained.release_rows(chained.commit().victim_rows)

        t_old = _time(old_step, args.iters)
        t_new = _time(new_step, args.iters)
        print(f"{batch:>9} {'step':>8} {t_old:>12.3f} {t_new:>12.3f} {t_old / t_new:>7.2f}")


if __name__ == "__main__":
    main()
