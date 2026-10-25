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

"""RowExchange.swap on a cache in HBM and a storage in host memory.

Random rows of a big host buffer, as a cache miss step has them, with rows of their own in
every iteration. Reports the time of a swap and the bandwidth it achieved: the bytes that
went out plus the bytes that came in, over the time. For reference, one way of the link
carries about 50 GB/s and both ways at once about 100 GB/s.

Usage: python bench_row_exchange.py [--storage-rows 26843545] [--swaps 100000,200000]
"""

import argparse

import torch
from dynamicemb.storage.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.storage.row_exchange import RowExchange


def _time(fn, iters=20, warmup=3):
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters):
        fn(warmup + i)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-rows", type=int, default=26843545)
    parser.add_argument("--cache-rows", type=int, default=2097152)
    parser.add_argument("--swaps", type=str, default="100000,200000")
    parser.add_argument("--dims", type=str, default="128,384", help="value dims: sgd, adam")
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"{torch.cuda.get_device_name()}  storage rows {args.storage_rows}")
    print(f"{'value_dim':>9} {'rows':>8} {'swap ms':>9} {'GB/s both ways':>15}")
    for vd in [int(x) for x in args.dims.split(",")]:
        cache = MixedDimBuffers([vd], [vd], [args.cache_rows], torch.float32, device, "fixed")
        storage = MixedDimBuffers([vd], [vd], [args.storage_rows], torch.float32, device, "host")
        ex = RowExchange(cache, storage)
        for n in [int(x) for x in args.swaps.split(",")]:
            sets = []
            for _ in range(8):
                rows = torch.randperm(args.storage_rows, device=device)[: 2 * n]
                nodes = torch.randperm(args.cache_rows, device=device)[:n]
                sets.append((nodes, rows[:n].contiguous(), rows[n:].contiguous()))
            tids = torch.zeros(n, dtype=torch.int64, device=device)
            t = _time(lambda i: ex.swap(sets[i % 8][0], tids, sets[i % 8][1], sets[i % 8][2]))
            gb = 2 * n * vd * 4 / 1e9
            print(f"{vd:>9} {n:>8} {t:>9.3f} {gb / (t / 1e3):>15.1f}")
        del cache, storage


if __name__ == "__main__":
    main()
