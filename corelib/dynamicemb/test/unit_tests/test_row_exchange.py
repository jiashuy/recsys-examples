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

"""RowExchange: swapping the rows of a cache with the rows of its storage."""

import pytest
import torch
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.row_exchange import RowExchange
from dynamicemb.types import CopyMode


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _pair(device, dims, dtype, storage_kind, cache_rows=24, storage_rows=64):
    cache = MixedDimBuffers(dims, dims, [cache_rows] * len(dims), dtype, device, "fixed")
    storage = MixedDimBuffers(dims, dims, [storage_rows] * len(dims), dtype, device, storage_kind)
    return cache, storage


def _fill(buffers, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(buffers.num_tables):
        buf = buffers.tensor(t)
        data = torch.randn(buf.shape, generator=g).to(buf.dtype)
        buf.copy_(data.to(buf.device))


def _snapshot(buffers):
    return [buffers.tensor(t).clone() for t in range(buffers.num_tables)]


@pytest.mark.parametrize("storage_kind", ["device", "host"])
@pytest.mark.parametrize(
    "dims, dtype",
    [
        ([8, 16, 12], torch.float32),  # 16-byte words
        ([6, 10], torch.float32),  # 4-byte words
        ([7, 9], torch.float16),  # 2-byte words
        ([8, 16], torch.bfloat16),
    ],
)
def test_a_swap_writes_back_the_old_row_and_brings_in_the_new_one(
    device, dims, dtype, storage_kind
):
    cache, storage = _pair(device, dims, dtype, storage_kind)
    _fill(cache, device, 1)
    _fill(storage, device, 2)
    before_cache, before_store = _snapshot(cache), _snapshot(storage)

    ex = RowExchange(cache, storage)
    rng = torch.Generator(device="cpu").manual_seed(3)
    n = 18
    tids = torch.randint(0, len(dims), (n,), generator=rng)
    nodes = torch.zeros(n, dtype=torch.int64)
    evict = torch.full((n,), -1, dtype=torch.int64)
    fetch = torch.full((n,), -1, dtype=torch.int64)
    for t in range(len(dims)):
        sel = (tids == t).nonzero().squeeze(1)
        k = sel.numel()
        nodes[sel] = torch.randperm(24, generator=rng)[:k]
        # storage rows are distinct across all the rows this table writes and reads
        rows = torch.randperm(64, generator=rng)[: 2 * k]
        evict[sel] = rows[:k]
        fetch[sel] = rows[k:]
    # every combination: both, only write-back, only fetch, neither, no node
    evict[0::4] = -1
    fetch[1::5] = -1
    nodes[2] = -1

    ex.swap(nodes.to(device), tids.to(device), evict.to(device), fetch.to(device))
    torch.cuda.synchronize()

    for i in range(n):
        t, node, ev, fe = int(tids[i]), int(nodes[i]), int(evict[i]), int(fetch[i])
        if node < 0:
            continue
        if ev >= 0:
            assert torch.equal(storage.tensor(t)[ev].cpu(), before_cache[t][node].cpu())
        if fe >= 0:
            assert torch.equal(cache.tensor(t)[node].cpu(), before_store[t][fe].cpu())
        elif ev >= 0 and fe < 0:
            # written back, nothing came in: the node still holds its row
            assert torch.equal(cache.tensor(t)[node].cpu(), before_cache[t][node].cpu())

    # Rows that no swap names are untouched.
    for t in range(len(dims)):
        named_nodes = {int(nodes[i]) for i in range(n) if int(tids[i]) == t and int(nodes[i]) >= 0
                       and (int(fetch[i]) >= 0)}
        named_rows = {int(evict[i]) for i in range(n) if int(tids[i]) == t and int(nodes[i]) >= 0
                      and int(evict[i]) >= 0}
        for node in range(24):
            if node not in named_nodes:
                assert torch.equal(cache.tensor(t)[node].cpu(), before_cache[t][node].cpu())
        for row in range(64):
            if row not in named_rows:
                assert torch.equal(storage.tensor(t)[row].cpu(), before_store[t][row].cpu())


def test_the_old_row_is_read_before_the_new_one_is_written(device):
    """A node swapping with the row that comes in must write back what it held, not what
    came in: the case a load-all-then-store-all in the wrong order gets wrong."""
    cache, storage = _pair(device, [8], torch.float32, "device")
    cache.tensor(0).fill_(1.0)
    storage.tensor(0).fill_(2.0)
    ex = RowExchange(cache, storage)
    z = torch.zeros(24, dtype=torch.int64, device=device)
    nodes = torch.arange(24, dtype=torch.int64, device=device)
    ex.swap(nodes, z, torch.arange(24, dtype=torch.int64, device=device),
            torch.arange(24, 48, dtype=torch.int64, device=device))
    assert cache.tensor(0).eq(2.0).all()
    assert storage.tensor(0)[:24].eq(1.0).all()


def test_the_storage_may_have_grown_since_the_exchange_was_made(device):
    cache, storage = _pair(device, [8, 16], torch.float32, "device", storage_rows=32)
    ex = RowExchange(cache, storage)
    storage.ensure_rows(1, 500)
    storage.tensor(1)[400].fill_(7.0)
    ex.swap(torch.tensor([3], dtype=torch.int64, device=device),
            torch.tensor([1], dtype=torch.int64, device=device),
            torch.tensor([-1], dtype=torch.int64, device=device),
            torch.tensor([400], dtype=torch.int64, device=device))
    assert cache.tensor(1)[3].eq(7.0).all()


def test_different_row_widths_are_refused(device):
    a = MixedDimBuffers([8], [8], [4], torch.float32, device, "fixed")
    b = MixedDimBuffers([8], [12], [4], torch.float32, device, "device")
    with pytest.raises(ValueError):
        RowExchange(a, b)
    c = MixedDimBuffers([8], [8], [4], torch.float16, device, "device")
    with pytest.raises(ValueError):
        RowExchange(a, c)


def test_through_the_loads_a_swap_agrees_with_load_and_store(device):
    """What the prefetch did in four steps, and what a swap does in one."""
    dims = [8, 12]
    cache, storage = _pair(device, dims, torch.float32, "device")
    cache2, storage2 = _pair(device, dims, torch.float32, "device")
    for b, s in ((cache, 1), (cache2, 1), (storage, 2), (storage2, 2)):
        _fill(b, device, s)
    nodes = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    tids = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
    evict = torch.tensor([10, 11, 12, 13], dtype=torch.int64, device=device)
    fetch = torch.tensor([20, 21, 22, 23], dtype=torch.int64, device=device)

    old = cache.load(nodes, tids, CopyMode.VALUE)
    storage.store(evict, tids, old)
    new = storage.load(fetch, tids, CopyMode.VALUE)
    cache.store(nodes, tids, new)

    RowExchange(cache2, storage2).swap(nodes, tids, evict, fetch)
    for t in range(2):
        assert torch.equal(cache.tensor(t), cache2.tensor(t))
        assert torch.equal(storage.tensor(t), storage2.tensor(t))


def test_many_steps_match_a_model(device):
    """Back-to-back swaps with rows that overlap from one step to the next, and two tables
    of different widths. A model in plain Python says what the cache and the storage hold at
    the end."""
    dims = [8, 12]
    n_tables, cache_rows, store_rows, steps, per_step = len(dims), 200, 400, 40, 120
    cache, storage = _pair(device, dims, torch.float32, "host", cache_rows, store_rows)
    ex = RowExchange(cache, storage)
    # a key's value is its identity, so a row that moves can be recognised wherever it lands
    store_model = [torch.arange(store_rows, dtype=torch.float32) * 1000 + t for t in range(n_tables)]
    cache_model = [torch.arange(cache_rows, dtype=torch.float32) * 1000 + 500 + t for t in range(n_tables)]
    for t in range(n_tables):
        storage.tensor(t).copy_(store_model[t].to(device)[:, None].expand(store_rows, dims[t]))
        cache.tensor(t).copy_(cache_model[t].to(device)[:, None].expand(cache_rows, dims[t]))
    rng = torch.Generator(device="cpu").manual_seed(11)
    for _ in range(steps):
        tids = torch.sort(torch.randint(0, n_tables, (per_step,), generator=rng)).values
        nodes = torch.zeros(per_step, dtype=torch.int64)
        evict = torch.zeros(per_step, dtype=torch.int64)
        fetch = torch.zeros(per_step, dtype=torch.int64)
        for t in range(n_tables):
            sel = (tids == t).nonzero().squeeze(1)
            k = sel.numel()
            nodes[sel] = torch.randperm(cache_rows, generator=rng)[:k]
            rows = torch.randperm(store_rows, generator=rng)[: 2 * k]
            evict[sel], fetch[sel] = rows[:k], rows[k:]
        evict[0::7] = -1
        fetch[3::5] = -1
        for i in range(per_step):
            t, nd, ev, fe = int(tids[i]), int(nodes[i]), int(evict[i]), int(fetch[i])
            old, new = cache_model[t][nd].clone(), store_model[t][fe].clone()
            if ev >= 0:
                store_model[t][ev] = old
            if fe >= 0:
                cache_model[t][nd] = new
        ex.swap(*(x.to(device) for x in (nodes, tids, evict, fetch)))
    torch.cuda.synchronize()
    for t in range(n_tables):
        assert torch.equal(storage.tensor(t)[:, 0].cpu(), store_model[t]), f"storage of table {t}"
        assert torch.equal(cache.tensor(t)[:, 0].cpu(), cache_model[t]), f"cache of table {t}"
