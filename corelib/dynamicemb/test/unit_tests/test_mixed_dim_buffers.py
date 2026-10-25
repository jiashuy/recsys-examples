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

"""MixedDimBuffers: the value rows of tables of different dimensions, on their own."""

import pytest
import torch
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.optimizer import (
    AdamDynamicEmbeddingOptimizer,
    OptimizerArgs,
    SGDDynamicEmbeddingOptimizer,
)
from dynamicemb.types import CopyMode

DIMS = [8, 16, 12]


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _buffers(device, optimizer, kind="device", rows=(32, 48, 40)):
    value_dims = [d + optimizer.get_state_dim(d) for d in DIMS]
    return MixedDimBuffers(DIMS, value_dims, rows, torch.float32, device, kind)


def _pairs(rows_per_table, device):
    rows = torch.cat(
        [torch.arange(n, dtype=torch.int64, device=device) for n in rows_per_table]
    )
    tids = torch.cat(
        [
            torch.full((n,), t, dtype=torch.int64, device=device)
            for t, n in enumerate(rows_per_table)
        ]
    )
    return rows, tids


@pytest.mark.parametrize("kind", ["device", "host", "fixed"])
def test_each_table_has_a_buffer_of_its_own_width(device, kind):
    opt = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    buf = _buffers(device, opt, kind)
    assert [tuple(buf.tensor(t).shape) for t in range(3)] == [(32, 8), (48, 16), (40, 12)]
    assert (buf.max_emb_dim, buf.max_value_dim) == (16, 16)
    assert buf.all_dims_vec4
    assert buf.on_device == (kind != "host")


def test_dims_that_are_not_multiples_of_four_turn_vec4_off(device):
    buf = MixedDimBuffers([8, 7], [8, 7], [4, 4], torch.float32, device, "device")
    assert not buf.all_dims_vec4


def test_store_then_load_round_trips_each_tables_own_width(device):
    opt = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    buf = _buffers(device, opt)
    rows, tids = _pairs([10, 20, 15], device)
    values = torch.randn(rows.numel(), buf.max_value_dim, device=device)
    buf.store(rows, tids, values)

    got = buf.load(rows, tids, CopyMode.VALUE)
    for t, d in enumerate(DIMS):
        sel = tids == t
        torch.testing.assert_close(got[sel][:, :d], values[sel][:, :d])
    emb = buf.load(rows, tids, CopyMode.EMBEDDING)
    assert emb.shape == (rows.numel(), buf.max_emb_dim)
    torch.testing.assert_close(emb[tids == 1], values[tids == 1][:, :16])


def test_one_table_by_its_own_compact_rows(device):
    opt = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    buf = _buffers(device, opt)
    rows = torch.tensor([3, 1, 7], dtype=torch.int64, device=device)
    values = torch.randn(3, 12, device=device)
    others = [buf.tensor(0).clone(), buf.tensor(1).clone()]
    buf.store_table(rows, 2, values)
    torch.testing.assert_close(buf.load_table(rows, 2), values)
    # The other tables were not touched.
    assert torch.equal(buf.tensor(0), others[0]) and torch.equal(buf.tensor(1), others[1])
    # Nothing to do is not an error.
    empty = torch.empty(0, dtype=torch.int64, device=device)
    buf.store_table(empty, 2, torch.empty(0, 12, device=device))
    assert buf.load_table(empty, 2).shape == (0, 12)


def test_growing_a_table_keeps_its_rows_and_refreshes_the_pointer(device):
    opt = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    buf = _buffers(device, opt)
    rows, tids = _pairs([32, 48, 40], device)
    values = torch.randn(rows.numel(), buf.max_value_dim, device=device)
    buf.store(rows, tids, values)

    assert not buf.ensure_rows(1, 48)  # already that big
    assert buf.ensure_rows(1, 500)
    assert buf.rows(1) >= 500 and buf.rows(0) == 32
    # The kernels address the rows through the refreshed pointers.
    got = buf.load(rows, tids, CopyMode.VALUE)
    for t, d in enumerate(DIMS):
        sel = tids == t
        torch.testing.assert_close(got[sel][:, :d], values[sel][:, :d])
    more = torch.tensor([400], dtype=torch.int64, device=device)
    buf.store(more, torch.tensor([1], dtype=torch.int64, device=device), torch.ones(1, 16, device=device))
    assert buf.tensor(1)[400].eq(1).all()


def test_a_fixed_pool_does_not_grow(device):
    opt = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    buf = _buffers(device, opt, "fixed")
    with pytest.raises(NotImplementedError):
        buf.ensure_rows(0, 64)


def test_update_rows_applies_the_optimizer_in_place(device):
    args = OptimizerArgs(learning_rate=0.5)
    opt = SGDDynamicEmbeddingOptimizer(args)
    buf = _buffers(device, opt)
    rows, tids = _pairs([4, 4, 4], device)
    start = torch.randn(rows.numel(), buf.max_value_dim, device=device)
    buf.store(rows, tids, start)
    grads = torch.ones(rows.numel(), buf.max_emb_dim, device=device)
    buf.update_rows(opt, grads, rows, tids)
    got = buf.load(rows, tids, CopyMode.VALUE)
    for t, d in enumerate(DIMS):
        sel = tids == t
        torch.testing.assert_close(got[sel][:, :d], start[sel][:, :d] - 0.5)


def test_the_padded_path_and_the_stored_path_agree(device):
    opt = AdamDynamicEmbeddingOptimizer(OptimizerArgs(learning_rate=0.1))
    buf = _buffers(device, opt)
    assert buf.max_value_dim > buf.max_emb_dim
    rows, tids = _pairs([5, 5, 5], device)
    init = torch.zeros(rows.numel(), buf.max_value_dim, device=device)
    init[:, : buf.max_emb_dim] = torch.randn(rows.numel(), buf.max_emb_dim, device=device)
    padded = init.clone()
    buf.reset_optimizer_states(opt, padded, tids)
    buf.store(rows, tids, padded)

    grads = torch.randn(rows.numel(), buf.max_emb_dim, device=device)
    opt.step()
    buf.update_rows(opt, grads, rows, tids)
    stored = buf.load(rows, tids, CopyMode.VALUE)

    through_padded = padded.clone()
    opt2 = AdamDynamicEmbeddingOptimizer(OptimizerArgs(learning_rate=0.1))
    opt2.step()
    buf.update_padded(opt2, grads, through_padded, tids)
    for t, d in enumerate(DIMS):
        sel = tids == t
        torch.testing.assert_close(
            stored[sel][:, :d], through_padded[sel][:, :d], rtol=1e-5, atol=1e-6
        )


def test_reset_optimizer_states_is_a_no_op_without_state(device):
    opt = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    buf = _buffers(device, opt)
    padded = torch.randn(6, buf.max_value_dim, device=device)
    before = padded.clone()
    buf.reset_optimizer_states(opt, padded, torch.zeros(6, dtype=torch.int64, device=device))
    assert torch.equal(padded, before)
