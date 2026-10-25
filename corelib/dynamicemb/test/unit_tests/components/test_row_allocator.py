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

import pytest
import torch
from dynamicemb.storage.row_allocator import RowAllocator


@pytest.fixture
def device():
    return torch.device("cuda", torch.cuda.current_device())


def _t(values, device):
    return torch.tensor(values, dtype=torch.int64, device=device)


def _take(alloc, keys, tids, device):
    """Rows for keys already sorted by table, as the placing step hands them over."""
    tstart = [0]
    for t in range(alloc.num_tables):
        tstart.append(tstart[-1] + tids.count(t))
    srows, abort = alloc.take(_t(keys, device), _t(tids, device), _t(tstart, device))
    return srows.tolist(), int(abort.item())


def test_the_top_of_the_stack_is_taken_in_order(device):
    alloc = RowAllocator([10], [False], [10], device)
    rows, abort = _take(alloc, [100, 101, 102], [0, 0, 0], device)
    assert (rows, abort) == ([7, 8, 9], 0)
    assert alloc.free_rows(0) == 7


def test_a_released_row_is_taken_again_first(device):
    alloc = RowAllocator([10], [False], [10], device)
    _take(alloc, [1, 2, 3], [0, 0, 0], device)
    alloc.release(_t([8], device), _t([0], device))
    assert alloc.free_rows(0) == 8
    assert _take(alloc, [4], [0], device) == ([8], 0)


def test_more_keys_than_free_rows_is_an_abort_and_nothing_is_taken(device):
    alloc = RowAllocator([4], [False], [4], device)
    _, abort = _take(alloc, [1, 2, 3, 4, 5], [0] * 5, device)
    assert abort == 1
    assert alloc.free_rows(0) == 4


def test_identity_rows_are_the_keys_and_the_pool_is_untouched(device):
    alloc = RowAllocator([8, 8], [True, False], [8, 8], device)
    rows, abort = _take(alloc, [5, 2, 7, 9], [0, 0, 1, 1], device)
    assert abort == 0 and rows[:2] == [5, 2]
    assert alloc.free_rows(0) == 8  # untouched
    assert alloc.free_rows(1) == 6
    alloc.release(_t([5, 2, 7], device), _t([0, 0, 1], device))
    assert alloc.free_rows(0) == 8 and alloc.free_rows(1) == 7  # none for table 0


@pytest.mark.parametrize("bad", [8, -1])
def test_a_key_that_is_not_an_identity_row_is_an_abort(device, bad):
    alloc = RowAllocator([8], [True], [8], device)
    assert _take(alloc, [3, bad], [0, 0], device)[1] == 2


def test_growing_puts_the_new_rows_on_top(device):
    alloc = RowAllocator([4], [False], [4], device)
    alloc.grow(0, 7)
    assert alloc.row_capacity_of(0) == 7
    assert alloc.free_rows(0) == 7
    assert _take(alloc, [1, 2], [0, 0], device) == ([5, 6], 0)
