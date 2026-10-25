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

"""Moving value rows between a cache and the storage behind it.

A cache node holds a copy of the value row of a key that the storage also holds.
Giving a node to another key is a *swap*: the old key's row goes back to the row it has
in the storage, and the new key's row, if it has one, comes from the storage into the
node. :class:`RowExchange` does a step's swaps together.

What the swap is built around is what reaching host memory costs. A GPU pays a fixed
price for each random row it touches across the link, whatever the width of the row, and
reaches rows in order about twice as fast; the CPU does not do better once its own memory
bandwidth is the limit, and the driver runs the host functions of all streams on one
thread, so a CPU gather and scatter cannot overlap. What works at every row width is to
visit the storage's rows in order, each side on its own: the victims are first gathered
from the cache (in HBM) into a staging buffer, in the order of the storage rows they go
back to, and one kernel then pairs the i-th victim with the i-th fetched row, writing the
one out and reading the other in, in the same pass, so that both directions of the link
are busy at once. See ``benchmark/bench_row_exchange.py`` on the branch
``row-exchange-experiments`` for what was measured.
"""

import torch
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.types import CopyMode
from dynamicemb_extensions import swap_staged as _swap_staged


class RowExchange:
    """The rows of a cache and of the storage behind it, which have the same tables and
    row widths. The buffers are read each time, so a buffer of the storage that has
    grown is seen."""

    def __init__(self, cache: MixedDimBuffers, storage: MixedDimBuffers):
        if cache.value_dims_list != storage.value_dims_list:
            raise ValueError("the cache and the storage hold rows of different widths")
        if cache.dtype != storage.dtype:
            raise ValueError("the cache and the storage hold different element types")
        self.cache = cache
        self.storage = storage
        elem = torch.empty((), dtype=cache.dtype).element_size()
        row_bytes = [v * elem for v in cache.value_dims_list]
        self._row_bytes = torch.tensor(row_bytes, dtype=torch.int64, device=cache.device)
        # The widest word every row is a whole number of.
        self._word_bytes = next(w for w in (16, 4, 2) if all(b % w == 0 for b in row_bytes))

    def swap(
        self,
        nodes: torch.Tensor,
        table_ids: torch.Tensor,
        evict_rows: torch.Tensor,
        fetch_rows: torch.Tensor,
    ) -> None:
        """For the i-th node of table ``table_ids[i]``: its row goes to the storage row
        ``evict_rows[i]``, and the storage row ``fetch_rows[i]`` comes into it. Either is
        skipped for -1, and a node of -1 is skipped altogether; a node that has nothing
        to fetch keeps its row."""
        if nodes.numel() == 0:
            return
        cache = self.cache
        gone = nodes < 0
        evict_rows = torch.where(gone, -1, evict_rows)
        fetch_rows = torch.where(gone, -1, fetch_rows)
        safe_nodes = nodes.clamp(min=0)
        evict_order = torch.argsort(evict_rows)
        fetch_order = torch.argsort(fetch_rows)
        staging = cache.load(
            safe_nodes[evict_order], table_ids[evict_order], CopyMode.VALUE
        )
        _swap_staged(
            cache.ptrs,
            self.storage.ptrs,
            self._row_bytes,
            table_ids[evict_order],
            table_ids[fetch_order],
            evict_rows[evict_order],
            fetch_rows[fetch_order],
            safe_nodes[fetch_order],
            staging,
            staging.shape[1] * staging.element_size(),
            self._word_bytes,
        )
