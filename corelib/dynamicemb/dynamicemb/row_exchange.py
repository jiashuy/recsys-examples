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
node. :class:`RowExchange` does the two together, node by node, in one kernel. The old
row has to be read before the new one is written, and doing it per node is what makes
that so, without a staging copy of every victim; and when the storage is in host memory
the write-back and the fetch cross the link in opposite directions at once.
"""

import torch
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb_extensions import swap_rows as _swap_rows


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
        skipped for -1; a node of -1 is skipped altogether."""
        _swap_rows(
            self.cache.ptrs,
            self.storage.ptrs,
            self._row_bytes,
            table_ids,
            nodes,
            evict_rows,
            fetch_rows,
            self._word_bytes,
        )
