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

"""Which row of a table's value buffer a key takes.

The index of a table places keys in slots; the rows its values live in are the
allocator's. A table has a pool of rows, a stack of the free ones: the keys placed in a
step take the top of it, and the rows of the keys that leave are put back once nothing
reads them. A table of identity rows has no pool: each key is the row of its own number.
Everything about where a key's row is lives here, so a different way of handing out rows
is a change to this module and its kernels, and not to the index.
"""

from typing import List, Sequence, Tuple

import torch
from dynamicemb_extensions import row_pool_make_views, row_pool_release, row_pool_take


class RowAllocator:
    def __init__(
        self,
        row_capacities: Sequence[int],
        identity: Sequence[bool],
        limits: Sequence[int],
        device: torch.device,
    ) -> None:
        """``row_capacities`` are the rows of each table. For a table with ``identity`` rows
        the keys are ``0 .. limits[t] - 1``."""
        self.device = device
        self.num_tables = len(row_capacities)
        self._row_cap = [int(r) for r in row_capacities]
        self.identity = [bool(x) for x in identity]
        self._limits = [int(x) for x in limits]
        assert len(self.identity) == self.num_tables == len(self._limits)
        # The pool of free rows is a stack; _tops[t] is how many are on it.
        self._free: List[torch.Tensor] = [
            torch.arange(r, dtype=torch.int32, device=device) for r in self._row_cap
        ]
        self._tops = torch.tensor(self._row_cap, dtype=torch.int32, device=device)
        self._rebuild_views()

    def _rebuild_views(self) -> None:
        self._views = row_pool_make_views(
            self._free,
            self._tops,
            [int(x) for x in self.identity],
            self._limits,
        )

    # -- queries --------------------------------------------------------------

    def row_capacity_of(self, table: int) -> int:
        """Rows the table's value buffer has to have."""
        return self._row_cap[table]

    def free_rows(self, table: int = 0) -> int:
        return int(self._tops[table].item())

    def free_counts(self) -> List[int]:
        return self._tops.tolist()

    # -- taking and giving back -----------------------------------------------

    def take(
        self, skeys: torch.Tensor, stids: torch.Tensor, tstart: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rows for the staged keys, sorted by table (``tstart`` is where each table's
        keys start): ``(rows in that order, abort)``.

        ``abort`` is a flag on the device that the placing step reads and reports: 0, 1
        when a table has fewer free rows than keys that need one, 2 when a key of a table
        of identity rows is not one of its rows. With an abort nothing is taken. No host
        synchronisation: the pool is checked where the keys are placed.
        """
        srows, abort = row_pool_take(self._views, skeys, stids, tstart, self.num_tables)
        return srows, abort

    def release(self, rows: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Give rows back to their pools, once nothing needs what they hold. A table of
        identity rows has no pool to give them to."""
        if rows.numel() == 0:
            return
        if self.num_tables == 1:
            if not self.identity[0]:
                row_pool_release(self._views, 0, rows)
            return
        for t in range(self.num_tables):
            if not self.identity[t]:
                row_pool_release(self._views, t, rows[table_ids == t])

    def grow(self, table: int, new_rows: int) -> None:
        """Give a table ``new_rows`` rows in all; the rows it had not are free."""
        old = self._row_cap[table]
        assert new_rows > old and not self.identity[table]
        top = int(self._tops[table].item())
        free = torch.empty(new_rows, dtype=torch.int32, device=self.device)
        free[:top] = self._free[table][:top]
        added = new_rows - old
        free[top : top + added] = torch.arange(
            old, new_rows, dtype=torch.int32, device=self.device
        )
        self._free[table] = free
        self._tops[table] = top + added
        self._row_cap[table] = new_rows
        self._rebuild_views()
