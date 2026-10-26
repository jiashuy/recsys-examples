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

With ``preinitialize`` the rows a key takes already hold their initial value. The pool is
then in three parts: the *clean* rows, on the stack, initialised; the *dirty* rows, given
back by keys that left, still holding what those keys had; and the rows never handed out,
from the watermark on. Rows are initialised in chunks, by the initializer the allocator
is given, when a step needs more clean rows than there are, from the dirty rows first and
then from the watermark on. A table of identity rows is initialised whole, once.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import torch
from dynamicemb_extensions import row_pool_make_views, row_pool_release, row_pool_take


class RowAllocator:
    def __init__(
        self,
        row_capacities: Sequence[int],
        identity: Sequence[bool],
        limits: Sequence[int],
        device: torch.device,
        preinitialize: bool = False,
        chunk: int = 1 << 16,
    ) -> None:
        """``row_capacities`` are the rows of each table. For a table with ``identity`` rows
        the keys are ``0 .. limits[t] - 1``. With ``preinitialize`` the rows are initialised
        before they are taken, ``chunk`` at a time, once :meth:`set_initializer` says how."""
        self.device = device
        self.num_tables = len(row_capacities)
        self._row_cap = [int(r) for r in row_capacities]
        self.identity = [bool(x) for x in identity]
        self._limits = [int(x) for x in limits]
        assert len(self.identity) == self.num_tables == len(self._limits)
        self.preinitialize = bool(preinitialize)
        self._chunk = int(chunk)
        self._init_rows: Optional[Callable[[torch.Tensor, int], None]] = None
        if self.preinitialize:
            # Nothing is clean yet: the stack is filled as rows are initialised. The rows
            # from _mark[t] on were never handed out; the dirty ones are on _dirty[t].
            self._free = [
                torch.empty(r, dtype=torch.int32, device=device) for r in self._row_cap
            ]
            self._tops = torch.zeros(self.num_tables, dtype=torch.int32, device=device)
            self._clean = [0] * self.num_tables
            self._mark = [0] * self.num_tables
            self._dirty: List[Optional[torch.Tensor]] = [None] * self.num_tables
            self._dirty_n = [0] * self.num_tables
        else:
            # The pool of free rows is a stack; _tops[t] is how many are on it.
            self._free = [
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
        """Rows the table could hand out, initialised or not."""
        if self.preinitialize:
            return self._clean[table] + self._dirty_n[table] + (
                self._row_cap[table] - self._mark[table]
            )
        return int(self._tops[table].item())

    def free_counts(self) -> List[int]:
        return [self.free_rows(t) for t in range(self.num_tables)]

    # -- initialising ---------------------------------------------------------

    def set_initializer(self, init_rows: Callable[[torch.Tensor, int], None]) -> None:
        """How a row gets its initial value: ``init_rows(rows, table)`` writes it into the
        given rows (int64, on the device) of a table. The rows of a table of identity rows
        are initialised now, all of them."""
        self._init_rows = init_rows
        if not self.preinitialize:
            return
        for t in range(self.num_tables):
            if self.identity[t]:
                for start in range(0, self._row_cap[t], self._chunk):
                    stop = min(start + self._chunk, self._row_cap[t])
                    init_rows(
                        torch.arange(start, stop, dtype=torch.int64, device=self.device), t
                    )

    def ensure(self, counts: Sequence[int]) -> None:
        """Make sure each table has at least ``counts[t]`` clean rows to take."""
        if not self.preinitialize:
            return
        for t, n in enumerate(counts):
            if n and not self.identity[t] and self._clean[t] < n:
                self._refill(t, int(n))

    def _refill(self, t: int, n: int) -> None:
        assert self._init_rows is not None, "the rows have no initializer yet"
        have = self._clean[t]
        available = self._dirty_n[t] + self._row_cap[t] - self._mark[t]
        if have + available < n:
            raise RuntimeError(
                f"table {t} has {have + available} free rows and {n} keys need one; "
                "increase row_capacity or release the victims of earlier steps"
            )
        want = min(max(n - have, self._chunk), available)
        from_dirty = min(self._dirty_n[t], want)
        parts = []
        if from_dirty:
            d = self._dirty_n[t]
            parts.append(self._dirty[t][d - from_dirty : d].to(torch.int64))
        fresh = want - from_dirty
        if fresh:
            parts.append(
                torch.arange(
                    self._mark[t], self._mark[t] + fresh, dtype=torch.int64, device=self.device
                )
            )
        rows = torch.cat(parts) if len(parts) > 1 else parts[0]
        self._init_rows(rows, t)
        self._free[t][have : have + want] = rows.to(torch.int32)
        self._tops[t] = have + want
        self._clean[t] = have + want
        self._dirty_n[t] -= from_dirty
        self._mark[t] += fresh

    def note_taken(self, counts: Sequence[int]) -> None:
        """The rows a step took, per table, so that the clean ones can be counted."""
        if not self.preinitialize:
            return
        for t, c in enumerate(counts):
            if not self.identity[t]:
                self._clean[t] -= int(c)

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
        for t in range(self.num_tables):
            if self.identity[t]:
                continue
            mine = rows if self.num_tables == 1 else rows[table_ids == t]
            if mine.numel() == 0:
                continue
            if self.preinitialize:
                self._make_dirty(t, mine)
            else:
                row_pool_release(self._views, t, mine)

    def _make_dirty(self, t: int, rows: torch.Tensor) -> None:
        if self._dirty[t] is None:
            self._dirty[t] = torch.empty(
                self._row_cap[t], dtype=torch.int32, device=self.device
            )
        d = self._dirty_n[t]
        self._dirty[t][d : d + rows.numel()] = rows.to(torch.int32)
        self._dirty_n[t] = d + rows.numel()

    def grow(self, table: int, new_rows: int) -> None:
        """Give a table ``new_rows`` rows in all; the rows it had not are free."""
        old = self._row_cap[table]
        assert new_rows > old and not self.identity[table]
        if self.preinitialize:
            # The new rows were never handed out: the watermark leaves them to be.
            free = torch.empty(new_rows, dtype=torch.int32, device=self.device)
            free[: self._clean[table]] = self._free[table][: self._clean[table]]
            self._free[table] = free
            if self._dirty[table] is not None:
                dirty = torch.empty(new_rows, dtype=torch.int32, device=self.device)
                dirty[: self._dirty_n[table]] = self._dirty[table][: self._dirty_n[table]]
                self._dirty[table] = dirty
            self._row_cap[table] = new_rows
            self._rebuild_views()
            return
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
