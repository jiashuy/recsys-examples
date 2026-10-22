# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential test for the multi-table path.

Same idea as test_differential.py — drive a sequence through the cache and
through a plain model, compare afterwards — with the things only multiple tables
can get wrong:

  * the same key value in two tables is two different entities, and must not
    collide. Node ids repeat across tables, so anything that drops the table id
    somewhere will surface here and almost nowhere else.
  * row widths differ per table, which is most of the reason the structures are
    split at all. A width mix-up shows up as a row of the wrong length or as
    values bleeding between tables.
  * capacity cannot move between tables. One table thrashing must leave its
    neighbour's hit rate untouched.

STATUS: verified on EOS (H100, sm_90). See src/lw_cache/README.md.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from dynamicemb_extensions import LwMultiCache, LwMultiStubBacking


def initial_row(key: int, width: int) -> np.ndarray:
    """Must match InitFill with derive_from_key=True."""
    return np.array([key * 100.0 + e for e in range(width)], dtype=np.float32)


class Model:
    """Per table, what the backing should hold if the cache were not there."""

    def __init__(self, widths: List[int]) -> None:
        self.widths = widths
        self.rows: List[Dict[int, np.ndarray]] = [{} for _ in widths]

    def touch(self, table: int, key: int, delta: float) -> None:
        rows = self.rows[table]
        if key not in rows:
            rows[key] = initial_row(key, self.widths[table])
        rows[key] += delta


def run_step(
    cache: LwMultiCache,
    backing: LwMultiStubBacking,
    per_table: List[List[int]],
    delta: float,
) -> Tuple[int, int]:
    """One step across all tables. Returns (hits, misses)."""
    dev = torch.device("cuda")

    flat: List[int] = []
    ranges = [0]
    for keys in per_table:
        assert len(set(keys)) == len(keys), "keys must be deduplicated per table"
        flat.extend(keys)
        ranges.append(len(flat))

    key_t = torch.tensor(flat, dtype=torch.int64, device=dev)
    range_t = torch.tensor(ranges, dtype=torch.int64, device=dev)

    nodes = cache.lookup(key_t, range_t)
    hits = int((nodes >= 0).sum().item())
    nodes = cache.allocate(key_t, range_t, nodes, backing)

    placed = nodes >= 0
    assert bool(placed.all()), (
        "a key could not be cached; size the trial so neither the backing fills "
        "nor the ring runs dry, or the comparison means nothing"
    )

    # forward would gather here; backward writes the optimizer result back.
    # Per-table pools, because the widths differ.
    for t in range(len(per_table)):
        lo, hi = ranges[t], ranges[t + 1]
        if lo == hi:
            continue
        rows = cache.rows(t)
        rows[nodes[lo:hi].long()] += delta

    cache.release(nodes, range_t)
    return hits, len(flat) - hits


def check(backing: LwMultiStubBacking, model: Model) -> None:
    for t, expect_rows in enumerate(model.rows):
        keys_t, rows_t = backing.snapshot(t)
        got = {int(k): rows_t[i].numpy() for i, k in enumerate(keys_t.tolist())}
        for key, expect in expect_rows.items():
            assert key in got, f"table {t}: key {key} missing from the backing"
            assert np.array_equal(got[key], expect), (
                f"table {t} key {key}: cache path {got[key]!r} != model {expect!r}"
            )


def test_key_collision_across_tables() -> None:
    """The same key value in two tables must stay two separate entities."""
    widths = [4, 4]
    cache = LwMultiCache(
        capacities=[8, 8], row_elems=widths, init_derive_from_key=True
    )
    backing = LwMultiStubBacking(slots=[64, 64], row_elems=widths)
    model = Model(widths)

    shared = [1, 2, 3]
    # table 0 gets +1 per step, table 1 gets +10, over the identical key values
    for _ in range(5):
        run_step(cache, backing, [shared, []], 1.0)
        for k in shared:
            model.touch(0, k, 1.0)
        run_step(cache, backing, [[], shared], 10.0)
        for k in shared:
            model.touch(1, k, 10.0)

    cache.flush(backing)
    check(backing, model)

    for k in shared:
        assert cache.contains(0, k) and cache.contains(1, k)


def test_differing_row_widths() -> None:
    widths = [2, 8]
    cache = LwMultiCache(
        capacities=[16, 16], row_elems=widths, init_derive_from_key=True
    )
    backing = LwMultiStubBacking(slots=[128, 128], row_elems=widths)
    model = Model(widths)

    assert cache.rows(0).shape[1] == 2
    assert cache.rows(1).shape[1] == 8

    for step in range(20):
        a = [100 + step % 7, 200 + step % 7]
        b = [300 + step % 5]
        run_step(cache, backing, [a, b], 1.0)
        for k in a:
            model.touch(0, k, 1.0)
        for k in b:
            model.touch(1, k, 1.0)

    cache.flush(backing)
    check(backing, model)


def test_one_table_thrashing() -> None:
    """Capacity is not shared: table 0 churning must not disturb table 1."""
    widths = [4, 4]
    cache = LwMultiCache(
        capacities=[2, 16], row_elems=widths, init_derive_from_key=True
    )
    backing = LwMultiStubBacking(slots=[256, 256], row_elems=widths)
    model = Model(widths)

    steady = [900, 901, 902]
    hot_hits = 0
    for step in range(60):
        churn = [1000 + step % 20]
        _, _ = run_step(cache, backing, [churn, steady], 1.0)
        for k in churn:
            model.touch(0, k, 1.0)
        for k in steady:
            model.touch(1, k, 1.0)
        if step > 0:
            # table 1's working set fits, so it should be resident throughout
            hot_hits += sum(1 for k in steady if cache.contains(1, k))

    assert hot_hits == 3 * 59, (
        "table 1's keys should never be evicted: its capacity is its own, and "
        "table 0's churn cannot touch it"
    )
    assert cache.count_referenced() == 0

    cache.flush(backing)
    check(backing, model)


def test_random_differential(seeds: int = 10, steps: int = 100) -> None:
    widths = [4, 6, 2]
    for seed in range(seeds):
        rng = random.Random(seed)
        cache = LwMultiCache(
            capacities=[4, 16, 8], row_elems=widths, init_derive_from_key=True
        )
        backing = LwMultiStubBacking(slots=[256, 256, 256], row_elems=widths)
        model = Model(widths)

        for _ in range(steps):
            per_table = []
            for t, universe in enumerate((30, 100, 12)):
                n = rng.randint(0, 3)
                base = (t + 1) * 10_000
                per_table.append(
                    [base + k for k in rng.sample(range(universe), n)]
                )
            run_step(cache, backing, per_table, 1.0)
            for t, keys in enumerate(per_table):
                for k in keys:
                    model.touch(t, k, 1.0)

        assert cache.count_referenced() == 0, f"seed {seed}: reference leak"
        cache.flush(backing)
        check(backing, model)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device; skipping")
        return 0
    for fn in (
        test_key_collision_across_tables,
        test_differing_row_widths,
        test_one_table_thrashing,
        test_random_differential,
    ):
        print(f"=== {fn.__name__} ===")
        fn()
        print("  PASS")
    print("all multi-table trials passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
