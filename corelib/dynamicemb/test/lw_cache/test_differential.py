# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P0 acceptance for lw_cache: a differential test.

The same sequence of key touches and updates is driven twice — once through the
cache, once through a plain dict — and the backing is compared against the model
at the end. Any disagreement is a real defect: the cache is supposed to be
invisible except for how long things take.

This replaces loss alignment as the P0 gate, and is stronger for the purpose.
It is deterministic, it shrinks (a failing seed reduces to a minimal sequence),
and a mismatch names the key and the element rather than reporting that a number
drifted somewhere.

Three axes matter, and each stresses something different:

    capacity vs working set   no eviction / constant eviction / thrashing
    key distribution          uniform / zipf / all in one bucket
    backing eviction          off / on

The tightest configuration is the most valuable: capacity 4 against a working
set of dozens evicts on nearly every step, so a few hundred steps visit every
state transition many times over. That is the one to keep as a regression.

STATUS: verified on EOS (H100, sm_90). See src/lw_cache/README.md.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from dynamicemb_extensions import LwCache, LwStubBacking

ROW_ELEMS = 4


def initial_row(key: int) -> np.ndarray:
    """Must match InitFill with derive_from_key=True."""
    return np.array([key * 100.0 + e for e in range(ROW_ELEMS)], dtype=np.float32)


@dataclass
class Model:
    """What the backing should contain if the cache were not there."""

    rows: Dict[int, np.ndarray] = field(default_factory=dict)

    def touch(self, key: int, delta: float) -> None:
        if key not in self.rows:
            self.rows[key] = initial_row(key)
        self.rows[key] += delta

    def forget(self, keys: List[int]) -> None:
        """The backing dropped these; their history is gone."""
        for k in keys:
            self.rows.pop(k, None)


@dataclass
class Stats:
    steps: int = 0
    hits: int = 0
    misses: int = 0
    bypassed: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def run_step(
    cache: LwCache,
    backing: LwStubBacking,
    keys: List[int],
    delta: float,
    stats: Stats,
) -> None:
    """One training step, in the order the kernels require."""
    assert len(set(keys)) == len(keys), (
        "keys must be deduplicated: reference counting is one +1 per unique key "
        "per step, and a repeat adds a reference the single release cannot undo"
    )

    dev = torch.device("cuda")
    key_t = torch.tensor(keys, dtype=torch.int64, device=dev)

    nodes = cache.lookup(key_t)
    hits = int((nodes >= 0).sum().item())

    nodes = cache.allocate(key_t, nodes, backing)
    placed = nodes >= 0
    bypassed = int((~placed).sum().item())

    # forward would gather here; backward writes the optimizer result back
    rows = cache.rows()
    sel = nodes[placed].long()
    rows[sel] += delta

    cache.release(nodes)

    stats.steps += 1
    stats.hits += hits
    stats.misses += len(keys) - hits
    stats.bypassed += bypassed


def sample_keys(rng: random.Random, spec: str, universe: int, count: int) -> List[int]:
    if spec == "uniform":
        pool = range(1, universe + 1)
        return rng.sample(list(pool), min(count, universe))
    if spec == "zipf":
        # Heavy tail, which is what recommender traffic looks like and what makes
        # the difference between FIFO and CLOCK visible.
        out: List[int] = []
        while len(out) < count:
            k = min(int(rng.paretovariate(1.2)), universe)
            if k not in out:
                out.append(k)
        return out
    if spec == "one_bucket":
        # Keys engineered to collide is the only way the chain code gets
        # exercised beyond length one.
        stride = 1 << 20
        return [1 + i * stride for i in range(min(count, universe))]
    raise ValueError(f"unknown key distribution: {spec}")


def check(
    backing: LwStubBacking, model: Model, label: str
) -> Optional[str]:
    keys_t, rows_t = backing.snapshot()
    got = {int(k): rows_t[i].numpy() for i, k in enumerate(keys_t.tolist())}

    for key, expect in model.rows.items():
        if key not in got:
            return f"{label}: key {key} missing from the backing"
        if not np.array_equal(got[key], expect):
            bad = int(np.argmax(got[key] != expect))
            return (
                f"{label}: key {key} element {bad}: "
                f"cache path {got[key][bad]!r}, model {expect[bad]!r}"
            )
    return None


def run_trial(
    seed: int,
    capacity: int,
    universe: int,
    per_step: int,
    steps: int,
    distribution: str,
    backing_slots: int,
    evict_every: int,
    replacement: str,
    verbose: bool = False,
) -> Optional[str]:
    """Returns None on success, or a description of the first mismatch."""
    rng = random.Random(seed)
    # CLOCK spends a ring entry on every second chance without getting a node
    # for it — the entry goes to the retry buffer and only returns after the
    # follow-up kernel — so it needs (1 + k) entries per allocating thread
    # against FIFO's 1. Scaling here rather than hand-picking capacities keeps
    # the two policies comparable at the same pressure.
    entries_per_thread = 2 if replacement == "clock" else 1
    cache = LwCache(
        capacity=capacity * entries_per_thread,
        row_elems=ROW_ELEMS,
        replacement=replacement,
        expected_unique_per_step=per_step,
        capacity_safety_factor=1.0,
        init_derive_from_key=True,
    )
    backing = LwStubBacking(slots=backing_slots, row_elems=ROW_ELEMS)
    model = Model()
    stats = Stats()

    for step in range(steps):
        keys = sample_keys(rng, distribution, universe, per_step)
        run_step(cache, backing, keys, 1.0, stats)
        for k in keys:
            model.touch(k, 1.0)

        if stats.bypassed:
            return (
                f"step {step}: {stats.bypassed} key(s) could not be cached. "
                "Either the backing is full or the ring ran dry; size the trial "
                "so neither happens, or the comparison is meaningless."
            )

        if evict_every and step and step % evict_every == 0:
            # Only keys the cache is *not* holding. A key dropped by the backing
            # while still resident has lost nothing: the cached copy is the
            # newest one that exists (INV-9) and will be written back to a
            # freshly allocated row. Its history is gone only when neither tier
            # has it, which is the case this is trying to produce.
            candidates = [
                k
                for k in model.rows
                if backing.contains(k) and not cache.contains(k)
            ]
            if candidates:
                victims = rng.sample(candidates, max(1, len(candidates) // 4))
                # The backing forgets them and rebuilds, which relocates every
                # surviving row as well. Nothing in the cache stores a row
                # index, so both halves of that must be invisible to it.
                backing.evict(victims)
                model.forget(victims)

    leaked = cache.count_referenced()
    if leaked:
        return (
            f"{leaked} node(s) still referenced after the last release: an "
            "unbalanced pin, which otherwise only shows up as a hit rate that "
            "sags over hours"
        )

    cache.flush(backing)
    if verbose:
        print(
            f"  seed={seed} cap={capacity * entries_per_thread} "
            f"set={universe} dist={distribution} "
            f"hit_rate={stats.hit_rate():.3f}"
        )
    return check(backing, model, f"seed {seed}")


def shrink(failing: dict) -> dict:
    """Reduce a failing configuration to something small enough to read."""
    best = dict(failing)
    for field_name, floor in (("steps", 1), ("per_step", 1), ("universe", 2)):
        while best[field_name] > floor:
            trial = dict(best)
            trial[field_name] = max(floor, trial[field_name] // 2)
            if run_trial(**trial) is not None:
                best = trial
            else:
                break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; skipping")
        return 0

    # (capacity, universe, per_step, distribution, backing_slots, evict_every)
    configs = [
        # no eviction at all: the cache holds everything
        (64, 16, 4, "uniform", 256, 0),
        # constant eviction
        (16, 64, 8, "uniform", 512, 0),
        # thrashing: nearly every step evicts
        (4, 40, 3, "uniform", 256, 0),
        # heavy tail, where replacement policy actually matters
        (16, 200, 8, "zipf", 1024, 0),
        # every key in one bucket: the only real test of the chain code
        (16, 32, 6, "one_bucket", 256, 0),
        # backing drops keys and relocates the rest
        (16, 64, 8, "uniform", 512, 7),
        (4, 40, 3, "uniform", 256, 5),
    ]
    if args.quick:
        configs = configs[:3]

    failures = 0
    for capacity, universe, per_step, dist, slots, evict in configs:
        for policy in ("fifo", "clock"):
            label = (
                f"cap={capacity} set={universe} n={per_step} {dist} "
                f"{policy} evict_every={evict}"
            )
            print(f"=== {label} ===")
            for seed in range(args.seeds):
                params = dict(
                    seed=seed,
                    capacity=capacity,
                    universe=universe,
                    per_step=per_step,
                    steps=args.steps,
                    distribution=dist,
                    backing_slots=slots,
                    evict_every=evict,
                    replacement=policy,
                )
                err = run_trial(**params)
                if err is not None:
                    print(f"  FAIL {err}")
                    small = shrink(params)
                    print(f"  minimal reproducer: {small}")
                    failures += 1
                    break
            else:
                print("  PASS")

    print("all differential trials passed" if not failures else f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
