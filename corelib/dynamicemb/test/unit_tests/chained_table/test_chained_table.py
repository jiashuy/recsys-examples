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

"""The chained bucket table, against a model of what it is meant to do.

The model is a few lines of python that apply the rule directly: a step finds its keys,
stages the misses, and commits the ones kept; a bucket gives free slots to its new keys
in order and evicts the lowest-scored members, skipping any that are protected, for the
rest. The table is checked against it slot for slot, rows included, after every step.
"""

import random

import pytest
import torch
from dynamicemb.chained_table import (
    ChainedTable,
    ChainedTables,
    ChainedTableFull,
    hash_py,
)
from dynamicemb_extensions import CacheCore

def _offs(values):
    """Offsets as the cache takes them: int64, on the device."""
    return torch.tensor(values, dtype=torch.int64, device="cuda")


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _t(values, device, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device=device)


class Model:
    """What a ChainedTable should hold, written out the plain way."""

    def __init__(self, num_buckets, cap, row_capacity):
        self.nb, self.cap = num_buckets, cap
        self.buckets = [[] for _ in range(num_buckets)]  # each: [key, score, row]
        self.free = list(range(row_capacity))
        self.staged = []  # (position, key, score)

    def bucket(self, key):
        return hash_py(key) % self.nb

    def find_and_stage(self, keys, scores, policy):
        rows = []
        self.staged = []
        for pos, key in enumerate(keys):
            score = None if scores is None else scores[pos]
            for entry in self.buckets[self.bucket(key)]:
                if entry[0] == key:
                    if policy == "assign":
                        entry[1] = score if score is not None else 0
                    elif policy == "accumulate":
                        entry[1] += score if score is not None else 1
                    rows.append(entry[2])
                    break
            else:
                rows.append(-1)
                init = score if score is not None else (1 if policy == "accumulate" else 0)
                self.staged.append((pos, key, init))
        return rows

    def commit(self, keep=None, protected=frozenset()):
        staged = self.staged
        if keep is not None:
            staged = [s for s, k in zip(staged, keep) if k]
        self.staged = []
        m = len(staged)
        pool = self.free[len(self.free) - m :] if m else []
        del self.free[len(self.free) - m :]
        # Rows go to the keys in the order they sort in: by bucket, then as staged.
        order = sorted(range(m), key=lambda j: (self.bucket(staged[j][1]), j))
        new_rows = [None] * m
        for i, j in enumerate(order):
            new_rows[j] = pool[i]
        by_bucket = {}
        for j in order:
            pos, key, score = staged[j]
            by_bucket.setdefault(self.bucket(key), []).append((key, score, new_rows[j]))
        victims = []
        for b in sorted(by_bucket):
            new = by_bucket[b]
            members = self.buckets[b]
            s0 = len(members)
            free_slots = self.cap - s0
            need = max(0, len(new) - free_slots)
            order = sorted(range(s0), key=lambda s: (members[s][1], s))
            chosen = [s for s in order if members[s][0] not in protected][:need]
            if len(chosen) < need:
                raise ChainedTableFull("model: not enough victims", None)
            for key, score, row in new[:free_slots]:
                members.append([key, score, row])
            for slot, (key, score, row) in zip(chosen, new[free_slots:]):
                victims.append((members[slot][0], members[slot][2]))
                members[slot] = [key, score, row]
        return new_rows, [p for p, _, _ in staged], victims

    def release(self, rows):
        self.free.extend(rows)

    def flat(self):
        return [tuple(e) for b in self.buckets for e in b]


def _state(table):
    keys, scores, rows = table.export()
    return list(zip(keys.tolist(), scores.tolist(), rows.tolist()))


def _step(table, model, keys, scores, policy, keep=None, guard=None, protected=frozenset()):
    dev = table.device
    rows = table.find_and_stage(
        _t(keys, dev),
        None if scores is None else _t(scores, dev),
        policy,
    )
    assert rows.tolist() == model.find_and_stage(keys, scores, policy)
    keep_t = None if keep is None else _t(keep, dev, torch.bool)
    out = table.commit(keep_t, guard=guard)
    new_rows, positions, victims = model.commit(keep, protected)
    assert out.rows.tolist() == new_rows
    assert out.positions.tolist() == positions
    got = list(zip(out.victim_keys.tolist(), out.victim_rows.tolist()))
    assert got == victims
    table.release_rows(out.victim_rows)
    model.release([r for _, r in victims])
    assert _state(table) == model.flat()
    return out


def test_empty_table_places_everything(device):
    table = ChainedTable(num_buckets=4, bucket_capacity=8, device=device)
    keys = [5, 6, 7, 8, 9]
    rows = table.find_and_stage(_t(keys, device))
    assert rows.tolist() == [-1] * 5
    assert table.num_staged == 5
    out = table.commit()
    assert out.victim_keys.numel() == 0
    assert sorted(out.rows.tolist()) == sorted(set(out.rows.tolist()))
    assert table.size() == 5
    assert table.lookup(_t(keys, device)).tolist() == out.rows.tolist()
    table.check_invariants()


def test_an_unknown_key_is_not_found_and_nothing_changes(device):
    table = ChainedTable(num_buckets=4, bucket_capacity=8, device=device)
    table.find_and_stage(_t([1, 2, 3], device))
    table.commit()
    before = _state(table)
    assert table.lookup(_t([4, 5], device)).tolist() == [-1, -1]
    table.find_and_stage(_t([4, 5], device))  # staged, never committed
    assert _state(table) == before
    table.check_invariants()


def test_dropped_keys_never_enter_the_table(device):
    table = ChainedTable(num_buckets=4, bucket_capacity=8, device=device)
    table.find_and_stage(_t([1, 2, 3, 4], device))
    out = table.commit(_t([True, False, True, False], device, torch.bool))
    assert out.positions.tolist() == [0, 2]
    assert table.lookup(_t([1, 2, 3, 4], device)).tolist()[1::2] == [-1, -1]
    assert table.size() == 2
    table.check_invariants()


def test_find_updates_the_score_of_a_hit(device):
    table = ChainedTable(num_buckets=2, bucket_capacity=8, device=device)
    table.find_and_stage(_t([10, 11], device), _t([5, 6], device))
    table.commit()
    table.find_and_stage(_t([10, 11], device), _t([100, 200], device), "assign")
    _, scores, _ = table.export()
    assert sorted(scores.tolist()) == [100, 200]
    table.find_and_stage(_t([10], device), None, "accumulate")
    table.find_and_stage(_t([10], device), None, "accumulate")
    _, scores, _ = table.export()
    assert sorted(scores.tolist()) == [102, 200]
    table.find_and_stage(_t([10, 11], device), None, "none")
    _, scores, _ = table.export()
    assert sorted(scores.tolist()) == [102, 200]


def test_the_lowest_scored_members_are_evicted(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=8, device=device)
    keys = list(range(100, 108))
    table.find_and_stage(_t(keys, device), _t([50, 10, 40, 20, 30, 70, 60, 80], device))
    table.commit()
    table.find_and_stage(_t([200, 201, 202], device), _t([1, 2, 3], device))
    out = table.commit()
    assert sorted(out.victim_keys.tolist()) == [101, 103, 104]  # scores 10, 20, 30
    assert table.lookup(_t([200, 201, 202], device)).min() >= 0
    assert table.lookup(_t([101, 103, 104], device)).tolist() == [-1, -1, -1]
    table.release_rows(out.victim_rows)
    table.check_invariants()


def test_a_survivor_keeps_its_row(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=8, device=device)
    table.find_and_stage(_t(list(range(8)), device), _t(list(range(8)), device))
    table.commit()
    before = dict(zip(*[x.tolist() for x in table.export()[::2]]))
    table.find_and_stage(_t([50, 51], device), _t([100, 101], device))
    out = table.commit()
    table.release_rows(out.victim_rows)
    after = dict(zip(*[x.tolist() for x in table.export()[::2]]))
    for key in set(before) & set(after):
        assert before[key] == after[key]
    table.check_invariants()


def test_victim_rows_hold_until_released(device):
    """A victim's row is not given to anyone until the caller releases it."""
    table = ChainedTable(num_buckets=1, bucket_capacity=4, device=device)
    table.find_and_stage(_t([1, 2, 3, 4], device), _t([1, 2, 3, 4], device))
    table.commit()
    table.find_and_stage(_t([9], device), _t([9], device))
    out = table.commit()
    assert out.victim_keys.tolist() == [1]
    held = out.victim_rows.item()
    assert held not in table.export()[2].tolist()
    free_before = table.free_rows()
    table.release_rows(out.victim_rows)
    assert table.free_rows() == free_before + 1


def test_uint64_keys(device):
    table = ChainedTable(num_buckets=4, bucket_capacity=8, device=device)
    keys = torch.tensor([2**63 + 5, 2**63 + 6, 7], dtype=torch.uint64, device=device)
    table.find_and_stage(keys)
    table.commit()
    assert bool((table.lookup(keys) >= 0).all())
    table.check_invariants()


def test_the_row_pool_can_run_out(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=4, row_capacity=4, device=device)
    table.find_and_stage(_t([1, 2, 3, 4], device))
    table.commit()
    table.find_and_stage(_t([5], device))
    with pytest.raises(RuntimeError, match="free rows"):
        table.commit()


@pytest.mark.parametrize("policy", ["assign", "accumulate", "none"])
# (buckets, bucket capacity, keys per step). The last three put more than 32 new keys
# in one bucket, which the kernel places a chunk of 32 at a time, and enough of them
# on one chain that several victims and several new keys share it.
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 6), (4, 16, 24), (16, 32, 150), (64, 128, 800), (1, 128, 100), (2, 256, 200),
        (1, 64, 60),
        # Buckets too large for the registers of a warp: a victim is found by a scan.
        (1, 600, 500), (3, 300, 400), (2, 1000, 900),
        # Links of 4 bytes: four times the capacity does not fit in 2.
        (1, 17000, 3000),
    ],
)
def test_matches_the_model_over_many_steps(device, shape, policy):
    nb, cap, per_step = shape
    rng = random.Random(1234 + nb)
    table = ChainedTable(num_buckets=nb, bucket_capacity=cap, device=device)
    model = Model(nb, cap, table.row_capacity)
    universe = nb * cap * 3  # three times the capacity: plenty of eviction
    for step in range(25):
        keys = rng.sample(range(1, universe + 1), per_step)
        scores = None if rng.random() < 0.2 else [rng.randrange(1, 1000) for _ in keys]
        keep = None
        if rng.random() < 0.5:
            keep = [rng.random() < 0.7 for _ in range(per_step)]
            # the mask is over the staged keys only; the model slices it the same way
            n_staged = sum(
                1 for k in keys if all(e[0] != k for e in model.buckets[model.bucket(k)])
            )
            keep = keep[:n_staged]
        _step(table, model, keys, scores, policy, keep)
        if step % 8 == 0:
            table.check_invariants()
    table.check_invariants()


def test_more_new_keys_than_a_bucket_can_make_room_for_is_an_error(device):
    """A new key never evicts another new key of the same step, so a bucket can take
    at most as many new keys as it has slots; more is a capacity error, and the bucket
    is left as it was."""
    table = ChainedTable(num_buckets=1, bucket_capacity=8, device=device)
    table.find_and_stage(_t(list(range(8)), device), _t([1] * 8, device))
    table.commit()
    before = _state(table)
    free_before = table.free_rows()
    table.find_and_stage(_t(list(range(100, 112)), device), _t([5] * 12, device))
    with pytest.raises(ChainedTableFull):
        table.commit()
    assert _state(table) == before
    assert table.free_rows() == free_before
    table.check_invariants()


# -- the cache's guard ---------------------------------------------------------


def _cache_keys(core, keys, device, hold):
    """Bind `keys` in the cache's only table. The first `hold` keep the reference a
    step in flight would; the rest are released, idle."""
    n = keys.numel()
    nodes = core.lookup_pin(keys, _offs([0, n]))
    _, _, new_nodes = core.allocate(
        keys, torch.arange(n, dtype=torch.int64, device=device), _offs([0, n]), nodes
    )
    assert bool((new_nodes >= 0).all())
    core.publish(new_nodes, _offs([0, n]))
    if n > hold:
        idle = new_nodes[hold:].contiguous()
        core.release(idle, _offs([0, idle.numel()]))
    torch.cuda.synchronize()


def test_a_referenced_key_is_never_evicted(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=128, device=device)
    core = CacheCore([4 * 128])
    stored = torch.arange(1, 101, dtype=torch.int64, device=device)
    table.find_and_stage(stored, stored.clone())  # score == key: 1 is the lowest
    table.commit()

    _cache_keys(core, torch.arange(1, 61, dtype=torch.int64, device=device), device, hold=30)
    held = set(range(1, 31))

    # 80 new keys into a bucket with 28 free slots: 52 evictions, with scores that put
    # the held keys first in line
    new = torch.arange(1000, 1080, dtype=torch.int64, device=device)
    table.find_and_stage(new, torch.full_like(new, 10_000))
    out = table.commit(guard=core.guard())
    victims = set(out.victim_keys.tolist())
    assert len(victims) == 52
    assert not (victims & held), "a referenced key was evicted"
    assert victims == set(range(31, 83))  # the lowest scores the guard lets go
    table.release_rows(out.victim_rows)
    table.check_invariants()


def test_without_the_guard_the_same_keys_go(device):
    """The control: the keys the guard protects are ones the score alone would take."""
    table = ChainedTable(num_buckets=1, bucket_capacity=128, device=device)
    stored = torch.arange(1, 101, dtype=torch.int64, device=device)
    table.find_and_stage(stored, stored.clone())
    table.commit()
    new = torch.arange(1000, 1080, dtype=torch.int64, device=device)
    table.find_and_stage(new, torch.full_like(new, 10_000))
    out = table.commit()
    assert set(out.victim_keys.tolist()) == set(range(1, 53))


def test_a_bucket_of_protected_members_is_an_error(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=128, device=device)
    core = CacheCore([4 * 128])
    stored = torch.arange(1, 129, dtype=torch.int64, device=device)
    table.find_and_stage(stored, stored.clone())
    table.commit()
    _cache_keys(core, stored, device, hold=128)  # every member is held

    free_before = table.free_rows()
    before = list(zip(*[x.tolist() for x in table.export()]))
    new = _t([9999], device)
    table.find_and_stage(new, new.clone())
    with pytest.raises(ChainedTableFull, match="in use by a step in flight") as info:
        table.commit(guard=core.guard())
    assert table.free_rows() == free_before, "the row of the unplaced key was not returned"
    assert list(zip(*[x.tolist() for x in table.export()])) == before
    assert info.value.partial.rows.numel() == 0
    table.check_invariants()


def test_other_buckets_still_place_when_one_is_full(device):
    nb, cap = 8, 16
    table = ChainedTable(num_buckets=nb, bucket_capacity=cap, device=device)
    pool = torch.arange(1, 20000, dtype=torch.int64, device=device)
    bucket = table.bucket_of(pool)

    def take(b, n, skip=0):
        return pool[bucket == b][skip : skip + n]

    # Fill every bucket exactly.
    stored = torch.cat([take(b, cap) for b in range(nb)])
    table.find_and_stage(stored, stored.clone())
    table.commit()
    assert table.size() == nb * cap

    # Hold all of bucket 0 in the cache.
    core = CacheCore([4 * 128])
    held = take(0, cap)
    _cache_keys(core, held, device, hold=cap)

    # Two new keys for bucket 0, which cannot make room, and two for bucket 1, which can.
    new0, new1 = take(0, 2, skip=cap), take(1, 2, skip=cap)
    new = torch.cat([new0, new1])
    table.find_and_stage(new, torch.full_like(new, 10**6))
    with pytest.raises(ChainedTableFull) as info:
        table.commit(guard=core.guard())
    partial = info.value.partial
    assert partial.rows.numel() == 2
    assert sorted(new[partial.positions].tolist()) == sorted(new1.tolist())
    assert bool((table.lookup(new1) >= 0).all())
    assert bool((table.lookup(new0) < 0).all())
    assert bool((table.lookup(held) >= 0).all())
    assert partial.victim_keys.numel() == 2
    table.release_rows(partial.victim_rows)
    table.check_invariants()


# -- several tables, growing, pinning ------------------------------------------


def _multi_state(tables, t):
    keys, scores, rows = tables.export(t)
    return list(zip(keys.tolist(), scores.tolist(), rows.tolist()))


@pytest.mark.parametrize("cap", [16, 300])
def test_several_tables_match_one_model_each(device, cap):
    nbs = [4, 8, 2]
    tables = ChainedTables(nbs, bucket_capacity=cap, device=device)
    models = [Model(nb, cap, tables.row_capacity_of(t)) for t, nb in enumerate(nbs)]
    rng = random.Random(7)
    for step in range(20):
        keys, tids = [], []
        for t, nb in enumerate(nbs):
            n = rng.randrange(1, nb * cap // 2)
            ks = rng.sample(range(1, nb * cap * 3), n)
            keys += ks
            tids += [t] * n
        order = list(range(len(keys)))
        rng.shuffle(order)  # tables interleaved in the input
        keys, tids = [keys[i] for i in order], [tids[i] for i in order]
        scores = [rng.randrange(1, 1000) for _ in keys]

        rows = tables.find_and_stage(_t(keys, device), _t(tids, device), _t(scores, device))
        expect = [-1] * len(keys)
        for t, model in enumerate(models):
            idx = [i for i, x in enumerate(tids) if x == t]
            r = model.find_and_stage([keys[i] for i in idx], [scores[i] for i in idx], "assign")
            for i, row in zip(idx, r):
                expect[i] = row
        assert rows.tolist() == expect

        out = tables.commit()
        vk, vr, vt = out.victim_keys.tolist(), out.victim_rows.tolist(), out.victim_tables.tolist()
        got_victims = {t: [] for t in range(len(nbs))}
        for k, r, t in zip(vk, vr, vt):
            got_victims[t].append((k, r))
        for t, model in enumerate(models):
            new_rows, _, victims = model.commit()
            sel = out.tables == t
            assert out.rows[sel].tolist() == new_rows
            assert got_victims[t] == victims
        tables.release_rows(out.victim_rows, out.victim_tables)
        for t, model in enumerate(models):
            model.release([r for _, r in got_victims[t]])
            assert _multi_state(tables, t) == model.flat()
    tables.check_invariants()


def test_the_same_key_in_two_tables_is_two_keys(device):
    tables = ChainedTables([4, 4], bucket_capacity=8, device=device)
    keys = _t([5, 5], device)
    tids = _t([0, 1], device)
    tables.find_and_stage(keys, tids, _t([1, 2], device))
    out = tables.commit()
    assert tables.size(0) == 1 and tables.size(1) == 1
    assert tables.lookup(keys, tids).tolist() == out.rows[torch.argsort(out.positions)].tolist()
    tables.check_invariants()


def test_growing_one_table_leaves_the_others_alone(device):
    tables = ChainedTables([2, 2], bucket_capacity=8, device=device)
    keys = _t(list(range(1, 9)), device)
    for t in (0, 1):
        tables.find_and_stage(keys, _t([t] * 8, device), keys.clone())
        tables.commit()
    state1 = _multi_state(tables, 1)
    ptr1 = tables._keys[1].data_ptr()
    tables.expand(0)
    assert tables.num_buckets_of(0) >= 4
    assert tables.num_buckets_of(1) == 2
    assert tables._keys[1].data_ptr() == ptr1, "the other table's arrays were replaced"
    assert _multi_state(tables, 1) == state1
    assert bool((tables.lookup(keys, _t([0] * 8, device)) >= 0).all())
    tables.check_invariants()


@pytest.mark.parametrize("factor", [None, 3, 16])
def test_growing_moves_the_keys_and_not_the_rows(device, factor):
    table = ChainedTable(num_buckets=4, bucket_capacity=16, device=device)
    keys = torch.arange(1, 33, dtype=torch.int64, device=device)
    table.find_and_stage(keys, keys * 7)
    table.commit()
    before = sorted(zip(*[x.tolist() for x in table.export()]))
    old_rows = table.row_capacity
    new_rows = table.expand(0, None if factor is None else 4 * factor)
    after = sorted(zip(*[x.tolist() for x in table.export()]))
    assert before == after, "a key, its score or its row changed"
    assert new_rows >= table.capacity and new_rows >= old_rows
    assert table.free_rows() == new_rows - 32
    assert bool((table.lookup(keys) >= 0).all())
    table.check_invariants()


def test_a_table_that_never_evicts_grows_instead(device):
    table = ChainedTable(num_buckets=2, bucket_capacity=8, no_eviction=True, device=device)
    seen = {}
    rng = random.Random(3)
    universe = list(range(1, 2000))
    rng.shuffle(universe)
    for step in range(10):
        keys = [universe.pop() for _ in range(30)]
        t = _t(keys, device)
        table.find_and_stage(t, t.clone())
        out = table.commit()
        assert out.victim_keys.numel() == 0, "a table that never evicts evicted"
        for k, r in zip(keys, table.lookup(t).tolist()):
            assert r >= 0
            seen[k] = r
    assert table.size() == 300
    assert table.num_buckets > 2
    got = dict(zip(*[x.tolist() for x in table.export()[::2]]))
    assert got == seen, "a key lost or changed its row while the table grew"
    table.check_invariants()


def test_a_bucket_with_more_new_keys_than_slots_grows_a_no_eviction_table(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=8, no_eviction=True, device=device)
    keys = _t(list(range(1, 30)), device)
    table.find_and_stage(keys, keys.clone())
    out = table.commit()
    assert out.victim_keys.numel() == 0
    assert table.size() == 29
    table.check_invariants()


def test_a_pinned_row_is_not_evicted(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=8, track_refs=True, device=device)
    keys = _t(list(range(1, 9)), device)
    table.find_and_stage(keys, keys.clone())  # score == key: 1 is the lowest
    table.commit()
    rows = table.lookup(_t([1, 2, 3], device))
    table.pin(rows)
    assert table.refs_of(0)[rows].tolist() == [1, 1, 1]

    table.find_and_stage(_t([100, 101], device), _t([50, 51], device))
    out = table.commit()
    assert sorted(out.victim_keys.tolist()) == [4, 5]  # 1, 2, 3 are pinned
    table.release_rows(out.victim_rows)

    table.unpin(rows)
    assert table.refs_of(0)[rows].tolist() == [0, 0, 0]
    table.find_and_stage(_t([102], device), _t([60], device))
    out = table.commit()
    assert out.victim_keys.tolist() == [1]
    table.release_rows(out.victim_rows)
    table.check_invariants()


def test_find_and_commit_can_pin(device):
    table = ChainedTable(num_buckets=2, bucket_capacity=8, track_refs=True, device=device)
    keys = _t([1, 2, 3], device)
    table.find_and_stage(keys, keys.clone())
    out = table.commit(pin=True)
    assert table.refs_of(0)[out.rows].tolist() == [1, 1, 1]
    rows = table.find_and_stage(keys, keys.clone(), pin=True)
    assert table.refs_of(0)[rows].tolist() == [2, 2, 2]
    table.unpin(rows)
    table.unpin(rows)
    assert table.refs_of(0)[rows].tolist() == [0, 0, 0]


def test_every_member_pinned_is_an_error_even_without_a_guard(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=4, track_refs=True, device=device)
    keys = _t([1, 2, 3, 4], device)
    table.find_and_stage(keys, keys.clone())
    table.commit(pin=True)
    table.find_and_stage(_t([9], device), _t([9], device))
    with pytest.raises(ChainedTableFull):
        table.commit()
    assert table.size() == 4
    table.check_invariants()


# -- erase ---------------------------------------------------------------------


def _model_erase(model, keys):
    """The rule the kernel applies: the bucket's last record fills the hole."""
    freed = []
    for key in keys:
        members = model.buckets[model.bucket(key)]
        for s, entry in enumerate(members):
            if entry[0] == key:
                freed.append(entry[2])
                members[s] = members[-1]
                members.pop()
                break
    return freed


@pytest.mark.parametrize(
    "shape", [(1, 8, 6), (4, 16, 24), (16, 32, 150), (1, 128, 100), (2, 700, 300), (1, 17000, 2000)]
)
def test_erase_matches_the_model(device, shape):
    nb, cap, per_step = shape
    rng = random.Random(99 + nb)
    table = ChainedTable(num_buckets=nb, bucket_capacity=cap, device=device)
    model = Model(nb, cap, table.row_capacity)
    universe = nb * cap * 2
    for step in range(20):
        keys = rng.sample(range(1, universe + 1), per_step)
        _step(table, model, keys, [rng.randrange(1, 1000) for _ in keys], "assign")
        # erase some keys that are there and some that are not
        present = [e[0] for b in model.buckets for e in b]
        gone = rng.sample(present, min(len(present), per_step // 2))
        gone += rng.sample(range(universe + 1, universe + 50), 5)  # never stored
        rng.shuffle(gone)
        rows = table.erase(_t(gone, device))
        freed = _model_erase(model, gone)
        assert [r for r in rows.tolist() if r >= 0] == freed
        model.release(freed)
        assert _state(table) == model.flat()
        for key in gone:
            assert table.lookup(_t([key], device)).item() == -1
    table.check_invariants()


def test_erase_leaves_the_others_findable(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=64, device=device)
    keys = _t(list(range(1, 61)), device)
    table.find_and_stage(keys, keys.clone())
    table.commit()
    rows_before = dict(zip(keys.tolist(), table.lookup(keys).tolist()))
    victims = _t(list(range(1, 61, 3)), device)
    table.erase(victims)
    left = [k for k in keys.tolist() if k not in set(victims.tolist())]
    after = table.lookup(_t(left, device)).tolist()
    assert after == [rows_before[k] for k in left]
    assert table.size() == len(left)
    table.check_invariants()


def test_a_rows_erased_come_back_to_the_pool(device):
    table = ChainedTable(num_buckets=2, bucket_capacity=8, device=device)
    keys = _t([1, 2, 3, 4], device)
    table.find_and_stage(keys, keys.clone())
    table.commit()
    free = table.free_rows()
    table.erase(_t([1, 2], device))
    assert table.free_rows() == free + 2
    table.erase(_t([1, 2], device))  # already gone
    assert table.free_rows() == free + 2


def test_two_runs_of_the_same_steps_end_identically(device):
    """Nothing in a step depends on the order threads happen to run in: the same input
    leaves the same keys, scores, rows, chains and free rows."""

    def run():
        rng = random.Random(5)
        table = ChainedTable(num_buckets=16, bucket_capacity=32, device=device)
        for _ in range(25):
            keys = rng.sample(range(1, 1500), 150)
            scores = [rng.randrange(1, 100) for _ in keys]  # plenty of ties
            table.find_and_stage(_t(keys, device), _t(scores, device))
            out = table.commit()
            table.release_rows(out.victim_rows)
            table.erase(_t(rng.sample(range(1, 1500), 20), device))
        return table

    a, b = run(), run()
    for name in ("_keys", "_scores", "_index", "_next", "_head", "_bsize", "_free"):
        assert torch.equal(getattr(a, name)[0], getattr(b, name)[0]), name
    assert a.free_rows() == b.free_rows()


# -- a (timestamp, frequency) pair ----------------------------------------------


class PairModel(Model):
    """Two words per key: the clock at the last access, and a count. The least used goes
    first, and of those the longest unused."""

    def find_and_stage(self, keys, freqs, ts):
        rows = []
        self.staged = []
        for pos, key in enumerate(keys):
            add = 1 if freqs is None else freqs[pos]
            for entry in self.buckets[self.bucket(key)]:
                if entry[0] == key:
                    entry[1] = (ts, entry[1][1] + add)
                    rows.append(entry[2])
                    break
            else:
                rows.append(-1)
                self.staged.append((pos, key, (ts, add)))
        return rows

    def commit(self, keep=None, protected=frozenset()):
        staged = self.staged
        if keep is not None:
            staged = [s for s, k in zip(staged, keep) if k]
        self.staged = []
        m = len(staged)
        pool = self.free[len(self.free) - m :] if m else []
        del self.free[len(self.free) - m :]
        order = sorted(range(m), key=lambda j: (self.bucket(staged[j][1]), j))
        new_rows = [None] * m
        for i, j in enumerate(order):
            new_rows[j] = pool[i]
        by_bucket = {}
        for j in order:
            pos, key, score = staged[j]
            by_bucket.setdefault(self.bucket(key), []).append((key, score, new_rows[j]))
        victims = []
        for b in sorted(by_bucket):
            new = by_bucket[b]
            members = self.buckets[b]
            s0 = len(members)
            free_slots = self.cap - s0
            need = max(0, len(new) - free_slots)
            # frequency, then timestamp, then slot
            rank = lambda s: (members[s][1][1], members[s][1][0], s)
            chosen = [s for s in sorted(range(s0), key=rank)][:need]
            for key, score, row in new[:free_slots]:
                members.append([key, score, row])
            for slot, (key, score, row) in zip(chosen, new[free_slots:]):
                victims.append((members[slot][0], members[slot][2]))
                members[slot] = [key, score, row]
        return new_rows, [p for p, _, _ in staged], victims

    def flat(self):
        return [(e[0], e[1][0], e[1][1], e[2]) for b in self.buckets for e in b]


def _pair_state(table):
    keys, scores, rows = table.export()
    return list(zip(keys.tolist(), scores[:, 0].tolist(), scores[:, 1].tolist(), rows.tolist()))


@pytest.mark.parametrize(
    "shape", [(1, 8, 6), (4, 16, 24), (16, 32, 150), (1, 128, 100), (1, 600, 500)]
)
def test_a_timestamp_and_frequency_pair_matches_the_model(device, shape):
    nb, cap, per_step = shape
    rng = random.Random(321 + nb)
    table = ChainedTable(num_buckets=nb, bucket_capacity=cap, device=device, num_scores=2)
    model = PairModel(nb, cap, table.row_capacity)
    universe = nb * cap * 3
    clock = 1000
    for step in range(25):
        clock += rng.randrange(1, 50)
        keys = rng.sample(range(1, universe + 1), per_step)
        freqs = None if rng.random() < 0.3 else [rng.randrange(1, 5) for _ in keys]
        rows = table.find_and_stage(
            _t(keys, device), None if freqs is None else _t(freqs, device), "lru_lfu", timestamp=clock
        )
        assert rows.tolist() == model.find_and_stage(keys, freqs, clock)
        out = table.commit()
        new_rows, positions, victims = model.commit()
        assert out.rows.tolist() == new_rows
        assert out.positions.tolist() == positions
        assert list(zip(out.victim_keys.tolist(), out.victim_rows.tolist())) == victims
        table.release_rows(out.victim_rows)
        model.release([r for _, r in victims])
        assert _pair_state(table) == model.flat()
    table.check_invariants()


def test_the_least_used_goes_first_and_of_those_the_oldest(device):
    table = ChainedTable(num_buckets=1, bucket_capacity=4, device=device, num_scores=2)
    keys = _t([1, 2, 3, 4], device)
    # key: (frequency, timestamp) -- 1: (5, 10), 2: (1, 40), 3: (1, 20), 4: (9, 5)
    for key, freq, ts in [(1, 5, 10), (2, 1, 40), (3, 1, 20), (4, 9, 5)]:
        table.find_and_stage(_t([key], device), _t([freq], device), "lru_lfu", timestamp=ts)
        table.commit()
    table.find_and_stage(_t([9], device), None, "lru_lfu", timestamp=100)
    out = table.commit()
    assert out.victim_keys.tolist() == [3]  # frequency 1, and older than key 2
    table.release_rows(out.victim_rows)


def test_a_hit_moves_its_timestamp_and_adds_to_its_frequency(device):
    table = ChainedTable(num_buckets=2, bucket_capacity=8, device=device, num_scores=2)
    k = _t([7], device)
    table.find_and_stage(k, _t([3], device), "lru_lfu", timestamp=10)
    table.commit()
    table.find_and_stage(k, _t([4], device), "lru_lfu", timestamp=99)
    _, scores, _ = table.export()
    assert scores.tolist() == [[99, 7]]


def test_erase_and_growth_keep_both_words(device):
    table = ChainedTable(num_buckets=2, bucket_capacity=16, device=device, num_scores=2)
    keys = list(range(1, 25))
    table.find_and_stage(
        _t(keys, device), _t([k * 3 for k in keys], device), "lru_lfu", timestamp=500
    )
    table.commit()
    table.erase(_t(keys[::4], device))
    table.expand(0)
    k, scores, _ = table.export()
    got = dict(zip(k.tolist(), scores.tolist()))
    for key in keys:
        if key in keys[::4]:
            assert key not in got
        else:
            assert got[key] == [500, key * 3]
    table.check_invariants()


@pytest.mark.parametrize(
    "cap, dtype",
    [(1, torch.int16), (128, torch.int16), (16383, torch.int16), (16384, torch.int32), (100000, torch.int32)],
)
def test_links_are_two_bytes_unless_four_times_the_capacity_does_not_fit(device, cap, dtype):
    table = ChainedTable(num_buckets=1, bucket_capacity=cap, device=device)
    assert table._link_dtype == dtype
    assert table._next[0].dtype == dtype and table._head[0].dtype == dtype
