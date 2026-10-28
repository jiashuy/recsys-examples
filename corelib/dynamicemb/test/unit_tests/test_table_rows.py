# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A table built with ``use_rows`` hands out value rows, not hash slots.

A slot takes the next row of its table the first time a key occupies it and
keeps that row for good: the key that evicts the occupant, or reuses the slot
after an erase, inherits it. So rows stay dense (``[0, num_rows)``) and a
rehash can carry them along instead of moving values.
"""

import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy

BUCKET = 16


def _table(capacity, use_rows=True):
    caps = [capacity] if isinstance(capacity, int) else list(capacity)
    return get_scored_table(
        capacity=caps,
        bucket_capacity=BUCKET,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
        use_rows=use_rows,
    )


def _args(table, keys, table_id=0, score=1):
    device = keys.device
    tids = torch.full((keys.numel(),), table_id, dtype=torch.int64, device=device)
    scores = torch.full((keys.numel(),), score, dtype=torch.uint64, device=device)
    return tids, ScoreArg(name="score", value=scores, policy=ScorePolicy.ASSIGN)


def _keys(lo, hi):
    return torch.arange(lo, hi, dtype=torch.int64, device="cuda")


def test_rows_are_dense_and_stable():
    table = _table(1024)
    keys = _keys(0, 300)
    tids, score = _args(table, keys)
    rows = table.insert(keys, tids, score)

    assert sorted(rows.tolist()) == list(range(300))
    assert table.num_rows(0) == 300
    _, founds, looked_up = table.lookup(keys, tids, score)
    assert bool(founds.all())
    assert torch.equal(looked_up, rows)

    # Re-inserting existing keys neither moves nor allocates.
    assert torch.equal(table.insert(keys, tids, score), rows)
    assert table.num_rows(0) == 300


def test_rows_are_per_table():
    table = _table([256, 256])
    k0, k1 = _keys(0, 40), _keys(1000, 1050)
    for tid, keys in ((0, k0), (1, k1)):
        tids, score = _args(table, keys, tid)
        rows = table.insert(keys, tids, score)
        assert sorted(rows.tolist()) == list(range(keys.numel()))
    assert (table.num_rows(0), table.num_rows(1)) == (40, 50)


def test_eviction_reuses_the_victims_row():
    # One bucket: filling it and then inserting more keys must evict, and the
    # newcomer takes the row of whoever it displaced.
    table = _table(BUCKET)
    first = _keys(0, BUCKET)
    tids, score = _args(table, first, score=1)
    table.insert(first, tids, score)
    assert table.num_rows(0) == BUCKET

    newcomers = _keys(100, 100 + BUCKET // 2)
    tids, score = _args(table, newcomers, score=10)
    rows, num_evicted, _, ev_rows, _, _ = table.insert_and_evict(newcomers, tids, score)
    assert num_evicted == newcomers.numel()
    assert table.num_rows(0) == BUCKET, "eviction must not allocate"
    # The victims held exactly the rows the newcomers now hold.
    assert sorted(rows.tolist()) == sorted(ev_rows.tolist())


def test_erased_slot_keeps_its_row_for_the_next_key():
    # One full bucket: erased slots are reclaimed, not empty, and with no empty
    # slot left the next keys must take exactly those, inheriting their rows.
    table = _table(BUCKET)
    keys = _keys(0, BUCKET)
    tids, score = _args(table, keys)
    rows = table.insert(keys, tids, score)
    assert table.num_rows(0) == BUCKET

    gone = keys[:4]
    gone_tids, _ = _args(table, gone)
    present = table.erase(gone, gone_tids, return_indices=True)
    assert bool((present >= 0).all())
    assert table.num_rows(0) == BUCKET, "erase does not hand rows back"
    again = table.erase(gone, gone_tids, return_indices=True)
    assert bool((again < 0).all()), "erasing a missing key reports -1"

    fresh = _keys(100, 104)
    tids, score = _args(table, fresh)
    fresh_rows = table.insert(fresh, tids, score)
    assert sorted(fresh_rows.tolist()) == sorted(rows[:4].tolist())
    assert table.num_rows(0) == BUCKET, "reclaimed rows are reused, not re-allocated"


def test_copy_table_from_keeps_rows_in_place():
    # Same capacity: every column is copied slot for slot, rows included.
    src = _table([BUCKET * 4, BUCKET * 4])
    keys = _keys(0, 30)
    tids, score = _args(src, keys, table_id=1)
    rows = src.insert(keys, tids, score)

    dst = _table([BUCKET * 4, BUCKET * 4])
    dst.copy_table_from(src, 1)
    _, founds, copied = dst.lookup(keys, tids, score)
    assert bool(founds.all())
    assert torch.equal(copied, rows)
    assert dst.num_rows(1) == src.num_rows(1) == 30
    assert dst.num_rows(0) == 0


@pytest.mark.parametrize("erased", [0, 100])
def test_rehash_carries_rows_and_reclaimed_slots(erased):
    # Erased keys leave reclaimed slots that still own their rows. A rehash must
    # keep every live key's row and re-place each reclaimed row in a reclaimed
    # slot of the new table, so no value has to move and no row is lost.
    old = _table(BUCKET * 64)
    keys = _keys(0, BUCKET * 16)
    tids, score = _args(old, keys)
    rows = old.insert(keys, tids, score)
    gone, live = keys[:erased], keys[erased:]
    if erased:
        old.erase(gone, _args(old, gone)[0])
    live_tids, live_score = _args(old, live)
    _, _, live_rows = old.lookup(live, live_tids, live_score)

    new = _table(BUCKET * 128)
    new.insert_carrying_rows(live, 0, live_score, live_rows)
    new.carry_rows_from(old, 0)

    # Reclaimed slots of the new table hold exactly the erased keys' rows.
    reclaim_key = -2
    keys_col = new.keys_.view(torch.int64)
    rows_col = new.rows_.view(new.num_buckets_, BUCKET)
    reclaimed = keys_col == reclaim_key
    assert int(reclaimed.sum()) == erased
    assert sorted(rows_col[reclaimed].tolist()) == sorted(rows[:erased].tolist())
    assert new.num_rows(0) == old.num_rows(0)

    # The live keys keep their rows ...
    _, founds, new_rows = new.lookup(live, live_tids, live_score)
    assert bool(founds.all()) and torch.equal(new_rows, live_rows)

    # ... and new keys never collide with them: every row is handed to one key.
    fresh = _keys(10_000, 10_000 + 50)
    tids, score = _args(new, fresh)
    fresh_rows = new.insert(fresh, tids, score)
    all_rows = fresh_rows.tolist() + live_rows.tolist()
    assert len(all_rows) == len(set(all_rows))
    assert max(all_rows) < new.num_rows(0)
