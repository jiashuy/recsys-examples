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

"""Table-level tests for the LruLfu score policy (2 AoS score words per key:
word 0 = last-access timestamp, word 1 = frequency; eviction ranks by frequency,
the timestamp serves time-based incremental dump). Scores are read back via the
gather_score_blocks primitive so the CUDA kernels are directly observable."""

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import DynamicEmbStorage, _expand_tables_impl
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _lru_lfu_table(capacity, bucket_capacity=128, key_type=torch.int64):
    return get_scored_table(
        capacity=[capacity],
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
    )


def _insert(table, keys, tids, freq, policy=ScorePolicy.LRU_LFU):
    n = keys.numel()
    score_out = torch.empty(n, dtype=torch.int64, device=keys.device)
    insert_results = torch.empty(n, dtype=table.result_type, device=keys.device).fill_(
        InsertResult.INIT.value
    )
    indices = table.insert(
        keys,
        tids,
        ScoreArg(name="frequency", value=freq, policy=policy),
        insert_results,
        score_out=score_out,
    )
    return indices, score_out


def test_lru_lfu_num_scores():
    """LruLfu occupies two physical score words per key."""
    table = _lru_lfu_table(4096)
    assert table.num_scores_ == 2


def test_lru_lfu_frequency_accumulation(current_device):
    """word 1 accumulates the frequency; word 0 is a (monotonic) timestamp; and a
    CONST lookup returns the frequency (word 1), not the timestamp (word 0)."""
    device = torch.cuda.current_device()
    table = _lru_lfu_table(4096)
    n = 100
    keys = torch.arange(1000, 1000 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.ones(n, dtype=torch.uint64, device=device)

    indices, score_out = _insert(table, keys, tids, freq)
    assert torch.all(score_out == 1), "frequency after first insert should be 1"

    blocks = table.gather_score_blocks(0, indices)  # [n, 2] = (timestamp, freq)
    assert tuple(blocks.shape) == (n, 2)
    ts0 = blocks[:, 0].clone()
    assert torch.all(blocks[:, 1] == 1), "word 1 (frequency) should be 1"
    assert torch.all(ts0 > 0), "word 0 (timestamp) should be a device timer value"

    # Each LruLfu lookup is an access: accumulate frequency and re-stamp time.
    out2, founds, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    )
    assert torch.all(founds)
    assert torch.all(out2 == 2), "frequency after one lookup should be 2"

    out3, _, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    )
    assert torch.all(out3 == 3)

    # CONST (eval) lookup must return the reduction score == frequency (word 1),
    # NOT word 0 (the timestamp).
    out_const, _, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert torch.all(
        out_const == 3
    ), "CONST lookup must return frequency (word 1), not the timestamp"

    blocks2 = table.gather_score_blocks(0, indices)
    assert torch.all(blocks2[:, 1] == 3)
    assert torch.all(
        blocks2[:, 0] >= ts0
    ), "timestamp (word 0) must be non-decreasing after further accesses"


def test_lru_lfu_eviction_by_frequency(current_device):
    """Eviction ranks by frequency (word 1): a small set of frequently-accessed
    keys survives even though newly inserted keys have more recent timestamps.
    Exercises the 2-score reduce() path. The hot set is kept well below capacity
    so it is never forced out just for lack of room -- only the frequency ranking
    should matter."""
    device = torch.cuda.current_device()
    bc = 128
    table = _lru_lfu_table(bc, bucket_capacity=bc)  # single bucket: all keys compete

    # Fill ~80% of the bucket with frequency-1 keys.
    n = 100
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx, _ = _insert(table, keys, tids, ones)

    # Boost a small "hot" subset (10 keys) far above the rest.
    n_hot = 10
    hot = keys[:n_hot]
    hot_tids = tids[:n_hot]
    hot_freq = torch.ones(n_hot, dtype=torch.uint64, device=device)
    for _ in range(20):
        table.lookup(
            hot,
            hot_tids,
            ScoreArg(name="frequency", value=hot_freq, policy=ScorePolicy.LRU_LFU),
        )

    # Sanity: the boost actually raised the hot keys' frequency (word 1).
    hot_blocks = table.gather_score_blocks(0, idx[:n_hot])
    assert torch.all(hot_blocks[:, 1] == 21), "hot keys should have frequency 1 + 20"

    # Insert enough brand-new (frequency 1) keys to overflow the bucket and force
    # eviction. There are many more frequency-1 keys (cold + new) than eviction
    # slots, so a correct LFU must evict only those, never the hot keys.
    n_new = 100
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )

    assert num_evicted > 0, "eviction should have occurred (bucket overflowed)"
    evicted_set = set(int(k) for k in evicted_keys.tolist())
    assert evicted_set.isdisjoint(
        set(int(k) for k in hot.tolist())
    ), "high-frequency (hot) keys must not be evicted by LFU"

    # The hot keys must still be present.
    _, founds, _ = table.lookup(
        hot, hot_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert torch.all(founds), "high-frequency keys must survive LFU eviction"


def test_lru_lfu_gather_scatter_roundtrip(current_device):
    """gather_score_blocks -> scatter_score_blocks -> gather_score_blocks preserves
    both score words (used by dump/load)."""
    device = torch.cuda.current_device()
    n = 50
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(
        torch.uint64
    )  # varied

    src = _lru_lfu_table(4096)
    idx_src, _ = _insert(src, keys, tids, freq)
    blocks = src.gather_score_blocks(0, idx_src)

    dst = _lru_lfu_table(4096)
    # Place keys without touching scores (CONST), then restore the full block.
    idx_dst, _ = _insert(dst, keys, tids, None, policy=ScorePolicy.CONST)
    dst.scatter_score_blocks(0, idx_dst, blocks)
    assert torch.equal(
        dst.gather_score_blocks(0, idx_dst), blocks
    ), "scatter then gather must round-trip both score words"


def test_lru_lfu_copy_score_blocks_roundtrip(current_device):
    """copy_score_blocks_from copies all score words between tables of different
    capacity (the rehash primitive)."""
    device = torch.cuda.current_device()
    n = 50
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(torch.uint64)

    src = _lru_lfu_table(4096)
    idx_src, _ = _insert(src, keys, tids, freq)
    blocks = src.gather_score_blocks(0, idx_src)

    # Rehash target: a larger table (keys re-hash into different buckets).
    dst = _lru_lfu_table(8192)
    idx_dst, _ = _insert(dst, keys, tids, None, policy=ScorePolicy.CONST)
    dst.copy_score_blocks_from(src, 0, idx_src, idx_dst)
    assert torch.equal(
        dst.gather_score_blocks(0, idx_dst), blocks
    ), "copy_score_blocks_from must preserve both score words across tables"


def _evict_high_freq_first(scores, cur_timestamp, gamma):
    # scores[0] = timestamp, scores[1] = frequency. Eviction removes the LOWEST
    # returned score, so returning -freq makes the HIGHEST-frequency key evict
    # first -- the opposite of the default LFU ranking. If this function is
    # actually used, the hot (high-frequency) keys are the ones evicted.
    return -float(scores[1])


def test_lru_lfu_custom_score_function_overrides_ranking(current_device):
    """A custom score_function (JIT-compiled + nvJitLink-linked) drives eviction
    instead of the default freq->ts policy. This decay evicts the HIGHEST
    frequency first, so the hot keys -- which the default LFU would keep -- are
    the ones evicted, proving the user function is used."""
    from dynamicemb.jit import register_score_function

    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    key = register_score_function(_evict_high_freq_first, cc[0], cc[1])

    bc = 128
    table = get_scored_table(
        capacity=[bc],
        bucket_capacity=bc,
        key_type=torch.int64,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
        score_fn_key=key,
    )

    n = 100
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx, _ = _insert(table, keys, tids, ones)

    # Boost a small hot subset far above the rest.
    n_hot = 10
    hot = keys[:n_hot]
    hot_tids = tids[:n_hot]
    for _ in range(20):
        table.lookup(
            hot,
            hot_tids,
            ScoreArg(name="frequency", value=ones[:n_hot], policy=ScorePolicy.LRU_LFU),
        )
    hot_blocks = table.gather_score_blocks(0, idx[:n_hot])
    assert torch.all(hot_blocks[:, 1] == 21), "hot keys should have frequency 21"

    # Overflow the bucket to force eviction.
    n_new = 100
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )
    assert num_evicted > 0, "eviction should have occurred"

    # Custom decay evicts highest frequency first -> the hot keys are gone
    # (the default LFU policy would have kept them).
    _, founds_hot, _ = table.lookup(
        hot, hot_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert not torch.any(
        founds_hot
    ), "custom score_function should evict the HIGH-frequency (hot) keys first"


def _evict_oldest(scores, cur_timestamp, gamma):
    # Pure LRU: rank by recency only, ignoring frequency. age = elapsed time
    # since last access = cur_timestamp - scores[0] (a non-negative uint64 -- note
    # the order: scores[0] - cur_timestamp would underflow). Eviction removes the
    # LOWEST score, so returning -age evicts the LARGEST age (the oldest key).
    return -float(cur_timestamp - scores[0])


def test_lru_lfu_custom_decay_by_timestamp(current_device):
    """A timestamp-based decay (age = cur_timestamp - scores[0]) evicts the OLDEST
    key even when it has the HIGHEST frequency -- exercises the timestamp
    subtraction path (must not underflow) and proves a recency decay overrides the
    default LFU ranking (which would keep the high-frequency key)."""
    from dynamicemb.jit import register_score_function

    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    key = register_score_function(_evict_oldest, cc[0], cc[1])

    bc = 128
    table = get_scored_table(
        capacity=[bc],
        bucket_capacity=bc,
        key_type=torch.int64,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
        score_fn_key=key,
    )

    n = 110
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx, _ = _insert(table, keys, tids, ones)

    # Make keys[0] "hot but old": boost its frequency early, then never touch it.
    hot = keys[:1]
    hot_tid = tids[:1]
    for _ in range(20):
        table.lookup(
            hot, hot_tid, ScoreArg(name="frequency", value=ones[:1], policy=ScorePolicy.LRU_LFU)
        )
    torch.cuda.synchronize()
    assert table.gather_score_blocks(0, idx[:1])[0, 1] == 21, "hot key freq == 21"

    # Re-access everyone else LATER so their last-access timestamps are newer than
    # keys[0]'s -- keys[0] is now strictly the oldest.
    rest = keys[1:]
    rest_tid = tids[1:]
    table.lookup(
        rest, rest_tid, ScoreArg(name="frequency", value=ones[1:], policy=ScorePolicy.LRU_LFU)
    )
    torch.cuda.synchronize()

    # Overflow the bucket. Pure-LRU decay evicts the oldest (keys[0]) despite its
    # frequency being the highest -- the default LFU policy would keep it.
    n_new = 60
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, _, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )
    assert num_evicted > 0, "eviction should have occurred"

    _, founds_hot, _ = table.lookup(
        hot, hot_tid, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert not torch.any(
        founds_hot
    ), "timestamp decay must evict the OLDEST key even though it has the highest frequency"


# ---------------------------------------------------------------------------
# Storage-level integration: dump/load round-trip and rehash preservation for
# the LruLfu layout (LFU + need_incremental_dump). These exercise the Python
# glue around the gather/scatter/copy kernels (export_keys_values_iter,
# _dump_table, _iter_batches_from_files, _load_key_values, _expand_tables_impl).
# ---------------------------------------------------------------------------


def _lru_lfu_storage(dim=8, max_capacity=4096, init_capacity=None):
    device_id = torch.cuda.current_device()
    opts = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=dim,
            max_capacity=max_capacity,
            # Constructing DynamicEmbStorage directly bypasses the batched layer's
            # planner/_create_score, so set init_capacity and the LFU evict
            # strategy explicitly (batched normally derives these).
            init_capacity=init_capacity if init_capacity is not None else max_capacity,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.LFU,
            evict_strategy=DynamicEmbEvictStrategy.LFU,
            need_incremental_dump=True,
        )
    ]
    return DynamicEmbStorage(opts, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))


def _read_score_blocks(storage, keys, tids):
    """Look up keys and gather their full [N, num_scores] score blocks."""
    name = storage._state.score_policy.name
    _, founds, idx = storage.key_index_map.lookup(
        keys, tids, ScoreArg(name=name, policy=ScorePolicy.CONST)
    )
    assert torch.all(founds), "all keys must be present"
    return storage.key_index_map.gather_score_blocks(0, idx)


def test_lru_lfu_storage_dump_load_roundtrip(current_device, tmp_path):
    """DynamicEmbStorage dump -> load preserves BOTH LruLfu score words
    (timestamp and frequency), not just word 0."""
    device = torch.cuda.current_device()
    dim = 8
    n = 60
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    values = torch.randn(n, dim, dtype=torch.float32, device=device)
    freq_ref = torch.arange(1, 1 + n, dtype=torch.int64, device=device)  # varied

    src = _lru_lfu_storage(dim=dim)
    src.insert(keys, tids, values, scores=freq_ref.to(torch.uint64))

    blocks_src = _read_score_blocks(src, keys, tids)
    assert torch.all(
        blocks_src[:, 1] == freq_ref
    ), "insert should set word 1 (frequency) to the provided value"

    d = str(tmp_path)
    paths = [f"{d}/meta.json", f"{d}/keys", f"{d}/emb", f"{d}/score", f"{d}/opt"]
    src.dump(0, *paths, include_optim=False)

    dst = _lru_lfu_storage(dim=dim)
    dst.load(0, *paths, include_optim=False)

    blocks_dst = _read_score_blocks(dst, keys, tids)
    assert torch.equal(
        blocks_dst, blocks_src
    ), "dump/load must preserve both score words (timestamp + frequency)"


def test_lru_lfu_storage_rehash_preserves_frequency(current_device):
    """Growing the table (_expand_tables_impl re-hashes keys into more buckets)
    must preserve both LruLfu score words -- a single-value re-insert would reset
    the frequency to 0."""
    device = torch.cuda.current_device()
    dim = 8
    bc = 128
    # Start with a single bucket, leave room to grow.
    storage = _lru_lfu_storage(dim=dim, max_capacity=8 * bc, init_capacity=bc)
    n = 80
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    values = torch.randn(n, dim, dtype=torch.float32, device=device)
    freq_ref = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    storage.insert(keys, tids, values, scores=freq_ref.to(torch.uint64))

    before = _read_score_blocks(storage, keys, tids)
    assert torch.all(before[:, 1] == freq_ref)

    # Rehash: grow the single logical table to 8 buckets; keys re-hash and are
    # re-inserted, and copy_score_blocks must carry both words across.
    _expand_tables_impl(storage._state, [True], [8 * bc])

    after = _read_score_blocks(storage, keys, tids)
    assert torch.equal(
        after, before
    ), "rehash must preserve both score words (timestamp + frequency)"


def _lru_lfu_strategy_storage(
    dim=8, max_capacity=4096, init_capacity=None, score_function=None, gamma=1.0
):
    """DynamicEmbStorage built with the LRU_LFU *strategy* (vs LFU +
    need_incremental_dump). Exercises create_table_state's LRU_LFU branch,
    including score_function auto-registration. Constructing storage directly
    bypasses the batched _create_score, so set evict_strategy=LFU explicitly."""
    device_id = torch.cuda.current_device()
    opts = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=dim,
            max_capacity=max_capacity,
            init_capacity=init_capacity if init_capacity is not None else max_capacity,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.LRU_LFU,
            evict_strategy=DynamicEmbEvictStrategy.LFU,
            score_function=score_function,
            lfu_decay_gamma=gamma,
        )
    ]
    return DynamicEmbStorage(opts, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))


def test_lru_lfu_strategy_create_table_state_wiring(current_device):
    """create_table_state maps LRU_LFU -> the compound LruLfu policy (2 score
    words); without a score_function the table uses the default evictor
    (score_fn_key == 0), and with one create_table_state auto-registers it
    (numba + nvJitLink) and threads a nonzero key + gamma to the table."""
    default = _lru_lfu_strategy_storage()
    assert default.key_index_map.num_scores_ == 2
    assert default.key_index_map.score_fn_key_ == 0, "no score_function => key 0"

    custom = _lru_lfu_strategy_storage(score_function=_evict_high_freq_first, gamma=0.9)
    assert custom.key_index_map.num_scores_ == 2
    assert (
        custom.key_index_map.score_fn_key_ != 0
    ), "score_function must be registered and routed via a nonzero key"
    assert custom.key_index_map.lfu_decay_gamma_ == 0.9


def test_score_function_requires_lru_lfu():
    """score_function is only valid with score_strategy=LRU_LFU."""
    with pytest.raises(ValueError):
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=8,
            max_capacity=128,
            score_strategy=DynamicEmbScoreStrategy.LFU,
            score_function=_evict_high_freq_first,
        )
