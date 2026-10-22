# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers for the capacity-expansion test of BatchedDynamicEmbeddingTablesV2.

Strict-expansion sizing constants, an FBGEMM reference fed the same debug
initial weights, and checks that a table's cache/storage export matches what a
lookup returns and what the reference holds.
"""

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from dynamicemb import DynamicEmbInitializerMode, DynamicEmbScoreStrategy
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import DEBUG_EMB_INITIALIZER_MOD, ScoreStrategy
from dynamicemb.key_value_table import (
    DynamicEmbCache,
    DynamicEmbStorage,
    DynamicEmbTableState,
    HybridStorage,
    Storage,
    _flat_row_indices_from_slots_and_scores,
    export_keys_values_iter,
    load_from_flat,
    load_from_flat_single_table,
)
from dynamicemb.types import CopyMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

# Strict table expansion: modest max_capacity so we can insert every slot exactly once.
_CACHE_RATIO = 0.25

_WORLD_SIZE = 1

_EMB_DIM_EXPANSION = 8

_BUCKET_CAPACITY_EXP = 128

# Global num_embeddings for single-table strict expansion (need not be multiples of _BUCKET_CAPACITY_EXP).
_STRICT_SINGLE_NUM_EMB = 20_000

# Per-table global num_embeddings (need not be multiples of _BUCKET_CAPACITY_EXP).
_STRICT_MULTI_NUM_EMB = (20_000, 30_000, 40_000)

# Same as ``DEBUG_EMB_INITIALIZER_MOD`` / ``debug_init`` (initializer.cu).
_DEBUG_EMB_MOD = DEBUG_EMB_INITIALIZER_MOD

# Expansion tests assume key_index_map rehash threshold at load factor 0.5.
_STRICT_MAX_LOAD_FACTOR = 0.5


def _init_capacity_strict(max_capacity: int, bucket_cap: int) -> int:
    """init_capacity ≈ max_capacity // 16, rounded down to a multiple of ``bucket_cap``; always < max_capacity."""
    if max_capacity <= bucket_cap:
        return bucket_cap
    q = max_capacity // 16
    init_cap = (q // bucket_cap) * bucket_cap
    if init_cap < bucket_cap:
        init_cap = bucket_cap
    if init_cap >= max_capacity:
        init_cap = ((max_capacity - 1) // bucket_cap) * bucket_cap
        if init_cap < bucket_cap:
            init_cap = bucket_cap
    return init_cap


def _embedding_configs_for_strict_expansion_test(
    multi_table: bool,
) -> Tuple[List[Any], List[str], List[int]]:
    from torchrec import DataType
    from torchrec.modules.embedding_configs import EmbeddingConfig

    if not multi_table:
        names = ["t0"]
        num_embs = [_STRICT_SINGLE_NUM_EMB]
        feature_table_map = [0]
    else:
        names = ["t0", "t1", "t2"]
        num_embs = list(_STRICT_MULTI_NUM_EMB)
        feature_table_map = [0, 1, 2]
    configs = [
        EmbeddingConfig(
            name=names[i],
            embedding_dim=_EMB_DIM_EXPANSION,
            num_embeddings=num_embs[i],
            feature_names=[f"f_{names[i]}"],
            data_type=DataType.FP32,
        )
        for i in range(len(names))
    ]
    return configs, names, feature_table_map


def _iter_dynamic_emb_table_states(
    bdebt: BatchedDynamicEmbeddingTablesV2,
) -> List[Tuple[str, DynamicEmbTableState]]:
    """All table states (cache + backing, or hybrid tiers)."""
    out: List[Tuple[str, DynamicEmbTableState]] = []
    cache = bdebt.cache
    if isinstance(cache, DynamicEmbCache):
        out.append(("cache", cache._state))
    st = bdebt.tables
    if isinstance(st, DynamicEmbStorage):
        out.append(("storage", st._state))
    elif isinstance(st, HybridStorage):
        out.append(("hbm", st.tables[0]))
        out.append(("host", st.tables[1]))
    return out


def _lfu_per_sample_weights(
    indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return torch.ones(indices.numel(), dtype=torch.int64, device=device)


# Chunk size when iterating keys 0..cmp_capacity-1; align with table ``bucket_capacity``.
_EXPANSION_STBE_TRAIN_BATCH = _BUCKET_CAPACITY_EXP


def _init_stbe_debug_embedding_weights(
    stbe: SplitTableBatchedEmbeddingBagsCodegen,
) -> None:
    """Match DynamicEmb ``DEBUG`` init: row ``i`` is filled with ``i % _DEBUG_EMB_MOD`` in every dim."""
    with torch.no_grad():
        for w in stbe.split_embedding_weights():
            n, d = w.shape
            if n == 0:
                continue
            vals = (
                torch.arange(n, device=w.device, dtype=torch.float32)
                % float(_DEBUG_EMB_MOD)
            ).view(n, 1)
            w.copy_(vals.expand(-1, d))


def _bdebt_forward_maybe_lfu(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    score_strategy: ScoreStrategy,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if score_strategy == DynamicEmbScoreStrategy.LFU:
        return bdebt(
            indices,
            offsets,
            frequency_counters=_lfu_per_sample_weights(indices, device),
        )
    return bdebt(indices, offsets)


def _iter_compare_batches_single(
    cmp_cap: int,
    batch_size: int,
    device: torch.device,
    key_type: torch.dtype,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for s in range(0, cmp_cap, batch_size):
        e = min(s + batch_size, cmp_cap)
        idx = torch.arange(s, e, device=device, dtype=key_type)
        off = torch.tensor([0, e - s], device=device, dtype=key_type)
        yield idx, off


def _iter_compare_batches_multi_three(
    cmp_caps: List[int],
    batch_size: int,
    device: torch.device,
    key_type: torch.dtype,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Cover local keys ``0 .. cmp_caps[t]-1`` on each of three tables using anchored triples.

    ``BatchedDynamicEmbeddingTablesV2`` / FBGEMM use the same layout as ``get_table_range``:
    for ``batch_size==1`` and three features (tables 0,1,2), ``indices`` must be **feature-major**
    — all keys for table 0, then all for table 1, then all for table 2 — with
    ``offsets = [0, n0, n0+n1, n0+n1+n2]``. Interleaved ``[a0,b0,c0,a1,...]`` with
    ``[0,n,2n,3n]`` is wrong: table 0 would incorrectly receive other tables' keys.
    """
    c0, c1, c2 = cmp_caps[0], cmp_caps[1], cmp_caps[2]
    min_c = min(cmp_caps)

    def pack_triples(
        rows: List[Tuple[int, int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack one micro-batch: rows are (key_table0, key_table1, key_table2) per sample."""
        n = len(rows)
        if n == 0:
            z = torch.tensor([], device=device, dtype=key_type)
            return z, torch.tensor([0, 0, 0, 0], device=device, dtype=key_type)
        col0 = [r[0] for r in rows]
        col1 = [r[1] for r in rows]
        col2 = [r[2] for r in rows]
        flat = col0 + col1 + col2
        idx = torch.tensor(flat, device=device, dtype=key_type)
        n0, n1 = len(col0), len(col1)
        off = torch.tensor(
            [0, n0, n0 + n1, n0 + n1 + len(col2)],
            device=device,
            dtype=key_type,
        )
        return idx, off

    for s in range(0, min_c, batch_size):
        e = min(s + batch_size, min_c)
        rows = [(k, k, k) for k in range(s, e)]
        yield pack_triples(rows)

    a1 = max(min_c - 1, 0)
    a2 = max(min_c - 1, 0)
    for s in range(min_c, c0, batch_size):
        e = min(s + batch_size, c0)
        rows = [(k, a1, a2) for k in range(s, e)]
        yield pack_triples(rows)

    a0 = max(c0 - 1, 0)
    for s in range(min_c, c1, batch_size):
        e = min(s + batch_size, c1)
        rows = [(a0, k, a2) for k in range(s, e)]
        yield pack_triples(rows)

    a1b = max(c1 - 1, 0)
    for s in range(min_c, c2, batch_size):
        e = min(s + batch_size, c2)
        rows = [(a0, a1b, k) for k in range(s, e)]
        yield pack_triples(rows)


def _print_hybrid_hbm_host_sizes(
    storage: Storage,
    table_names: List[str],
    *,
    phase: str,
    only_table_id: Optional[int] = None,
) -> None:
    """Print ``key_index_map.size()`` (occupied) and ``capacity()`` per tier for Hybrid backing."""
    if not isinstance(storage, HybridStorage):
        return
    hbm_state, host_state = storage.tables
    kim_h, kim_o = hbm_state.key_index_map, host_state.key_index_map
    indices = (
        [only_table_id] if only_table_id is not None else list(range(len(table_names)))
    )
    for tid in indices:
        if tid < 0 or tid >= len(table_names):
            continue
        name = table_names[tid]
        hbm_n, host_n = kim_h.size(tid), kim_o.size(tid)
        hbm_cap, host_cap = kim_h.capacity(tid), kim_o.capacity(tid)
        print(
            f"[bdebt vs stbe] {phase} table {name!r} (table_id={tid}):\n"
            f"  Hybrid HBM:   size()={hbm_n}, capacity={hbm_cap}\n"
            f"  Hybrid host:  size()={host_n}, capacity={host_cap}\n"
            f"  sum size()={hbm_n + host_n}",
            flush=True,
        )


def _bdebt_storage_entry_count_for_table(
    storage: Storage, table_id: int
) -> Tuple[int, str]:
    """Count occupied slots and capacity in backing storage for one logical table (post-flush)."""
    if isinstance(storage, HybridStorage):
        hbm_state, host_state = storage.tables
        kim_h, kim_o = hbm_state.key_index_map, host_state.key_index_map
        hbm_n, host_n = kim_h.size(table_id), kim_o.size(table_id)
        hbm_cap, host_cap = kim_h.capacity(table_id), kim_o.capacity(table_id)
        total = hbm_n + host_n
        return total, (
            f"HybridStorage table_id={table_id}: "
            f"HBM size={hbm_n}, capacity={hbm_cap}; "
            f"host size={host_n}, capacity={host_cap}; sum_size={total}"
        )
    if isinstance(storage, DynamicEmbStorage):
        kim = storage.key_index_map
        n, cap = kim.size(table_id), kim.capacity(table_id)
        return n, f"DynamicEmbStorage table_id={table_id}: size={n}, capacity={cap}"
    return -1, f"unknown storage type {type(storage).__name__}"


def _kim_occupied_sizes_per_table(km: Any, num_tables: int) -> List[int]:
    """``key_index_map.size(t)`` per logical table (handles int or 0-dim tensor)."""
    out: List[int] = []
    for t in range(num_tables):
        n_t = km.size(t)
        out.append(int(n_t.item() if isinstance(n_t, torch.Tensor) else n_t))
    return out


def _kim_capacities_per_table(km: Any, num_tables: int) -> List[int]:
    """``key_index_map.capacity(t)`` per logical table (handles int or 0-dim tensor)."""
    out: List[int] = []
    for t in range(num_tables):
        c_t = km.capacity(t)
        out.append(int(c_t.item() if isinstance(c_t, torch.Tensor) else c_t))
    return out


def _print_test_expansion_storage_cache_sizes_after_train_step(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    *,
    outer_iter: int,
    batch_index: int,
) -> None:
    """Print storage/cache KIM ``size`` and ``capacity`` per table after one forward+backward."""
    tag = (
        f"[test_table_expansion_capacity_growth] outer_iter={outer_iter} "
        f"train_batch={batch_index}"
    )
    storage = bdebt.tables
    if isinstance(storage, DynamicEmbStorage):
        km = storage.key_index_map
        nt = storage._state.num_tables
        sz = _kim_occupied_sizes_per_table(km, nt)
        cap = _kim_capacities_per_table(km, nt)
        print(
            f"{tag} storage KIM size (per table): {sz}, capacity (per table): {cap}",
            flush=True,
        )
    elif isinstance(storage, HybridStorage):
        hbm, host = storage.tables
        hkm, okm = hbm.key_index_map, host.key_index_map
        nt_h, nt_o = hbm.num_tables, host.num_tables
        hs, os_ = _kim_occupied_sizes_per_table(
            hkm, nt_h
        ), _kim_occupied_sizes_per_table(okm, nt_o)
        hc, oc = _kim_capacities_per_table(hkm, nt_h), _kim_capacities_per_table(
            okm, nt_o
        )
        print(
            f"{tag} storage HBM KIM size: {hs}, capacity: {hc}; "
            f"host KIM size: {os_}, capacity: {oc}",
            flush=True,
        )
    else:
        print(f"{tag} storage type: {type(storage).__name__}", flush=True)

    cache = bdebt.cache
    if isinstance(cache, DynamicEmbCache):
        ckm = cache.key_index_map
        nt_c = cache._state.num_tables
        csz = _kim_occupied_sizes_per_table(ckm, nt_c)
        ccap = _kim_capacities_per_table(ckm, nt_c)
        print(
            f"{tag} cache KIM size (per table): {csz}, "
            f"capacity (per table): {ccap}",
            flush=True,
        )
    else:
        print(f"{tag} cache: disabled", flush=True)


def _stbe_size_capacity_for_table(
    stbe: SplitTableBatchedEmbeddingBagsCodegen, logical_table_id: int
) -> Tuple[int, int]:
    """Static TBE: allocated rows (``size``) and declared ``rows_per_table`` (``capacity``).

    For dense split tables both are the number of embedding rows; ``rows_per_table`` comes
    from the first feature mapped to ``logical_table_id``.
    """
    w = stbe.split_embedding_weights()[logical_table_id]
    weight_rows = int(w.shape[0])
    ftm = stbe.feature_table_map
    try:
        f_idx = next(i for i, t in enumerate(ftm) if t == logical_table_id)
        declared = int(stbe.rows_per_table[f_idx].item())
    except (StopIteration, AttributeError):
        declared = weight_rows
    return weight_rows, declared


def _dump_bde_stbe_embedding_samples(
    *,
    table_name: str,
    by_key: Dict[int, torch.Tensor],
    w_stbe: torch.Tensor,
    cmp_cap: int,
    max_keys: int = 5,
) -> None:
    """Print a few key rows for BDE export vs STBE (debug: single-table + caching)."""
    candidates = [0, 1, 2, cmp_cap // 2, cmp_cap - 1]
    seen: List[int] = []
    for k in candidates:
        if k in seen or k < 0 or k >= cmp_cap:
            continue
        seen.append(k)
        if len(seen) >= max_keys:
            break
    print(
        f"[bde vs stbe dump] table {table_name!r} sample embeddings "
        f"(cmp_cap={cmp_cap}, showing up to {len(seen)} keys):",
        flush=True,
    )
    for k in seen:
        stbe_row = w_stbe[k].tolist()
        if k in by_key:
            bde_row = by_key[k].tolist()
            print(
                f"  key={k}\n    BDE export: {bde_row}\n    STBE row:   {stbe_row}",
                flush=True,
            )
        else:
            print(
                f"  key={k}\n    BDE export: <missing>\n    STBE row:   {stbe_row}",
                flush=True,
            )


def _assert_cache_export_matches_lookup_embeddings(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    device: torch.device,
    *,
    batch_size: int = 65536,
) -> None:
    """Check GPU cache: exported (key, slot) rows match ``lookup`` + ``load_from_flat``.

    Uses ``training=False`` on cache state so lookup does not mutate scores.
    No-op when ``bdebt.cache`` is None.
    """
    cache = bdebt.cache
    if not isinstance(cache, DynamicEmbCache):
        return
    state = cache._state
    saved_training = state.training
    cache.training = False
    try:
        for tid in range(state.num_tables):
            for keys, emb_exp, _, _ in export_keys_values_iter(
                state, device, batch_size=batch_size, table_id=tid
            ):
                if keys.numel() == 0:
                    continue
                tids = torch.full(
                    (keys.numel(),), tid, dtype=torch.int64, device=keys.device
                )
                _, founds, indices = cache.lookup(keys, tids, None)
                assert bool(
                    founds.all()
                ), f"cache table_id={tid}: every exported key must be found via lookup"
                emb_slots = load_from_flat(
                    state, indices, tids, copy_mode=CopyMode.EMBEDDING
                )
                torch.testing.assert_close(
                    emb_exp.to(dtype=emb_slots.dtype),
                    emb_slots,
                    rtol=1e-5,
                    atol=1e-5,
                )
    finally:
        cache.training = saved_training


def _assert_storage_export_matches_find_embeddings(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    device: torch.device,
    *,
    batch_size: int = 65536,
) -> None:
    """After flush/insert into backing ``DynamicEmbStorage``, check key↔embedding consistency.

    For each export batch, ``find(keys)`` must return the same embeddings as
    ``export_keys_values_iter`` (read-only lookup via ``training=False``).
    If the table uses ``DynamicEmbInitializerMode.DEBUG``, also checks that both
    export and find embeddings equal ``key % DEBUG_EMB_INITIALIZER_MOD`` per dim.
    Skips non-``DynamicEmbStorage`` backends (e.g. ``HybridStorage``).
    """
    storage = bdebt.tables
    if not isinstance(storage, DynamicEmbStorage):
        return
    state = storage._state
    saved_training = state.training
    storage.training = False
    try:
        for tid in range(state.num_tables):
            debug_table = (
                state.options_list[tid].initializer_args.mode
                == DynamicEmbInitializerMode.DEBUG
            )
            emb_dim_t = state.table_emb_dims_cpu[tid]
            for keys, emb_exp, _, _ in storage.export_keys_values(
                device, batch_size, tid
            ):
                if keys.numel() == 0:
                    continue
                tids = torch.full(
                    (keys.numel(),), tid, dtype=torch.int64, device=keys.device
                )
                (
                    h_miss,
                    _,
                    _,
                    _,
                    _,
                    founds,
                    _,
                    emb_find,
                ) = storage.find(keys, tids, CopyMode.EMBEDDING)
                assert h_miss == 0, (
                    f"storage table_id={tid}: find reports {h_miss} missing keys "
                    "that export still lists"
                )
                assert bool(
                    founds.all()
                ), f"storage table_id={tid}: find must locate every exported key"
                expected_dbg: Optional[torch.Tensor] = None
                if debug_table:
                    mod = torch.tensor(
                        float(DEBUG_EMB_INITIALIZER_MOD),
                        device=keys.device,
                        dtype=torch.float32,
                    )
                    kf = keys.to(dtype=torch.float32)
                    expected_dbg = (kf % mod).unsqueeze(1).expand(-1, emb_dim_t)
                if os.environ.get("DYNAMICEMB_PRINT_STORAGE_EXPORT_FIND") == "1":
                    ze = emb_exp.to(dtype=emb_find.dtype)
                    parts = [
                        "[storage export vs find]",
                        f"table_id={tid}",
                        f"batch_n={keys.numel()}",
                        f"max_abs(exp-find)={(ze - emb_find).abs().max().item()}",
                    ]
                    if expected_dbg is not None:
                        ee32 = emb_exp.to(device=keys.device, dtype=torch.float32)
                        ef32 = emb_find.to(device=keys.device, dtype=torch.float32)
                        parts.append(
                            f"max_abs(exp-key%mod)={(ee32 - expected_dbg).abs().max().item()}"
                        )
                        parts.append(
                            f"max_abs(find-key%mod)={(ef32 - expected_dbg).abs().max().item()}"
                        )
                    print(*parts, flush=True)

                torch.testing.assert_close(
                    emb_exp.to(dtype=emb_find.dtype),
                    emb_find,
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    emb_find.to(dtype=emb_find.dtype),
                    expected_dbg,
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    emb_exp.to(dtype=emb_find.dtype),
                    expected_dbg,
                    rtol=1e-5,
                    atol=1e-5,
                )

                if debug_table:
                    assert expected_dbg is not None
                    torch.testing.assert_close(
                        emb_exp.to(device=keys.device, dtype=torch.float32),
                        expected_dbg,
                        rtol=0.0,
                        atol=0.0,
                        msg=lambda m: (
                            f"storage table_id={tid}: export embedding != key % "
                            f"{DEBUG_EMB_INITIALIZER_MOD}: {m}"
                        ),
                    )
                    torch.testing.assert_close(
                        emb_find.to(device=keys.device, dtype=torch.float32),
                        expected_dbg,
                        rtol=0.0,
                        atol=0.0,
                        msg=lambda m: (
                            f"storage table_id={tid}: find embedding != key % "
                            f"{DEBUG_EMB_INITIALIZER_MOD}: {m}"
                        ),
                    )
    finally:
        storage.training = saved_training


def _kim_scan_keys_embeddings_dict(
    state: DynamicEmbTableState,
    device: torch.device,
    *,
    batch_size: int = 65536,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Scan KIM + value buffer: (table_id, key) -> embedding.

    Does not call ``Storage.export_keys_values`` / ``export_keys_values_iter``;
    uses ``_batched_export_keys_scores`` and :func:`load_from_flat_single_table` only.
    """
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    score_name = state.score_policy.name
    for tid in range(state.num_tables):
        emb_dim_t = state.table_emb_dims_cpu[tid]
        for (
            keys,
            named_scores,
            indices,
        ) in state.key_index_map._batched_export_keys_scores(
            [score_name],
            state.device,
            batch_size=batch_size,
            return_index=True,
            table_id=tid,
        ):
            if keys.numel() == 0:
                continue
            scores = named_scores[score_name]
            flat_rows = _flat_row_indices_from_slots_and_scores(state, indices, scores)
            values = load_from_flat_single_table(state, flat_rows, tid)
            embeddings = values[:, :emb_dim_t].contiguous().to(device)
            keys_dev = keys.to(device=device)
            for i in range(keys.numel()):
                k_int = int(keys_dev[i].item())
                out[(tid, k_int)] = embeddings[i].detach().clone()
    return out


def _assert_bdebt_storage_cache_merged_export_matches_lookup(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    device: torch.device,
    *,
    batch_size: int = 65536,
    verify_batch_size: int = 4096,
) -> None:
    """No flush: merge storage then cache embeddings into a dict; check vs live lookup/find.

    - Fills a dict from backing ``DynamicEmbStorage`` KIM scan, then overlays cache scan
      (cache wins on key collision).
    - Verifies every merged (table_id, key) matches ``cache.lookup`` + ``load_from_flat``
      when present in cache, else ``storage.find`` (embedding column).
    """
    storage = bdebt.tables
    cache = bdebt.cache
    if not isinstance(storage, DynamicEmbStorage) or not isinstance(
        cache, DynamicEmbCache
    ):
        return

    saved_st = storage.training
    saved_ca = cache.training
    storage.training = False
    cache.training = False
    try:
        storage_dict = _kim_scan_keys_embeddings_dict(
            storage._state, device, batch_size=batch_size
        )
        cache_dict = _kim_scan_keys_embeddings_dict(
            cache._state, device, batch_size=batch_size
        )
        merged: Dict[Tuple[int, int], torch.Tensor] = {**storage_dict, **cache_dict}

        for k, v in cache_dict.items():
            assert k in merged, f"cache key {k} missing from merged dict"
            torch.testing.assert_close(merged[k], v, rtol=1e-5, atol=1e-5)
        for k, v in storage_dict.items():
            if k in cache_dict:
                torch.testing.assert_close(
                    merged[k], cache_dict[k], rtol=1e-5, atol=1e-5
                )
            else:
                torch.testing.assert_close(merged[k], v, rtol=1e-5, atol=1e-5)

        st = storage._state
        ca = cache._state
        key_type = st.key_index_map.key_type
        by_tid: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
        for (tid, k_int), emb in merged.items():
            by_tid.setdefault(tid, []).append((k_int, emb))

        for tid, pairs in by_tid.items():
            for s in range(0, len(pairs), verify_batch_size):
                chunk = pairs[s : s + verify_batch_size]
                ks = [p[0] for p in chunk]
                ref = torch.stack([p[1] for p in chunk])
                key_t = torch.tensor(ks, device=device, dtype=key_type)
                tid_t = torch.full((len(ks),), tid, dtype=torch.int64, device=device)
                _, founds_c, ix_c = cache.lookup(key_t, tid_t, None)
                emb_live = torch.empty_like(ref)
                if bool(founds_c.any()):
                    emb_live[founds_c] = load_from_flat(
                        ca,
                        ix_c[founds_c],
                        tid_t[founds_c],
                        copy_mode=CopyMode.EMBEDDING,
                    )
                miss = ~founds_c
                if bool(miss.any()):
                    (
                        h_miss,
                        _,
                        _,
                        _,
                        _,
                        sf,
                        _,
                        sv,
                    ) = storage.find(
                        key_t[miss],
                        tid_t[miss],
                        copy_mode=CopyMode.EMBEDDING,
                    )
                    assert h_miss == 0, (
                        f"table_id={tid}: storage.find reports {h_miss} missing "
                        f"among keys not in cache"
                    )
                    assert bool(
                        sf.all()
                    ), f"table_id={tid}: storage.find must hit all keys absent from cache"
                    emb_live[miss] = sv
                torch.testing.assert_close(ref, emb_live, rtol=1e-5, atol=1e-5)
    finally:
        storage.training = saved_st
        cache.training = saved_ca


def _compare_bdebt_export_to_stbe_weights(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    stbe: SplitTableBatchedEmbeddingBagsCodegen,
    table_names: List[str],
    cmp_caps: List[int],
    device: torch.device,
    *,
    dump_sample_embeddings: bool = False,
) -> None:
    # Match export_keys_values: flush so storage counts reflect the same backing state.
    bdebt.flush()
    torch.cuda.synchronize()
    # Key↔embedding mapping: cache tier (still hot after flush) then backing storage.
    _assert_cache_export_matches_lookup_embeddings(bdebt, device, batch_size=65536)
    _assert_storage_export_matches_find_embeddings(bdebt, device, batch_size=65536)
    _print_hybrid_hbm_host_sizes(
        bdebt.tables,
        table_names,
        phase="Hybrid tier sizes (after flush, before export)",
    )
    for tid, name in enumerate(table_names):
        _, st_detail = _bdebt_storage_entry_count_for_table(bdebt.tables, tid)
        if bdebt.cache is not None:
            ckim = bdebt.cache.key_index_map
            cache_sz, cache_cap = ckim.size(tid), ckim.capacity(tid)
            cache_line = (
                f"  Cache:    table_id={tid} size={cache_sz}, capacity={cache_cap}\n"
            )
        else:
            cache_line = "  Cache:    (none)\n"
        stbe_sz, stbe_cap = _stbe_size_capacity_for_table(stbe, tid)
        print(
            f"[bdebt vs stbe] table {name!r} (after flush, before export):\n"
            f"{cache_line}"
            f"  Storage:  {st_detail}\n"
            f"  STBE:     size={stbe_sz}, capacity={stbe_cap} "
            f"(split_embedding_weights rows={stbe_sz}, rows_per_table={stbe_cap})\n"
            f"  cmp_cap={cmp_caps[tid]}",
            flush=True,
        )

    for tid, name in enumerate(table_names):
        cmp_cap = cmp_caps[tid]
        keys_t, emb_t = bdebt.export_keys_values(name, device, batch_size=65536)
        exported_n = int(keys_t.numel())
        stbe_sz, stbe_cap = _stbe_size_capacity_for_table(stbe, tid)
        print(
            f"[bdebt vs stbe] table {name!r} (after export_keys_values):\n"
            f"  BDE export: size={exported_n} (unique keys in backing export)\n"
            f"  STBE:       size={stbe_sz}, capacity={stbe_cap}\n"
            f"  cmp_cap={cmp_cap}",
            flush=True,
        )
        _print_hybrid_hbm_host_sizes(
            bdebt.tables,
            table_names,
            phase="Hybrid tier sizes (after export_keys_values)",
            only_table_id=tid,
        )
        by_key: Dict[int, torch.Tensor] = {}
        for i in range(keys_t.numel()):
            by_key[int(keys_t[i].item())] = emb_t[i].detach().cpu()
        w = stbe.split_embedding_weights()[tid].detach().cpu()
        if dump_sample_embeddings:
            _dump_bde_stbe_embedding_samples(
                table_name=name,
                by_key=by_key,
                w_stbe=w,
                cmp_cap=cmp_cap,
                max_keys=5,
            )
        for k in range(cmp_cap):
            assert k in by_key, f"missing key {k} in exported table {name}"
            torch.testing.assert_close(by_key[k], w[k], rtol=1e-5, atol=1e-5)


def _assert_expected_storage_backend(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    *,
    caching: bool,
    hbm_budget_ratio: Optional[float],
) -> None:
    st = bdebt.tables
    if caching:
        assert bdebt.cache is not None
        assert isinstance(st, DynamicEmbStorage)
    else:
        assert bdebt.cache is None
        if hbm_budget_ratio is not None and hbm_budget_ratio < 1.0:
            assert isinstance(st, HybridStorage)
        else:
            assert isinstance(st, DynamicEmbStorage)
