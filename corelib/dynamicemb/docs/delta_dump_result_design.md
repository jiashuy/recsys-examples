# incremental_dump → DeltaDumpResult Refactor · Design & Change Document

> Branch: `feat/dynamicemb-retain-evicted-keys` (on top of the completed
> retain-evicted-keys feature).
> Goal: refactor `incremental_dump`'s return value from the loose
> `(ret_tensors, ret_scores)` pair of nested dicts into a structured result —
> **one `DeltaDumpResult` per embedding collection** — returning that
> collection's full delta at once: upsert (`keys`/`values`) + evict
> (`evicted_keys`) + per-table `meta` (including the `slot_index` needed for
> precise write-back).
> **This round covers the dump side only; `replay_increment` is left to the next
> round.**

---

## 1. Background & Motivation

Current state (three layers):
- **model level** `incremental_dump(model, score_threshold, pg)`
  (`incremental_dump.py:217`)
  → `(ret_tensors: {collection:{table:(keys,values)}}, ret_scores: {collection:{table:score}})`
- **module level** `BatchedDynamicEmbeddingTablesV2.incremental_dump`
  (`batched:1455`) → `({table:(keys,values)}, {table:score})`
- **storage level** `DynamicEmbStorage/HybridStorage.incremental_dump`
  (`key_value_table:2077/2519`) → `(keys_cat, values_cat)`; internally
  `key_index_map.incremental_dump(return_index=True)` already has `indices`
  (key slot), and `_flat_row_indices_from_slots_and_scores` gives `flat_rows`
  (value row).

Problem: the return is scattered (keys/values and score across two dicts),
carries no slot information for "precise write-back", and does not fold the
already-implemented retain-evicted-keys into one "delta" result. This refactor
unifies them.

---

## 2. Confirmed Design Decisions

| Decision | Conclusion |
|---|---|
| Return shape | `incremental_dump` → `Dict[collection_path, DeltaDumpResult]` (**breaking change**) |
| Evicted integration | `DeltaDumpResult.evicted_keys` as a **new second outlet**; the standalone `pop_evicted_keys` API **coexists**, underlying mechanism untouched |
| Config naming | Keep **`retain_evicted_keys`** (no new `pack_evicted_keys`); `evicted_keys[i]` is present only if that table has `retain_evicted_keys=True`, else `None` |
| pop return | `pop_evicted_keys` returns a **host tensor** (still GPU `unique` / NCCL gather internally, then `.cpu()`) |
| Gather view | **Option B (global)**: `keys/values/evicted/slot_index` are all `all_gather`ed within the given `pg`; `pg=None` / `world_size==1` is a per-rank view |
| Rank reconstruction | replay reconstructs rank from the key (rank not stored); **only `roundrobin`/`hash_roundrobin` supported** (`(key or hash(key)) % world_size`), `continuous` **unsupported** (`raise` on encounter) |
| slot_index encoding | Single-tier `DynamicEmbStorage`: normal = `key_slot` (== value_row), NO_EVICTION = `(key_slot << 32) | value_row`. `HybridStorage` (**NO_EVICTION unsupported -> `raise` at construction**): `bit63=tier \| bits0-62=key_slot`. See §4 |
| This round's scope | Dump side only; `replay_increment` next round |

---

## 3. `DeltaDumpResult` Definition

Added to `dynamicemb/incremental_dump.py` (same file as `incremental_dump`). The
only change relative to the initial draft is renaming the `pack_evicted_keys`
wording in the docstring to `retain_evicted_keys`.

```python
@dataclass
class DeltaDumpResult:
    """Incremental-dump result for one embedding collection. All lists are
    column-aligned by index: element ``i`` of every list refers to the same
    table, ``table_names[i]``.

    table_names : List[str]
    keys        : List[torch.Tensor]            # per-table matched keys on host
    values      : List[torch.Tensor]            # per-table matched values on host
    evicted_keys: List[Optional[torch.Tensor]]  # per-table evicted keys on host,
        # None for a table without retain_evicted_keys=True; otherwise the table's
        # retained evicted keys (returning them drains that table's buffer, each
        # evicted key reported exactly once across successive incremental_dump).
    meta        : List[Dict[str, Any]]          # per-table, flat dict:
        #   "current_score":    int
        #   "slot_index":       torch.Tensor (int64 host, aligned with keys[i])
        #   "current_capacity": int
        #   "world_size":       int
        #   "table_options":    DynamicEmbTableOptions
    """
    table_names: List[str] = field(default_factory=list)
    keys: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    evicted_keys: List[Optional[torch.Tensor]] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)
```

---

## 4. `slot_index` Encoding (computed in the storage layer)

The storage layer already has two indices on hand (`key_value_table.py:2094`/`2102`):
- `indices`  = the key's **slot** in `key_index_map` (key_slot)
- `flat_rows` = the **value row** from
  `_flat_row_indices_from_slots_and_scores(state, indices, scores)`
  - Normal: `flat_rows == indices` (slot is the row)
  - NO_EVICTION: `flat_rows == stored_score` (auto-increment, independent of the slot)

The int64 `slot_index` has two layouts depending on the storage:

- **Single-tier `DynamicEmbStorage`** (`tier=None`):
  - normal:      `key_slot` (== value_row; one value locates both)
  - NO_EVICTION: `(key_slot << 32) | value_row` (each 32 bits, asserted < 2³²)
- **`HybridStorage`** (`tier` per tier, HBM=0 / host=1). HybridStorage does NOT
  support NO_EVICTION (rejected at construction with a `ValueError`), so every
  tier is a normal policy with `key_slot == value_row`:
  - `bit 63 = tier | bits 0..62 = key_slot`
  - Only the tier bit is added, so replay can tell the two tiers' independent
    key_index_map slot spaces apart.

```python
def _encode_slot_index(state, indices, flat_rows, tier=None):
    indices = indices.to(torch.int64)
    if tier is not None:                                  # HybridStorage
        assert indices.max() < (1 << 63)                  # key_slot in bits 0..62
        return indices | (torch.ones_like(indices) << 63) if int(tier) else indices
    if state.no_eviction_next_index is None:
        return indices                                    # normal: key_slot == value_row
    vr = flat_rows.to(torch.int64)
    assert indices.max() < (1 << 32) and vr.max() < (1 << 32)
    return (indices << 32) | vr                           # NO_EVICTION
```
replay (next round) picks the layout from the storage type / score strategy, then
unpacks with unsigned 32-bit masks.

---

## 5. Three-layer Changes

### 5.1 storage level (`key_value_table.py` · `DynamicEmbStorage` + `HybridStorage`)
`incremental_dump(table_id, threshold, pg)`'s return goes from `(keys_cat,
values_cat)` to `(keys_cat, values_cat, slot_index_cat)`:
- Compute `slot_index` (§4), **column-aligned** with keys/values.
- `dist_type == "continuous"` → `raise NotImplementedError` (unsupported here).
- **gather extension**: `_all_gather_dumped_keys_values` (`:75`) grows from
  gathering 2 tensors to gathering the keys/values/**slot_index** triple (same
  variable-length all_gather, kept aligned). The two call sites `:2130`/`:2571`
  are updated accordingly.
- host-ify: `.cpu()` on the non-gather branch; `.cpu()` after the gather on the
  gather branch (DeltaDumpResult wants host).
- `current_capacity` from `s.key_index_map.capacity(table_id)`
  (`scored_hashtable:1421`).

### 5.2 module level (`batched_dynamicemb_tables.py:1455`)
`incremental_dump(named_thresholds, pg)`'s return goes from `(ret_tensors,
ret_scores)` to **one `DeltaDumpResult`** (this module's tables):
```python
res = DeltaDumpResult()
for table_name, threshold in named_thresholds.items():
    table_id = self._table_names.index(table_name)
    keys, values, slot_index = storage.incremental_dump(table_id, threshold, pg)
    opt = self._dynamicemb_options[table_id]
    # evicted: retain-enabled tables only; drain + host + aggregate within the
    # same pg (reusing _all_gather_evicted_keys)
    if opt.retain_evicted_keys and hasattr(storage, "pop_evicted_keys"):
        ev = storage.pop_evicted_keys(table_id)          # device tensor (local)
        ev = _all_gather_evicted_keys(ev, pg) if pg else ev
        ev = ev.cpu()                                    # host
    else:
        ev = None
    current_score = device_timestamp() if score_strategy_has_timestamp_column(
        opt.score_strategy) else self._scores[table_name]
    res.table_names.append(table_name)
    res.keys.append(keys); res.values.append(values)
    res.evicted_keys.append(ev)
    res.meta.append({
        "current_score": current_score,
        "slot_index": slot_index,
        "current_capacity": storage.key_index_map.capacity(table_id),
        # shard world_size recorded at table creation (dist.get_world_size() in
        # __init__), NOT the gather pg -- replay needs the sharding fan-out.
        # dynamicemb only does ROW_WISE sharding over the global WORLD (no 2D /
        # sub-pg), so this global size equals input_dist's pg.size() (the
        # key->rank modulo base). Chose to mirror it rather than thread pg into
        # the table ctor (which currently receives no pg).
        "world_size": self._shard_world_size,
        "table_options": opt,
    })
return res
```
> `current_score` reuses the existing logic (`device_timestamp()` /
> `self._scores`); it just moves into `meta`.

### 5.3 model level (`incremental_dump.py:217`)
`incremental_dump(model, score_threshold, pg)`'s return goes from `(ret_tensors,
ret_scores)` to `Dict[collection_path, DeltaDumpResult]`: for each collection,
walk its dynamic-emb modules and **merge** each module's `DeltaDumpResult`
(`extend` the five lists) into one `DeltaDumpResult` for the collection, keyed by
`collection_path`. The filter/threading semantics (`score_threshold` an int or
`{collection:{table:threshold}}`) are unchanged.

---

## 6. `pop_evicted_keys` → host (small change, independent of the above)
`_pop_state_evicted_keys` (GPU `unique`) stays on device internally; the
module-/model-level `pop_evicted_keys` outlets `.cpu()` before returning, and
`_all_gather_evicted_keys` `.cpu()`s after the gather. The API shape and
semantics (read-and-clear, pg aggregation) are unchanged.

---

## 7. Callers / Test Updates (breaking change)
- `test/unit_tests/incremental_dump/test_distributed_dynamicemb.py`
  `ret_tensors, ret_scores = incremental_dump(...)` → consume
  `Dict[str, DeltaDumpResult]` (`res["model"].keys[i]` /
  `.meta[i]["current_score"]`, etc.).
- `test/unit_tests/incremental_dump/test_batched_dynamicemb_tables.py`
  `ret_tensors, score = model.incremental_dump(...)` → consume a single
  `DeltaDumpResult`.
- `test/unit_tests/table_operation/test_table_dump_load.py` uses
  **`LinearBucketTable.incremental_dump`** (the low-level one, returning
  `(keys, named_scores, indices)`), which is **out of scope** and untouched.

---

## 8. Out of Scope / Next Round
- **`replay_increment`**: precise write-back by slot_index (normal = a plain
  insert; NO_EVICTION = an insert with slot_index's low 32 bits as the score and
  the high 32 bits validating the key slot), which needs a new "write-back by
  slot" primitive — next round.
- `continuous` dist_type: explicitly unsupported here.

---

## 9. Changed Files
| File | Change |
|---|---|
| `dynamicemb/incremental_dump.py` | new `DeltaDumpResult`; model-level `incremental_dump` returns `Dict[str, DeltaDumpResult]`; `pop_evicted_keys` / `_all_gather_evicted_keys` return host |
| `dynamicemb/key_value_table.py` | storage-level `incremental_dump` adds `slot_index` (§4 encoding + continuous raise); `_all_gather_dumped_keys_values` grows to a triple; `_pop_state_evicted_keys` |
| `dynamicemb/batched_dynamicemb_tables.py` | module-level `incremental_dump` returns `DeltaDumpResult` (assembling keys/values/evicted/meta); record `_shard_world_size` in `__init__` |
| `dynamicemb/__init__.py` | export `DeltaDumpResult` |
| `test/unit_tests/incremental_dump/test_distributed_dynamicemb.py` | adapt to the new return shape |
| `test/unit_tests/incremental_dump/test_batched_dynamicemb_tables.py` | adapt to the new return shape |
| `test/unit_tests/retain_evicted_keys/*` | `pop_evicted_keys` now returns host tensors (`.tolist()`/`.numel()` assertions unaffected) |

---

## 10. Resolved Points
1. `DeltaDumpResult` lives in `incremental_dump.py` (same file as
   `incremental_dump`). **Confirmed.**
2. module-level `incremental_dump` keeps requiring `named_thresholds` (behavior
   unchanged; no "dump all tables on None"). **Confirmed.**
3. `evicted_keys` aggregation uses the **same** `pg` as keys/values.
   **Confirmed.**
