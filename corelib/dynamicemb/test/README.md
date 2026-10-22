# dynamicemb tests

Two levels, one rule each:

| Level | Directory | What it tests | Process model |
|---|---|---|---|
| **unit** | `unit/` | one component, on its own: a kernel, the table, the storage, one module, the planner, one optimizer | one process, at most one GPU (a few table tests declare `DISTRIBUTED_NPROCS` because the component itself is rank-aware) |
| **integration** | `integration/` | the components together, the way a user wires them: planner + sharder + `DistributedModelParallel` + training / checkpointing, compared against TorchRec or a reference | several ranks, `torchrun` |

A test that builds a `DistributedModelParallel` or drives a train loop through
TorchRec is integration, wherever it started life.

## Running

```bash
cd corelib/dynamicemb
pytest test                          # everything; needs 8 GPUs for the big cases
pytest test -m unit                  # components only
pytest test -m integration           # end to end only
pytest test -m load_dump             # checkpoint / dump-load group (CI)
pytest test -m fwd_bwd               # everything else (CI)
pytest test/unit/table -k lru        # any pytest selection works
test/unit_test.sh [fwd_bwd|load_dump|unit|integration]   # CI wrapper, nothing but the marker mapping
```

Cases that need more GPUs than are visible are **skipped**, not failed.

## How multi-GPU is expressed (no shell loops)

* A test file that must run on several ranks declares one module constant:
  `DISTRIBUTED_NPROCS = (1, 8)`. `conftest.py` then collects it as one item per
  process count (`test_x.py[1-gpu]`, `test_x.py[8-gpu]`); each item re-runs the
  file under `torchrun --standalone -m pytest`. Inside, the file is an ordinary
  rank-aware pytest file using the `backend_session` fixture. File-level
  `pytestmark = pytest.mark.load_dump` is honoured.
* A standalone driver (several phases, many CLI flags) lives in
  `integration/workers/` and is started by `common.launch.run_worker(script, nproc, args)`
  from a parametrized test; the case table is Python (`pytest.param(..., id=...)`).
* CI groups are markers: `load_dump`, otherwise `fwd_bwd`. Put
  `pytest.param(..., marks=pytest.mark.load_dump)` on a case that is about checkpoints.

## Shared code (`common/`)

| Module | Provides |
|---|---|
| `launch.py` | `run_worker`, `require_gpus`, `gpu_count`, torchrun command |
| `features.py` | sparse-feature / random-index generators, table & feature naming |
| `sharding.py`, `gpu_platform.py` | `get_planner`, comm precision, platform/optimizer kwargs |
| `dynamicemb_model.py` | build a sharded model (`create_model`, `apply_dmp`) + checkpoint assertions |
| `frequency_features.py`, `lfu_checks.py` | deterministic frequency-tracking inputs; LFU score validation |
| `tables.py`, `scores.py`, `incremental.py` | table builders (`lru_lfu_table`), score helpers, incremental-dump helpers |
| `bdet_builders.py`, `bdet_expansion.py` | `BatchedDynamicEmbeddingTablesV2` builders / FBGEMM reference, expansion helpers |
| `planner_fixtures.py` | `make_topology` |
| `cli_args.py`, `debug.py` | worker command-line helpers (`as_flags`, `str2bool`), `Debugger` |

Fixtures in `conftest.py`: `current_device`, `backend_session`. Never redefine them in a file.

## Catalog: what is tested where

### Unit

| File | Purpose |
|---|---|
| `unit/ops/test_unique_op.py` | segmented unique, table-id expansion, flagged compact CUDA ops |
| `unit/ops/test_vmm_tensor.py` | VMM-backed extendable device / host buffers |
| `unit/ops/test_hash_roundrobin.py` | key → rank assignment: roundrobin vs hash_roundrobin, CPU model vs CUDA kernel |
| `unit/table/test_table_operation.py` | scored hash table kernels: layout, `bucketize_keys`, insert / lookup / erase / evict, counter overflow |
| `unit/table/test_lru_lfu.py` | LRU_LFU compound score policy and custom score functions (table level) |
| `unit/table/test_table_dump_load.py` | table dump / load and threshold incremental dump (1 and 8 ranks, `load_dump`) |
| `unit/table/test_hybrid_storage_export.py` | `HybridStorage` HBM+host tiers driven directly; tier bit of incremental dump |
| `unit/table/test_no_eviction_row_indexing.py` | NO_EVICTION value-row indexing, caching spill |
| `unit/table/test_expansion_value_capacity.py` | key-map expansion keeps a value row behind each slot |
| `unit/eviction/test_insert_collect_evicted.py` | retain-evicted-keys at table / storage level |
| `unit/eviction/test_pop_evicted_keys.py` | `pop_evicted_keys` at module level |
| `unit/incremental_dump/test_batched_dynamicemb_tables.py` | which keys each incremental-dump window returns (`load_dump`) |
| `unit/incremental_dump/test_replay_increment.py` | `replay_increment`: apply a delta to another model + validation (`load_dump`) |
| `unit/module/test_forward_backward.py` | `BatchedDynamicEmbeddingTablesV2` fwd/bwd vs FBGEMM; deterministic insert; empty batch; mixed dims; admission |
| `unit/module/test_batched_table_backward.py` | one-GPU backward, pooled and sequence |
| `unit/module/test_weighted_pooled_embedding_v2.py` | weighted SUM pooling vs unsharded EBC |
| `unit/module/test_cache_flush.py` | cache prefetch and flush |
| `unit/module/test_capacity_expansion.py` | capacity growth while inserting |
| `unit/module/test_fill_tables.py` | fill to a load factor |
| `unit/module/test_dump_load.py` | module-level checkpoint round trip, dist_type metadata |
| `unit/module/test_dump_value_layout.py` | checkpoint precision / optimizer-state dtype / row width, legacy checkpoints |
| `unit/module/test_embedding_collection_context.py` | `DynamicEmbeddingCollectionContext` defaults (CPU) |
| `unit/optimizer/test_ftrl_optimizer.py` | FTRL vs the paper, group sparsity, flat vs padded |
| `unit/optimizer/test_padded_buffer_optimizer.py` | padded-buffer slot addressing for Adam / FTRL |
| `unit/optimizer/test_rowwise_adagrad_state_precision.py` | fp32 accumulator in fp16 / bf16 tables, checkpoint conversion |
| `unit/admission/test_probabilistic_admission.py` | probabilistic admission strategy |
| `unit/admission/test_admission_counter_deprecation.py` | deprecated `admission_counter` spelling |
| `unit/planner/test_planner_placement.py` | which node a table-row-wise table lands on (CPU) |
| `unit/planner/test_planner_plan_cost.py` | per-rank cost of DynamicEmb tables, what TorchRec is left with (CPU) |

### Integration

| File (worker) | Purpose | Matrix |
|---|---|---|
| `integration/test_embedding_forward_backward.py` (`workers/embedding_forward.py`) | sharded pooled / sequence / multi-collection forward vs reference | dim, dedup, dtype, #tables, multi-hot, 2/4/8 GPUs |
| `integration/test_twin_module.py` | sharded dynamicemb model trains identically to plain TorchRec | optimizer, pooling, weighted; 1 and 8 ranks |
| `integration/test_alignment.py` | planner + DMP table capacity / HBM alignment | 1, 4, 8 ranks |
| `integration/test_model_dump_load.py` (`workers/model_dump_load.py`) | dump a sharded model and load it back, incl. resharding 1↔8, hash_roundrobin | optimizer × optim-state × counter × score × world sizes (`load_dump`) |
| `integration/test_lfu_scores.py` (`workers/lfu_scores.py`) | LFU scores of a sharded model | dedup × storage mode × GPUs × optimizer |
| `integration/test_embedding_admission.py` (`workers/embedding_admission.py`) | frequency admission end to end | storage × optimizer × score × GPUs (`load_dump`) |
| `integration/test_table_row_wise.py` (`workers/table_row_wise.py`) | table-row-wise: plan, parity, jagged, pooling, grad, checkpoints | topology × dist_type; checkpoints are `load_dump` |
| `integration/test_incremental_dump_distributed.py` | `incremental_dump` / `replay_increment` on sharded models | 1 and 8 ranks (`load_dump`) |
| `integration/test_pop_evicted_keys_distributed.py` | `pop_evicted_keys` on a sharded model | 1 and 8 ranks |

## Adding a test

1. Does it need only one component? → `unit/<area>/`. Does it need a DMP / sharder /
   train loop? → `integration/`. Pick the area directory by the component, not the feature.
2. Check this catalog and the gap list first; extend a parametrization before writing a new function.
3. Need ranks? `DISTRIBUTED_NPROCS`. Need a standalone driver? a worker + a case table.
4. Shared setup goes in `common/<topic>.py`; never copy a helper between test files.
5. The module docstring states `Level / Needs / Covers / Does not cover (see …)`.
   Update the catalog row.

## Known gaps (not covered by any test)

* Table-level `HybridStorage` dump/load; `ScorePolicy.CONST`; compound LRU_LFU at table-level dump/load.
* Overflow-bucket behaviour with eviction / caching beyond one counter case.
* `replay_increment` on multi-rank models; incremental dump with per-table differing score strategies.
* `BatchedDynamicEmbeddingTablesV2` eval mode with cache + external storage together; padded-buffer kernels for Adagrad / SGD.
* Pooled MEAN backward; backward with optimizers other than SGD at module level.
* fp16 / bf16 output dtype and low-precision all-to-all are accepted by the worker but never applied.
* `use_torch_opt` for pooled / sequence; `data_parallel_embeddings`; sharding types other than row-wise (outside table-row-wise).
* Twin test: adam / adamw entries exist but are not parametrized.
* Dump/load: `dist_type=continuous`, `adagrad`, cross-world-size hash_roundrobin, 4-GPU resharding.
* LFU / admission with compound LRU_LFU on cache or hybrid storage.
* `DynamicEmbeddingShardingPlanner` end to end through DMP.
