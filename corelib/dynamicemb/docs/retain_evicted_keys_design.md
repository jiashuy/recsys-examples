# Retain Evicted Keys — 设计与修改文档

> 分支：`feat/dynamicemb-retain-evicted-keys`（基于 PR #437 的 LruLfu / JIT 基础设施）
> 目标：让**最后一级存储**在驱逐 key 时可选地保留被驱逐的 key，并暴露
> `pop_evicted_keys(...)` API，按 table 取回「过去一段时间内被驱逐、去重后的 key」。

---

## 1. 背景与动机

DynamicEmb 的 key→slot 哈希表在 bucket 满时会按 score 逐出「受害者 key」。驱逐分两类，行为完全不同：

- **`table_insert_and_evict`**（GPU 主表 / cache / HBM tier 用）：被驱逐 key 会写进
  `evicted_keys/scores/indices/table_ids` 输出缓冲，然后由 Python 层**搬到下一级**（host/backing）。
  这些 key 并未真正丢失。
- **`table_insert`**（**最后一级** host / backing 表用）：evict 输出指针传 `nullptr`
  （`kernels.cuh` 中 `insert_body` 目前把 `evict_key_out` 传 `nullptr`），被驱逐 key 被
  **静默覆盖丢弃**——这是 key 真正从系统中消失的唯一位置，而这里当前没有任何缓冲区保留它。

本功能就是给这条「最后一级 `insert`」路径加一个**可选的收集钩子**。

---

## 2. 已确定的设计决策（前期讨论结论）

| 决策点 | 选择 |
|---|---|
| 触发级别 | **仅最后一级**（真正丢弃 key 的那级）。中间层驱逐是搬运，不记录 |
| 收集载体 | **`insert_body` 的编译期模板 sink**（不是 numba-JIT）。numba 那条线保留给 `user_score_fn` 数值函数 |
| 收集内容 | 只 **key + table_id**（不含 score / value / index），最省内存、契合需求 |
| 覆盖范围 | **所有 score policy**：AoT (`DefaultEvictor`) 与 LruLfu cubin (`RankedEvictor`) 两条 insert 路径都支持 |
| 去重 & 内存 | **append buffer + 读时 `torch.unique`**（追加期不做任何压缩，只在 `pop` 时去重） |
| 时间窗口 | **`pop` 读取即清空 / 增量**：返回上次调用以来新驱逐的 unique key |
| 分布式 | `pg=None` 每 rank 返回本地部分（row-wise sharding → 天然 disjoint，零通信）；给 `pg` 则组内 `all_gather` 并集。**清空永远只清本 rank buffer** |
| 配置项 | `DynamicEmbTableOptions.retain_evicted_keys: bool = False`（默认关闭，零开销） |
| API 名 | `pop_evicted_keys` |

**待你确认的开放点**见 §9。

---

## 3. 端到端数据流

```
最后一级 storage 的一次 insert（retain 开启）
  key_value_table._insert_key_values(state, keys, table_ids, values, ...)
    └─ state.key_index_map.insert(..., collect_evicted=True)          # LinearBucketTable
         └─ C++ table_insert_collect_evicted(...)                     # 新增 host 函数
              ├─ AoT:    table_insert_collect_kernel                  # 非 LruLfu
              └─ LruLfu: dyn_emb_insert_collect_entry (cubin)         # via jit_link
                   └─ insert_body<Table, Evictor, CollectEvicted=true>
                        · 驱逐点取 victim key
                        · compaction 写 evicted_keys[out] = victim
                                        evicted_table_ids[out] = table_ids[i]
         → 返回 (indices, num_evicted, evicted_keys, evicted_table_ids)
    └─ 取前 num_evicted 个，append 到 state.evicted_key_buffer (key 列 + table_id 列)
       （超过软阈值则就地 torch.unique 压缩）

用户调用（任意时刻）
  dynamicemb.pop_evicted_keys(model, table_names=None, pg=None)
    └─ 遍历 collection → 最后一级 storage → 每个 table_id：
         本地 keys = unique(buffer[table_id==tid])
         pg 给定 → all_gather + concat（disjoint，无需再去重）
         清空本 rank buffer
    → {collection_path: {table_name: 1-D int64 keys}}
```

---

## 4. C++ 层改动（逐文件）

### 4.1 `src/table_operation/kernels.cuh`

**(a) `insert_body` 加模板 sink + 输出参数**（模板参数追加在末尾，向后兼容；输出参数带默认 `nullptr`）

```cpp
template <typename Table, typename KernelTraits,
          typename Evictor = DefaultEvictor, bool CollectEvicted = false>
__device__ __forceinline__ void
insert_body(..., int32_t *__restrict__ counter, uint64_t cur_ts = 0,
            typename Table::KeyType *__restrict__ evicted_keys = nullptr,
            int64_t *__restrict__ evicted_table_ids = nullptr,
            CounterType *evicted_counter = nullptr) {
```

- 现有实例化 `insert_body<Table, KernelTraits, DefaultEvictor>(..., counter, 0)` 与
  `insert_body<..., RankedEvictor<C>>(..., counter, cur_ts)` **不受影响**（`CollectEvicted`
  默认 false、新参数默认 nullptr）。

**(b) 驱逐点取 victim key**（当前把 `evict_key_out` 传 `nullptr`）

```cpp
    KeyType evict_key = KeyType();
    insert<ReductionGroupSize, BufferDim, Policy, Evictor>(
        bucket, key, score, ..., &iter, &result,
        CollectEvicted ? &evict_key : static_cast<KeyType *>(nullptr),  // was nullptr
        static_cast<ScoreType *>(nullptr), counter, counter_offset, cur_ts);
```
`CollectEvicted` 是编译期常量，false 分支折叠回 `nullptr` → 非收集路径代码不变。

**(c) 循环体末尾加 compaction 段（`if constexpr` 包裹，只写 key + table_id）**

```cpp
    if constexpr (CollectEvicted) {
      // CompactTileSize == 1 for every insert instantiation
      // (InsertKernelTraits<...,1,...> in insert.cu / evict_lrulfu.cu), so this
      // size-1 tile ballot needs no cross-thread convergence; the `continue` on
      // table_cap==0 above is harmless (those threads evicted nothing).
      auto g = cg::tiled_partition<KernelTraits::CompactTileSize>(
          cg::this_thread_block());
      bool evicted = (result == InsertResult::Evict);
      uint32_t vote = g.ballot(evicted);
      int group_cnt = __popc(vote);
      CounterType group_offset = 0;
      if (g.thread_rank() == 0)
        group_offset =
            atomicAdd(evicted_counter, static_cast<CounterType>(group_cnt));
      group_offset = g.shfl(group_offset, 0);
      int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
      int64_t out_id = group_offset + previous_cnt;
      if (evicted) {
        evicted_keys[out_id] = evict_key;
        evicted_table_ids[out_id] = table_ids[i];
      }
    }
```
> **只收集 `InsertResult::Evict`**（表中已存在的老 key 被逐出）。**不**收集 `Busy`（新 key 因竞争/满桶未插入——那是插入失败，属 `safe_check_mode` 范畴，且该 key 会在后续 batch 再次出现）。【决策 1】

**(d) 新增 AoT 收集版 `__global__`**（紧邻 `table_insert_kernel`）

```cpp
template <typename Table, typename KernelTraits>
__global__ void table_insert_collect_kernel(
    /* 同 table_insert_kernel 参数 */ ..., int32_t *counter,
    typename Table::KeyType *evicted_keys, int64_t *evicted_table_ids,
    CounterType *evicted_counter) {
  insert_body<Table, KernelTraits, DefaultEvictor, /*CollectEvicted=*/true>(
      ..., counter, /*cur_ts=*/0, evicted_keys, evicted_table_ids,
      evicted_counter);
}
```

### 4.2 `src/jit/evict_lrulfu.cu`

新增 LruLfu cubin 的收集 entry（复用现有 `EvictParams` 已有字段 `evicted_keys / evicted_table_ids / evicted_counter`）：

```cpp
__device__ __forceinline__ void run_insert_collect(const EvictParams &p,
                                                   uint64_t cur_ts) {
  using KernelTraits =
      InsertKernelTraits<256, 1, 1, /*CompactTileSize=*/1, 8,
                         ScorePolicyType::LruLfu, /*OutputScore=*/true>;
  EvictTable table(p.table_storage, p.num_buckets, p.bucket_capacity,
                   p.num_scores);
  insert_body<EvictTable, KernelTraits, RankedEvictor<EvictComparator>,
              /*CollectEvicted=*/true>(
      table, p.table_bucket_offsets, p.bucket_sizes, p.batch, p.input_keys,
      p.table_ids, reinterpret_cast<InsertResult *>(p.insert_results),
      p.indices, p.score_input, p.score_output, p.table_key_slots, p.counter,
      cur_ts, reinterpret_cast<KeyType *>(p.evicted_keys), p.evicted_table_ids,
      p.evicted_counter);
}
extern "C" __global__ void dyn_emb_insert_collect_entry(EvictParams p) {
  run_insert_collect(p, read_globaltimer());
}
```
> `EvictParams` 结构体**不改**（字段已齐）。这条 entry 进 `evict_lrulfu_lex.fatbin`（AoT，默认 Lex evictor）与 custom fatbin 两份，与现有 3 个 entry 同批编译。

### 4.3 `src/jit/jit_link.h` / `jit_link.cpp`

- `ModuleEntries` 增加 `CUfunction insert_collect;`
- 加载时 `cuModuleGetFunction(&m.insert_collect, m.module, "dyn_emb_insert_collect_entry")`
  （放入现有「任一 entry 缺失即卸载」的校验块）。
- 新增 getter：
  ```cpp
  CUfunction demb_get_insert_collect_fn(int64_t key);  // 同 key 空间 / module
  ```
- launch 复用现有 `demb_launch_evict(fn, EvictParams, batch, stream)`，无需新 launcher。

### 4.4 `src/table_operation/insert.cu`

- 新增 `launch_table_insert_collect_kernel<Table, PolicyTypeV, OutputScoreV>(...)`：与
  `launch_table_insert_kernel` 同构，但
  - LruLfu 分支：填 `EvictParams` 的 `evicted_keys/evicted_table_ids/evicted_counter`，
    调 `demb_get_insert_collect_fn(score_fn_key)`；
  - AoT 分支：调 `table_insert_collect_kernel`。
  - 之后同样跟一个 `table_unlock_kernel`。
- 新增 host 入口（**不改动现有 `table_insert` 签名**）：
  ```cpp
  // 返回 (indices, num_evicted, evicted_keys, evicted_table_ids)
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  table_insert_collect_evicted(/* 同 table_insert 的入参 */ ...);
  ```
  内部分配 `evicted_keys[num_total]`、`evicted_table_ids[num_total]`（`int64`）、
  `evicted_counter[1]=0`，dispatch 到 `launch_table_insert_collect_kernel`，返回缓冲与计数
  （Python 侧按 `num_evicted` 切片）。缓冲上界 = 本 batch key 数（每个输入 key 至多逐出一个 victim）。

### 4.5 `src/table_operation/table.cu`

在 `bind_table_operation` 中绑定 `table_insert_collect_evicted`（签名/返回与 4.4 一致）。

---

## 5. Python 层改动（逐文件）

### 5.1 `dynamicemb/dynamicemb_config.py`
`DynamicEmbTableOptions` 新增字段：
```python
retain_evicted_keys: bool = False
```
文档字符串说明：仅对最后一级存储生效；开启后可用 `pop_evicted_keys` 取回被驱逐 key。

### 5.2 `dynamicemb/scored_hashtable.py`
`LinearBucketTable.insert` 增加可选参数（默认关，保持现返回类型）：
```python
def insert(self, keys, table_ids, score, insert_results=None, score_out=None,
           collect_evicted: bool = False):
    ...
    if collect_evicted:
        indices, num_evicted, ev_keys, ev_tids = table_insert_collect_evicted(
            ..., score_fn_key=self.score_fn_key_)
        return indices, num_evicted, ev_keys, ev_tids
    indices = table_insert(...)   # 现状
    return indices
```
> `_deterministic_insert`（`DEMB_DETERMINISM_MODE`）路径**不支持**收集：`collect_evicted=True`
> 命中该分支时抛明确的 `RuntimeError`（不静默回退）。【决策 3】

### 5.3 `dynamicemb/key_value_table.py`（核心）

**(a) `DynamicEmbTableState` 增加 retain 状态**（`create_table_state` 里按 `retain_evicted_keys` 初始化，仅最后一级 state）：
- `retain_evicted_keys: bool`
- `evicted_key_buffer: Optional[DeviceExtendableBuffer]`（1-D int64，存 key）
- `evicted_tid_buffer: Optional[DeviceExtendableBuffer]`（1-D int32/64，存 table_id）

> 追加期**不做**任何压缩/去重（【决策 2】）；去重只在 `pop` 时 `torch.unique`。

**(b) `_insert_key_values`（最后一级统一入口，:1021）** 改为：
```python
if state.retain_evicted_keys:
    indices, num_evicted, ev_keys, ev_tids = state.key_index_map.insert(
        unique_keys, table_ids, score_arg, score_out=score_out_flat,
        collect_evicted=True)
    if num_evicted > 0:
        _append_evicted(state, ev_keys[:num_evicted], ev_tids[:num_evicted])
else:
    indices = state.key_index_map.insert(...)   # 现状
...
store_to_flat(state, flat_indices, table_ids, unique_values)   # 不变
```
`_append_evicted` 只把 (key, tid) 追加到两个 buffer（纯 append，无压缩）。

**(c) `pop` 辅助**（storage 层）
- `DynamicEmbStorage.pop_evicted_keys(table_id) -> Tensor`：从其 state buffer 取
  `key[tid==table_id]` → `torch.unique` → 清空**该 rank**的 buffer（整体清，或按 tid 分段清；见 §7）。
- `HybridStorage.pop_evicted_keys(table_id)`：只读 `self._host` 的 buffer（最后一级）。
- `DynamicEmbCache`：不参与（非最后一级）。

**(d) 最后一级判定**（`retain_evicted_keys` 只挂在真正会丢 key 的 state）
- 非 caching 单层 `DynamicEmbStorage` → 其 state 是最后一级 ✔
- `HybridStorage` → `self._host` state 是最后一级 ✔；`self._hbm` ✘（走 insert_and_evict 溢出）
- caching → backing `self._storage`（`DynamicEmbStorage` 或 external PS）是最后一级 ✔；
  `self._cache` ✘

### 5.4 `dynamicemb/batched_dynamicemb_tables.py`
- 构造时把每表的 `retain_evicted_keys` 传给**最后一级** storage 的 `create_table_state`
  （在 `_create_cache_storage` 里按上面判定分派）。
- 新增方法：
  ```python
  def pop_evicted_keys(self, table_names=None, pg=None) -> Dict[str, torch.Tensor]:
      # 仅遍历开启 retain 的表；table_name→table_id = self._table_names.index(name)
      # 调 self._storage.pop_evicted_keys(table_id)，pg 给定则 all_gather+concat
  ```
  （对 caching：从 backing `self._storage` 取。）

### 5.5 `dynamicemb/incremental_dump.py`（或新增 `dynamicemb/evicted_keys.py`）
model 级函数，风格对齐 `incremental_dump`：
```python
def pop_evicted_keys(
    model: torch.nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,  # None = 所有开启 retain 的表
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # {collection_path: {table_name: 1-D int64 unique evicted keys}}
```
复用 `find_sharded_modules` / `get_dynamic_emb_module` 遍历，逐 `BatchedDynamicEmbeddingTablesV2`
调其 `pop_evicted_keys`。

### 5.6 `dynamicemb/__init__.py`
`from .incremental_dump import pop_evicted_keys`（或新模块），并加入 `__all__`。

---

## 6. `pop_evicted_keys` API 规格

```python
def pop_evicted_keys(model, table_names=None, pg=None)
    -> Dict[str, Dict[str, torch.Tensor]]
```
- **返回**：`{collection_path: {table_name: keys}}`，`keys` 为 1-D `int64`、table 内 unique。
- **`pg=None`**：每 rank 返回本地 shard 上被驱逐的 key（零通信）。
- **`pg` 给定**：组内 `all_gather` 后 concat（各 rank disjoint，直接拼即全局 unique）。
- **增量语义**：返回「上次调用以来」新驱逐的 key；调用后**清空本 rank buffer**（无论是否聚合，只清自己的）。
- 未开启 `retain_evicted_keys` 的表：**从返回中省略**（不返回空张量）。【决策 4】

---

## 7. 内存与语义细节

- **收集上界**：一次 insert 的 evicted 数 ≤ 输入 key 数；C++ 侧按 `num_total` 预分配，
  `evicted_counter` 给真实数量，Python 侧切片。
- **buffer 增长**：`DeviceExtendableBuffer` 按需 `extend`，追加期不去重；内存 ≈ O(两次 pop 之间
  被逐出的 key 总数，含重复)。去重只在 `pop` 时做（【决策 2】）。
- **清空粒度**：`pop` 单表时理想是只清该 table 的条目。实现上 buffer 混存多表 → 用
  `mask = tid != table_id` 保留其余、重建 buffer；或一次 `pop` 所有请求表后统一重建。
- **disjoint 保证**：row-wise sharding 下同一 key 只落一个 rank，故跨 rank 聚合只需 concat。
- **与 `incremental_dump` 正交**：两者独立；retain 记录「已离开表」的 key，incremental_dump
  导出「仍在表内、score 达阈值」的 key。

---

## 8. 测试计划

- **table 级**（`test/unit_tests/table_operation/`）：小容量表，构造必然驱逐的输入，
  校验 `table_insert_collect_evicted` 返回的 (evicted_keys, evicted_table_ids) 与 oracle 一致；
  **AoT 与 LruLfu 两路**各测；多逻辑表共享存储时按 table_id 分流正确。
- **storage / module 级**（`test/unit_tests/`）：`retain_evicted_keys=True`，喂多个 batch
  触发最后一级驱逐，`pop_evicted_keys` 返回 unique 正确、二次调用为增量（已清空）、
  未开启的表不返回。覆盖 caching / HybridStorage / 单层三种最后一级形态。
- **分布式**（`test_distributed_dynamicemb.py`）：`pg=None` 各 rank disjoint；给 `pg`
  时组内并集正确；清空只影响本 rank。
- **回归**：`retain_evicted_keys=False`（默认）下所有现有 insert / evict 测试不变。

---

## 9. 决策记录（已确认）

1. **收集范围**：只收集 `InsertResult::Evict`；**Busy 不算**。
2. **无中途压缩**：追加期不压缩，去重只在 `pop` 时 `torch.unique`。
3. **确定性模式**：`DEMB_DETERMINISM_MODE` 不支持收集，`collect_evicted=True` 命中即抛
   `RuntimeError`。
4. **未开启表**：从 `pop` 结果中省略（不返回空张量）。
5. **API 落点**：`pop_evicted_keys` 放进 `dynamicemb/incremental_dump.py`。
6. **无 `clear=False`**：只保留「读取即清空」语义，不加只读变体。

---

## 10. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/table_operation/kernels.cuh` | `insert_body` 加 sink 模板 + compaction；新增 `table_insert_collect_kernel` |
| `src/jit/evict_lrulfu.cu` | 新增 `dyn_emb_insert_collect_entry` |
| `src/jit/jit_link.h` / `.cpp` | 缓存新 entry + `demb_get_insert_collect_fn` |
| `src/table_operation/insert.cu` | `launch_table_insert_collect_kernel` + host `table_insert_collect_evicted` |
| `src/table_operation/table.cu` | 绑定 `table_insert_collect_evicted` |
| `dynamicemb/dynamicemb_config.py` | `retain_evicted_keys` 配置项 |
| `dynamicemb/scored_hashtable.py` | `LinearBucketTable.insert(collect_evicted=...)` |
| `dynamicemb/key_value_table.py` | state retain buffer + `_insert_key_values` 收集 + storage `pop_evicted_keys` |
| `dynamicemb/batched_dynamicemb_tables.py` | 传导配置 + module 级 `pop_evicted_keys` |
| `dynamicemb/incremental_dump.py`（或新文件） | model 级 `pop_evicted_keys` |
| `dynamicemb/__init__.py` | 导出 `pop_evicted_keys` |
| `test/...` | table / storage / 分布式 测试 |
