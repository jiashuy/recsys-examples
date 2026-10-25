# dynamicemb 组件重构设计

本文记录我们讨论并确定的组件划分。目标：**一个组件只做一件事，接口清楚，没有无意义的参数**。链式索引替换旧路径之后，训练路径的结构已经简单了很多，这一轮是在它的基础上继续拆分。

状态标记：【已实现】、【已确定，待实现】、【待讨论】。

## 一、现状：哪里混在一起

| 位置 | 混在一起的事 |
|---|---|
| `ChainedStorage`（约 1200 行） | 索引 + 值缓冲 + score 策略 + 下一层（PS）+ 淘汰保留 + checkpoint + 增量 dump + replay + 扩容 |
| `BatchedDynamicEmbeddingTablesV2`（约 2300 行） | 构造三种存储布局、forward / prefetch、dump / load、增量 dump、replay、内存报告 |
| `key_value_table.py` | 状态 + checkpoint 读写 + 缓存 + eval 查找（已拆：状态在 `table_state.py`，缓存在 `cache.py`，eval 查找在 `eval_lookup.py`，剩下的是 checkpoint 文件读写，等 step 5 重排） |
| `batched_dynamicemb_function.py` | 去重、`PrefetchState`、prefetch、eval 前向、通用前向、autograd |
| score 逻辑 | 分散在 `score_args`、`_scores_to_age`、`score_load_permutation`、`_fresh_score_block`、kernel 的模式码 |
| checkpoint | `key_value_table.py`、`incremental_dump.py`、`batched_dynamicemb_tables.py` 三处 |

## 二、组件与职责

### 数据结构

| 组件 | 职责 | 状态 |
|---|---|---|
| `ChainedTables` | key 到 row 的索引：桶、链、行的归属、桶内淘汰、引用保护。只管 key，不碰 value | 已实现 |
| `CacheCore` | 缓存的 key 到 node 的放置、引用计数、回收队列。一次发射处理所有表，offsets 在设备上 | 已实现 |
| `MixedDimBuffers` | 各表维度不同的 value 缓冲：缓冲本身、各表维度、基址数组、行读写、增长。**不含优化器** | 已实现，已不含优化器（见 2.2） |
| `RowExchange` | 缓存的 node 与存储的 row 之间的数据交换，核心是 **swap** | 已实现 |
| `RowAllocator` | row 的生命周期：干净行 / 脏行、预初始化、1:1 映射 | 待实现（最后做） |

### 2.1 `RowExchange` 与 swap【已实现】

缓存里一个 node 存着某个 key 的 value 副本。把 node 让给新 key 是两件作用在**同一个 node** 上的事：旧 key 的行（缓存里可能已经被训练）写回它在存储里的行；新 key 的行（如果有）从存储读进 node。

**最终做法：排序暂存。** 先把被顶掉的行从缓存（HBM）按它们存储行号的顺序 gather 到暂存区（同时腾出 node）；一个融合 kernel 取写回和读入两条列表各自按存储行号排序后的第 i 项：第 i 个被顶掉的行写到它的存储行，同时第 i 个读入的存储行写进它的 node。两个方向在同一个 kernel 里同时进行，访问存储都是有序的。接口 `swap(nodes, table_ids, evict_rows, fetch_rows)`：`-1` 跳过该半边，没有要读入的 node 保持原样，node 为 -1 整项跳过；存储缓冲增长后无需重建。

**为什么是这个做法（H100 上的测量，host 存储 2680 万行，20 万行一步）：**
- 链路峰值：`cudaMemcpy` 单向约 52 GiB/s，双向 94 GiB/s；kernel 零拷贝双向 78 GiB/s。连续 pinned 缓冲上的配对 kernel 能到 84 GB/s 双向。
- GPU 访问 host 表的随机行很慢：host 表 256 MB 以内随机读每行 10 ns（链路满速），13 GB 时 33.7 ns，是 TLB 受限（翻译缓存覆盖约 1 GB 量级）。排序后降到 16.6 ns。大页、`cuMemCreate` host 分配、NUMA 绑定都没有效果；`cudaMemcpyBatchAsync`（让 DMA 引擎搬随机行）每行 544 ns，不可行。
- 融合 swap 把一个 node 的写回行和读入行配对，没法同时让两个列表都有序，所以随机访问；两个方向也不重叠（swap 时间等于读加写之和）。
- CPU 做随机行的 gather / scatter、GPU 只访问连续缓冲：窄行上略好（约 10%），宽行上更差。CPU 的 scatter 用非临时写比普通写快 2.5 倍；CUDA 驱动把所有 stream 的 host 回调放在一个线程上，gather 和 scatter 不能并行，换成专用线程后两者又会争抢 CPU 内存带宽（总量到连续拷贝峰值约 99 GB/s 就到顶）；adam 一步 600 MB 负载，CPU 路径比 GPU 排序暂存慢约 45%。
- 结果（有效双向带宽，写回加读入字节数除以时间）：sgd 20 万行 融合 16 GB/s、排序暂存 38 GB/s；adam 20 万行 融合 41 GB/s、排序暂存 **68 GB/s（约 64 GiB/s，约 68% 双向峰值）**。

完整的实验代码（融合 kernel 变体、CPU 路径、分块流水线、专用线程加标志位等待、各类微基准和探针）在分支 `row-exchange-experiments` 上，不在主线。

与上游（同一台 H100，同样的 benchmark 配置）相比，训练一步：缓存 adam 命中率 0.8 快 2.0 倍、命中率 1.0 快 3.5 倍，sgd 快 2.3 和 3.8 倍；hybrid 快 1.2 到 1.3 倍；全 HBM 快 1.05 倍。

读缓存（forward 的 `load`）和写缓存（优化器更新）仍然是对 `MixedDimBuffers` 的操作，不经过 `RowExchange`；它只管**两个缓冲之间**的搬运。

### 2.2 优化器与 `MixedDimBuffers`【已实现】

优化器本身（超参数、step 计数）没有逐行状态；权重和优化器状态统一存在底层存储的 value 行里。所以：

- **所有涉及优化器的操作都不走 `MixedDimBuffers`**，由优化器处理。现在 `MixedDimBuffers` 里的 `update_rows`、`update_padded`、`reset_optimizer_states` 要移到优化器那一侧。
- `MixedDimBuffers` 提供一个**布局视图**：每张表的 embedding 宽度、状态区的位置、行基址、是否按 4 对齐。优化器据此准确分出哪些是 embedding、哪些是状态。
- 实现：`MixedDimBuffers.layout` 返回冻结的 `RowLayout`（基址、dtype、各表 value/emb 宽度、最宽值、是否按 4 对齐）；优化器基类有 `update_rows(layout, grads, rows, table_ids)`、`update_padded(layout, grads, padded, table_ids)`、`init_states(layout, padded, table_ids)`，内部转调各优化器已有的 kernel 入口。`RowLayout` 里的基址在某个缓冲增长后失效，所以每次调用现取。
- 优化器的输入有两种：直接作用于存储里的行（行号加基址），或作用于已经读出的补齐缓冲；两种都用同一个布局对象描述，不再用 7 个散参数。
- **一行的初始化 = initializer 填 embedding + `optimizer.init_states` 填状态**。初始状态归优化器。

### 2.3 initializer、预初始化与 `RowAllocator`【已确定，待实现】

- **不再支持 debug 初始化**（`debug_init` 那种按 key 取值的）。因此初始化只依赖行，不依赖 key，可以预先做。
- initializer 是一套算法，输入是 `MixedDimBuffers`（的某些行）。
- **预初始化**：创建表时不再逐个分配时初始化。`RowAllocator` 持有"干净行"和"脏行"两个栈：
  - 新建表时只初始化一个前缀（一个水位），不是整张表（100GB 的 host 表一次初始化要很久）；
  - `take(n)` 保证返回的行已初始化，干净行不够时按块补；
  - 被淘汰释放的行进脏栈，攒到一定数量批量重新初始化。
- **1:1 映射**：`RowAllocator` 支持传入 key 到 row 的 1:1 映射（key 就是行号）。配合 `bucket_capacity` 设到最大（整张表一个桶，容量等于行数），没有淘汰，DynamicEmb 和 TorchRec 的表在行的层面一一对应，可以逐行直接对比。
- **`MixedDimBuffers` 的 set 接口**：按表设置一张表的 value（把 TorchRec 的权重灌进来）。表名由 `BatchedDynamicEmbeddingTablesV2` 处理：它提供一个按表名的 `set` 接口，把表名换成下标再调用 `MixedDimBuffers`；`MixedDimBuffers` 自己不记表名。
- **和索引的关系**：现在空闲行栈在 `ChainedTables` 里，commit 的 kernel 直接从栈上取行。抽成独立的 `RowAllocator` 需要 commit 改成向它要行，会动 C++，所以放在最后。

### 2.4 准入（admitter）【调用点已并入存储】

准入是一个有状态的策略（计数器、阈值、自己的 checkpoint），**代码不并入存储，调用点并入存储**：存储内部在"这批 key 缺失"这一步调用 `admitter.filter(...)`，prefetch 流程里看不到它。有无 PS 的差异在存储里处理（PS 里已有的 key 视为已准入）。

实现：`ChainedStorage.set_admitter(admitter)`（模块创建 admitter 后挂上）和 `resolve_missing(keys, table_ids, frequency) -> Missing(from_lower, lower_values, admit_mask)`：先问下一层，再对下一层没有的 key 问 admitter。`prefetch_cache_path` 和 `prefetch_hbm_direct_path` 不再接收 admitter，只调 `storage.resolve_missing`。Hybrid 与通用路径里的 `_apply_admission` 没动。

实现：`ChainedStorage.set_admitter(admitter)`（`BatchedDynamicEmbeddingTablesV2._create_admitter` 创建后设置）和 `ChainedStorage.resolve_missing(keys, table_ids, frequency) -> Missing(from_lower, lower_values, admit_mask)`：对存储里没有的 key，先问下一层，再对下一层也没有的 key 问 admitter。两条 prefetch 路径（`prefetch_cache_path`、`prefetch_hbm_direct_path`）不再接收 admitter，也不再自己拼这两步。Hybrid 与通用路径（`batched_dynamicemb_function` 里的 `_apply_admission`）仍沿用函数链里传下来的 admitter，没有并入。

### 2.5 下一层（`lower`）与淘汰去向（`EvictionSink`）【已实现】

PS、Hybrid、retain evicted keys 统一成接口相同的不同后端。它们其实是两件事：

- **下一层 `lower`**：找不到时去哪里取、被淘汰时往哪里放。PS、host 层（Hybrid）都是它的实现。接口：`find` / `insert` / `erase` / 导出。
- **淘汰去向 `EvictionSink`**：丢弃、写到下一层、记录 key。**retain 是对任意去向的装饰**，不是单独的一种。

实现：`dynamicemb/eviction_sink.py`，`EvictionSink.put(keys, table_ids, values, scores)` 加 `wants_values`（不要值的去向，调用方不去加载）；`Discard`、`ToLower(lower)`、`ToTier(tier)`（带 score 写到同类的旁边一层）、`Retain(state, inner)`（记录 key 后转给 `inner`）。`ChainedStorage` 构造时按 `spill_to` / `lower` / `evicted_item_mode` 选出 sink，`retain_and_spill` 改名为 `evict`，只加载值再交给 sink。`lower` 一侧：`dynamicemb/lower_tier.py` 的 `LowerTier(storage)` 把用户传入的 `external_storage`（任何 `Storage`）包成窄接口：`fetch(keys, table_ids, frequency) -> Fetched(found, scores, values)`、`insert`、`fill`（原 `_fill_from_lower`）。其余代码不再按位置去取 `Storage.find` 返回元组的第 5、7 项。`ChainedStorage.lower` 返回 `LowerTier`，底层的 `Storage` 在 `.storage`。Hybrid 的 host 层不是 `lower`，而是 `ToTier` 的去向，另有自己的查找路径，没有并进来：它和 PS 的语义不同（host 层的 key 与 HBM 层不相交，查找要带 score 和行号，更新就地进行）。

用户实现的 `Storage` 抽象（`external_storage` 的契约）没有改。

于是 Hybrid 是 `lower = HostTier`，PS 是 `lower = PSTier`，retain 是在去向外面包一层记录。`ChainedStorage` 里的 `retain_and_spill` 去掉。

### 2.6 score 组件【Python 侧已实现，kernel 接口未动】

一个单独的组件回答所有与 score 有关的问题，其他组件都来问它：

- 新 key 的初始 score、命中时怎么更新；
- 淘汰怎么排序；
- 增量 dump 的阈值列、timestamp 到 age 的转换；
- checkpoint 的列顺序（逻辑到物理的置换）；
- replay 怎么重建 score。

实现：`dynamicemb/score.py` 的 `ScoreRules(strategy)`，每个存储一个（`ChainedStorage._score`）。`access()` 给出一次访问的 score 值和更新模式（原 `score_args`）；`to_logical`/`to_physical`、`to_age`/`from_age`、`dump_block`/`replay_block`、`fresh_block` 负责 checkpoint 列顺序、timestamp 与 age 的转换和 replay 的 score 重建。原来散在 `key_value_table.py`（`_scores_to_age`、`_age_to_scores`、`_fresh_score_block`、`_dump_score_block`、`_replay_score_block`）和 `chained_storage.py`（`_is_lru_lfu`）的代码已删。`dynamicemb_config.py` 里的三个排列函数保留（`jit/score_jit.py` 和配置检查还在用），`ScoreRules` 在它们之上。

实现：`dynamicemb/score.py` 的 `ScoreRules(strategy)`：`access`（一次访问给的值和更新模式）、`to_logical`/`to_physical`（列顺序）、`to_age`/`from_age`、`dump_block`/`replay_block`（含宽度检查）、`fresh_block`。原来散在 `chained_storage.score_args`、`key_value_table` 的 `_scores_to_age`、`_age_to_scores`、`_fresh_score_block`、`_dump_score_block`、`_replay_score_block`、`_timestamp_score_columns` 的逻辑都收到这里；配置里的 `get_physical_score_order` 和两个置换函数仍在 `dynamicemb_config.py`（`jit/score_jit.py` 也用）。

淘汰排序最终要在 kernel 里执行，不能回调 Python，所以组件的接口是**声明式**的：给出 score 字数、更新模式、排序规则，由索引的 kernel 解释；其余问题由 Python 侧直接问它。

### 2.7 checkpoint【`RowChunk` 已落地；文件格式层与 `Checkpointer` 待做】

三层，中间有一个共同的数据单位：

- **`RowChunk`**：（keys、emb、opt、scores，加可选的被删除 key）。存储只提供两个原语：`export_chunks(filter)` 和 `import_chunk(chunk, mode)`。
- **文件格式层**：命名、meta、列布局、dtype、`dist_type`、旧版兼容。只管读写 `RowChunk`，不碰表。
- **`Checkpointer`**：多表、按 rank 分片、跨 rank 汇聚、计数器；用 score 组件处理列顺序，用优化器处理状态的存盘形式。

进度：`dynamicemb/row_chunk.py` 的 `RowChunk(keys, embeddings, opt_states, scores)` 是 `NamedTuple`，所以元组解包的旧调用点照常工作。`export_keys_values_iter`、`Storage.export_keys_values`、`ChainedHybridStorage.export_keys_values` 现在产出 `RowChunk`，`_dump_table` 按字段名使用。

现有函数与三层的对应（接下来的拆分按这张表做，每一步可单独测试）：

| 现有 | 去向 |
|---|---|
| `export_keys_values_iter`、`incremental_dump` 的取行、`_split_value_row` | 存储的 `export_chunks(filter)` |
| `_load_key_values`、`_replay_write_values`、`_replay_score_block`（已并入 `ScoreRules`） | 存储的 `import_chunk(chunk, mode)`，mode 区分整表加载（覆盖）与 replay（合并） |
| `_dump_table` 里的文件写、`_iter_batches_from_files`、meta json、`encode_*_path`、`find_files`、`get_loading_files` | 文件格式层，只读写 `RowChunk` |
| `BatchedDynamicEmbeddingTablesV2.dump/load/incremental_dump/replay_increment` 与 `_plan_replay`、`_apply_replay`、`_check_delta_shapes`、跨 rank 的 `_all_gather_dumped_columns` | `Checkpointer` |

整份 dump 是"不过滤"的导出，增量 dump 是"score 大于阈值"的导出，replay 是带合并语义的导入；三者共用同一组原语，现在分散的三处逻辑合成一条路径。

### 2.8 `BatchedDynamicEmbeddingTablesV2` 与训练

它应该**只负责和训练相关的事情**，即 `nn.Module` 该做的：

- forward / backward 的编排：prefetch 流水线、取值、更新、score 的推进；
- 持有各组件（存储、缓存、优化器、initializer、admitter）并把它们接起来；
- 对外的少量模块接口（按表名的 `set`、`flush`、score 设置等）。

它**不再**负责：构造存储布局（交给 `storage_layout.py` 里的 builder，已实现）、checkpoint / 增量 dump / replay（交给 `Checkpointer`）、内存报告（已拆到 `dynamicemb/memory_report.py` 的 `print_memory_consume`）。

### 2.9 存储布局与 builder【已实现】

`storage_layout.py` 持有 `StorageLayout`（HBM_ONLY / HOST_ONLY / HYBRID / CACHING）和 `build_storage(options, optimizer, device_id)`，后者按各表的 value 大小与 HBM 预算选定布局并构造出来，返回 `StorageBundle(layout, storage, cache, exchange)`。`BatchedDynamicEmbeddingTablesV2` 只是调用它，拿到各部分后接起来；各层的容量换算、桶容量规则、external storage 的挂接、NO_EVICTION 的限制都在这个文件里。

## 三、迭代顺序

1. **`RowExchange` 与 swap**【已实现】
2. 优化器的布局视图：把 7 个散参数换成视图对象；优化器操作移出 `MixedDimBuffers`
3. score 组件：先把 Python 侧收拢，kernel 接口不动
4. `lower` 和 `EvictionSink`：拆 `ChainedStorage` 的第一刀
5. checkpoint 三层
6. `RowAllocator` 与预初始化、1:1 映射

## 四、待讨论

- admitter 在有 PS 时的具体语义（PS 里已有的 key 是否一律绕过准入）。
- `RowExchange` 是否也提供读缓存、写缓存的接口，还是继续留在 `MixedDimBuffers`。
- `MixedDimBuffers` 按下标管理表、表名由 `BatchedDynamicEmbeddingTablesV2` 处理，这一点是否足够。

## 五、代码的目录组织

内部的训练组件按职责放进三个子包，对外的入口模块留在顶层：

| 位置 | 内容 |
|---|---|
| `dynamicemb/storage/` | 行与索引、层级：`chained_table`、`chained_storage`、`storage_layout`、`table_state`、`mixed_dim_buffers`、`extendable_tensor`、`row_exchange`、`eviction_sink`、`lower_tier`、`cache`、`score`、`memory_report` |
| `dynamicemb/checkpoint/` | `row_chunk`、`key_value_table`（checkpoint 文件读写；名字过时，step 5 重排时改） |
| `dynamicemb/training/` | 训练一步的编排：`chained_prefetch`、`batched_dynamicemb_function`、`eval_lookup`、`lookup_layout` |
| 顶层（公开入口，路径不变） | `batched_dynamicemb_tables`、`dump_load`、`incremental_dump`、`dynamicemb_config`、`types`、`optimizer`、`initializer`、`embedding_admission`、`utils`、`batched_dynamicemb_compute_kernel`、`construct_twin_module`、`key_ownership` |
| 顶层（推理，不动） | `exportable_tables`、`scored_hashtable`、`lookup_meta`、`index_range_meta`、`jit/` |

新组件的单元测试在 `test/unit_tests/components/`。
