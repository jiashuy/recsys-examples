# dynamicemb 组件重构设计

本文记录我们讨论并确定的组件划分。目标：**一个组件只做一件事，接口清楚，没有无意义的参数**。链式索引替换旧路径之后，训练路径的结构已经简单了很多，这一轮是在它的基础上继续拆分。

状态标记：【已实现】、【已确定，待实现】、【待讨论】。

## 一、现状：哪里混在一起

| 位置 | 混在一起的事 |
|---|---|
| `ChainedStorage`（约 1200 行） | 索引 + 值缓冲 + score 策略 + 下一层（PS）+ 淘汰保留 + checkpoint + 增量 dump + replay + 扩容 |
| `BatchedDynamicEmbeddingTablesV2`（约 2300 行） | 构造三种存储布局、forward / prefetch、dump / load、增量 dump、replay、内存报告 |
| `key_value_table.py` | 状态 + checkpoint 读写 + 缓存 + eval 查找 |
| `batched_dynamicemb_function.py` | 去重、`PrefetchState`、prefetch、eval 前向、通用前向、autograd |
| score 逻辑 | 分散在 `score_args`、`_scores_to_age`、`score_load_permutation`、`_fresh_score_block`、kernel 的模式码 |
| checkpoint | `key_value_table.py`、`incremental_dump.py`、`batched_dynamicemb_tables.py` 三处 |

## 二、组件与职责

### 数据结构

| 组件 | 职责 | 状态 |
|---|---|---|
| `ChainedTables` | key 到 row 的索引：桶、链、行的归属、桶内淘汰、引用保护。只管 key，不碰 value | 已实现 |
| `CacheCore` | 缓存的 key 到 node 的放置、引用计数、回收队列。一次发射处理所有表，offsets 在设备上 | 已实现 |
| `MixedDimBuffers` | 各表维度不同的 value 缓冲：缓冲本身、各表维度、基址数组、行读写、增长。**不含优化器** | 已实现，待收窄（见 2.2） |
| `RowExchange` | 缓存的 node 与存储的 row 之间的数据交换，核心是 **swap** | 已实现 |
| `RowAllocator` | row 的生命周期：干净行 / 脏行、预初始化、1:1 映射 | 待实现（最后做） |

### 2.1 `RowExchange` 与 swap【已实现】

缓存里一个 node 存着某个 key 的 value 副本。把 node 让给新 key 是两件作用在**同一个 node** 上的事：旧 key 的行（缓存里可能已经被训练）写回它在存储里的行；新 key 的行（如果有）从存储读进 node。读旧值必须先于写新值。

做成一个 kernel，一个 block 处理一个 node：每个线程先读旧行和新行的各自那几列，再写。这样：
- 不需要为所有 victim 准备临时缓冲；
- 两个方向（写回、读入）同时在 PCIe 上传输，用满双向带宽；
- 行按 16、4 或 2 字节的字拷贝（取所有表行字节数的最大公约字），和元素类型无关。

接口：`swap(nodes, table_ids, evict_rows, fetch_rows)`，`-1` 表示跳过该半边；存储缓冲增长后无需重建（每次调用读当前指针）。

读缓存（forward 的 `load`）和写缓存（优化器更新）仍然是对 `MixedDimBuffers` 的操作，不经过 `RowExchange`；它只管**两个缓冲之间**的搬运。

### 2.2 优化器与 `MixedDimBuffers`【已确定，待实现】

优化器本身（超参数、step 计数）没有逐行状态；权重和优化器状态统一存在底层存储的 value 行里。所以：

- **所有涉及优化器的操作都不走 `MixedDimBuffers`**，由优化器处理。现在 `MixedDimBuffers` 里的 `update_rows`、`update_padded`、`reset_optimizer_states` 要移到优化器那一侧。
- `MixedDimBuffers` 提供一个**布局视图**：每张表的 embedding 宽度、状态区的位置、行基址、是否按 4 对齐。优化器据此准确分出哪些是 embedding、哪些是状态。
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

### 2.4 准入（admitter）【已确定方向，细节待讨论】

准入是一个有状态的策略（计数器、阈值、自己的 checkpoint），**代码不并入存储，调用点并入存储**：存储内部在"这批 key 缺失"这一步调用 `admitter.filter(...)`，prefetch 流程里看不到它。有无 PS 的差异在存储里处理（PS 里已有的 key 视为已准入）。

### 2.5 下一层（`lower`）与淘汰去向（`EvictionSink`）【已确定，待实现】

PS、Hybrid、retain evicted keys 统一成接口相同的不同后端。它们其实是两件事：

- **下一层 `lower`**：找不到时去哪里取、被淘汰时往哪里放。PS、host 层（Hybrid）都是它的实现。接口：`find` / `insert` / `erase` / 导出。
- **淘汰去向 `EvictionSink`**：丢弃、写到下一层、记录 key。**retain 是对任意去向的装饰**，不是单独的一种。

于是 Hybrid 是 `lower = HostTier`，PS 是 `lower = PSTier`，retain 是在去向外面包一层记录。`ChainedStorage` 里的 `retain_and_spill` 去掉。

### 2.6 score 组件【已确定，待实现】

一个单独的组件回答所有与 score 有关的问题，其他组件都来问它：

- 新 key 的初始 score、命中时怎么更新；
- 淘汰怎么排序；
- 增量 dump 的阈值列、timestamp 到 age 的转换；
- checkpoint 的列顺序（逻辑到物理的置换）；
- replay 怎么重建 score。

淘汰排序最终要在 kernel 里执行，不能回调 Python，所以组件的接口是**声明式**的：给出 score 字数、更新模式、排序规则，由索引的 kernel 解释；其余问题由 Python 侧直接问它。

### 2.7 checkpoint【已确定，待实现】

三层，中间有一个共同的数据单位：

- **`RowChunk`**：（keys、emb、opt、scores，加可选的被删除 key）。存储只提供两个原语：`export_chunks(filter)` 和 `import_chunk(chunk, mode)`。
- **文件格式层**：命名、meta、列布局、dtype、`dist_type`、旧版兼容。只管读写 `RowChunk`，不碰表。
- **`Checkpointer`**：多表、按 rank 分片、跨 rank 汇聚、计数器；用 score 组件处理列顺序，用优化器处理状态的存盘形式。

整份 dump 是"不过滤"的导出，增量 dump 是"score 大于阈值"的导出，replay 是带合并语义的导入；三者共用同一组原语，现在分散的三处逻辑合成一条路径。

### 2.8 `BatchedDynamicEmbeddingTablesV2` 与训练

它应该**只负责和训练相关的事情**，即 `nn.Module` 该做的：

- forward / backward 的编排：prefetch 流水线、取值、更新、score 的推进；
- 持有各组件（存储、缓存、优化器、initializer、admitter）并把它们接起来；
- 对外的少量模块接口（按表名的 `set`、`flush`、score 设置等）。

它**不再**负责：构造三种存储布局（交给一个 builder）、checkpoint / 增量 dump / replay（交给 `Checkpointer`）、内存报告（独立的报告函数）。

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
- 存储布局的 builder 放在哪里，以及它和 `BatchedDynamicEmbeddingTablesV2` 的边界。
