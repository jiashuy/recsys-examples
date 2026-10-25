# 链式桶索引（key 与 value 行分离）设计

状态：**设计讨论稿，尚未实现。** 记录的是与 Jiashu 讨论后确定的方案，以及讨论中发现的潜在问题。文中的数字除非注明"实测"，都是推演或估算，没有在代码上测过。

这是一套**新实现，与现有的 `key_index_map`（`scored_hashtable.py` 的 `LinearBucketTable` 及 `src/table_operation`）并存，不修改现有实现**。现有实现被 `exportable_tables.py`（Torch export 推理路径）依赖，保持不动。

修订记录：
- v1：桶内链、行号全局分配、先插入后淘汰、`bucket_size` 找空槽。
- v2：链头统一命名为 `head`；桶内偏移用 2 B（因为活动空间节点也要链到表上）；输入先按桶排序，同一桶由一个 worker 独占处理（不再需要原子操作）；桶内未被 pin 的节点不够时活动空间节点留在表里，NO_EVICTION 则扩容。
- v5（本版）：待定项收尾：默认淘汰范围为桶内 128；trim 只腾出本步所需；**确定性只在设置 `DEMB_DETERMINISM_MODE` 时保证，且只要求同一输入两次运行结果一致，不要求与旧实现对齐；不开启时怎么快怎么做**；**不兼容旧 checkpoint 和旧增量 dump 格式**（需要兼容的用户使用旧版本）。
- v4：**一步 prefetch 拆成 `find_and_stage` 和 `commit` 两段**：所有未命中的新 key 先暂存在活动空间（不入表、不分配行号），PS 查找、admission 和扩容夹在两段之间；`commit` 里才分配行号、放置到桶内空槽、不够再淘汰（§4.2、4.3、4.10）。
- v3：**活动空间用共享的按桶连续段**（不采用"每桶多留槽"）；**第一版 pin 沿用缓存的 `EvictionGuard`**，不在槽上额外存 pin 信息（"最近 pin 的步号"只作为将来的优化，见 §5.3）；HBM_DIRECT 没有缓存可查，只在这个模式下分配按行号的引用计数。

## 1. 动机

现有实现里 **key 的槽位（slot）与 value 的行（row）是同一个数**：槽位下标直接作为 value 缓冲的行号。这带来一串复杂性：

| 问题 | 现状 |
|---|---|
| rehash / 扩容 | 槽位变了，value 行必须跟着搬（`_expand_tables_impl` 里 `load_from_flat_single_table` / `store_to_flat_single_table`） |
| 淘汰 | 新 key 占用被淘汰 key 的槽位，也就是同一行，所以必须**先把旧值拿出来**再写新值 |
| 同一次 insert 内先插入的 key 又被后面的淘汰 | 需要 `take_values_of_keys_evicted_in_the_same_insert` 这类补丁 |
| NO_EVICTION | 槽位与行必须分开，于是把行号塞进 score 字，`slot_index` 要打包成高低 32 位，HybridStorage 因此不支持 |
| 多表合并在一个 map 里 | 扩容要重建整个 map，要读 `table_bucket_offsets` |
| 每个 key 各自做一次桶内规约 | 同一桶里的多个新 key 要各自规约、争用同一个 victim |

目标：key 到 value 行的映射**解耦**，使得 rehash 不搬 value、prefetch 可以先全部插入再统一淘汰、淘汰范围可配置，并尽量少花内存。

## 2. 方案概览

保留现有的"桶 + 桶内连续 score"布局，在每个桶**内部**加一层链，行号由全局行池分配：

```
key --hash--> 桶 b (每桶 bucket_capacity 个槽，默认 128)
          --hash2--> 桶内第 c 条链 (每桶 bucket_capacity 条链)
          head[c] → 槽 → next → 槽 → …              (链只在桶内)
槽 s 里存: key, score[S], index(= value 行号, 全局分配), next
value 行: 全局行池, 与槽位无关
```

要点：

1. **链在桶内**：每桶 `bucket_capacity` 个 `head`，每槽一个 `next`，没有 `prev`（省空间；删除时从 `head` 走）。`head` 和 `next` 是**桶内偏移，2 B**；偏移空间同时覆盖本桶的槽和本桶的活动空间节点（见 3.2）。
2. **输入先按桶排序**，同一个桶的 key 由一个 worker 一起处理。桶内数据只有这个 worker 在写，**不需要原子操作**，`bucket_size`、`head`、`next` 都是普通读写。
3. **行号全局分配**，与槽位无关，在 `commit` 时才分配（此时已经知道哪些新 key 会被留下）。
4. **先暂存，后放置**：`find_and_stage` 只查找命中的 key，把未命中的 key 暂存到活动空间，**不改动表**；PS 查找、admission、扩容之后，`commit` 把留下的新 key 放进桶内空槽，放不下的才在桶内规约**一次**选出全部 victim，由暂存的节点占用 victim 的槽位。
5. **桶内未被 pin 的节点不够**时，多出的活动空间节点留在表里（软超限），等它们不再被 pin、再次被 trim 处理时才归位或被淘汰；**NO_EVICTION 模式则扩容**。
6. **淘汰范围可配置**：桶内（默认，128）、整表 top-K，以后可加采样。
7. **每表一个 map**：扩容只动需要扩容的表；kernel 里按表读设备端指针数组，一次 launch 仍可处理多表。

## 3. 数据布局

### 3.1 每表的数组

| 数组 | 类型 | 大小 | 说明 |
|---|---|---|---|
| `keys` | int64 / uint64 | 槽数 | 沿用现有 |
| `scores` | uint64 × S | 槽数 | 沿用现有的 AoS；整个数组连续，所以桶内规约和整表扫描都是连续读 |
| `index` | uint32 | 槽数 | value 行号 |
| `next` | uint16 | 槽数 | 桶内下一个节点的偏移，`0xFFFF` 为空 |
| `head` | uint16 | 槽数（每桶 `bucket_capacity` 个） | 链头，桶内偏移 |
| `refs` | int32 | **行数** | 按行号的 pin 引用计数，**只在 HBM_DIRECT 模式分配**（缓存路径用守卫，不需要），见 §5 |
| `bucket_size` | uint16 | 桶数 | 已占用槽数，**占用的槽恒为前缀 `[0, size)`** |
| 行池空闲栈 | uint32 | 行数 | 空闲行号 |
| 活动空间（workspace） | 见下 | 每步新 key 数量级 | 每步暂存所有未命中的新 key，不链入表；只有放不下的残留节点才链入（§4.4） |

`digest` 数组**去掉**：它的作用（用一个缓存行的指纹向量快速比较）被 `head` 取代。现有的 `_ref_counter`（每槽 4 B，不论用不用都分配）也不再无条件分配，见 §5。

活动空间节点：`{key, score[S], index, next}`，桶号由所在的段隐含。

### 3.2 桶内偏移的编码（2 B）

| 偏移 | 含义 |
|---|---|
| `[0, bucket_capacity)` | 本桶的第 o 个槽 |
| `[bucket_capacity, 0xFFFE)` | 本桶活动空间段的第 `o - bucket_capacity` 个节点，实际位置是 `该桶的段起点 + (o - bucket_capacity)`，段起点来自**每步的段描述表**，见下 |
| `0xFFFF` | 空 |

活动空间段是**按桶连续**的：因为输入已经按桶排好序，同一个桶这一步的新节点天然相邻。残留节点（§4.4）每一步被复制前移到新的段里，见 §8.1。

**段起点不需要每桶存一份。** 排序的输出本来就是一张"这一步处理哪些桶"的表（排序后的唯一桶号列表，加上每个桶在活动空间里的 `(段起点, 节点数)`），负责某个桶的 worker 直接从自己的那一行读到。所以：
- 正常路径：worker 沿链走，遇到偏移 ≥ `bucket_capacity` 就用自己这一行的段起点解析；trim 也从这一行得到活动空间节点数。
- 每步的段描述表大小是"这一步触及的桶数 + 残留桶数"，不随表大小增长；每步重建，不需要清理。
- 不经过排序流水线的查找（例如 eval 路径）：只有走到残留节点时才需要段起点，此时对**残留桶表**（按桶号排序，通常为空或很小）做二分。

### 3.3 id 的宽度

- 行号、活动空间下标：**uint32**，哨兵 `0xFFFFFFFF`。上限约 42.9 亿。
- int32（约 21.5 亿）在 80 GB 显卡上基本够用（索引在 21.5 亿行之前就占满显存），但 B200（约 192 GB）、GH200 等大显存卡上 56 B/key 的索引可达约 34 亿行，所以选 uint32。同样 4 字节，没有额外内存。
- **字节偏移一律 int64**：`行号 × 维度 × 元素字节数` 在 32 位里会溢出。
- 构造时检查容量，超过 2^32 时给出明确的容量错误。

### 3.4 内存账（每个存活 key 的索引开销，不含 value）

本方案每 key：key 8 + score 8S + index 4 + next 2 + head 2 + `bucket_size` 约 0.02。

| | S=1 | S=2 |
|---|---|---|
| 现有（key 8 + digest 1 + score 8S + `_ref_counter` 4，后者无条件分配） | 21 B | 29 B |
| 本方案，缓存路径（不含行池空闲栈） | 约 24 B | 约 32 B |
| 本方案，缓存路径（含行池空闲栈 4 B） | **约 28 B** | **约 36 B** |
| 本方案，HBM_DIRECT（再加按行号的 `refs` 4 B） | 约 32 B | 约 40 B |

对 value 较小的表影响仍然可见（dim 16 fp32 SGD，value 64 B）：现有约 85 B/key，本方案缓存路径约 92 B/key（约 +8%）；dim 128 以上基本无感。

## 4. 操作

文中的 **trim** 指 `commit` 里"空槽不够、在桶内淘汰"的那一部分。

一步 prefetch 的流程分两段，中间夹着 PS 查找、admission 和扩容（见 §4.10）：

**排序 → `find_and_stage` →〔PS 查找 ‖ 扩容〕→〔admission〕→〔缓存 allocate（pin）〕→ 排序 → `commit` →（调用方）读 victim 值、写下层 → `release_rows`**；backward 结束时 **unpin**。

### 4.1 排序与分段

- 把这一步的 `(table_id, key)` 按 `(table_id, 桶号)` 排序（保留原位置，用于把结果还原成输入顺序），再切成"每个桶一段"。
- 一步里会排序**两次**：`find_and_stage` 对本步所有 unique key 排一次；`commit` 对留下来的新 key 再排一次（此时可能已经扩容，桶数变了，所以不能沿用第一次的桶号）。第二次只排新 key，规模小。
- 现有 `bucketize_keys`（`table_operation/bucketize.cu`，确定性模式在用）做了相近的事，但它按"波次"排，输出形态不同，需要新写按桶分段的排序；复用程度待核对。
- 之后的 kernel 都是**每个桶由一个 worker（warp 或 block）独占处理一段**：桶内的 `bucket_size`、`head`、`next`、`scores` 只有它写，不需要原子操作，也就不需要对 2 B 字段做原子。

### 4.2 `find_and_stage(keys, table_ids) -> rows, hit, staged`

每个桶 worker 对本桶的 key 段：
1. `hash → 链 → 沿 next 查找`（桶内的槽，以及残留的活动空间节点）。
2. **命中**：更新 score（HBM_DIRECT 下再 `refs[row] += 1`），返回它的行号。
3. **未命中**：把 `{key, table_id, 初始 score}` 写入活动空间（用前缀和压紧，保持输入顺序），**不分配行号、不占槽、不链入表**。

这一步除了命中 key 的 score 更新，**不改动表**。因此在 `commit` 之前任何一步失败（PS 出错、容量错误），表都不需要回滚；被 admission 拒绝的 key 也只是从活动空间里删掉，没有任何要撤销的东西。

命中得到的是**行号**而不是槽位；行号不会因为之后的扩容而改变，所以这一步的结果在扩容之后依然有效（现有实现里槽位就是行，扩容会搬动，所以扩容必须排在查找之前）。

### 4.3 `commit(keep_mask, guard) -> new_rows, victims`

输入是 admission 之后留下来的新 key（PS 里找到的 key 直接留下，不经过 admission；其余按 admission 结果）。

1. **压紧**：按 `keep_mask` 把活动空间里被拒绝的节点去掉。
2. **排序**：对留下的新 key 按 `(table_id, 桶号)` 排序（扩容后的桶数），切成每桶一段。
3. **分配行号**：默认对全局行池空闲栈做原子弹栈（单遍）；确定性模式（`DEMB_DETERMINISM_MODE`）下改为先标记、前缀和、再按序取，行号确定；返回每个新 key 的行号。
4. **放置**：每个桶一个 worker 独占，对本桶的 `m` 个新 key：
   - 桶内空槽数 `f = bucket_capacity - bucket_size`；前 `min(m, f)` 个新 key 直接放到槽 `bucket_size, bucket_size+1, …`，写 `{key, score, index}`，接入 `head` 链。**不需要淘汰。**
   - 还剩 `k = m - f > 0` 个时，对桶内全部成员按排名函数**规约一次**：按 `(排名, 位置)` 排好序，从最小的开始依次取，**取到一个才检查它是否被 pin**（守卫或 `refs`，见 §5），被 pin 的跳过，取满 `k` 个为止。只对被选中的候选查 pin，不对 128 个成员逐个查。同一桶里所有新 key 引起的淘汰**一次选完**，不会互相争用同一个 victim，结果确定。
   - victim 从链里摘掉（没有 `prev`，从 `head` 走），由一个暂存的新 key **占用该槽位**（覆盖 key、score、index，接入链）。
5. 没有足够的非 pin 候选时，剩下的新 key 成为残留节点，见 §4.4。
6. 返回 victim 的 `(key, table_id, row)`；**行不立即归还**。

### 4.4 残留节点（软超限）

- 桶内没被 pin 的成员不够 `k` 个时，多出的暂存节点**留在表里，不释放**：此时才把它们**链入**所在桶的链（偏移落在活动空间段），lookup 照常能找到。正常路径上暂存节点从不链入表，所以 2 B 偏移里"≥ `bucket_capacity`"的部分只服务于这条少见路径。
- 这个桶登记进"残留桶列表"。之后的每一步 trim 都把残留桶也纳入处理：此时它们的 pin 可能已经解除（缓存的节点已被释放），成为可被淘汰的候选，或者占用 victim 腾出的槽位而"归位"。
- 残留节点与新节点同处一个按桶连续的段：每一步重建活动空间时，把残留桶的旧段**复制**到新段前面再接上新节点（见 §8.1）。

### 4.5 NO_EVICTION 模式

没有 victim。`commit` 里如果某个桶放不下全部新 key，**扩容**：该表重新分桶，只搬 `(key, score, index)` 记录，value 不动；扩容后对放不下的新 key 重新排序、放置。这是在现有按负载因子触发扩容（`get_expand_info`）之外的兜底：桶是定长的，即使整体负载因子不高，个别桶也可能放不下。

### 4.6 `release_rows` 与 `unpin`

- `release_rows(rows)`：把 victim 的行号放回全局空闲栈（调用方读完 victim 的值之后）。
- `unpin`：缓存路径下**存储什么也不做**（pin 是缓存节点的引用计数，由缓存的 `release` 在 backward 里解除）；HBM_DIRECT 下对这一步的行号 `refs[row] -= 1`。

### 4.7 删除与空槽管理

- 空槽 = 下标 `bucket_size`：**占用的槽恒为前缀 `[0, size)`，不留洞**，不需要位图，也不需要扫描。
- 常见路径（活动空间节点占用 victim 槽位）不产生洞。
- 会产生洞的操作：显式 `erase`、淘汰数多于活动空间节点数（整表 top-K、低水位批量淘汰）。处理方式：**用队尾记录补洞**（swap-with-last），重接被搬动记录的链（用它的 key 重新算出链头，沿链找到前驱改 `next`），`bucket_size` 减小。
- 补洞发生在 trim 的同一个 worker 内，顺序：先"占用 victim 槽位"，再补洞。

### 4.8 扩容

- 按表独立；只需要扩容的表重建。
- 重新分桶时只搬记录，**value 不动**；行池追加行（`ExtendableBuffer` 已支持）。
- 行池大小 = 设定容量 + 余量（≥ 在途步骤的 unique key 数），host 在 insert 前就知道每表的 unique 数（`unique_size_per_table`），不够就追加。

### 4.9 淘汰范围（可配置）

| 范围 | 做法 |
|---|---|
| **桶内（默认，128）** | 上面的 trim。语义最接近现有实现 |
| **整表 top-K** | score 数组整体连续，对它做两遍直方图（基数）选择，找出第 k 小的排名值并淘汰；需要水位线（高水位触发、淘汰到低水位）来摊薄 O(N) 的扫描 |
| 采样淘汰（可选，后续） | 每次随机取 m 个，淘汰最小的，O(1) 但近似 |

多个 score 列：按列原样存储，淘汰时用**排名函数** `rank(score 列…) → uint64` 得出比较键（LRU 用时间戳，LFU 用频率，LruLfu 沿用现有的打包规则）。可沿用现有 JIT 机制（`score_fn_key`、`evict_lrulfu`）。

### 4.10 与 PS、admission、扩容的配合

现有缓存路径是串行的：存储查找 → PS 查找 → admission → 缓存 allocate → 存储 insert。新流程把"存储 insert"拆成 `find_and_stage`（最前）和 `commit`（最后），中间的几步可以重叠：

| 步骤 | 依赖 | 能与谁并行 |
|---|---|---|
| `find_and_stage` | 无 | – |
| PS 查找（只查未命中的 key） | staged 列表 | **与扩容、初始值生成并行**（PS 查找不碰表） |
| 扩容（按 `unique_size_per_table` 和负载因子决定） | 无 | 与 PS 查找并行；表此时没有被暂存节点改动，行号稳定 |
| admission（只对 PS 里没找到的 key） | PS 查找结果 | – |
| 缓存 allocate（pin） | admission 结果 | – |
| `commit` | 以上全部 | – |

- **只在 `commit` 之前等 PS 的结果**。PS 里找到的 key 直接留下（值来自 PS）；没找到的走 admission，被拒绝的从活动空间删掉，不占槽、不占行、不入表。
- **所有新 key 统一先进活动空间**，不区分"有没有 admission"。不准入的 key 因此不需要从表里删除（否则会在桶里留洞，要补洞）。
- **`commit` 必须在缓存 allocate 之后**：新 key 先被缓存 pin 住，守卫才能保护它们（§5.2）。
- 收益的量级要实事求是：prefetch 本来就与上一步的 forward/backward 重叠，PS 延迟只有在 prefetch 比它们更长时才会暴露；稳态下扩容很少发生，所以重叠的收益主要在扩容步和 PS 慢的场景。**这个拆分更主要的价值是结构上的**：失败不需要回滚、被拒绝的 key 不碰表、行号推迟到确定留下之后才分配。

## 5. pin：第一版沿用缓存的守卫

### 5.1 现状与取舍

淘汰要避开"在途步骤正在使用的 key"。现在这件事分两处：缓存路径下，存储 insert 时通过 `EvictionGuard` 去查缓存节点的引用计数；HBM_DIRECT 路径下，存储自己的 `_ref_counter`（按槽位，每槽 4 B，不论用不用都分配）。

第一版**沿用这个分工**，在新实现里去掉无条件的 4 B：

| 模式 | pin 的来源 | 额外内存 |
|---|---|---|
| 缓存路径（存储在缓存之后） | 缓存节点的引用计数，经 `EvictionGuard`（`src/embedding_cache/guard.h`）查询 | **0** |
| HBM_DIRECT（没有缓存） | `refs[row]`，按**行号**索引的 int32 | 每行 4 B，**只在该模式分配** |

`refs` 按行号而不是槽位：行号是稳定的，trim 搬动记录（活动空间节点占用 victim 槽位、补洞）不会让它失效。

### 5.2 守卫的使用方式（影响 trim 的实现）

- **懒检查**：trim 把桶内成员按排名排好序，从最小的开始取，**只对取到的候选查守卫**，被保护的跳过取下一个，取满 `k` 个为止。不对桶内 128 个成员逐个查缓存的哈希表。预期查询数约等于 `k` 加上被保护的个数，很小。
- **顺序要求**：这一步的所有 key（包括新 key）必须在 trim 之前就已经有缓存节点并被缓存 pin 住，否则新 key 会被当成可淘汰候选。现有流程里缓存 `allocate` 在存储 insert 之前，已经满足；新流程要保持"缓存 allocate → 存储 `commit`（其中的淘汰部分即 trim）"的顺序。
- trim kernel 接受和现有一致的 `EvictionGuardView`（设备端指针加表数，0 表示无守卫），可以直接复用现有的 `guard.h`。

### 5.3 将来可以考虑的优化（不在第一版）

**每槽存"最近一次 pin 它的步号"（1 B）**，设备端一个标量 `oldest_inflight`：
- 一个 key 被 pin，当且仅当它的步号落在 `[oldest_inflight, 当前步号]`（模 256，要求在途步数小于 128）。
- unpin 只写这个标量，不碰任何 key 或桶；trim 搬动记录只是复制这 1 个字节。
- 收益：存储自己知道 pin，不再查缓存，可以去掉守卫；HBM_DIRECT 也不再需要 `refs`。代价：每 key 多 1 B，以及"backward 按 prefetch 顺序结束"的前提。
- 暂缓的原因：多出的 1 B 开销，以及守卫已经可用。

**不采用每桶 bitmap**（讨论过）：pin 是计数，要每个在途步骤一份 bitmap；而且 unpin（backward 流）清 bit 与 trim（prefetch 流）搬动记录时复制 bit 之间没有顺序保证，会残留永远清不掉的 pin；要避免只能让 backward 等 prefetch 的 trim，与流水线重叠的目标相冲突。

### 5.4 风险

- 守卫的查询是对缓存哈希表的随机访问；懒检查把次数压到约 `k`，但每个被选中的候选都有一次，需要实测。
- trim 与缓存的耦合保持现状：存储 kernel 依赖缓存的数据结构布局（`EvictionGuardView`）。
- HBM_DIRECT 下对被选中的候选 gather `refs[row]`，同样是懒检查。

## 6. 与现有组件的关系

- **与现有 `key_index_map` 并存**：不改 `exportable_tables.py`、`INFERENCE_EMB::table_lookup`、`lookup_meta.py`，推理导出路径不受影响。
- **与 `embedding_cache` 共用底层**：`allocator.h`（显存分配契约）等可以复用。**不能**直接复用 `reclaim_queue`：它是无 score 的 FIFO，而存储要按 score 选 victim。
- **与缓存叠用时**：第一版沿用 `EvictionGuard`（§5），缓存的 `invalidate` 在 trim 之后对被淘汰的 key 调用（与现有流程一致）。
- **与下层 PS**：victim 在 trim 之后、`release_rows` 之前读出并写到下层；这一步新插入的 key 都被 pin，不可能成为本步 victim。

## 7. 与旧实现相比的变化

| | 旧实现 | 本方案 |
|---|---|---|
| key→value 关系 | slot 即 row | index 字段，全局行池 |
| rehash | 搬 value | 只搬记录 |
| 淘汰 | 先读旧值再写新值 | 新 key 取新行，victim 行完好，之后统一读出 |
| 同一批内 insert 又被淘汰 | 需要补丁 | 不可能（新 key 被 pin） |
| 被 admission 拒绝的 key | 不入表（现有也是在 insert 前过滤） | 同样不入表，且 `find_and_stage` 之后表没有被改动，任何一步失败都无需回滚 |
| 桶满 | insert 内即时淘汰，每个 key 各自规约 | `find_and_stage` 不改表，`commit` 先占空槽、不够再每桶规约一次 |
| 并发 | 桶内原子 + 锁 | 按桶排序后一个桶一个 worker，无原子 |
| 确定性 | 逐波 insert | 槽位、行号、victim 都确定（见 8.7） |
| pin | 槽位上的 `_ref_counter`（无条件分配）+ 缓存的 `EvictionGuard` | 缓存路径：守卫，无额外内存；HBM_DIRECT：按行号的 `refs`，只在该模式分配 |
| 查找 | digest 向量 + 读 key | `head` + 读 key，**复杂度与桶大小无关** |
| NO_EVICTION | 行号塞在 score 里，打包 slot_index | 无需特殊处理，桶放不下则扩容 |

## 8. 潜在问题与风险

"已知"表示已有设计应对；"待验证"表示依赖未测的假设。

### 8.1 活动空间的 2 B 寻址与残留节点的复制前移（待验证，复杂度高）
- 2 B 偏移要覆盖"本桶的槽 + 本桶的活动空间节点"，所以活动空间必须**按桶连续**，段起点由每步的段描述表给出（见 3.2）。
- 残留节点跨步存在，而新一步的新节点要和它们在同一个连续段里，所以每步要把残留桶的旧段复制到新段，再接新节点；活动空间需要双缓冲。
- 正常路径上残留为零，所以这是少见路径，但它必须正确，是 trim 里最复杂的一块。
- 讨论过的备选是每桶物理多留 `E` 个槽（偏移直接是槽下标，不需要段描述表和复制，代价是槽内存增加 `E/128`，E=16 约 +12.5%），**决定不采用**，用共享的活动空间。

### 8.2 排序的代价与"一次规约"的收益（待验证）
- 排序的代价：每步一次按 `(table, 桶)` 的基数排序，加上把结果还原成输入顺序的一次散射。量级预计不大，需要实测。
- "同一桶的 key 一次规约"的**摊销收益取决于每桶新 key 数**：大表（例如 1 亿行、约 78 万个桶，一步 10 万个新 key）平均每桶不到 1 个新 key，几乎没有摊销；收益主要在**小表或热点桶**。
- 但无论摊销与否，"一个桶一个 worker"带来了**无原子、确定性、同桶多个新 key 不再争用同一个 victim** 这些确定的好处。
- 大表下大部分桶段只有 0 到 3 个 key，worker 应取 warp 粒度（warp 内的 lane 对 128 个槽并行规约）而不是整个 block。

### 8.3 行池分配（已知）
- 全局行池在多个桶 worker 间仍是共享的。默认模式用原子弹栈（单遍，更快）；确定性模式用"先标记 miss、前缀和、再按序分配"的两遍做法，让行号确定（见 8.7）。
- 行池空闲栈每行 4 B（**已定**）；替代做法是把空闲行串成链放在 value 行自己的内存里，省掉这 4 B，但要碰 value 内存，在 host 上不划算，所以不采用。

### 8.4 补洞（swap-with-last）的正确性（待验证，复杂度高）
- 要重新算被搬动记录的链头、找前驱、改 `next`；与"活动空间节点占用 victim 槽位"在同一个 worker 里的顺序必须严格。
- 需要专门的单元测试：随机序列的 insert / evict / erase 之后，每个桶里链的结构一致、前缀无洞、每个 key 可查。

### 8.5 守卫的代价与顺序要求（待验证）
- 守卫要对每个被选中的候选查缓存哈希表。懒检查把查询数压到约 `k`，但仍是随机访问，需要实测；如果成为瓶颈，再考虑 §5.3 的步号方案。
- trim 之前，这一步的所有 key 必须已经被缓存 pin 住（缓存 `allocate` 先于存储 trim），否则新 key 会成为候选。重构 `_prefetch_cache_path` 时要保住这个顺序。
- HBM_DIRECT 下 `refs[row]` 要在 prefetch 加、backward 减，与现有 `_ref_counter` 的使用一致，只是索引从槽位换成行号。

### 8.6 lookup 性能（待验证）
- 理论上"`head` + 读 key"和"digest 一行 + 读 key"访存次数相近，未命中时 `head` 为空的概率约 37%，应当更便宜；命中且链长大于 1 时要多走几步。
- **没有实测**。需要和现有实现对比，桶容量取 128 与 1024（缓存选项里出现过 1024）。活动空间节点本来就挂在桶内的链上，lookup 沿链走即可，**不需要单独检查活动空间**；只有走到偏移 ≥ `bucket_capacity` 的节点时才需要查段起点（残留节点，少见路径）。

### 8.7 确定性（已定）
- **只有设置了环境变量 `DEMB_DETERMINISM_MODE` 才保证确定性**；要求是**同一输入两次运行的结果一致**，**不要求与旧实现的结果对齐**。不开启时怎么快怎么做。
- 两种模式的差别只在几个点：

| | 默认（快） | 确定性模式 |
|---|---|---|
| 行号分配 | 对全局空闲栈做原子弹栈，单遍 | 先标记、前缀和，再按序分配（两遍） |
| 桶内排序、victim 的平局 | 不要求稳定 | 按 `(排名, 位置)` 全序，排序稳定 |
| 其余（按桶排序、桶内独占处理） | 相同 | 相同 |

- 按桶排序后每个桶由一个 worker 独占，新 key 落在哪个槽、选哪个 victim 本来就由输入顺序和比较键决定，**所以不再需要现有的逐波 `_deterministic_insert`**。
- 测试：确定性模式只需要"同一输入跑两次，比较槽位、行号、被淘汰的 key 是否一致"，不需要沿用旧实现的结果。

### 8.8 增量 dump / replay 与 checkpoint 的格式（已定：不兼容）
- 现有的增量 dump 和 replay（`replay_increment_design.md`）依赖"slot_index 既是 key 槽位也是 value 行"，NO_EVICTION 靠高低 32 位打包。新实现里 key 槽位与行号分离，`slot_index` 要携带行号，格式随之改变。
- **不兼容旧 checkpoint 和旧增量 dump**：兼容很难，新实现这次改完之后保持自己的格式一致即可；需要兼容的用户使用旧版本（旧实现并存，不受影响）。
- 因此新实现的 dump / load / 增量 dump / replay 可以直接按新布局设计，不需要迁移或读取旧格式的代码。

### 8.9 淘汰语义变化
- 旧：insert 时桶满即时淘汰。新：insert 之后 trim 淘汰，此时 score 已经被这一步的访问更新，victim 的选择可能与旧实现不同。
- 影响断言精确淘汰结果的测试（如 `test_lru_lfu`、`test_table_operation`、`retain_evicted_keys`），需要重新核对而不是直接沿用。
- 整表 top-K 是新语义，桶内是接近旧语义的默认。

### 8.10 小 value 表的内存开销与大容量 host 表
- dim 16 的表，索引开销占比从约 25% 升到约 30%（约 +8% 的总体积，缓存路径）。
- 大容量、value 放在 host 的表，索引在显存里，容量上限由索引决定：现有约 21 B/key，80 GB 可索引约 38 亿；本方案约 28 B/key（缓存路径），约 28 亿。若目标场景是单卡几十亿行的 host 表，需要再压缩或把索引放到 host。

### 8.11 整表 top-K 的成本
- 每次 trim 是 O(N) 的扫描，必须用高低水位摊薄；桶间不均衡时，整表 top-K 的结果与桶内也不同。

### 8.12 每表一个 map 的 kernel 间接访问
- kernel 要按表读设备端指针数组（桶基址、桶数、各数组指针）。与现有读 `table_bucket_offsets` 同量级，但要确认多表融合的 kernel 吞吐不回退。

### 8.13 NO_EVICTION 下的扩容频率（待验证）
- 桶放不下就扩容，如果整体负载因子较高，个别桶溢出会频繁触发扩容。估算：负载因子 0.9、桶 128 时，约 11% 的桶会超过 128，所以必须靠整体负载因子阈值提前扩容，把"桶溢出触发扩容"只当作兜底。

### 8.15 两次排序与扩容后桶号失效（待验证）
- 暂存的新 key 在 `find_and_stage` 时算出的桶号，在扩容之后就不对了（桶数变了），所以 `commit` 要重新排序；`find_and_stage` 的排序也不能沿用到 `commit`。
- 代价：多一次针对新 key 的排序，规模小于第一次，预计不大，需要实测。
- 如果扩容发生在 `find_and_stage` 之前（现有的顺序），就能省掉这次重排，但也就失去了"PS 查找与扩容并行"。扩容在稳态下很少发生，这个并行的收益有限，实现时可以先按现有顺序（扩容在前）做，之后再决定要不要打开重叠。

### 8.16 暂存节点不入链带来的约束（已知）
- 暂存节点在 `commit` 之前对 lookup 不可见。同一步内不会有人去查同一步新插入的 key（key 已经去重），下一步的 lookup 在 `commit` 之后，所以正常路径没有问题。
- 需要核对：缓存路径里 `_prefetch_cache_path` 在 `commit` 之前对存储做的操作（victim 写回时的 `_find_keys`）只涉及此前就已存在于存储里的 key，不涉及本步的新 key。读代码的结论是成立的（本步新建的缓存节点已被 pin，不会成为 victim），但需要用测试确认。
- 行号推迟到 `commit` 才分配，所以 HBM_DIRECT 路径里"写初始值或 PS 的值"必须排在 `commit` 之后。

### 8.14 不属于本方案、但会被问到的
- 推理导出（`exportable_tables`）仍用旧实现；如果以后要导出新格式，`INFERENCE_EMB::table_lookup` 的 schema 要再变，旧导出模型不兼容。
- 已知未解决问题与本方案无关但要记着：PS 的 score 历史在回到前层表时丢失；`erase` 不会传到 PS；dump/load 不包含 PS。

## 9. 待定项

| # | 问题 | 倾向 |
|---|---|---|
| 1 | ~~活动空间：共享段还是每桶多留槽~~ | **已定：共享的按桶连续段**；残留节点的复制前移分阶段实现，见 §11 |
| 2 | ~~空闲行用 4 B 空闲栈还是串在 value 行里~~ | **已定：空闲栈**（每行 4 B，借用 value 内存串链要碰 host 内存，不划算） |
| 3 | ~~默认淘汰范围~~ | **已定：桶内 128** |
| 4 | ~~trim 的水位~~ | **已定：只腾出本步所需**，整表 top-K 时才用水位 |
| 5 | ~~确定性模式要求到哪一层~~ | **已定：只在设置 `DEMB_DETERMINISM_MODE` 时保证，只要求两次运行自洽**（见 8.7） |
| 6 | ~~旧 checkpoint / 增量 dump 是否兼容~~ | **已定：不兼容**（见 8.8） |
| 8 | ~~新 key 是否统一先进活动空间~~ | **已定：统一先进**，不区分有无 admission；`commit` 时先占空槽，不够再淘汰 |
| 7 | ~~pin：步号方案还是引用计数~~ | **已定：第一版沿用守卫**，HBM_DIRECT 用按行号的 `refs`；步号方案留作将来的优化（§5.3） |

## 10. 验证计划

1. **微基准**：新链式查找 vs 现有 digest 查找，桶容量 128 与 1024，命中率分别取 0、50%、100%。
2. **桶占用模拟**：给定表容量、批大小、在途步数、key 分布，统计每桶新 key 数分布和"残留"比例，确定 8.1、8.2 的估算是否成立。
3. **排序开销**：每步排序加散射还原在典型批大小（10 万到 100 万 key）下的耗时。
4. **正确性**：随机序列的 insert / trim / erase，检查链结构一致、前缀无洞、每个 key 可查、被守卫保护的 key 不会成为 victim（含同一个 key 被连续多步 pin）。
5. **内存**：用 `memory_usage` 实测每 key 开销，与 3.4 的表对账。
6. **与旧实现的等价性**：桶内淘汰范围下，同样的输入序列，被淘汰的 key 集合与旧实现对比（预期会有可解释的差异，见 8.9）。

## 11. 实施阶段（建议）

### 第一阶段：已实现

代码：`src/chained_table/chained_table.cu`（`find` 与 `place` 两个 kernel）、`dynamicemb/chained_table.py`（`ChainedTable`）、`test/unit_tests/chained_table/test_chained_table.py`（26 个测试）、`benchmark/bench_chained_table.py`。

做了什么：单表，`find_and_stage` / `commit` / `release_rows`，守卫沿用 `EvictionGuardView`，分区内一次规约选出全部 victim，行池空闲栈，出现残留时抛 `ChainedTableFull`（带上已放置部分的 `partial` 结果，未放置的行已归还）。

测试：以几行 python 写的模型为对照（桶内槽位、行号、victim 及其顺序逐项比较），覆盖 4 种规模、3 种 score 策略、随机 admission 掩码；另有守卫（用真实的 `EmbeddingCacheCore`）、全部被保护的桶、行池耗尽、uint64 key、链结构不变量检查。EOS（H100）上 26 个全部通过；与之相邻的 `test_eviction_guard`、`test_ps_tier`、`test_embedding_cache_core` 仍通过。

与设计相比的差别（有意为之，后续阶段再收口）：
- **`find` 没有按桶排序**：key 已去重，每个线程只写自己命中的那个槽，没有写冲突，所以直接一个线程一个 key。按桶排序只用在 `commit`（放置阶段）。§4.1 说的"`find_and_stage` 前排序"在这一阶段没有做，是否值得做取决于后面的实测。
- **排序、分段、行池栈都在 python 层**，用 `torch.argsort`、`unique_consecutive`、`nonzero` 完成，每步有几次主机同步；之后换成 C++/cub。
- `bucket_size` 用 int32，`next` / `head` 用 int16 张量按 uint16 使用，行号用 int32 张量按 uint32 使用。
- 只有单 score 字（S=1），score 策略为 `assign` / `accumulate` / `none`，排名就是 score 升序；LruLfu 的两字 score 和 JIT 比较器放到后面。
- 单表；`table_id` 只用于守卫。
- **一个桶一步内的新 key 数不能超过桶容量**：新 key 不会淘汰同一步的其他新 key，所以 victim 只能来自已存在的成员，新 key 数大于桶容量就是容量错误（§8.2 讨论过的情形，这里作为明确的错误处理）。

### 微基准（EOS，H100，单表 1677 万槽，桶 128，约 96% 满）

对比现有的 `LinearBucketTable`；时间是每次调用的毫秒数，现有实现的 lookup 计时里包含它每次分配 `table_ids` 的开销，所以对它略不利。

| 批大小 | 操作 | 现有 ms | 链式 ms | 现有/链式 |
|---|---|---|---|---|
| 65536 | 命中 lookup | 0.030 | 0.008 | 3.8× |
| 65536 | 未命中 lookup | 0.054 | 0.006 | 9.0× |
| 524288 | 命中 lookup | 0.112 | 0.063 | 1.8× |
| 524288 | 未命中 lookup | 0.188 | 0.046 | 4.0× |
| 65536 | 整步（全是新 key，满表） | 0.218 | 0.429 | 0.5× |
| 524288 | 整步（全是新 key，满表） | 1.026 | 0.781 | 1.3× |

解读：
- **lookup 比现有实现快**：与 §8.6 的担心相反，`head` 加读 key 的路径在这个规模下没有比 digest 向量慢，未命中尤其便宜。
- **整步在小批量时更慢**：链式的整步包含 python 层的排序、分段和几次主机同步（见上），小批量被这些固定开销主导；批量增大后已经比现有实现快。这一部分是第二阶段要用 C++ 实现替换的对象，不代表最终水平。
- 只测了单表、批内全是新 key 这一种整步；命中占比、分布倾斜和多表都还没有测。

### 第二阶段：已实现

做了什么：
- **每表一个 map**：每张表有自己的数组和自己的行池；kernel 通过设备端的 `TableView` 数组（每表一项，含各数组指针、桶数、行池栈和栈顶）一次 launch 服务多张表；`find` / `place` 按每个 key 的 `table_id` 取对应的视图。
- **扩容只搬记录**：`expand(table)` 重建一张表，只搬 `(key, score, index)`，每个 key 保持原行号，value 不动；其他表的数组原封不动（测试里核对了指针没变）。扩容后行池容量随之增长，新行入栈，调用方的 value 池要跟上（`row_capacity_of`）。某个桶装不下时（负载太高）会自动再翻倍。
- **NO_EVICTION 扩容兜底**：`no_eviction` 的表在 `commit` 时，若某个桶放不下它的新 key，就扩容并重新排序，直到放得下，不淘汰任何 key。
- **按行号的 pin**：`track_refs` 的表带一个按行号的引用计数；`find_and_stage(pin=True)` 对命中的行加引用，`commit(pin=True)` 对新放置的行置 1，`pin` / `unpin` 显式增减。引用数大于 0 的行不会被选为 victim（与缓存的守卫并列检查）。
- **排序、分段、行池栈挪进 C++**：`chained_stage`（压紧未命中的 key）、`chained_plan`（按桶基数排序、按桶切段、每段需要的 victim 数）、`chained_apply`（分配行号、放置、把行号还原成输入顺序）、`chained_release`。行池栈的栈顶放在设备上，行号分配和放置之间没有主机同步；一步只有两次小的主机同步（`find_and_stage` 之后读未命中个数，`commit` 之后读统计）。池耗尽在设备上标记，整个 `commit` 跳过。

测试：沿用第一阶段的模型逐槽位对照，行号改为按"桶序"分配（与排序一致）；新增多表对照、同一个 key 在两张表里是两个 key、扩容保持 key/score/行号、不淘汰的表随写入增长、桶一步内新 key 超过槽数时不淘汰的表扩容、pin 保护、全部被 pin 的桶报错。EOS 上 37 个测试全部通过，相邻的测试仍通过（共 71）。

与设计相比的差别：
- 行号分配的顺序是"桶序"（排序后的顺序），而不是输入顺序；每个表内是确定的。
- 默认模式也用同一套确定的分配（没有单独做"原子弹栈"的快路径，因为这一步已经不是瓶颈）。
- `find` 仍然不按桶排序（见第一阶段）。
- 释放行时不检查是否超过容量（设备上没有廉价的检查）；多表时 `release_rows` 对每张表各调用一次。

### 微基准（第二阶段后，同第一阶段的配置）

| 批大小 | 操作 | 现有 ms | 链式 ms | 现有/链式 |
|---|---|---|---|---|
| 65536 | 命中 lookup | 0.029 | 0.011 | 2.7× |
| 65536 | 未命中 lookup | 0.055 | 0.010 | 5.3× |
| 524288 | 命中 lookup | 0.106 | 0.069 | 1.5× |
| 524288 | 未命中 lookup | 0.189 | 0.053 | 3.6× |
| 65536 | 整步（全是新 key，满表） | 0.217 | 0.319 | 0.68× |
| 524288 | 整步（全是新 key，满表） | 1.028 | 0.771 | 1.33× |

与第一阶段相比，小批量的整步从 0.429 ms 降到 0.319 ms（现有实现 0.217 ms）；大批量基本没变（0.781 → 0.771 ms），因为那里的时间不在 python 层。lookup 的数字波动在测量误差之内（两次运行之间有差别，没有做重复统计）。

**瓶颈是 `place_kernel`**：524288 个新 key 的一步里，GPU 总时间约 0.67 ms，其中 `place_kernel` 约 0.40 ms。拆开看（同一个批大小）：表只有 25% 满、不需要淘汰时它也要约 0.17 ms，淘汰时 0.40 ms。原因是每个桶由一个 warp 里的 lane 0 顺序地写入新 key、摘链、接链，每个桶都是一串依赖的访存；选 victim 的循环又增加了约 0.23 ms。

### `place_kernel` 的第一轮优化

做了两处：
- **选 victim**：每个 lane 把自己负责的槽（槽 j 属于 lane `j % 32`）的 score 读进寄存器，之后每个 victim 只要一次 warp 规约，不再读内存、不再用共享内存和同步；是否被保护由持有该槽的 lane 判断并广播，不再由 lane 0 串行。
- **放置**：32 个新 key 一组，每个 lane 一个。先把 victim 从各自的链上摘掉（同一条链由其中编号最小的 lane 走一遍链、跳过已标记的节点，不同链并行），再写记录，最后接链（同一条链上的 key 按 lane 次序依次接，没有原子操作，次序确定）。

结果（同一配置）：

| | 之前 | 之后 |
|---|---|---|
| `place_kernel`，表 25% 满、不淘汰 | 170 µs | 134 µs |
| `place_kernel`，表 99% 满、淘汰 | 401 µs | 308 µs |
| 整步，65536 个新 key | 0.319 ms | 0.305 ms（现有 0.219 ms） |
| 整步，524288 个新 key | 0.771 ms | 0.664 ms（现有 1.026 ms，**1.54×**） |

测试：新增三个规模（最后三个桶内新 key 超过 32 个，kernel 分多组放置，多个 victim 和新 key 落在同一条链上），EOS 上 46 个测试全部通过，仍然逐槽位对照模型。

收益比预期小（`place_kernel` 约快 25%）。不淘汰时仍要 134 µs，说明剩下的时间不在选 victim 或链操作的串行度上，更可能是每个桶要访问的数据分散在多条缓存行、warp 的驻留数不够；要确认需要用 ncu 看访存和占用率，这一步**还没做**。

### 第四阶段：prefetch 接入（已实现，实验性）

模块用 `ChainedStorage`（`dynamicemb/chained_storage.py`）作训练存储（最初由环境变量 `DEMB_CHAINED_INDEX` 开关，旧路径删除后已无此开关）；prefetch 的两条路径在 `dynamicemb/chained_prefetch.py`。

覆盖范围：
- **缓存路径**（HBM 缓存加 host 后备，后备用链式索引）和 **HBM-direct 路径**（没有缓存），都可以带下层 PS。
- 训练的 prefetch、forward、backward，`flush`，以及 eval 的 `find`。
- 单 score 字的策略：TIMESTAMP、STEP、CUSTOMIZED、LFU、NO_EVICTION；LruLfu（两字 score）直接抛 `NotImplementedError`。
- 扩容：不淘汰的表在 `commit` 里增长；其他表按 `max_load_factor` 增长到 `max_capacity`；value 缓冲随行池一起追加（`sync_value_buffers`）。

没有覆盖：host-only 与 hybrid 布局（仍用旧存储）、dump / load / 导出 / 增量 dump / replay / `fill_tables`（`ChainedStorage` 对 dump / load 抛 `NotImplementedError`，第五阶段）。

两条路径按 §4.10 的顺序重写：缓存路径是"缓存 pin → `find_and_stage` → 下层 PS → admission → 缓存 allocate → 缓存 victim 写回 → 载入已有的行、初始化新行 → `commit`（传入缓存的守卫）→ 被淘汰的 key 离开缓存 → publish"；HBM-direct 路径在 `find_and_stage` 时 pin 命中的行，`commit(pin=True)` 给新行加引用，backward 里 `unpin`。因为新 key 拿的是新行，victim 的行在释放前原封不动，所以：
- 不再需要 `take_values_of_keys_evicted_in_the_same_insert` 那一套，也不再有 `select_insert_failed_values`；victim 的值直接从它的行里读；
- 缓存 victim 写回时，所有被缓存的 key 在存储里都必须有行，否则抛 `RuntimeError`（旧代码里的 "gone" 分支）：这里没有 load / erase 能打破这个不变量。

与旧实现的差别：
- TIMESTAMP 用每次访问递增的计数（单调即可），不是设备时钟。
- 选 victim 只在桶内、按 score 升序，被守卫或引用计数保护的跳过（与旧实现的桶内淘汰语义一致，具体选出哪些 key 可能不同，见 §8.9）。

测试（`test_chained_prefetch.py`，24 个，EOS 上全部通过）：
- **端到端检查**（沿用 `test_ps_tier`）：每个 key 从 1.0 开始，loss 是输出之和，每次出现梯度为 1，SGD 之后 key 应该是 `1 - lr × 出现次数`，无论它被淘汰、送到下层、再取回多少次。HBM-direct 与缓存两种布局，TIMESTAMP / LFU / STEP 三种策略；一个控制组（下层把收到的东西都丢掉，检查必须失败）；一个"容量够装下所有 key"；一个"不淘汰的表随 key 增长"；一个"淘汰的表从 `init_capacity` 增长到不超过 `max_capacity`"。
- **与现有索引逐步对照**：两张表（dim 8 和 16）、SUM 池化，同样的输入，每一步 forward 的输出必须逐位相同（`torch.equal`）；HBM-direct 与缓存、TIMESTAMP 与 LFU、有无频次 admission（阈值 2）共八个组合；admission 的用例还核对了"确实有 key 被拒绝"，否则这个检查没有意义。
- **带单独 prefetch 流的流水线**：下一步在另一条流上 prefetch，而上一步的 backward 还排在一个长 kernel 后面。缓存布局下，去掉 prefetch 状态的流记录（`record_stream`）就会非法访存，加上则 120 步没有一个错的 key（用与旧路径相同的实验确认过）；HBM-direct 布局两种情况下都没有现象，所以它不是对照。
- 每个测试后用 `check_invariants` 核对没有行丢失或重复。

另外把主测试文件 `test_batched_dynamic_embedding_tables_v2.py` 在 `DEMB_CHAINED_INDEX` 下整体跑了一遍（仅放宽两处只认旧存储类型的断言）：
- `test_forward_train_eval`（160 个，含 SUM / MEAN / NONE 池化、dim 7 / 8 / 11 / 13 / 16 / 32、sgd / adam、缓存带或不带 PS）全部通过，其中缓存与 HBM-only 两种布局（约 160 个中的 96 个）走链式路径，host-only 与 hybrid 仍走旧存储；
- `test_backward`、`test_empty_batch`、`test_export_keys_values_empty_table`（97 个）、`test_prefetch_flush_in_cache`、`test_multi_table_mixed_dims_forward_backward`、`test_multi_table_with_admission`、`test_multi_table_caching_flush` 全部通过；
- 失败的都是针对旧实现内部或 checkpoint 的用例，本阶段不适用：dump / load 相关（约 40 个，抛 `NotImplementedError`，第五阶段）、`test_fill_tables`（12 个）、`test_table_expansion_capacity_growth`（32 个）、`test_deterministic_insert`（4 个），后三者直接读写 `key_index_map` 或断言存储类型，对链式路径需要另写（扩容已由上面的端到端用例覆盖）。

旧路径的全套分片回归结果与之前一致。

**没有测的**：多 GPU、旧实现内部相关的那几类（`fill_tables`、确定性插入）在链式路径下的对应测试。

### 微基准（端到端，H100，一张表 419 万行、dim 64、SGD、每步 65536 个 key，key 取自 4 倍表大小的全集，表已满）

| 布局 | 现有 ms/步 | 链式 ms/步 | 现有/链式 |
|---|---|---|---|
| HBM-direct | 1.209 | 1.154 | 1.05× |
| 缓存（HBM 缓存为 1/4，host 后备） | 3.597 | 2.480 | 1.45× |

一步包含 prefetch、forward 和 backward。HBM-direct 上几乎持平（forward / backward 占了大头，索引的差别被稀释）；缓存路径上快 45%，说明后备的 `find_and_stage` + `commit` 比旧的"查找 + 带守卫的 insert + 搬值"省得多。只测了这一个规模和这一种分布（均匀）；倾斜分布和更小的批没有测。

### 第五阶段：checkpoint、增量 dump 与 replay、host-only（已实现，实验性）

**checkpoint（dump / load）**：文件格式不变（key、embedding、score、优化器状态、meta json 各一个文件）。这些格式本来就不含槽位，所以不需要新格式；`ChainedStorage` 给自己的 `state.key_index_map` 装了一个视图（`_ChainedIndexView`），它回答 checkpoint 代码要读的几件事：一张表的 key 与 score 与行号、一个返回行号的 `insert`、容量与大小。于是现有的 `_dump_table`、`_load_key_values`、`_validate_load_meta`、`export_keys_values_iter` 不用改，直接复用。TIMESTAMP 策略的 score 用设备时钟（与旧实现相同），这样 checkpoint 里的"年龄 = 时间戳 − score"含义一致。
- 主测试文件里 dump / load 相关的 43 个用例（单表、多表、`dist_type`、旧版 meta、优化器状态精度和宽度等）在 `DEMB_CHAINED_INDEX` 下全部通过。

**erase**：补上了设计里的"补洞"：`ChainedTables.erase`，一个线程负责一个桶的一段 key，摘链后用桶内最后一条记录填洞，并把被搬动记录的链指向新槽，`bucket_size` 减一，行还回行池。与模型逐槽位对照（4 种规模，每步随机插入再随机删除，含不存在的 key）。

**增量 dump 与 replay**：
- `incremental_dump` 直接复用现有存储的实现（它只读 `_state`），`slot_index` 在这里是行号，仅供参考。
- `replay_increment` 改为**按 key**：已有的 key 取 delta 的值（和 score，按 `content`），没有的 key 放进表里，槽位和行号由表分配。现有的 replay 是逐槽位的，只在布局相同的两张表之间才有意义，所以这里不检查容量、行容量、桶容量，只检查维度、score 策略、world size、`dist_type`。
- **被源端淘汰的 key 也会在副本里删除**（先删除，再写 delta 里的 key，所以一个窗口内被淘汰又回来的 key 会保留）。现有实现里写槽位本身就会复现淘汰，所以它不处理 evicted keys，而这里必须处理。
- 测试（`test_chained_incremental.py`）：副本与源逐位给出相同的答案、持有相同的 key 集合（HBM-direct 与缓存、TIMESTAMP / LFU / STEP）；源端淘汰过的 key 不出现在副本里；源端 erase 的 key 在副本里也被删；一个窗口里被 erase 又写回的 key 保留。

**host-only 布局**：`_make_storage` 现在也服务 host-only（没有缓存、value 在 host），走原有的通用 forward 路径（`find` / `insert`）。`test_forward_train_eval` 的 160 个用例（含 host-only 的 32 个）全部通过。

**确定性**：`commit` 与 `erase` 本身没有原子操作和依赖线程调度的步骤；同样的输入两次运行，key、score、行号、链和空闲行完全一致（带大量 score 平局的 25 步随机序列，含 erase），已加为测试。

**evicted keys 的保留**（`evicted_item_mode=RETAIN_KEY`）：`pop_evicted_keys` / `pop_erased_keys` 可用；测试核对"每个用过但不在表里的 key 都被报告过、没有报告没用过的 key、pop 之后清空"，HBM-direct 与缓存两种布局。

测试总数：链式目录 88 个全部通过；旧路径全套回归与改动前一致；主测试文件在 `DEMB_CHAINED_INDEX` 下：`test_forward_train_eval`（160）、`test_backward` 等（97）、dump / load（43）全部通过。

**仍然不支持或不适用**（用现有套件在链式下跑，失败的都是下列原因）：
- **hybrid 布局**（HBM 层加 host 层）：仍用旧存储。
- **复合 score 策略**（如 `(TIMESTAMP, LFU)`，即 LruLfu 两字 score）：构造时抛 `NotImplementedError`；增量 dump / retain 里大部分现有用例用的就是它们。
- **桶容量大于 256**：构造时抛 `ValueError`（有一个现有用例用 1024 的存储桶）。
- **逐槽位 replay 的语义**：现有的"布局不匹配必须拒绝""evicted keys 不能删除任何东西"这类用例对按 key 的 replay 不适用。
- `fill_tables`、`test_table_expansion_capacity_growth`（读写 `key_index_map` 或假设旧的扩容阈值）、`test_deterministic_insert`（比较两个旧索引）：对链式路径需要另写，部分已由上面的用例覆盖。
- 多 GPU / 分布式的 incremental dump 与 replay 没有测（单卡路径通过 `pg=None`）。

### LruLfu：两字 score（已实现）

复合策略 `(TIMESTAMP, LFU)`（任一顺序）现在可用：每个槽两个 score 字，字 0 是最近一次访问的时间戳（设备时钟），字 1 是累计频次；与旧实现相同，物理布局固定，用户给的顺序只影响 checkpoint 的列序。

- **排名**：频次最小的先被淘汰，频次相同的先淘汰时间戳最老的，再相同取槽位小的（与旧实现默认的 `LexFreqTsComparator` 一致，外加槽位保证确定）。选 victim 时每个 lane 把自己槽的两个字读进寄存器，一次规约比较 `(频次, 时间戳, 槽位)`。
- **更新**：命中时时间戳置为当前时钟、频次加上这一步统计的次数（没有则加 1）；新 key 以 `(当前时钟, 次数)` 进入。
- **C++**：`TableView` 带 `num_scores`；`find`、暂存、排序、放置、erase 都按每槽两个字处理。
- **checkpoint 与增量 dump**：旧代码里取多字 score 的辅助函数按槽位索引，而这里导出的"索引"是行号，所以适配视图维护一张"行号→槽位"的映射（表发生变化时重建，版本号由 `ChainedTables.version` 提供），`gather_score_blocks` / `scatter_score_blocks` 经由它读写整块 score。增量 dump 对时间戳字设阈值（时间窗口），与旧实现一致。
- **不支持**：自定义 `score_function`（旧实现用 JIT 编译成 cubin 做比较器），构造时抛 `NotImplementedError`。

测试：
- 与模型逐槽位对照（4 种规模，每步随机的时钟和频次，含大量平局），以及"频次小的先走，同频次时间戳老的先走"、命中同时更新两个字、erase 与扩容保留两个字。
- 端到端：带下层 PS 的复合策略表（HBM-direct 与缓存）不丢任何更新；与现有索引逐步对照（有无频次 admission）。
- **用现有套件在链式下重跑**：增量 dump（`test_batched_dynamicemb_tables.py`）18 个全部通过（之前 18 个失败）；`test_pop_evicted_keys.py` 5 个、`test_insert_collect_evicted.py` 13 个全部通过；分布式增量 dump 在 1 卡和 8 卡下各 16 个全部通过；`test_replay_increment.py` 27 个中 22 个通过，其余 5 个测的是旧实现逐槽位 replay 的语义（布局不匹配必须拒绝、evicted keys 不能删除任何东西）或用了超过 256 的桶容量。

**已知限制**：
- 一个桶在**一步里的新 key 数不能超过桶容量**（新 key 不会淘汰同一步的其他新 key）。表比每步的 unique key 还小的玩具配置会抛 `ChainedTableFull`，旧实现则会在步内淘汰；分布式的 `test_distributed_pop_evicted_keys.py` 在 8 卡下就是这个原因失败（每个 rank 的表只有一个桶、一步 256 个新 key）。要消除这个限制需要 §4.4 的残留节点。
- 桶容量最大 256。

### hybrid 布局（已实现）

`ChainedHybridStorage`（`chained_storage.py`）：HBM 层与 host 层各是一个 `ChainedStorage`，**key 不相交**。新 key 进 HBM 层；HBM 层淘汰的 key 连同它们的 score 下沉到 host 层（`CommitResult` 现在带着 `victim_scores`，由 `place_kernel` 在覆盖槽位之前读出）；host 层淘汰的被丢弃（或按 `evicted_item_mode` 保留）。查找先问 HBM 层再问 host 层；更新就地写行，不移动 key（`preserve_existing`）。

- **checkpoint**：先写 host 层，再把 HBM 层追加到同一组文件里（与旧的 hybrid 存储一致）；加载走 HBM 层，它的淘汰自然下沉到 host 层。带两字 score 的 hybrid 仍然不支持 dump（两层共用一个 score 文件），与旧实现相同。
- **增量 dump 与 replay**：把两层的结果拼起来；replay 对 host 层已有的 key 在原处更新，其余的走 HBM 层。
- **`ChainedStorage.find` 现在在训练态更新 score**：之前它是只读的（对 HBM-direct 与缓存路径没有影响，因为它们走 `find_and_stage`），但 host-only 与 hybrid 用的通用 forward 路径靠 `find` 更新被访问 key 的 score，否则淘汰顺序不反映访问。这是之前的一个缺陷，现在已修，并加了对应的测试。
- **桶容量**：旧 hybrid 的 HBM 层用 1024 的桶；链式索引上限 256，所以这一层改用 128。

测试：新增 hybrid 的端到端用例（每个 key 恰好在一个层里、两层 key 不相交、什么都没丢、TIMESTAMP / LFU / LRU_LFU 三种策略、两层都装不下时被丢弃的与保留的都账目对得上）、host-only 下的 score 更新、hybrid 与 host-only 的增量 dump 与 replay。**全部五种布局**（HBM-only、host-only、缓存带或不带 PS、hybrid）下，主测试文件的 `test_forward_train_eval` 160 个、`test_backward` 等 97 个、dump / load 43 个在链式下全部通过；链式目录 107 个测试通过。

### 后续阶段

3. 残留节点的复制前移（软超限，同时解除"一个桶一步里的新 key 数不能超过桶容量"的限制）；整表 top-K 淘汰、排名函数与自定义 `score_function`；`place_kernel` 的进一步优化；小批量的固定开销。
6. （已完成，见下）删除旧的训练路径。

### 删除旧的训练路径（已完成，分支 `chained-only`）

训练只走链式索引：`DynamicEmbStorage`、`HybridStorage`、旧的两条 prefetch 路径（`_prefetch_cache_path` / `_prefetch_hbm_direct_path`）、`no_eviction_scores.cu` 及其绑定、以及 `key_value_table.py` 里只被它们用到的辅助函数都已删除（`key_value_table.py` 从约 4400 行降到约 1900 行）。`DEMB_CHAINED_INDEX` 不再存在，`BatchedDynamicEmbeddingTablesV2` 总是构造 `ChainedStorage` / `ChainedHybridStorage`。

**保留**（推理或其他模块仍在用）：`exportable_tables.py`、`lookup_meta.py`、`lookup_torch_binding.cu`、`table_insert` / `INFERENCE_EMB::table_lookup`（schema 与上游一致）、`scored_hashtable.py`（`KVCounter` 用）、`_iter_batches_from_files` / `load_from_json` / `resolve_checkpoint_value_layout`。这些与上游逐字节一致或行为一致，见 `debug_inference_check`（本地）里的对比探针。原分支 `lw-cache-wip`（只替换 Cache）与 `lw-cache-squashed` 不动，可随时回到那里。

**测试的处理**：依赖旧实现内部的用例删除——`test_no_eviction_row_indexing`、`test_hybrid_storage_export`、`test_expansion_value_capacity`、`test_deterministic_insert`、`test_table_expansion_capacity_growth`、`test_fill_tables`、以及"链式对旧索引逐步比对"的差分用例（旧索引已不存在）；其余改用 `ChainedStorage`。`test_alignment` 的桶容量参数改为 128 / 256（链式上限 256），值缓冲的行数断言改为"大于索引容量"（缓冲含每步新 key 用的备用行）。

**迁移时测出并修掉的缺陷**：`ChainedStorage.load` 对两字 score 的 checkpoint 把 `num_scores` 写死为 1，且没有把按用户配置顺序存的列换回物理顺序（`score_load_permutation`）——LFU 在前的复合策略 dump → load 后两个字互换。主测试文件没有覆盖这一点，由 `test_lru_lfu_storage_dump_load_roundtrip` 测出。

### 桶容量与链接宽度（已实现）

- **`bucket_capacity` 取任意正整数**。不小于每卡行数时（`MAX_BUCKET_CAPACITY = 2**63 - 1` 总是如此）整张表就是一个桶，大小正好是每卡行数，不补齐；否则每卡行数向上补到 `bucket_capacity` 的倍数。16 的倍数、不超过 256 这两条限制都已去掉（16 的倍数只是旧 `ScoredHashTable` 的要求，它自己会取整）。`BUCKET_ALIGNMENT` 一并删除。
- **同一存储内的表共用 `bucket_capacity`**：它进了表的分组键，所以"整表一个桶"时规模不同的表会分到不同模块。hybrid 的两层各自的桶：用户的桶容量不小于整表的 `max_capacity` 时，每层是一个桶、大小为该层的行数，否则沿用用户的值。
- **链接（`head` / `next`）宽度**：默认 2 字节；`4 × bucket_capacity` 放不下 2 字节（含"无"这个值）时用 4 字节，即桶容量超过 16383。所有表同宽。
- **大桶的淘汰**：桶容量不超过 256 时仍在寄存器里选 victim（快路径不变）；更大的桶每选一个 victim 就扫一遍桶，找排名在上一个之后的最小者，所以被守卫或引用拒绝的成员不用额外标记。代价是一步里每个桶的淘汰开销约为 `victims × 桶大小 / 32`，大桶适合小表。
- **淘汰标记**：离开的 key 的槽位用其行号的最高位标记（原来是按 256 开的共享内存数组），所以行数上限变为 2^31 - 1，没有东西再按桶大小开辟。

### 混合维度的值缓冲（`MixedDimBuffers`）

一个模块合并的表维度可以不同，每张表有自己的 `[行数, value_dim]` 缓冲。`dynamicemb/mixed_dim_buffers.py` 的 `MixedDimBuffers` 把这些缓冲、每张表的维度（设备、host、列表三份）、最宽维度、是否都能按 4 对齐、各缓冲基址的设备数组，以及对它们的操作和 kernel 调用放在一起：`load` / `store`（整行或只取 embedding，补齐到最宽）、`load_table` / `store_table`（单表紧凑行）、`ensure_rows`（缓冲增长并刷新基址）、`update_rows`（优化器原地更新存储的行）、`update_padded`（对已读出的补齐行更新）、`reset_optimizer_states`。`DynamicEmbTableState` 只持有一个 `values`；原来散在状态里的 11 个字段和 `load_from_flat` 等模块函数、`get_table_ptrs` 都没有了，缓冲增长时基址的刷新只在这一处。有单独的单元测试 `test_mixed_dim_buffers.py`。
