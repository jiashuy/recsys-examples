# 轻量化 Embedding Cache 设计提案（索引融合方案）

> 状态：草案（Draft）
> 适用范围：DynamicEmb **训练**路径的 GPU 侧 embedding 缓存
> 关联代码：`corelib/dynamicemb/`
> 与 `lightweight_cache_design.md` 的关系：两者的 ring、替换策略、写回、状态字设计完全一致，差异仅在**索引层**。本文档自足，可独立阅读。

---

## 0. 摘要

本文提出一套 GPU embedding 缓存设计，其核心特征是**缓存自身不持有任何索引结构**。

常规做法是让缓存维护一张自己的哈希表（key → 缓存槽位），与 backing 表的索引（key → host 行号）并存。本方案把两者合并：**在 backing 索引的每个槽位上增加一个 `node_id` 字段**，于是一次探测同时回答两个问题——这个 key 在 host 的哪一行，以及它当前是否在缓存里、在哪个 node。

由此得到三个结果：

1. **缓存侧的索引结构完全消失**——没有哈希表、没有桶链、没有页表。node 元数据降到 12 B/node（含 ring）。
2. **一次探测完成三分类**：命中 / 老 key 未缓存（需 H2D）/ 全新 key（走初始化，无 PCIe）。常规方案需要查两个结构才能得到这个分类。
3. **淘汰时的索引失效是一次 O(1) 直接写**。node 里存 backing 槽号作为反向指针，不需要哈希、不需要探测、不需要摘链。

代价是 backing 索引每槽增加 4 B，以及一个结构性约束：**backing 索引重哈希会使所有反向指针失效**（§12.1）。

替换（ring + FIFO/CLOCK）、写回（仅淘汰时、与加载合并为双向搬运）、引用计数与并发协议均与常规方案相同，本文一并完整给出。

---

## 1. 背景与动机

DynamicEmb 的 embedding 表规模远超单卡显存，因此采用两层结构：全量数据在 host，GPU 侧缓存热行。缓存需要回答的问题是"这个 key 现在在 GPU 的哪一行"。

常规实现为此维护一张 GPU 侧哈希表。但注意一个事实：**缓存 miss 时，本来就必须去 backing 索引查一次**（要拿到 host 行号才能加载）。于是每次 miss 都发生两次查找——先查缓存哈希（失败），再查 backing 索引。

更进一步，如果缓存里的每个 key 必然在 backing 中有表项（本设计通过"准入即分配"保证，见 §9.3），那么 backing 索引已经是一张覆盖全部 key 的完备索引，**缓存哈希表所提供的信息是它的子集**。

本方案的出发点即在于此：与其维护两张索引再保持一致，不如在唯一的那张上多存 4 个字节。

### 1.1 何时适用

本方案的成本按 backing 槽数（正比于总 key 数 `N`）计费，收益按缓存容量 `C` 计费。因此它适用于 **`N/C` 较小**的场景——缓存能容纳相当比例的全量 key。

`N/C` 很大时（例如 `N = 10⁹`、`C = 4M`），`4 B × 槽数` 的增量会远超缓存侧省下的结构，且命中路径被迫探测一个数十 GB 的结构，局部性显著劣于探测一张几十 MB 的独立缓存索引。此种情形下应采用独立缓存索引的方案（见 `lightweight_cache_design.md`）。

**选型判据见 §12.2，这是采用本方案前必须先确认的事项。**

---

## 2. 设计目标与非目标

### 2.1 目标

| # | 目标 | 验收方式 |
|---|---|---|
| G1 | 缓存侧无索引结构，node 元数据 ≤ 16 B/node | 静态计算 |
| G2 | 一次探测完成"命中 / 未缓存 / 全新"三分类 | 代码审查 |
| G3 | 淘汰时的索引失效为 O(1)，不含探测 | 代码审查 |
| G4 | 淘汰不影响任何正在被引用的 key | 不变量断言 + 压力测试 |
| G5 | 替换策略可配置且对训练结果透明 | FIFO / CLOCK 切换后 loss 逐位一致 |
| G6 | 反向路径不写 backing 索引 | nsys / 代码审查 |
| G7 | E2E step time 不低于现有实现 | HSTU / MovieLens benchmark |

### 2.2 非目标

- **不覆盖推理部署场景。** 本设计的若干前提在纯前向路径下不成立：`ref` 的生命周期绑定在"前向 pin、反向释放"上；行布局包含 optimizer state；写回时机依赖反向产生的脏行；prefetch 提前一步依赖 key 在计算前已知。推理侧的 embedding 缓存需要单独设计。文中提到的 eval（§9.4）指训练过程中的周期性验证。
- 不改动 backing 行数据（host 侧）的存储方式
- 不做 KV 量化 / 压缩
- 不做跨 rank 的缓存共享或一致性
- 不做运行时的跨表容量再平衡

---

## 3. 总体架构

```
┌──────────────────────────── GPU ─────────────────────────────┐
│                                                              │
│  ┌── backing 索引（唯一的索引结构）──────────────────────┐   │
│  │   槽位 s :  key[s]  digest[s]  node_id[s]             │   │
│  │             ↑槽位即 host 行号      ↑ -1 = 不在缓存    │   │
│  └───────────────────────────────────────────────────────┘   │
│                            │ node_id                          │
│                            ▼                                  │
│  ┌── 缓存（无索引）───────────────────────────────────────┐  │
│  │   emb_table [C, D]   行池                              │  │
│  │   slot      [C]      node → backing 槽号（反向指针）   │  │
│  │   state     [C]      状态 / dirty / in_ring / used/ref │  │
│  │   ring      [C]      可回收 node 队列                  │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────┘
                            │ PCIe（zero-copy）
┌───────────────────────────┴──────────────────────────────────┐
│                          Host                                │
│   行数据 [S, D]：embedding + optimizer state                 │
│   mapped pinned 内存，kernel 直接寻址，按槽号 s 索引         │
└──────────────────────────────────────────────────────────────┘
```

三条关键的分层决策：

**backing 索引在 GPU，backing 行数据在 host。** miss 路径因此完全没有 CPU 参与——GPU 侧探测得到槽号，warp 通过 zero-copy 直接把该行从 host 拉进 `emb_table`。这与现有实现的分层一致（`create_table_state` 中 `key_index_map` 无条件建在 `device`，值缓冲走 `HostExtendableBuffer` / `HostVMMTensor`，见 `dynamicemb/extendable_tensor.py:131-149`）。

**槽位即行号。** backing 索引的槽位下标直接作为 host 行数据的行号，因此不需要额外存储行号字段。现有实现即采用此约定（`_flat_row_indices_for_value_load` 将命中槽位翻译为扁平行号）。

**`node_id` 与 `slot` 构成双向指针。** 前者由索引指向缓存，后者由缓存指回索引。两者的一致性是本设计的核心不变量（INV-8）。

---

## 4. 数据结构

### 4.1 backing 索引

采用开放寻址 + digest 预筛（与仓库现有 `LinearBucketTable` 同构）：

| 字段 | 类型 / 形状 | 含义 |
|---|---|---|
| `key` | `int64 [S]` | 槽位承载的 key，空槽为哨兵值 |
| `digest` | `uint8 [S]` | key 的 1 字节摘要，用于批量预筛，避免逐个比对 8 字节 key |
| `node_id` | `int32 [S]` | **本方案新增**。该 key 当前所在的缓存 node，`-1` 表示不在缓存 |

`S` 为槽数，`S = N / 负载因子`。`node_id` 的增量内存为 `4 B × S`。

> `node_id` 是纯缓存状态，**不得持久化**。dump 时应跳过该字段，load 后统一置 `-1`。

### 4.2 缓存侧结构

设缓存容量 `C`（node 数），行宽 `D = emb_dim + optim_state_dim`。

| 名称 | 类型 / 形状 | 含义 |
|---|---|---|
| `emb_table` | `[C, D]` | 行池。行号即 node id；embedding 与 optimizer state **同行连续** |
| `slot` | `int32 [C]` | 该 node 对应的 backing 槽号，`-1` 表示 node 空闲 |
| `state` | `uint32 [C]` | 打包状态字，见 §4.3 |
| `ring` | `int32 [C]` | 可回收 node 队列 |
| `ring_head` / `ring_tail` | `uint64` 标量 | 单调递增计数器 |

**注意 node 中不存 `key`。** 淘汰时需要"从 node 反查索引项"，而 `slot` 直接给出索引项的下标，无需由 key 重新哈希探测。`slot` 一个字段身兼三职：

1. 淘汰时定位要清除的 `node_id[slot]`
2. miss 加载时的 host 行号
3. 脏行回写时的 host 行号

**embedding 与 optimizer state 同行**是刻意的：反向的优化器更新是访存密集型，两者分离会让访存次数翻倍且都退化为非合并访问；同时淘汰回写必须携带整行，同行布局让回写成为一次连续搬运。

### 4.3 状态字位布局

```
 31 30 | 29    | 28      | 27   | 26 .......... 0
 状态  | dirty | in_ring | used | ref_counter
```

| 字段 | 宽度 | 含义 |
|---|---|---|
| 状态 | 2 bit | `EMPTY` / `LOADING` / `RESIDENT` / `EVICTING` |
| `dirty` | 1 bit | 反向更新过、尚未回写 |
| `in_ring` | 1 bit | 当前在 ring 中存在一条记录 |
| `used` | 1 bit | 自上次被表针扫过以来是否被命中（CLOCK 用，见 §8） |
| `ref_counter` | 27 bit | 当前有多少个 in-flight step 引用该 node |

**打包成单字是必需的，不是优化。** 命中时的"钉住"（ref+1）与淘汰时的"占用"（判定 ref==0 并改状态）必须互斥。若 ref 与状态位分处两个字，就存在"刚 ref+1 完、同时被判定为可淘汰"的窗口。打包后两者 CAS 同一个字，互斥由硬件保证。

顺带的收益：CLOCK 的 `used` 位可直接合并进 pin 的那次 CAS，**不产生任何额外访存**。

⚠️ 由此推出一个硬化要求：`atomicSub(&state[n], 1)` 是对**整个字**减一，仅当 `ref > 0` 时才等价于"ref 减一"。若 INV-6 被破坏、出现多减，`ref` 会从 0 借位到高位，**`used`、`in_ring`、状态位同时被腐蚀**。debug 构建的释放路径应断言 `REF(old) > 0`。

### 4.4 状态机

```
EMPTY ──分配──> LOADING ──加载完成──> RESIDENT ──被占用──> EVICTING ──> LOADING
                                          │                              (复用)
                                          └── ref 在 RESIDENT 内部 0↔N 变化
```

- `EMPTY`：初始状态，`slot = -1`，行内无有效数据
- `LOADING`：已绑定某槽位，H2D 或初始化未完成，**既不可消费也不可淘汰**
- `RESIDENT`：数据有效。`ref > 0` 表示在用；`ref == 0` 表示空闲，但**仍然可以命中**——`node_id[slot]` 依旧指向它
- `EVICTING`：已被某个分配线程占用，此时 pin 必须失败

### 4.5 多表

每张逻辑表持有独立的一套缓存结构（`emb_table` / `slot` / `state` / `ring`）与独立的 backing 索引段。好处是 `D_t` 逐表独立（无行内碎片）、`ring_head`/`ring_tail` 拆成 T 个（原子竞争摊薄）、容量可逐表配置。

所有表在同一个 kernel 内处理，device 侧传入指针与参数数组（沿用 `dynamicemb/key_value_table.py:160` 的 `get_table_ptrs` 模式）：

```
emb_ptrs, slot_ptrs, state_ptrs, ring_ptrs      : 指针数组 [T]
bkey_ptrs, bdigest_ptrs, bnodeid_ptrs           : 指针数组 [T]
ring_head, ring_tail                            : uint64 [T]
C, S, D                                         : int32  [T]
```

输入为 `unique_keys[U]` 与 `table_range[T+1]`，后者由 `segmented_unique_cuda` 直接给出（`dynamicemb/batched_dynamicemb_function.py:74`）。

线程到表的映射采用 **block 级快路径**：`unique_keys` 按表分段，同一 block 内的线程绝大多数属于同一张表。由 block 的 0 号线程二分出本 block 首尾元素的 table id——若相同，整个 block 走快路径，把该表参数一次读进 shared memory；只有跨越分段边界的 block 需要逐线程二分，这样的 block 至多 `T-1` 个。

代价是容量无法跨表共享，需要容量规划环节（§10.1）。

---

## 5. 不变量

以下不变量应在 debug 构建中以断言形式落地。

| # | 内容 | 保障手段 |
|---|---|---|
| INV-1 | `ref > 0` 的 node 永不被淘汰 | 出队时 CAS 校验 |
| INV-2 | node 仅在 `ref` 发生 `1→0` 跃迁且 `in_ring == 0` 时入队 | `atomicOr(IN_RING)` 的返回值判定 |
| INV-3 | 从 ring 弹出后必须重新校验，`CAS(RESIDENT\|ref=0 → EVICTING)` 成功才算占用 | 与并发 pin 互斥 |
| INV-4 | `tail - head ≤ C` | 由 INV-2 推出，入队端无需检查 |
| INV-5 | `head ≤ tail` | 出队越界时必须 `atomicSub` 归还 |
| INV-6 | 每个 unique key 每步恰好 `+1 / -1` | 前向必须先去重 |
| INV-7 | 替换策略对训练结果完全透明 | 见下 |
| **INV-8** | **双向指针一致：`node_id[s] == n` ⟺ `slot[n] == s`** | **本方案特有，见下** |

**关于 INV-6。** 最容易在后续迭代中被破坏的一条。一旦某条路径（被 mask 的 token、提前退出的分支、异常路径）漏掉 `-1`，该 node 的 `ref` 将永久大于 0，从此无法被淘汰——症状是命中率随训练时长单调下降。更严重的是反向情形：多减一次会因借位腐蚀整个状态字（§4.3）。建议实现周期性巡检 kernel，统计 `ref > 0` 的 node 数并与预期 in-flight 数比对。

**关于 INV-7。** 缓存是纯性能设施：被淘汰的 key 会带着完整的行回写 host，重新加载后值不变。因此 FIFO 与 CLOCK 的切换只影响 PCIe 流量与 step time，不影响 loss 与收敛。这决定了该配置项**只需性能测试，不需要数值对齐测试**；同时它是一条回归防线——若切换策略导致 loss 变化，说明缓存存在真实缺陷。

**关于 INV-8。** 这是本方案的核心不变量，也是它相对"两张独立索引"方案的结构性优势所在：常规方案中缓存哈希与 backing 索引是两个可能失步的结构，而这里只有一对互指的字段，一致性检查是 O(C) 的线性扫描：

```
for n in 0..C:
    s = slot[n]
    assert (s == -1 && state[n] is EMPTY) || node_id[s] == n
```

建议作为周期性巡检的一部分。破坏 INV-8 的典型路径是淘汰时"清了旧 `node_id` 却未更新 `slot`"，或反之。

---

## 6. 运行时流程

### 6.1 三分类查找

前向的第一步是去重（`segmented_unique_cuda`），得到 `unique_keys[U]`、`inverse[N]`、`table_range[T+1]`。

去重是**必须的**，不只为省带宽：`ref_counter` 的语义是"有多少个 in-flight step 引用了这个 key"，若不去重，同一 key 在 batch 内出现 50 次就会 +50，反向必须精确 -50；任何路径不对称都会造成计数泄漏或状态字腐蚀。**每个 unique key 每步恰好 +1 / -1**，这是可审计的。

随后每个 unique key 一个线程，**一次 backing 探测完成三分类**：

```cuda
int s = probe(key);                  // digest 预筛 + key 比对

if (s < 0) {                         // ① 全新 key
    s = insert(key);                 //    插入 backing，分配槽位
    node_id[s] = -1;
    // → miss；host[s] 无历史值 → 走 initializer，不产生 H2D
}
else if (node_id[s] < 0) {           // ② 老 key，当前不在缓存
    // → miss；host[s] 有历史值 → 需要 H2D
}
else {                               // ③ 命中候选
    n = node_id[s];
}
```

三点说明：

- 情况 ① 即"准入即在 backing 分配槽位"。这不是优化而是**结构性前提**——缓存中的 key 必然在 backing 中有槽位，否则 `slot` 反向指针无处可指。它同时消除了"淘汰或 checkpoint 时批量插入 backing 导致扩容尖峰"这一常见问题：淘汰路径退化为纯值写，checkpoint 退化为纯值 dump。
- 情况 ① 与 ② 的区分是免费的，而它决定了**要不要发起 H2D**。常规方案需要额外的存在性判断才能得到这个信息。
- `s` 顺带就是后续加载与回写要用的 host 行号，不需要二次查询。

### 6.2 前向

| 步 | 操作 | 可复用 |
|---|---|---|
| 1 | 去重 → `unique_keys[U]`、`inverse[N]`、`table_range` | `src/unique_op.cu` |
| 2 | backing 探测三分类 + 命中者 pin | 新写 |
| 3 | miss 压紧后从 ring 弹出 node | 新写 |
| 4 | 淘汰受害者：清 `node_id[s_old]`、记录脏行 | 新写 |
| 5 | 建立新映射：`slot[n] = s_new`、`node_id[s_new] = n` | 新写 |
| 6 | 双向搬运：受害者行 → host，新行 host → `emb_table`；全新 key 走初始化 | `src/initializer.cu` |
| 7 | `out[i] = emb_table[node_ids[inverse[i]], :emb_dim]` | `src/lookup_forward.cu` |

**Step 2 的 pin：**

```cuda
if (!try_pin(n)) {                   // CAS：状态==RESIDENT 且 ref+1
    → 降级为 miss;                   //    撞上 EVICTING
}
if (slot[n] != s) {                  // ★ 身份校验，见 §6.4
    unpin(n);
    → 降级为 miss;
}
```

pin 失败时**降级为 miss，不自旋**。代价仅是多一次 H2D，且发生概率极低；在 GPU 上自旋等待一个可能夹着 PCIe 传输的操作是死路。

**命中的定义包含"躺在 ring 里的 node"。** `ref` 归零不等于数据失效——`node_id[slot]` 仍指向它，只要还没被弹出复用，探测就会返回情况 ③。这是 ring 相对"归零即释放"的核心收益。

**Step 1–6 整体提前一个 step 执行（prefetch）**，与上一步的 dense 计算重叠，用于掩盖 PCIe 延迟。

### 6.3 反向

| 步 | 操作 | 可复用 |
|---|---|---|
| 1 | 按 `inverse` 规约梯度 → `[U, emb_dim]` | `src/lookup_backward.cu` |
| 2 | 原地优化器更新：读整行 → 计算 → 写回原行 | `src/optimizer_kernel.cuh` |
| 3 | 置 `dirty`；`ref -1`；若 `1→0` 且 `in_ring` CAS 成功则入队尾 | 新写 |

```cuda
atomicOr(&state[n], DIRTY);
uint32_t old = atomicSub(&state[n], 1);
assert(REF(old) > 0);                                   // debug：防借位腐蚀
if (REF(old) == 1) {
    if (atomicOr(&state[n], IN_RING) & IN_RING) return; // 已在环中
    uint64_t pos = atomicAdd(&ring_tail, 1ULL);
    ring[pos & MASK] = n;
}
```

> **反向路径完全不触碰 backing 索引。** `node_id` 只在 miss 时被写一次、淘汰时被清一次——每步对 backing 的写次数是 `M`（miss 数）而非 `U`（访问数）。热行的反复更新只动 `emb_table` 与 `state`，均为缓存侧结构。这是 G6。

### 6.4 pin 的身份竞态

读到 `node_id[s] = n` 之后、CAS pin 之前，node `n` 可能已被淘汰并改派给**另一个** key。此时它的状态又是 `RESIDENT`，pin 的 CAS 会**成功**——线程以为拿到了 key K，实际持有的是 K′ 的行。

两道防线，建议都上：

**① 阶段分离（主防线）。** "查找 + pin" 与 "分配 + 淘汰" 处于**不同的 kernel**，由流序保证不并发。这与 ring 的读写相分离约束（§7.2）是同一套纪律，可合并为一条设计原则：

> **索引的读（查找）与写（淘汰 / 建立映射）在时间上相分离，由流序保证。**

这条原则同时解决了三件事：ring 的 MPMC 竞态、pin 的身份竞态、以及 `node_id` 读写竞态。

**② pin 后校验身份（兜底）。** `slot[n] != s` 即说明该 node 已改派，`unpin` 并降级为 miss。这只是一次 4 字节读，几乎免费。

---

## 7. Ring 协议

### 7.1 表示与初始化

```
ring = arange(C);   head = 0;   tail = C
所有 node：state = EMPTY | in_ring=1 | ref=0，slot = -1
```

`arange` 初始化让**"首次分配"与"回收复用"成为同一条代码路径**。传统实现需分别维护 free-list（从未使用的槽）与 eviction-queue（用过待回收的槽）；这里两者合一——环中装的一律是"可被占用的 node"，只是有的尚未绑定槽位（`EMPTY`，`slot = -1`），有的装着有效但无人引用的数据（`RESIDENT`，命中它即缓存命中）。分配方只关心 `ref == 0`。

元素个数为 `tail - head`。使用无符号单调计数器配合取模索引（`C` 向上取到 2 的幂），比维护独立 size 变量更好——后者在并发下需要额外原子操作。计数器使用 **uint64**：uint32 靠无符号减法也能处理回绕，但需要额外论证，而 uint64 在任何真实训练时长内都不会绕。

### 7.2 核心约束：读写在时间上相分离

> **同一时刻，ring 只被"纯出队 kernel"或"纯入队 kernel"中的一种访问，由流序保证。**

若出队与入队真正并发，就会撞上经典的 MPMC 环形队列问题：入队线程先 `atomicAdd(tail, 1)` 占位 `p`，尚未把 node id 写入 `ring[p]`；此时出队线程看到 `tail` 已增长，认为 `p` 有效，读到的却是上一轮的陈旧值。修复需要为每个槽引入序列号（Vyukov 方案），复杂度与调试成本陡增。

训练循环天然满足该约束：

```
step N:   [prefetch / allocate ← 只出队]  forward  backward  [release ← 只入队]
step N+1: [prefetch / allocate ← 只出队]  ...
```

即便启用 prefetch 流水，step N+1 的 allocate 早于 step N 的 release 发起，两者仍处于不同 kernel、有流序或 event 依赖。代价仅是"本步释放的 node 下一步用不上"——这是容量问题而非正确性问题。

该约束带来一个直接好处：**出队 kernel 中 `tail` 是常量，入队 kernel 中 `head` 是常量**，边界检查可在 kernel 开头读一次快照，全程精确。

### 7.3 出队

```cuda
__device__ int ring_pop(uint64_t tail_snapshot) {
    for (;;) {
        uint64_t pos = atomicAdd(&ring_head, 1ULL);
        if (pos >= tail_snapshot) {
            atomicSub(&ring_head, 1ULL);   // 归还，维持 INV-5
            return -1;                     // 环已抽干 → 上层报错 / 反压
        }
        int cand = ring[pos & MASK];
        uint32_t st = state[cand];

        if (REF(st) != 0) {                // 陈旧条目：该 node 已被重新命中
            clear_in_ring(cand);
            continue;                      // 丢弃，重领
        }
        if (try_claim(cand)) {             // INV-3
            clear_in_ring(cand);
            return cand;
        }
        clear_in_ring(cand);               // 刚被 pin 走，同样丢弃
    }
}
```

三个易错点：

**陈旧条目是丢弃，而非重新入队。** 直觉上会想"它还有效，放回队尾"，但这会在缓存被 pin 满时造成活锁——所有条目 `ref > 0`，线程将绕环无限循环。丢弃则使环单调收缩，最坏情况是抽干并返回 `-1`，上层立即得到明确错误。丢弃不会泄漏：该 node 会在下次真正归零时重新入队（因为 `in_ring` 已被清零）。

**越界时必须把 head 减回去。** 否则多个线程同时撞到空环会把 head 推过 tail，`tail - head` 下溢为巨大的无符号数，后续所有判断全部失效。此缺陷仅在缓存打满时触发，需要 INV-5 断言兜底。

**CAS 失败与陈旧条目走同一处理。** 两者都意味着"该 node 现在不归我"。

### 7.4 `in_ring` 位的必要性

> node X 的 `ref` 归零 → 入 ring。此后 X 被再次命中，`ref 0→1`（**X 仍留在 ring 中，成为陈旧条目**）。反向释放，`ref 1→0`。若此时无条件再次入队，ring 中就有了 X 的**两条**记录。第一条被弹出、成功淘汰并复用后，第二条即成为悬垂引用——会把一个正在服役的 node 当作垃圾回收。

`in_ring` 位保证任一 node 在环中至多一条记录：入队走 `CAS(in_ring: 0→1)`，出队清零。等价方案是为每个 node 维护 generation 计数、ring 中存 `(node_id, gen)` 二元组，出队时比对；`in_ring` 方案节省一半 ring 内存。

**入队无需溢出检查**，由 INV-4 保证：`in_ring` 使任一 node 在环中至多一条记录，因此记录数不超过 `C`；而出队只会减少记录数。

---

## 8. 替换策略

### 8.1 FIFO 与 CLOCK

- **FIFO** 的淘汰依据是：*进来多久了*
- **CLOCK** 的淘汰依据是：*进来之后有没有被再用过*

CLOCK 只有两条规则：

1. **插入时** `used = 0`；**命中时** `used = 1`
2. **淘汰时**检查队头：`used == 1` 则清零并放回队尾（第二次机会），继续看下一个；`used == 0` 才淘汰

推论：**一个 node 要被淘汰，必须在表针走完一整圈期间一次都没被命中过。** 对照关系：`ring_head` 即表针，`ring` 数组即钟面。

访问序列 `A B C D A E A F`、容量 4、A 为热 key 时的对照：

| 步 | 访问 | FIFO | CLOCK |
|---|---|---|---|
| 1–4 | A B C D | 依次装入，满 | 依次装入，`used` 全为 0 |
| 5 | **A** | **hit**，顺序不变 | **hit**，`used(A)=1` |
| 6 | E | miss → 淘汰队头 **A** ❌ | miss → A `used=1`，清零放回队尾；B `used=0` → 淘汰 B ✓ |
| 7 | **A** | **miss**（刚被扔掉） | **hit** ✓ |
| 8 | F | miss → 淘汰 C | miss → C `used=0` → 淘汰 C ✓ |

规模效应显著：**FIFO 的表针每转一圈就把整个缓存换一遍，热 key 也不例外**，命中率大致等于「缓存容量 ÷ 一圈内出现的 unique key 数」，几乎与热度分布无关——等于把分布倾斜的红利全部浪费。推荐场景的 key 分布是重尾的，倾斜越强，浪费越大。

### 8.2 实现

```python
class ReplacementPolicy(Enum):
    FIFO  = 0    # 默认
    CLOCK = 1
```

**运行期开关，而非模板参数。** 出队循环不是热路径（每个 miss 走一次，而非每个 key 走一次），一个全 kernel 一致、完美预测的分支开销可忽略；模板化的唯一实质收益是省下 `retry_buf` 显存，而按需分配已解决。作用域为全局而非逐表——逐表混用缺乏实际需求，且会让命中率回归的排查空间乘以 `2^T`。

CLOCK 开启后，出队循环增加一个分支：

```cuda
if (st & USED) {
    atomicAnd(&state[cand], ~USED);
    retry_buf[atomicAdd(&retry_cnt, 1)] = cand;   // 攒起来，稍后统一入队尾
    continue;
}
```

配合一个跟随的小 kernel 把 `retry_buf` 顺序追加到 `ring_tail`。**不能在出队 kernel 内直接写 `tail`**，那会破坏 §7.2 的相分离约束。

### 8.3 终止性：必须限次回退

> 进入 `retry_buf` 的条目在本次出队 kernel 内**不可见**。若 ring 中的活跃条目恰好全部 `used == 1`，它们会被逐个搬进 `retry_buf`，`head` 一路推到 `tail_snapshot`——`ring_pop` 返回 `-1`，误报"缓存打满"。

经典 CLOCK 没有这个问题——它的表针在**固定**环上循环，转一圈清零后第二圈必然找到受害者；而本设计的 ring 是队列，`head` 推进即意味着条目被消费。

修复采用**限次回退**：

```
每个线程最多授予 k 次第二次机会（k = 1 或 2）；
超过后无视 used 位，直接 claim 下一个 ref == 0 的条目。
```

保证循环有界，且降级优雅：**本步退化为 FIFO，下一步恢复 CLOCK 语义**（第一次扫描已把那些 `used` 位清零）。真正触发降级的场景是"缓存中所有内容都是热的"，此时任何替换策略表现都一样差。

### 8.4 `used` 位无条件置位：一个免费的在线指标

即便运行在 FIFO 模式，`used` 位也照常置位——它合并进 pin 的 CAS，零额外访存。

> **在出队时统计 `used == 1` 的条目占比，即等于"若启用 CLOCK，有多少次淘汰会被推迟"。**

无需离线模拟、无需 trace dump，跑一次正常训练即可得到：比例接近 0 说明访问几乎没有复用，CLOCK 无意义；40%+ 说明大量热 key 正被 FIFO 误杀。它是**筛选指标而非精确预测**（推迟一次淘汰会改变后续受害者选择），但作为"值不值得试"的判据完全够用。

建议与命中率、ring 抽干次数、三分类占比一并做成固定的健康度指标组，异步拷出、每数百步读取一次，**热路径上不得有任何 host 同步**。

---

## 9. 写回

### 9.1 触发时机：仅淘汰与 checkpoint

不做后台主动刷。理由：

> 训练中最热的行，特点是**每步都被更新、但永远不会被淘汰**。它们在缓存中反复累积更新，这正是缓存存在的全部意义——把 N 次更新吸收为最终淘汰时的 1 次回写。后台主动刷会把这些行反复写回 host，而每次写都会被下一步的更新作废。这是纯粹的带宽浪费，且越热的行浪费越多。

同一逻辑排除了"`ref` 归零时立即回写"的方案：那会让每步 D2H 行数从 `M`（miss / 淘汰数）涨到 `U`（unique key 数），而 `M = U × (1 - 命中率)`——命中率 90% 时流量是 10 倍，**缓存越有效，惩罚越重**。

### 9.2 与加载合并为双向 kernel

稳态下每次 miss 恰好触发一次淘汰，而训练中一个 step 内被访问的 key 基本都会在反向被更新，因此几乎每个受害者都是脏的：

```
每步 PCIe 流量：H2D = M 行（miss 加载），D2H ≈ M 行（脏行回写）
```

方向相反、量级相等——全双工的理想输入。合并后每个 warp 处理一对，`s_old` 与 `s_new` 均已在手：

```
  ① emb_table[n] ─────────────→ 寄存器（旧行）   device 读
  ② host[s_new]  ──── H2D ────→ 寄存器（新行）  ┐  ← 情况 ②
     或 initializer 生成                        │  ← 情况 ①，无 PCIe
  ③ 寄存器（旧行）──── D2H ───→ host[s_old]    ┘  仅当 dirty
  ④ 寄存器（新行）────────────→ emb_table[n]     device 写
```

寄存器开销不成问题：`dim=128` fp32 加 Adam 两个 state 为 1536 B/行 = 96 个 `float4`，摊到 32 lane 是每 lane 3 个 `float4`，两行共 24 个寄存器。可按 chunk 流水以进一步重叠。

旧行在 ① 就已进入寄存器，④ 随时可覆写——**D2H 不阻塞 H2D**，不存在"等待写回"。

⚠️ 该结构以 **PCIe 全双工有效**为前提，须由 §12.3 的实验确认。若双向互相干扰，退化为两次串行 launch，kernel 结构不变。

### 9.3 淘汰路径退化为纯数据搬运

由于槽位在准入时即已分配（§6.1 情况 ①），淘汰路径上不存在任何"可能失败、可能触发分配"的操作：

```cuda
int s_old = slot[n];
node_id[s_old] = -1;            // O(1) 直接写，无哈希、无探测
// 若 dirty，host[s_old] 即回写目标
```

对比"惰性分配"（新 key 只活在缓存中、淘汰或 flush 时才插入 backing）：那会让 checkpoint 时海量新 key 同时要求插入，backing 必须先扩容，产生内存尖峰。本设计不存在该问题——**淘汰是纯值写，checkpoint 是纯值 dump**。

### 9.4 训练内 eval 路径：只读缓存，不 bypass

指训练过程中周期性的验证前向（`module.eval()`）。

eval 不能绕过缓存直读 host——缓存中存在脏行，绕过会读到陈旧值。正确做法是只读缓存路径：

```
eval：探测 → node_id >= 0 则 pin 并读；否则直接从 host[s] zero-copy 读
      不准入、不淘汰、不置 dirty、不占用 node
```

既避免陈旧读，也规避扫描污染（不准入，全量遍历不会冲刷热数据），热 key 依然享受命中。相比"先全量 flush 再 bypass"省掉一整次脏行 dump。

### 9.5 崩溃一致性与 checkpoint

仅在淘汰时回写意味着崩溃会丢失缓存中的全部脏数据。但训练本就依赖周期性 checkpoint 恢复，"丢失上次 checkpoint 以来的进度"是既有语义。不需要 WAL 或额外持久化机制。

checkpoint 时的全量 flush：最坏情况整个缓存皆脏，即 `C × D` 的 D2H。4M node × 1536 B ≈ 6 GB，按 20 GB/s 约 300 ms，而 checkpoint 间隔通常是数千步——可忽略。

⚠️ **`node_id` 字段不得写入 checkpoint**，load 后统一置 `-1`（缓存冷启动）。同理 `slot` / `state` / `ring` 均为运行期状态，不持久化。

### 9.6 回写粒度

整行（embedding + optimizer state），`dirty` 位按 row 维护。

**只回写 embedding 会导致 optimizer state 在淘汰时静默丢失**——模型照常训练，只是收敛变差。这类缺陷极难定位，实现时应有针对性测试。

---

## 10. 容量与内存

### 10.1 容量约束

```
C_t ≥ (表 t 单步 unique key 数) × (prefetch 流水深度) × 安全系数
```

应在配置阶段校验并给出明确报错，而非运行时撞到 `ring_pop` 返回 `-1` 才失败。由于容量无法跨表共享，需要容量分配环节：初版按各表预估 key 数或历史访问频次静态按比例分配；P1 观测每表命中率与 ring 抽干次数做离线再平衡；不做运行时动态再平衡。现有 `dynamicemb/planner/` 已在做类似的分片规划，可挂靠。

### 10.2 内存账

**缓存侧**（按 node 计）：

| 项 | 字节/node |
|---|---|
| `slot` | 4 |
| `state` | 4 |
| `ring` | 4 |
| **合计** | **12** |

对比典型行宽（dim=128、fp32、Adam）1536 B，占比约 **0.8%**。

**backing 侧**（按槽计）：

```
增量 = 4 B × S = 4 B × N / 负载因子
```

| N | 负载因子 0.5 时的增量 |
|---|---|
| 10⁷ | 80 MB |
| 10⁸ | 800 MB |
| 10⁹ | 8 GB |

**这是本方案的主要成本，且与缓存容量无关。** 见 §12.2 的选型判据。

---

## 11. 接口

### 11.1 配置

```python
@dataclass
class FusedIndexCacheOptions:
    capacity: int                      # 每表 node 数 C_t
    replacement: ReplacementPolicy = ReplacementPolicy.FIFO
    clock_max_second_chances: int = 1  # §8.3 的 k
    prefetch_depth: int = 1
    record_metrics: bool = False
    presize_backing: bool = True       # §12.1：预分配以避免 rehash
```

### 11.2 核心 API

```python
lookup(unique_keys, table_range)     -> node_ids, slots, kind   # kind ∈ {命中, 未缓存, 全新}
allocate(miss_idx, slots, ...)       -> node_ids                # 含淘汰与双向搬运
release(node_ids, table_range)       -> None
flush(table_id=None)                 -> None                    # checkpoint 用
metrics()                            -> CacheMetrics
```

接口面积保持在 5 个方法以内。缓存不提供 `expand` / `dump` / `load`——这些属于 backing 表。

---

## 12. 风险与待验证

### 12.1 ⚠️ backing 重哈希会使所有 `slot` 失效

**本方案最主要的结构性风险。** `slot` 是反向指针，只有在 backing 槽位稳定时才有意义。一旦 backing 因插入新 key 而扩容重哈希，**所有 node 的 `slot` 同时变成野指针**，且由于 node 中不存 `key`，无法自行修复。

三个处理方式：

| | 做法 | 代价 |
|---|---|---|
| **A** | backing 预分配到容量上界，**永不 rehash** | 显存一次性占满；需预先知道 key 数上界 |
| **B** | rehash 时**整体 flush + 清空缓存** | 一次全量脏行 dump（~300 ms）+ 缓存重新预热 |
| **C** | node 额外存 `key`（8 B），rehash 后按 key 重新探测修正所有 `slot` | +8 B/node（12 → 20 B，元数据涨 67%）；rehash 时 C 次探测；**缓存内容不丢** |

**推荐 A + B 组合**：默认预分配，把 rehash 当作异常路径，真发生了就走 B。理由是 rehash 在训练中本就应罕见——若频繁发生，说明 backing 容量规划有误，那是配置问题而非缓存问题。C 虽最稳，但为一个应当罕见的事件常驻 8 B/node（占总元数据的 40%），不划算。

> **待定**：该取舍依赖"能否预先给出 key 数上界"。若不能，应改用 C。

### 12.2 ⚠️ 选型判据：`N / C`

本方案的成本按 `N` 计费（`4 B × 槽数`），收益按 `C` 计费（省下缓存侧索引结构）。因此：

| `N / C` | 判断 |
|---|---|
| **≤ 3** | 本方案明显更优——backing 索引本就不比独立缓存索引大多少，两者合一是净简化 |
| **~10** | 需实测探测代价才能定 |
| **≥ 50** | 本方案不适用。`4 B × 槽数` 的增量远超缓存侧节省，且命中路径被迫探测数十 GB 结构，局部性显著劣于探测几十 MB 的独立索引 |

补充约束：本方案**锁死了"backing 索引常驻 GPU"**。若 `N` 大到索引必须下沉 host，每次缓存命中都要走一趟 PCIe 探测，方案彻底不可用。而 `N` 大恰恰又是 `N/C` 大的成因——两个条件同向，本方案在不适用的场景里会以两种方式同时失效。

**采用本方案前必须先确认 `N` 与 `C` 的量级。**

### 12.3 PCIe 双向带宽

**测什么**：`examples/pcie_bidir_bw/pcie_bidir_gather.cu` 的 `gather-only` / `scatter-only` / `bidir` 三种配置在目标平台上的实测带宽。

**如何影响设计**：若 `bidir` 接近两个单向之和，§9.2 的融合双向 kernel 成立；否则退化为两次串行 launch。同一 benchmark 的 `do_sort` 开关（`argv[5]`）可顺带量化"按 `s_old` / `s_new` 排序回写行"的收益。

### 12.4 其他风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| INV-6 被破坏导致 ref 泄漏或状态字腐蚀 | 命中率单调下降；或 node 状态错乱 | 周期性巡检；`assert(REF(old) > 0)` |
| INV-8 被破坏（双向指针失步） | 命中到错误的行，数值错误 | O(C) 巡检（§5）；阶段分离 |
| 容量规划不当，热表 ring 抽干 | 训练中断 | 配置期校验；抽干计数作为告警 |
| CLOCK 出队循环退化 | 分配延迟抖动 | §8.3 限次回退；默认 FIFO |

---

## 13. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | 单表、FIFO、串行加载 / 回写；单元测试覆盖 INV-1..8 | 断言全通过；与无缓存路径 loss 逐位一致 |
| **P1** | 多表融合 kernel；双向搬运；prefetch 流水；健康度指标 | HSTU / MovieLens E2E 不低于现有实现（G7） |
| **P2** | CLOCK 可选；容量规划挂靠 planner；回写排序（视 §12.3） | 命中率与 step time 对比报告 |

---

## 附录 A：可复用的现有代码

| 用途 | 位置 |
|---|---|
| 分段去重（含 table_range 输出） | `src/unique_op.cu`、`dynamicemb/batched_dynamicemb_function.py:74` |
| 开放寻址 + digest 的哈希表（backing 索引可改造自此） | `dynamicemb/scored_hashtable.py:284` |
| 全新 key 的初始化 | `src/initializer.cu`、`src/initializer.cuh` |
| 优化器原地更新 | `src/optimizer_kernel.cuh` |
| 前 / 反向 gather-scatter | `src/lookup_forward.cu`、`src/lookup_backward.cu` |
| host 侧可扩展缓冲（mapped pinned） | `dynamicemb/extendable_tensor.py:131`、`src/vmm_tensor.cu` |
| 逐表指针数组模式 | `dynamicemb/key_value_table.py:160` |
| PCIe 双向带宽基准 | `examples/pcie_bidir_bw/pcie_bidir_gather.cu` |

需要新写的部分：backing 索引的 `node_id` 字段与三分类探测 kernel、ring 的领取与校验、状态字的原子协议、双向搬运 kernel。

---

## 附录 B：完整示例走查

用一个极小的配置走完四个 training step，覆盖三分类的全部三种情况、`node_id` 的建立与清除、陈旧条目丢弃、淘汰回写、以及被淘汰数据的完整往返。

### B.0 例子设定

```
C = 4                缓存 node n0 .. n3
S = 8                backing 槽位 s0 .. s7，hash(k) = k & 7，线性探测
D = 6                emb 4 列 + optimizer state 2 列
host 行表 [8, 6]     按槽号索引
单表，prefetch 深度 = 1，替换策略 = FIFO
```

**初始状态**（`ring = arange(4)`，全部 `EMPTY`、`slot = -1`、`in_ring = 1`）：

```
backing 索引                              缓存
┌────┬───────┬──────────┐        ┌────┬──────┬───────┬─────┬────────┐
│槽位│  key  │ node_id  │        │ id │ slot │ state │ ref │in_ring │
├────┼───────┼──────────┤        ├────┼──────┼───────┼─────┼────────┤
│ s0 │  空   │    -1    │        │ n0 │  -1  │ EMPTY │  0  │   1    │
│ s1 │  空   │    -1    │        │ n1 │  -1  │ EMPTY │  0  │   1    │
│ s2 │  空   │    -1    │        │ n2 │  -1  │ EMPTY │  0  │   1    │
│ s3 │  空   │    -1    │        │ n3 │  -1  │ EMPTY │  0  │   1    │
│ .. │  空   │    -1    │        └────┴──────┴───────┴─────┴────────┘
└────┴───────┴──────────┘
                                  ring   idx  0    1    2    3
host 行表：8 行未初始化                      ┌────┬────┬────┬────┐
                                             │ n0 │ n1 │ n2 │ n3 │
                                             └────┴────┴────┴────┘
                                               ↑head=0        tail=4
```

---

### B.1 Step 1 —— 三个全新 key（三分类的情况 ①）

**输入** `keys = [10, 18, 3, 10, 18]`

**① 去重**

```
输入 keys : [ 10, 18,  3, 10, 18 ]
                 ↓
unique    : [ 10, 18,  3 ]              U = 3
inverse   : [  0,  1,  2,  0,  1 ]
```

**② 三分类探测** —— 表为空，三个 key 全部落入**情况 ①（全新）**，插入时发生线性探测：

```
key 10:  hash = 2  →  s2 空  →  插入 s2
key 18:  hash = 2  →  s2 已被 key 10 占用  →  探测 s3 空  →  插入 s3
key  3:  hash = 3  →  s3 已被 key 18 占用  →  探测 s4 空  →  插入 s4

三者 node_id 均置 -1  →  miss，且 host 无历史值 → 走 initializer，不产生 H2D
```

**③ ring pop ×3**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │
           └────┴────┴────┴────┘
             ↑    ↑    ↑
           pop  pop  pop        head: 0 → 1 → 2 → 3      tail = 4

三个候选都是 EMPTY 且 ref == 0，CAS claim 全部成功
```

**④ 建立双向指针**

```
  slot[n0] = 2    ⇄    node_id[s2] = n0        (key 10)
  slot[n1] = 3    ⇄    node_id[s3] = n1        (key 18)
  slot[n2] = 4    ⇄    node_id[s4] = n2        (key  3)

state: EMPTY → LOADING → RESIDENT,  ref = 1
```

```
backing 索引                              缓存
┌────┬───────┬──────────┐        ┌────┬──────┬──────────┬─────┐
│ s2 │  10   │    n0    │◄──┐    │ n0 │  2   │ RESIDENT │  1  │
│ s3 │  18   │    n1    │◄─┐│    │ n1 │  3   │ RESIDENT │  1  │
│ s4 │   3   │    n2    │◄┐││    │ n2 │  4   │ RESIDENT │  1  │
└────┴───────┴──────────┘ │││    │ n3 │ -1   │  EMPTY   │  0  │
                          │││    └────┴──────┴──────────┘
        slot 反向指针 ────┴┴┘             双向互指 = INV-8
```

**⑤ 初始化** —— 三行由 `initializer` 在 GPU 上直接生成，**不产生任何 PCIe 流量**。这正是情况 ① 与情况 ② 区分开的价值。

**⑥ 前向** `out[i] = emb_table[node_ids[inverse[i]], :4]` —— 5 个输入位置由 3 行数据供给。

**⑦ 反向** —— 优化器原地更新三行，置 `dirty`，`ref 1→0` 后入队：

```
push n0 → ring[4 & 3] = ring[0] = n0     tail: 4 → 5
push n1 → ring[5 & 3] = ring[1] = n1     tail: 5 → 6
push n2 → ring[6 & 3] = ring[2] = n2     tail: 6 → 7
```

**Step 1 结束**

```
ring   idx  0    1    2    3
          ┌────┬────┬────┬────┐
          │ n0 │ n1 │ n2 │ n3 │      head = 3    tail = 7
          └────┴────┴────┴────┘
逻辑顺序（含环绕）：idx3 → idx0 → idx1 → idx2
                     n3     n0     n1     n2
                    (空)  (key10)(key18)(key3)
                    队头 ←──────────────→ 队尾

n0/n1/n2 均为 RESIDENT ⚑dirty，ref = 0
```

---

### B.2 Step 2 —— 缓存命中（情况 ③）+ `in_ring` 防重复入队

**输入** `keys = [10, 25]`

**① 三分类探测**

```
key 10:  hash = 2 → s2，key 匹配 ✓
         node_id[s2] = n0 ≥ 0   →  ★ 情况 ③：命中
         pin: CAS state(n0): RESIDENT|ref=0 → RESIDENT|ref=1|used=1
         身份校验: slot[n0] == 2 ✓

key 25:  hash = 1 → s1 空  →  情况 ①：全新，插入 s1，node_id[s1] = -1
```

> n0 的 `ref` 归零后一直躺在 ring 里，但 `node_id[s2]` 始终指着它——**数据有效，命中它零 PCIe 流量**。这是 ring 相对"归零即释放"的核心收益。

**② ring pop ×1**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │
           └────┴────┴────┴────┘
                            ↑ head=3 → pop 得 ring[3&3] = n3
           校验：ref(n3) == 0 ✓  state == EMPTY ✓ → claim 成功，head: 3 → 4
```

**③ 建立映射并初始化**

```
  slot[n3] = 1    ⇄    node_id[s1] = n3        (key 25)
  情况 ① → initializer 生成，无 H2D
```

**④ 反向释放** —— 两个 node 都被更新并置 `dirty`，但入队行为不同：

```
n0:  ref 1 → 0
     atomicOr(state, IN_RING) 返回值中 IN_RING 已置位
       →  ★ 不重复入队。Step 1 放进 ring 的那条记录（idx0）仍然有效

n3:  ref 1 → 0，IN_RING 为 0  →  CAS 成功
       push n3 → ring[7 & 3] = ring[3] = n3     tail: 7 → 8
```

★ 若此处无条件入队，ring 中会出现 n0 的**两条**记录，第一条被淘汰复用后，第二条即成为悬垂引用。

**Step 2 结束**

```
backing                            ring   idx  0    1    2    3
┌────┬───────┬──────────┐                    ┌────┬────┬────┬────┐
│ s1 │  25   │    n3    │                    │ n0 │ n1 │ n2 │ n3 │
│ s2 │  10   │    n0    │                    └────┴────┴────┴────┘
│ s3 │  18   │    n1    │                      ↑head=4         tail=8
│ s4 │   3   │    n2    │
└────┴───────┴──────────┘         逻辑顺序：n0 → n1 → n2 → n3
                                            队头 ←────────→ 队尾
四个 node 全部 RESIDENT ⚑dirty，ref = 0，环满（tail - head = 4 = C，满足 INV-4）
```

---

### B.3 Step 3 —— 陈旧条目丢弃 + 淘汰 + 双向回写

**输入** `keys = [10, 41]`

**① 三分类探测**

```
key 10:  s2 命中 → node_id[s2] = n0 → pin，ref 0→1，used=1
key 41:  hash = 1 → s1(key 25) ✗ → s2(key 10) ✗ → s3(key 18) ✗
                  → s4(key 3) ✗ → s5 空 → 插入 s5，node_id[s5] = -1
                  → 情况 ①：全新
```

> ★ 关键：n0 此刻 `ref = 1`，而它在 ring 的 **idx0 —— 正是 head 指向的位置**。它成了一条陈旧条目。

**② ring pop —— 撞上陈旧条目**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │
           └────┴────┴────┴────┘
             ↑ head=4 → pop 得 ring[4&3] = ring[0] = n0

  校验 ref(n0) = 1 ≠ 0  →  ★ 陈旧条目
       clear in_ring(n0)，【丢弃该记录，不放回队尾】，head: 4 → 5
       （in_ring 已清零，n0 会在下次真正归零时重新入队，不泄漏）

             ↑ head=5 → pop 得 ring[5&3] = ring[1] = n1   (key 18, ref=0, dirty)
  校验 ref(n1) = 0 ✓
       CAS(state: RESIDENT|ref=0 → EVICTING) 成功       ← INV-3
       clear in_ring(n1)，head: 5 → 6
```

**③ 淘汰 n1 —— O(1) 索引失效**

```
s_old = slot[n1] = 3

  node_id[s3] = -1          ← ★ 一次直接写。不需要哈希、不需要探测、不需要摘链

backing 变化：
  ┌────┬───────┬──────────┐          ┌────┬───────┬──────────┐
  │ s3 │  18   │    n1    │   ──→    │ s3 │  18   │    -1    │
  └────┴───────┴──────────┘          └────┴───────┴──────────┘
                                       key 仍在，只是不在缓存了
                                       —— 这正是情况 ② 的形态
n1 为 dirty → 需回写 host[3]
```

**④ 建立新映射并双向搬运**

```
slot[n1] = 5    ⇄    node_id[s5] = n1        (key 41)

                        一个 warp 内完成
   ┌──────────────────────────────────────────────────────────┐
   │ ① emb_table[1] ─────────────→ 寄存器（key 18 的旧行）    │
   │                                                          │
   │ ② key 41 是情况 ① → initializer 生成新行（无 H2D）      │
   │                                                          │
   │ ③ 寄存器（旧行）──── D2H ───→ host[3]   ← key 18 的数据 │
   │                                            完整落盘      │
   │ ④ 新行 ─────────────────────→ emb_table[1]              │
   └──────────────────────────────────────────────────────────┘
```

**⑤ 反向释放**

```
n0:  ref 1→0，in_ring 已在 ② 中被清零  →  CAS 成功，重新入队
       push n0 → ring[8 & 3] = ring[0] = n0     tail: 8 → 9
n1:  ref 1→0，in_ring = 0  →  push ring[9 & 3] = ring[1] = n1   tail: 9 → 10
```

**Step 3 结束**

```
backing                                缓存
┌────┬───────┬──────────┐     ┌────┬──────┬──────────┬─────┐
│ s1 │  25   │    n3    │     │ n0 │  2   │RESIDENT⚑ │  0  │  key 10
│ s2 │  10   │    n0    │     │ n1 │  5   │RESIDENT⚑ │  0  │  key 41
│ s3 │  18   │    -1    │◄─── │ n2 │  4   │RESIDENT⚑ │  0  │  key  3
│ s4 │   3   │    n2    │  │  │ n3 │  1   │RESIDENT⚑ │  0  │  key 25
│ s5 │  41   │    n1    │  │  └────┴──────┴──────────┴─────┘
└────┴───────┴──────────┘  │
                           └── key 18 已被淘汰，数据在 host[3]

ring   head = 6   tail = 10    逻辑顺序：n2 → n3 → n0 → n1
```

---

### B.4 Step 4 —— 老 key 未缓存（情况 ②），数据完整往返

**输入** `keys = [18]`

**① 三分类探测**

```
key 18:  hash = 2 → s2(key 10) ✗ → s3，key 匹配 ✓
         node_id[s3] = -1   →  ★ 情况 ②：老 key，当前不在缓存
                                host[3] 有历史值 → 需要 H2D
```

> 这一步展示了本方案的核心价值：**一次探测同时得知"不在缓存"和"host 行号是 3"**。常规方案需要先查缓存哈希（失败），再查 backing 索引（拿到行号）。

**② ring pop**

```
             ↑ head=6 → pop 得 ring[6&3] = ring[2] = n2   (key 3, slot=4, ref=0, dirty)
  校验 ref == 0 ✓ → CAS claim 成功，head: 6 → 7
```

**③ 淘汰 n2 并建立新映射**

```
s_old = slot[n2] = 4        →  node_id[s4] = -1    （key 3 退出缓存，数据回写 host[4]）
s_new = 3                   →  slot[n2] = 3，node_id[s3] = n2
```

**④ 双向搬运 —— 两个方向同时在飞**

```
   ┌──────────────────────────────────────────────────────────┐
   │ ① emb_table[2] ─────────────→ 寄存器（key 3 的旧行）     │
   │                                                          │
   │ ② host[3]     ──── H2D ────→ 寄存器  ┐                  │
   │                  ↑                    │ PCIe 双向同时在飞 │
   │              key 18 的数据            │                  │
   │              —— Step 3 刚写下去的     │                  │
   │                                       │                  │
   │ ③ 寄存器（旧行）──── D2H ───→ host[4]┘  key 3 落盘      │
   │                                                          │
   │ ④ 寄存器（新行）────────────→ emb_table[2]              │
   └──────────────────────────────────────────────────────────┘
```

★ key 18 的数据在 Step 3 被写到 `host[3]`，在 Step 4 又从 `host[3]` 读回——**淘汰是无损的**，这正是 INV-7（替换策略对训练结果透明）的具体体现。

**⑤ 反向释放** —— `n2` 的 `ref 1→0`，入队 `ring[10 & 3] = ring[2] = n2`，`tail: 10 → 11`

**Step 4 结束**

```
backing 索引                           缓存
┌────┬───────┬──────────┐     ┌────┬──────┬──────────┬─────┬───────┐
│ s1 │  25   │    n3    │     │ id │ slot │  state   │ ref │  key  │
│ s2 │  10   │    n0    │     ├────┼──────┼──────────┼─────┼───────┤
│ s3 │  18   │    n2    │     │ n0 │  2   │RESIDENT⚑ │  0  │  10   │
│ s4 │   3   │    -1    │     │ n1 │  5   │RESIDENT⚑ │  0  │  41   │
│ s5 │  41   │    n1    │     │ n2 │  3   │RESIDENT⚑ │  0  │  18   │
└────┴───────┴──────────┘     │ n3 │  1   │RESIDENT⚑ │  0  │  25   │
                              └────┴──────┴──────────┴─────┴───────┘
key 3 已退出缓存，数据在 host[4]        （key 列为示意，node 中并不存储）

ring   head = 7   tail = 11    逻辑顺序：n3 → n0 → n1 → n2
```

---

### B.5 一个 step 内的时序全貌

```
   ┌──── prefetch（只出队 / 只写索引）────┐  ┌── fwd/bwd ──┐  ┌─ release（只入队）─┐
   │ ① 去重                               │  │             │  │                    │
   │ ② 探测三分类 + pin        ← 只读索引 │  │ ⑦ gather    │  │ ⑨ 置 dirty         │
   │ ③ ring pop（含校验 / 丢弃）          │  │             │  │ ⑩ ref - 1          │
   │ ④ 清 node_id[s_old]       ← 写索引   │  │ ⑧ 梯度规约  │  │ ⑪ 若 1→0 入队尾    │
   │ ⑤ 置 slot / node_id[s_new]← 写索引   │  │    优化器   │  │                    │
   │ ⑥ 双向搬运（H2D + D2H）              │  │    原地更新 │  │                    │
   └──────────────────────────────────────┘  └─────────────┘  └────────────────────┘
       写 ring_head                              只读/写            写 ring_tail
       读写 node_id / slot                      emb_table            写 state

   └──── 由流序保证：索引的读（②）与写（④⑤）不并发，出队与入队不并发 ────┘

   与 step N-1 的 dense 计算重叠 ↗
```

注意 ② 与 ④⑤ 虽同在 prefetch 阶段，但必须是**不同的 kernel**——这是 §6.4 pin 身份竞态的主防线。`ring_head` 只在 prefetch 阶段被写、`ring_tail` 只在 release 阶段被写，则是 §7.2 那条约束的具体形态。

---

### B.6 不变量在本例中的落点

| 不变量 | 在哪一步体现 |
|---|---|
| INV-1 `ref > 0` 永不被淘汰 | B.3 ② —— n0 因 `ref = 1` 被跳过 |
| INV-2 仅在 `1→0` 且 `in_ring == 0` 时入队 | B.2 ④ —— n0 因 `in_ring` 已置位而不重复入队 |
| INV-3 弹出后 CAS 重新校验 | B.3 ② —— n1 经 `CAS(RESIDENT\|ref=0 → EVICTING)` 才被占用 |
| INV-4 `tail - head ≤ C` | B.2 结束时取等（环满），全程未超 |
| INV-5 `head ≤ tail` | 本例未触发抽干；若所有 node 都被 pin，`ring_pop` 将返回 `-1` 并归还 head |
| INV-6 每 unique key 每步恰好 `+1 / -1` | key 10 在 Step 1/2/3 各 `+1/-1` 一次，`ref` 始终回到 0 |
| INV-7 替换策略对结果透明 | B.4 ④ —— key 18 的整行（emb + optim state）经 host[3] 完整往返，值不变 |
| **INV-8 双向指针一致** | 全程 `node_id[s] == n ⟺ slot[n] == s`；B.3 ③ 与 B.4 ③ 是唯二打破再重建它的时刻 |

**若启用 CLOCK**：B.3 ② 中 n0 会先被检查 `used` 位——但 `ref != 0` 的判定在前，仍按陈旧条目丢弃，结果相同。差异出现在另一种情形：若 n0 当时 `ref == 0` 且 `used == 1`，FIFO 会直接淘汰它，而 CLOCK 会清零 `used`、把它送进 `retry_buf` 放回队尾，转而淘汰下一个——key 10 得以留在缓存中。

---

### B.7 三分类在本例中的分布

| Step | key | 分类 | PCIe 行为 |
|---|---|---|---|
| 1 | 10, 18, 3 | ① 全新 | 无（initializer 生成） |
| 2 | 10 | ③ 命中 | 无 |
| 2 | 25 | ① 全新 | 无 |
| 3 | 10 | ③ 命中 | 无 |
| 3 | 41 | ① 全新 | D2H 一行（受害者 key 18） |
| 4 | 18 | ② 未缓存 | H2D 一行 + D2H 一行（受害者 key 3），**双向同时** |

Step 4 是稳态的典型形态：**一次 miss 触发一次淘汰，H2D 与 D2H 各一行，方向相反、数量相等**——这正是 §9.2 主张把两者合并进一个双向 kernel 的依据。
