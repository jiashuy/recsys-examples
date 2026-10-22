# lw_cache

GPU 侧的 embedding 缓存，训练路径专用。设计见
[`docs/lightweight_cache_design.md`](../../docs/lightweight_cache_design.md)。

## 结构

| 文件 | 内容 |
|---|---|
| `backing.h` | backing 契约。缓存只要求两件事：把 key 解析成行号、给出行的可寻址指针 |
| `hash.h` | `mix64` |
| `node_state.h` | 状态字与全部原子协议 |
| `ring.h` | 可回收 node 队列 |
| `hash_chain.h` | key → node 的桶链 |
| `cache.h` | 单表的行池、node 元数据、配置与校验 |
| `ops.h` | 每 key 的操作，按阶段划分 |
| `kernels.cuh` | kernel 体（grid-stride，可在 host 打桩下直接调用） |
| `multi_cache.h` | 每表一套结构 + 元素到表的分派 |
| `multi_kernels.cuh` | 多表 kernel（一次 launch 覆盖全部表） |
| `launch.cuh` | `<<<>>>` 启动与流序纪律 |
| `binding.cu` | Python 绑定（单表 / 多表）。allocate 路径全程无 host 同步 |
| `testing/stub_backing.h` | 测试夹具 backing |

测试在 `../../test/lw_cache/`——刻意放在 `src/` 之外，因为构建脚本会递归收集
`src/**/*.cpp` 编进扩展模块，7 个 `main()` 会直接链接失败。

## 跑测试

```bash
# C++ 层：不需要 GPU，不需要 CUDA toolkit
cd ../../test/lw_cache && ./run_host_tests.sh

# 差分测试：需要 GPU 与已编译的扩展模块
python ../../test/lw_cache/test_differential.py   # 单表
python ../../test/lw_cache/test_multi_table.py    # 多表
```

## Python 接口

```python
from dynamicemb_extensions import LwMultiCache, LwMultiStubBacking

cache   = LwMultiCache(capacities=[64, 128], row_elems=[4, 8])
backing = LwMultiStubBacking(slots=[4096, 4096], row_elems=[4, 8])

# keys 按表分段且段内去重（segmented_unique 的输出形状）
nodes = cache.lookup(keys, table_range)
nodes = cache.allocate(keys, table_range, nodes, backing)
rows  = cache.rows(t)          # 每表一个行池；宽度不同，没有统一的堆叠视图
rows[nodes[lo:hi].long()] += grad
cache.release(nodes, table_range)
```

单表用 `LwCache` / `LwStubBacking`，接口相同但不带 `table_range`。

## 验证状态

**已在 H100 上实测**（EOS，sm_90，CUDA 13.2，torch 2.12；`test/lw_cache/gpu/eos_job.sh`）

C++ / CUDA：

- nvcc 编译整套头文件，以及 `binding.cu` 经 nvcc + torch 构建出扩展模块
- **pin 与 claim 的互斥**：160 万次竞争，0 个不可能状态字，最终 `ref == 0`
- **桶锁**：512 node 挤 8 个桶并发插入 → 512/512 可达、双向指针一致、无环；并发摘除后无残留
- **ring 并发**：8192 线程抢 4096 node → 0 重复、0 丢失，全部推回后 `tail - head == C`
- **多表分派**：表边界刻意错开 block 大小，shared memory 快路径与跨表 per-thread 回退**两条都走到**；三张表共用同一批 key 值、行宽 4/6/2，400 行全对
- **端到端**：真实 launch，FIFO / CLOCK / 抖动三种配置与纯 map 模型逐位一致
- **prefetch 深度 1 / 2 / 3 / 4**：多步在飞、`ref > 1`（一个 node 被多个 step 钉住）、两条 stream 之间**只靠 event 排序**（allocate 路径无 host 同步，否则同步会替 event 排好顺序、测了等于没测）
- `compute-sanitizer --tool racecheck`：**0 hazards**
- `compute-sanitizer --tool memcheck`：**0 errors**
- 全速、racecheck、memcheck 三种速度下结果一致——时序相关的缺陷会在这里现形

Python：

- 绑定导入并可用（`LwCache` / `LwMultiCache`）
- **单表差分测试 14 组配置 × 10 seed × 200 步全通过**：容量与工作集之比三档、均匀 / zipf / 全撞同一桶三种分布、backing 淘汰开关，FIFO 与 CLOCK 各一遍
- 多表差分 4/4：跨表同名 key、混合行宽、单表抖动的隔离性、随机多表

**已验证**（host 打桩，单线程；`test/lw_cache/run_host_tests.sh`，任何机器可跑，不需要 GPU）

状态字与全部转移、ring 协议、桶链、容量校验、ops 与 kernel 两条路径的全部分支、多表分派、指标的每个 bump 点。

**仍未验证**

- **真实 backing**：目前只有测试夹具实现。契约（`backing.h`）已定，接上去是剩下的主要工作。

PCIe 的两项（全双工、回写排序）已实测，结论见设计文档 §12.1 / §12.2——排序是主导因素，且与双向融合并不冲突。

### 上机撞出来的四个缺陷（均已修）

**四个全都出在测试脚手架与真实环境的落差上**，被测代码本身的设计没有被推翻。前三个是脚手架比真机宽容，第四个是脚手架自己写错了异步复用：

1. **CUDA 没有 64 位 `atomicSub`**（只有 int / unsigned int）。host 打桩自己加了一个不存在的重载，把 `ring_pop` 归还路径的编译错误掩盖了整整九组绿灯测试。打桩现已与 CUDA 的重载集一致。
2. **`absorb_retry_kernel` 未重置 `retry.count`**：计数逐步累加，下一步把上一步的条目再推一次 → 同一 node 在环中出现两次 → 破坏 INV-4 → 被两个线程同时取走 → 数据损坏。只在 CLOCK + 真并发下现形（全速 45 个 key 错，racecheck 下 0 错）。掩盖它的是 `test_kernels.cpp` 里一句手工 `*retry.count = 0`。
3. **`MultiCacheView::table()` 不能在 host 调用**：它读的每个字段都是设备指针数组，host 侧调用直接段错误。现已改为 `__device__` only，把误用变成编译错误。打桩下"设备指针"就是普通 host 指针，这类错误测不出来。
4. **CLOCK 的容量需求是 FIFO 的 `(1+k)` 倍**：一次第二次机会消耗一个环条目却不换来 node（条目进了 retry buffer，本次 kernel 内不可见）。设计文档 §10.1 的公式与 `CacheConfig::validate()` 都漏了这一项，表现为按 FIFO 定容量、切到 CLOCK 后跑几百步中途抽干。
5. **深度测试自身的缓冲区复用竞态**：`release` 异步发在主 stream 上，而下一步的预取在另一条 stream 上覆写同一批 `d_nodes`，release 可能读到被覆写的值。现已按真实训练循环该有的写法修正——每槽一个 event，复用前 wait。

值得一提的是发现方式：#2 是**全速下错、racecheck 下对**，#5 恰好相反。两个方向的时序敏感缺陷都撞上了，所以"全速 / racecheck / memcheck 各跑一遍"不是仪式。

## 尚未实现

- P1：加载与回写合并成双向 kernel（前提是实测 PCIe 全双工）
- 把 backing 接口接到真实的 host 表上（目前只有测试夹具实现）

## 在 EOS 上重跑

```bash
rsync -az corelib/dynamicemb/ login-eos:~/lw_cache_test/dynamicemb/
ssh login-eos "cd ~/lw_cache_test && sbatch dynamicemb/test/lw_cache/gpu/eos_job.sh"
```

用 `sbatch` 而非 `salloc`：交互式分配要求客户端一直连着，而这件事的产物本来就是一份日志。
账户 `coreai_devtech_all`（`coreai_devtech_china` 会撞 `QOSGrpNodeLimit`），
EOS 的 GPU 不是 slurm gres，所以整节点 `--exclusive` 而不是 `--gres=gpu:N`。
