# 缓存与 host 存储之间交换行：探索记录（含 CPU 路径）

本文记录为"让缓存的 swap（被顶掉的行写回存储、新行读入缓存）尽可能接近 PCIe 双向带宽"做过的所有探索，包括没有被采用的 CPU 路径。目的是以后有时间可以从任何一个点接着做，不必重新测。

- **最终采用**：GPU 排序暂存的 `RowExchange.swap`，在 `chained-only`（`3446646`）。
- **所有实验代码**：分支 `row-exchange-experiments`（`83d3761`、`6169d90`）。其中 `6169d90` 含非临时写、专用线程加标志位等待，以及全部探针程序（`benchmark/host_link_probes/`）。
- **设计背景**：`docs/component_redesign.md` 第 2.1 节。

## 一、结论摘要

1. 链路能跑多快：`cudaMemcpy` 单向约 52 GiB/s、双向 94 GiB/s；kernel 零拷贝双向 78 GiB/s。连续缓冲上的配对 kernel 能到 84 GB/s 双向。
2. GPU 经 PCIe 随机访问 host 表很慢，**瓶颈是 GPU 对 host 内存的地址翻译覆盖不够**：表在 256 MB 以内每行 10 ns（链路满速），13 GB 时 34 ns；排序降到 16.6 ns。行宽无关。
3. 大页、`cuMemCreate` host 分配、`cudaHostAlloc`、NUMA 绑定对 GPU 随机访问**都没有效果**；DMA 引擎批量拷贝随机行（`cudaMemcpyBatchAsync`）慢 50 倍。
4. 融合 kernel 把一个 node 的写回行和读入行配对，没法让两条行号列表都有序，两个方向也不重叠（swap 等于读加写之和）。
5. CPU 随机访问 host 内存比 GPU 经 PCIe 快得多，但是：scatter 要用非临时写才不浪费带宽（快 2.5 倍）；CUDA 驱动把所有 stream 的 host 回调放在一个线程上；两个方向的 CPU 拷贝同时跑会争抢内存带宽。**最好的 CPU 路径只在窄行（512 B）上比 GPU 排序暂存好约 10%，宽行（1536 B）上差 40% 以上。**
6. 采用 GPU 排序暂存：adam 20 万行 9.0 ms、双向 68 GB/s（约 68% 的 94 GiB/s 峰值），sgd 20 万行 5.4 ms。

## 二、环境与方法

**硬件和软件**（EOS 节点）：8 × H100 80GB HBM3；2 × Xeon Platinum 8480C（每插槽 56 核、112 线程，共 224 CPU）；2 个 NUMA 节点各约 1 TB；GPU 0 到 3 在节点 0（CPU 0 到 55 和 112 到 167），GPU 4 到 7 在节点 1；CPU 与 GPU 之间 PCIe Gen5 x16；CUDA 13.3；PyTorch devel 容器。

**host 存储的实际形态**：`HostVMMTensor`（`src/vmm_tensor.cu`）是 `mmap` 匿名内存加 `mlock` 加 `cudaHostRegister(mapped, portable)`；UVA 下 `data_ptr()` 同时是 CPU 能直接用的指针（CPU 路径依赖这一点）。扩容用 `mremap`，会移动基地址，所以 `extend()` 开头有 `torch.cuda.synchronize()`，任何异步 CPU 工作都必须在它之前结束。

**微基准设置**：host 存储 26,843,545 行（sgd 的 value_dim 128，行 512 B，表 13.7 GB；adam 的 value_dim 384，行 1536 B，表 41 GB）；缓存 2,097,152 行；一步交换 10 万或 20 万行（模块 benchmark 的缓存命中率 0.8 时一步约 20 万行）；行号均匀随机。"有效双向带宽" = 写回字节数加读入字节数，除以总时间。

**方法上的教训（都实际踩过）：**
- **热缓存假象**：用同一批行号反复测，10 万行乘 512 B 约 51 MB 放得进 CPU 末级缓存（每插槽约 105 MB），CPU gather 被测成 124 GB/s，真实冷缓存约 65 GB/s。**所有 CPU 数字必须每次迭代用新的随机行**（微基准现在预生成 12 组轮流用）。
- **构建脚本吞错**：`build.sh` 把输出截成最后 30 行、不返回编译器退出码，编译失败时 `.so` 还是旧的，后面的测试只报导入错误。现在保存完整日志、打印所有 error、返回编译器退出码。
- **nvcc 只报 "Compilation aborted"**：`.cu` 里放 `std::thread`、`std::function`、`cudaLaunchHostFunc` 这类纯 host 代码会失败且没有错误信息（构建命令里有 `-w`）。把 host 代码拆到独立的 `.cpp`（`host_rows.cpp`）解决。
- **旧完成标记**：新作业复用旧输出目录时，等待命令一看到旧的 `DONE` 就返回了旧结果。提交前先删旧的标记。
- **一个作业一个目录**：作业运行时往它的目录同步文件或重编译，会让一次运行里混用两个版本的代码。
- 事件要用 `enable_timing=True` 才能量时间；`torch.device("cuda")` 没有序号，host VMM 缓冲会因设备号为 `None` 失败。
- 微基准里的时间含 Python 开销和每次迭代的同步，没有和训练的其他计算重叠。

## 三、链路峰值

`debug_pcie_bidir_bw/pcie_bidir_bw.cu` 和 `pcie_bidir_kernel.cu`（现在在 `benchmark/host_link_probes/`），256 MiB 缓冲：

| | GiB/s |
|---|---|
| `cudaMemcpy` H2D（pinned） | 51.98 |
| `cudaMemcpy` D2H | 51.14 |
| `cudaMemcpy` 同时（两个 stream） | **94.17** |
| kernel 零拷贝读 | 47.84 |
| kernel 零拷贝写 | 48.88 |
| kernel 零拷贝同时读写 | **78.33** |

`asyncEngineCount = 3`。双向加速比 1.81 倍（memcpy）、1.60 倍（kernel）。

## 四、GPU 直接访问 host 表

### 4.1 基线：融合 swap（一个 block 一个 node，读旧行、读新行，再写）

同一 node 的写回行和读入行配对（`swap_rows_kernel`，在 `391e7c3`）。时间（ms）：

| 行宽 | 行数 | swap | 只写回 | 只读入 | 每方向 GB/s |
|---|---|---|---|---|---|
| 128 | 10 万 | 6.41 到 6.45 | 3.36 到 3.38 | 3.35 到 3.38 | 8.0 |
| 128 | 20 万 | 12.8 到 12.9 | 6.73 到 6.76 | 6.7 | 8.0 |
| 384 | 10 万 | 7.45 到 7.50 | 3.78 到 3.80 | 3.72 到 3.75 | 20.6 |
| 384 | 20 万 | 14.9 到 15.0 | 7.55 到 7.6 | 7.45 | 20.6 |

两点：**swap 时间等于只写回加只读入，两个方向没有重叠**；**每行耗时几乎恒定**（512 B 的行 33.7 ns、1536 B 的行 37.3 ns），所以窄行的带宽很低，宽行单向接近链路峰值（41 GB/s）。

### 4.2 线程组织、排序

| 变体（ms） | 128 / 10 万 | 128 / 20 万 | 384 / 10 万 | 384 / 20 万 |
|---|---|---|---|---|
| block 每行，随机 | 6.44 | 12.9 | 7.45 | 14.9 |
| warp 每行，随机 | 6.46 | 12.9 | 7.46 | 14.9 |
| block，按读入行排序（配对保持） | 5.47 | 10.3 | 6.83 | 13.05 |
| block，两个列表各自排序（上界） | 3.25 到 3.41 | 4.5 到 4.95 | 5.1 到 5.4 | 8.2 到 8.8 |
| warp，两个列表各自排序 | 3.87 | 4.83 | 5.8 | 8.8 |

- warp 每行和 block 每行**没有区别**，线程组织不是因素。
- 只按读入行排序（配对保持）收益有限；两个列表各自排序快 1.9 到 2.6 倍，但这样配对丢失，不能直接用，是排序暂存方案的动机。

### 4.3 host 内存的分配方式没有影响

`host_alloc_rows.cu`，13 GB 表，20 万行，读、写、swap（ms）：

| 分配方式 | 512 B 读 / 写 / swap | 1536 B 读 / 写 / swap |
|---|---|---|
| `cudaHostAlloc` | 6.60 / 6.64 / 12.64 | 7.10 / 7.22 / 14.01 |
| `mmap` 加 `cudaHostRegister`（现在用的） | 6.67 / 6.69 / 12.75 | 7.10 / 7.23 / 14.03 |
| 同上加 `MADV_HUGEPAGE` | 6.64 / 6.66 / 12.69 | 7.09 / 7.22 / 14.02 |
| `cuMemCreate` host（粒度 2 MB） | 6.63 / 6.66 / 12.69 | 7.06 / 7.18 / 13.94 |
| `cuMemCreate` host NUMA | 6.64 / 6.67 / 12.70 | 7.00 / 7.12 / 13.81 |

在真实模块路径上也试了 `MADV_HUGEPAGE`（13 GB 确实在大页上，`AnonHugePages` 13106 MB）：随机 swap 12.89 对 12.77 ms，adam 14.91 对 14.89 ms，没变；排序后 4.9 对 4.35 ms 稍好。系统的透明大页是 `madvise` 模式。

### 4.4 TLB 覆盖范围

`tlb_probe.cu`，20 万个随机 512 B 行，kernel 读，host 表大小扫描（ns 每行；随机 / 排序）：

| 表大小 | 随机 | 排序 |
|---|---|---|
| 16 MB | 9.9 | 11.7 |
| 64 MB | 10.0 | 10.0 |
| 256 MB | 10.0 | 10.0 |
| 1 GB | 14.7 | 10.0 |
| 4 GB | 27.5 | 10.0 |
| 13 GB | 33.7 | 16.6 |

10 ns 每行正好是链路单向峰值（512 B 乘 1 亿行每秒约 51 GB/s）。随机访问在 1 GB 以上开始变慢；排序后到 4 GB 仍满速。说明 GPU 对这块内存的翻译缓存覆盖大约是 1 GB 量级；上面大页和分配方式没有效果，说明 GPU 一侧本来就按大页翻译，换分配方式没有改变它。

### 4.5 让 DMA 引擎搬随机行

`cudaMemcpyBatchAsync`（CUDA 13.3 的签名没有 `failIdx`），20 万个 512 B 源和目的地址、13 GB 表：**108.7 ms，每行 544 ns**，比 kernel（33.7 ns）慢 16 倍，比满速（10 ns）慢 54 倍。不可行。

## 五、CPU 的随机访问能力

`cpu_copy_bw.cpp`，节点 0（`numactl --cpunodebind=0 --membind=0`），13 GB 表，20 万行，每次迭代新行。

**连续拷贝（4 GB，payload GB/s）**：4 线程 44.5、8 线程 71.2、16 线程 92.1、32 线程 98.9（峰值）、56 线程 97.4、112 线程 96.1。两个节点交错分配（`--interleave=0,1`）：最高 128.7 GB/s（56 线程）。

**随机行（payload GB/s，括号是占连续拷贝峰值 99 的比例）**：

| | 线程 | 512 B 行 | 1536 B 行 |
|---|---|---|---|
| gather，普通写 | 32 / 56 | 54.4（55%）/ 50.4 | 60.3 / 65.9（67%） |
| gather，加软件预取（下 8 行） | 32 / 56 | 62.3（63%） | 60.1 / 67.1（68%） |
| scatter，普通写 | 32 / 56 | 37.2（38%）/ 38.7 | 51.6 / 59.9（61%） |
| scatter，**非临时写**（512 位流式存储） | 16 / 32 | **101.8（103%）** / 87.7 | 90.8 / **98.8（100%）** |
| scatter，非临时写加预取 | 32 | 50.7 | 57.4 |

- scatter 用非临时写之后达到连续拷贝的速率，是普通写的 2.6 到 2.7 倍；原因是普通写要先把目的行读进缓存（read-for-ownership），流量翻倍。
- **非临时写加预取反而慢了**（预取把行拉进缓存和流式写相冲）。
- 线程数 16 到 32 最好，64 以上变慢。
- 交错分配对随机行没有帮助（gather 54.2，scatter NT 89.5 对 87.7）。
- NUMA 绑定对整条 CPU 路径没有影响（`numa1`：plain / bound0 / bound1 三次结果几乎一样）。

## 六、CPU 路径的实现

代码在分支 `row-exchange-experiments`：`src/row_exchange/host_rows.cpp`（线程池、拷贝、调度）、`src/row_exchange/row_exchange.cu`（`wait_flag_kernel`）、`dynamicemb/row_exchange.py`（`swap_via_cpu`、`swap_pipelined`）。

### 6.1 数据流

读入：行号 D2H 到 pinned；CPU 线程按行号从 host 表 gather 到 pinned 连续缓冲（`inn`，行步长 = 最宽表的行字节）；一次 H2D 到 HBM 暂存；按 node 写进缓存。
写回：先把被顶掉的行从缓存 gather 到 HBM 暂存（必须在读入写 node 之前）；D2H 到 pinned 连续缓冲（`out`）；CPU 线程按行号 scatter 到 host 表。
读入失败（行号 -1，新 key）的 node 会被写入缓冲里的残留值，要求调用方后面用初始化覆盖（prefetch 里确实如此）。多表、不同行宽：缓冲是统一步长的填充行，CPU 只拷贝每张表自己的行字节数。

### 6.2 几个版本

| 版本 | 做法 | 方法名 |
|---|---|---|
| v1 | 两个 stream（读入一个、写回一个），整块，CPU 部分用 `cudaLaunchHostFunc` 放进 stream 顺序里 | `swap_via_cpu` |
| v2 | 分块（2、4、8、16 块），四个 stream（gather、H2D、D2H 加被顶掉行的 gather、scatter），按块用事件衔接 | `swap_pipelined` |
| v3 | scatter 用非临时写 | `copy_row`（`_mm_stream_si128`，目的 16 字节对齐、行宽是 16 的倍数时，否则回退 `memcpy`；每线程做完 `_mm_sfence`） |
| v4 | gather 和 scatter 各用一个专用调度线程加各自的线程池；host 回调只把任务交给调度线程；做完写 pinned 标志位；stream 里放 `wait_flag_kernel` 自旋等标志 | `Dispatcher`、`Flags` |

### 6.3 事件顺序（正确性靠这些）

1. 旧值必须先读出，node 才能被覆盖：被顶掉行的 gather 完成记事件 `gathered`，当前 stream 写 node 前等它。
2. 所有 stream 开始前等当前 stream 上的 `ready`。
3. CPU 工作放在 stream 里：host 回调（v1、v2）排在行号和数据的 D2H 之后；v4 用标志位。
4. 读入这一侧开始前等上一步的 `write_done`（上一步 scatter 之后的事件），因为上一步写回的行可能就是这一步要读的。
5. **跨步缓冲复用**（分块版本发现的两处竞态，已修）：下一步的 gather 要等上一步的读入完成（`_fetch_done`），下一步的 D2H 要等上一步的 scatter 完成（`_write_done`）。不分块版本因为同一方向在同一个 stream，天然有序。
6. 存储增长：`extend()` 的 `torch.cuda.synchronize()` 等所有 stream 和 host 回调做完；基地址每次调用现取。
7. 缓冲保留在 `RowExchange` 里；增长时先同步相关 stream 再换缓冲。

### 6.4 非临时写的正确性

非临时写写的值和普通写一样，只是弱序：需要 `sfence` 保证"之前的所有写对所有人可见"后再发布。每个工作线程在返回计数之前 `sfence`；调度线程做完后再 `sfence`，才写标志位（release）。gather 的目的地（pinned 缓冲）由 DMA 引擎读，scatter 的目的地（host 表）由下一步的 CPU gather 或 GPU 读；这两个读者都排在 `sfence` 之后。各线程写互不重叠的行。目的地址不对齐时回退普通 `memcpy`。弱内存序的问题是概率性的，测试通过不能证明没有问题，靠的是"`sfence` 在发布之前"这条规则；建议以后再加一个故意提高概率的压力测试（让 scatter 和下一步 gather 反复访问同一批行）。

### 6.5 CUDA 驱动对 host 回调的行为（实测）

在两个 stream 上各放一个 gather 和 scatter 的 host 回调，没有依赖，"同时跑"对"单独相加"（ms）：

| | 单独 gather | 单独 scatter | 同时 | 相加 |
|---|---|---|---|---|
| sgd 10 万（普通写） | 0.52 | 1.28 | 2.25 | 1.81 |
| sgd 20 万（普通写） | 1.54 | 2.98 | 5.01 | 4.52 |
| adam 20 万（普通写） | 6.00 | 7.25 | 13.7 | 13.2 |

同时跑的时间约等于相加（甚至更大），说明驱动在一个线程上串行执行所有 stream 的 host 回调。v4 改用专用调度线程后（非临时写 scatter）：sgd 20 万 gather 1.60、scatter 1.25，同时 2.65（相加 2.85）；adam 20 万 5.96、3.92，同时 8.93（相加 9.88）。**能同时跑了，但同时跑几乎没有省时间**，说明两者争抢的是 CPU 内存带宽（合计 600 MB 在 8.9 ms 内约 67 GB/s，接近一个插槽的连续拷贝峰值）。

### 6.6 分块的固定开销

分块越多越慢：sgd 10 万行 pipelined c2 / c4 / c8 / c16 = 3.05 / 3.21 / 5.07 / 5.89 ms，每块多约 0.2 ms（host 回调调度、事件、线程池唤醒）。分块只在 10 万行 sgd 上有收益。

## 七、CPU 路径的测量结果（按版本）

时间单位 ms，每次迭代用新行（除非特别说明）。

| 版本 | sgd 10 万 | sgd 20 万 | adam 10 万 | adam 20 万 |
|---|---|---|---|---|
| 基线：GPU 融合 swap | 6.44 | 12.9 | 7.48 | 14.9 |
| v1 整块（普通写，host 回调） | 4.9 到 5.1 | 5.6 到 5.9 | 7.7 | 16.2 到 16.6 |
| v2 分块，最好一档 | 2.7 到 2.9（2 块） | 5.7（2 块）到 6.1 | 7.6 到 8.2 | 16.3 到 17.8 |
| v3 整块（scatter 和 gather 都非临时写） | 3.83 | 4.71 | 7.04 | 14.2 |
| v3 分块（2 块，32 线程） | **2.17** | 4.29 | 6.04 | 12.4 |
| v4 整块（专用线程，gather 普通写） | 3.75 | 4.64 | 7.01 | 15.2 |
| v4 分块（2 块或 4 块） | 2.68 | 4.63 | 6.62 | 13.1 |
| **GPU 排序暂存（采用）** | 3.62 到 3.77 | 5.10 到 5.43 | 5.65 到 5.70 | **9.05 到 9.12** |

**单独的 CPU 拷贝（20 万行，ms）**：普通写 gather 1.48、scatter 3.0（sgd），5.9 和 7.1（adam）；v3 的非临时写 scatter 1.2 和 3.8；v3 把 gather 也改成非临时写后，sgd gather 反而从 1.48 变慢到 1.95（连续目的地用普通写本来就顺），adam 5.9 变 5.6 几乎没变，所以 v4 把 gather 改回普通写。

**时间线**（一步里每块完成的时间点，v2、4 块、32 线程，sgd 20 万行，v1 的普通写）：gather 完成在 2.30 / 3.75 / 5.09 / 6.41 ms，一块 1.3 ms（单独 gather 全部只要 1.48 ms），读入完成 7.36，scatter 完成 7.32 ms。分块没有带来重叠：host 回调串行，一步就是 gather 加 scatter 加传输一个接一个。v4（专用线程）下 gather 完成在 0.63 / 1.34 / 2.39 / 3.64 ms，读入 4.37，scatter 4.22 ms。

**其他做过的 CPU 相关变体**：
- 单 stream 配对 kernel：CPU gather 到 pinned 缓冲 → 配对 kernel 原地交换 node 与缓冲 → CPU scatter。kernel 单独 1.23 / 2.43 / 3.66 / 7.28 ms（每方向 42 GB/s，双向 84 GB/s，等于 kernel 零拷贝双向峰值），但整体串行：sgd 20 万 6.8 到 7.2，adam 20 万 20.7（比两个 stream 慢 20% 到 25%）。
- GPU 暂存原型（用 Python 拼现有 load / store kernel，两个 stream）：sgd 20 万 8.2、adam 20 万 13.0，远不如融合有序 kernel 的上界，因为两个各自占满 GPU 的 kernel 不能真正重叠，加上两次 `argsort` 和几次额外 kernel 的开销（后来用单个融合 kernel 解决，见第八节）。

## 八、为什么 CPU 路径没有赢，以及最终采用的做法

1. **CPU 路径的瓶颈是 CPU 内存带宽**：一步要在 host 表上做随机行 gather 加 scatter，payload 合计 200 MB（sgd）和 600 MB（adam）。按 99 GB/s 的连续拷贝峰值算，下限分别是约 2 ms 和约 6 ms，再加上和链路传输之间不能完全重叠。
2. **宽行上不划算**：adam 一步搬 600 MB，CPU 要拷两遍（host 表到 pinned，pinned 再 DMA），GPU 排序暂存只走一遍链路。
3. **窄行收益很小**：sgd 20 万行最好 4.3 到 4.6 ms，GPU 排序暂存 5.1 到 5.4 ms，约 10%，在噪声边缘；10 万行 2.2 到 2.7 对 3.6 到 3.8 ms，约 1.4 到 1.7 倍，但这是比较小的步。
4. **复杂度**：线程池、调度线程、pinned 缓冲、标志位、自旋 kernel、四个 stream、跨步事件，还有 `mremap` 与异步 CPU 工作的交互。
5. **GPU 排序暂存**：先把被顶掉的行在 HBM 里按存储行号排序 gather 到暂存区；一个融合 kernel 取两条各自排序的列表的第 i 项，第 i 个被顶掉的行写出去，第 i 个读入的行写进它的 node。不用配对（暂存区已经保存了旧行），两个方向在同一个 kernel 里同时进行，访问存储都是有序的。设备和 host 存储都可用。

| 20 万行（ms，有效双向带宽 GB/s） | sgd | adam |
|---|---|---|
| GPU 融合（旧） | 12.9，16 | 14.9，41 |
| CPU，最好一版 | 4.3 到 4.6，44 到 48 | 12.4 到 13.1，47 到 50 |
| **GPU 排序暂存** | 5.1 到 5.4，38 到 40 | **9.0 到 9.1，68** |
| 参照：连续 pinned 缓冲上的配对 kernel | 2.43，84 | 7.28，84 |

adam 上 68 GB/s 约 64 GiB/s，是 `cudaMemcpy` 双向峰值 94 GiB/s 的约 68%。

### 模块里的情况

- 模块 benchmark（T10，D128，缓存命中率 0.8，sgd）：swap 之前 forward 9.98 ms；融合 swap（`391e7c3`）9.20 ms；排序暂存 8.22 ms。adam：14.03、10.84、10.79（forward 基本没变）。
- profile：融合版的 `swap_rows_kernel` 平均 7.5 ms 每次，占 GPU 总时间 80%；排序暂存的 `swap_staged_kernel` 平均 6.05 ms 每次，占 74%，一步 GPU 总时间 8.0 ms。其余 kernel（取值、归约、更新）各 0.2 到 0.3 ms。
- **模块里比微基准慢约 26% 的原因（已查清）**：模块的 host 存储是 10 张表、每张 3020 万行、行 512 B，共约 155 GB；微基准只有一张 13.7 GB 的表。内核时间跟一步里碰到的不同 2 MB 区域数相关：每边 1.5 万个区域 5.2 ms，3.1 万个 6.1 ms，14 万个 14.4 ms。模块里的行分布有聚集（排序后 33% 的写回行和 53% 的读入行相邻，每边约 1.5 万到 1.6 万个区域），所以只比单表情形高约 18%，远低于完全随机的 14.4 ms。其余因素都排除了：暂存 gather 加 argsort 一共约 0.33 ms，稳态下没有 `-1`，内核运行期间没有别的 stream 或拷贝引擎在工作。微基准的 4.8 ms 偏乐观，同一节点上等价情形实测 5.18 ms。可做的小改动：按（表，行）而不是只按行排序，内核 6.11 变 5.92 ms，排序多 0.03 ms，约 3%，没有采用。真正的杠杆是第九节第 1 条（存储行局部性）：区域数从 14 万降到 3.1 万，内核从 14.4 降到 6.1 ms。
- 对比上游（同一台 H100）：缓存 adam 命中率 0.8 快 2.0 倍（23.7 到 11.8 ms），命中率 1.0 快 3.5 倍；sgd 快 2.3 和 3.8 倍；hybrid 快 1.2 到 1.3 倍；全 HBM 快约 1.05 倍。

## 九、没有探索的方向（按我认为的价值排序）

1. **存储行的局部性（`RowAllocator` 相关，最有潜力）**。TLB 探针显示表在 256 MB 以内随机访问也是 10 ns 每行（链路满速）。如果一步里要读写的行能集中在一个不超过约 1 GB 的区域，随机访问的代价就从 34 ns 降到 10 ns，GPU 路径可能直接接近链路峰值，不需要 CPU。做法的方向：让经常在缓存与存储之间来回的 key（最近被淘汰的、将要被再次读入的）的存储行集中分配（热区），而不是散在整张表里。需要设计分配策略和热区的回收。
2. **行换位**（victim 直接占用被读入 key 的存储行）：一对 swap 只访问一行（读后写同一行），翻译次数和访问次数减半，也不需要给 victim 另找行。代价是"被缓存的 key 在存储里没有行"，改变了"缓存是存储的副本"的约定，影响索引、checkpoint、flush、PS；和上面的热区策略可以结合（被释放的行自然就是热区）。
3. **CPU 路径只用于窄行**：行字节数不大于约 1 KB 时走 CPU，其余走 GPU 排序暂存。预期窄行小步有 1.4 到 1.7 倍、20 万行约 10%，要同时维护两套，优先级低。
4. **改进 CPU 路径本身**（如果以后要做）：
   - gather 目的地用 512 位流式存储或加软件预取（预取在 gather 上 63% 对 55%，在 scatter 非临时写上反而有害）；现在的 `copy_row` 用的是 128 位流式存储，探针里 512 位的 scatter 达到 100%，128 位的可能还有 10% 到 20% 的空间；
   - 用 `cuStreamWaitValue32`（stream 内存操作）代替自旋 kernel，不占一个 SM；
   - 分块循环放进 C++，减少每块 0.2 ms 的固定开销；
   - 线程绑定到 GPU 所在插槽的核；线程数 16 到 32 最好；
   - 表跨两个 NUMA 节点交错放：连续拷贝峰值 129 对 99 GB/s，随机 scatter 非临时写 112 GB/s（32 线程、1536 B 行）；但 GPU 经 PCIe 访问另一个插槽的内存要过 UPI，没测过；
   - **和训练流水线重叠**：微基准测的是单次 swap 的延迟。真实训练里预取在自己的 stream 上提前一步做，CPU 路径的 gather 和 scatter 可以完全藏在前一步的 GPU 计算后面（CPU 不占 SM、也不抢 HBM），而 GPU 路径的 kernel 会和前一步的计算抢 HBM 带宽。这是我没有量化的最大变量，CPU 路径在端到端里可能比微基准显示的更有竞争力。
5. **Grace Hopper**：CPU 与 GPU 之间是 NVLink-C2C，地址翻译走 ATS，是完全不同的状态，上面所有关于 PCIe、TLB 覆盖和 CPU 路径的结论都要重测。
6. **更精确的 TLB 覆盖**：在 256 MB 到 1 GB 之间细扫，确定覆盖的确切大小和翻译粒度，还可以试 `cudaDeviceSetLimit`、`cuMemAdvise`、HMM 的相关提示；也可以试 Hopper 的 TMA 或 `cp.async.bulk` 从 host 映射内存做批量拷贝（没有试过，不知道是否减少翻译开销）。
7. **减少搬运的字节**：宽行（adam）一步搬 1536 B 乘 20 万行，链路时间下限 6.2 ms。只搬必要的状态、用 fp16 或 bf16 存优化器状态，是和这个问题正交的杠杆。
8. 排序后相邻的行合并成连续区间（行密度高时才有意义，现在密度 0.75% 几乎没有相邻的行），用 `cudaMemcpyAsync` 的区间拷贝。

## 十、复现

**代码位置（分支 `row-exchange-experiments`）：**
- `dynamicemb/row_exchange.py`：`swap`（融合随机）、`swap_via_cpu`、`swap_pipelined`、`swap_sorted`（后者是主线采用的）。
- `src/row_exchange/row_exchange.cu`：融合 kernel（含 warp 变体）、`swap_staged_kernel`、`wait_flag_kernel`。
- `src/row_exchange/host_rows.cpp`：线程池、`copy_row`（非临时写）、调度线程、标志位、`enqueue_cpu_rows`、`cpu_copy_rows`。
- `test/unit_tests/test_row_exchange.py`：逐位比较各方法、跨步顺序、40 步对照模型的压力测试（对三种 CPU 和 GPU 方法都参数化）。
- `benchmark/bench_row_exchange.py`：新行的整套对比（含 CPU 单独 gather / scatter、host 回调并发检查、各版本、时间线）；`benchmark/bench_host_pages.py`：大页对比。
- `benchmark/host_link_probes/`：`pcie_bidir_bw.cu`、`pcie_bidir_kernel.cu`（链路峰值）、`host_alloc_rows.cu`（分配方式）、`tlb_probe.cu`（TLB 覆盖和 DMA 批量拷贝）、`cpu_copy_bw.cpp`（CPU 拷贝带宽）、`pcie_density_sweep.cu`、`pcie_bidir_gather.cu` 以及它们的作业脚本。

**探针的构建**：`nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 -o x x.cu`（`host_alloc_rows.cu` 要加 `-lcuda`）；`g++ -O3 -march=native -pthread -o cpu_copy_bw cpu_copy_bw.cpp`；CPU 探针用 `numactl --cpunodebind=0 --membind=0` 运行。

**EOS 作业的做法**：`sbatch` 一个无人值守的批作业，账号 `coreai_devtech_all`，作业名要以 `coreai_devtech_all-` 开头；容器镜像 `gitlab-master.nvidia.com#devtech-compute/distributed-recommender:devel_latest`，把一个独立的目录挂到 `/workspace/GR-B`；脚本里先 `rm -rf` 自己的输出目录，用 `progress` 文件记录每一步的退出码，最后写 `DONE`。一个作业一个目录；运行期间不往它的目录同步或重编译。

## 十一、遇到的问题与修复清单

| 问题 | 原因 | 修复 |
|---|---|---|
| 测试报 `cannot import name 'enqueue_cpu_rows'`，但构建"成功" | 构建脚本截断输出、不返回退出码 | 脚本保存完整日志、打印 error、返回编译器退出码 |
| nvcc 只报 "Compilation aborted" | `.cu` 里放了 host 线程代码 | 拆到 `host_rows.cpp` |
| 等到的是旧结果 | 复用输出目录，旧 `DONE` 还在 | 提交前删旧标记 |
| 分块版跨步竞态 | 复用的 pinned 缓冲在上一步的 H2D 或 scatter 还在用时被覆盖 | 下一步的 gather 等 `_fetch_done`，D2H 等 `_write_done` |
| 时间线报错 | 事件没开计时 | `enable_timing=self.trace` |
| host 缓冲创建失败 | `torch.device("cuda")` 没有序号 | 用 `torch.cuda.current_device()` |
| CPU 数字偏高 | 同一批行号反复测，缓存是热的 | 每次迭代新行 |
| 同步脚本破坏正在跑的作业 | 往运行中作业的目录同步或重编译 | 一个作业一个目录 |
