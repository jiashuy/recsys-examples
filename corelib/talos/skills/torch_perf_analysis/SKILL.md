---
name: torch-perf-analysis
description: "Analyze an existing nsys trace of a PyTorch workload (training or inference, NVTX-instrumented) and produce an end-to-end performance diagnosis. Uses the bundled `veloq` CLI for evidence extraction."
argument-hint: "<nsys_trace_file>"
---

# torch-perf-analysis: end-to-end PyTorch performance analysis

## Skill Directory

`${SKILL_DIR}` refers to the directory containing this SKILL.md.

## Overview

This skill is for analyzing a PyTorch workflow (training or inference) captured under nsys.

A companion skill, **`nsys-profile-analysis`**, covers veloq usage and generic NSys timeline patterns (iter regression, sync stalls, CUDA-graph drilldown, hotspot, bandwidth — see its `references/cookbook.md`). Consult it for veloq mechanics and general NSys workflows; this skill stays focused on PyTorch-specific recipes.

`veloq` is bundled inside `talos` — once you `pip install talos`, the binary is on `PATH`. Confirm with `veloq --version`.

---

## Prerequisites

The user captures the `.nsys-rep` before invoking this skill (the skill does **not** capture it). Two steps:

1. **Instrument the loop** — no model argument; a process-global profiler hooks every `nn.Module`. Call `enable()` once up front, bracket the window with `start()`/`stop()`, and mark each step with `step()`:

   ```python
   from talos.torch_perf_analysis import talos_profiler, nvtx_range

   talos_profiler.enable()    # optional: pre-load the observer up front

   for step, batch in enumerate(loader):
       if step == 50: talos_profiler.start()   # open capture window
       if step == 60: talos_profiler.stop()
       talos_profiler.step()                    # mark a step boundary
       loss = loss_fn(model(batch)); loss.backward()
       optimizer.step(); optimizer.zero_grad()
   ```

   Every outermost dispatched op that launches device work gets tagged with its module path + user call site (cwd-relative path:line); each kernel is later attributed to the line that launched it (forward and backward alike). Emission is a **C++ RecordFunction observer** (`talos_observer.cpp`, JIT-compiled on first use, ~1 min then cached; pre-build at install time with `python -m talos.compile_observer`) registered process-globally at the ATen dispatcher — it fires on *every* thread (framework workers and the C++ autograd threads), and measured overhead is ~4% of step time without nsys / ~5% marginal on top of an active nsys capture. `enable()` just pre-loads the extension so the mid-training `start()` stays clean. Wrap non-`nn.Module` code (an optimizer, a free helper) in `nvtx_range("name", "Class")` to give it its own attribution. Gated by `TORCH_PERF_ANALYSIS_ENABLE` — unset, every call is a no-op, so they can stay in the training script permanently. (Mechanics: `${SKILL_DIR}/python/profiler.py`.)

2. **Run under nsys** — `--capture-range=cudaProfilerApi` records only the `start()`→`stop()` window. Use the minimal-overhead flag set (CPU sampling and context-switch tracing off, trace only `cuda,nvtx` — no cudnn/cublas/osrt):

   ```bash
   TORCH_PERF_ANALYSIS_ENABLE=1 \
   nsys profile --sample none --cpuctxsw=none \
       --trace=cuda,nvtx \
       --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
       -f true -o profile_output python3 <train_script>.py
   ```

   (`--capture-range-end=stop-shutdown` terminates the process after the window; use `stop` + an external kill if the training loop must keep running.)

The user passes the resulting `.nsys-rep` via `$ARGUMENTS`.

---

## Workflow

Two phases. Phase 1 is a fixed `stats.py` invocation that produces a workspace of structured JSON evidence; Phase 2 is the agent reading that workspace and producing the diagnosis (see the analysis directions in Step 2).

Workspace location is set with `--workspace` (default: `.torch_perf_analysis` in cwd). When running inside a `torch-optimize` loop, point it at that loop's folder (`--workspace <root>/.torch_optimize/loop_<n>`) so the evidence lands beside the round folders.

---

## Step 1: Stats

```bash
TRACE="$ARGUMENTS"
python3 -m talos.torch_perf_analysis.python.stats "$TRACE"
```

The script:

1. Runs `veloq summary "$TRACE"` to materialize the parquet cache at `<trace>.veloq/parquetdir/`.
2. **Synthesizes backward attribution rows** into `NVTX_EVENTS.parquet`: for each backward-node event (`XxxBackward0`, seq carried in the NVTX int64 payload; legacy `XxxBackward, seq = N` text is also recognized), it looks up the forward event with the same `seq` (the `talos::` range's own payload or a nested `talos_seq:` mark) and the `talos::<func>#<path>@<Class>!!<file>:<line>` range that enclosed it, and stamps that label *verbatim* over the **enclosing `autograd::engine::evaluate_function` range** — the full backward of that one forward op, which also covers the gradient-accumulate ops (`aten::add` …) sitting outside `XxxBackward`. (Placed strictly inside the `evaluate_function`, so analysis doesn't mistake it for the `loss.backward()` driver.) So a backward kernel keeps the forward's module + source line; direction is read from the surrounding `autograd::` range, not the label. Idempotent.
3. Invalidates veloq's NVTX-derived caches (`meta.bin`, `nvtx-tree.parquet`, `nvtx-parent.parquet`) so subsequent queries pick up the synthesized rows. The CUPTI correlation cache (`correlation.bin`) stays — it doesn't depend on NVTX_EVENTS.
4. Runs veloq queries for the run-level + movement evidence (`summary`, `hardware`, `slices` for steps, `stats --group-by nvtx-path` for memcpy/sync, `gaps` for idle), and builds the per-stream GPU timeline **directly from the parquet** (`correlate.build_timeline` — kernel→NVTX launch correlation). Writes the per-step JSON artifacts above.
   (A rendered `perf_analysis_report.md` is not emitted for now — the agent reads the JSON directly.)

Progress lines go to stderr (`[stats] 1/4 ...` through `[stats] 4/4 ...`).

### Workspace artifacts produced

All JSON outputs **and** veloq's working caches (parquet + correlation index) land inside `.torch_perf_analysis/`. The original `.nsys-rep` is left untouched; stats.py symlinks it into the workspace so veloq's auto-generated `<trace>.veloq/` directory sits there too. (See Step 2 for what each file means and how to use it.)

```
.torch_perf_analysis/
├── <trace_basename>.nsys-rep       symlink to the user's trace
├── <trace_basename>.nsys-rep.veloq/
│   ├── parquetdir/                 nsys-exported per-table parquet (includes the
│   │                                augmented NVTX_EVENTS.parquet)
│   ├── meta.bin                    veloq's NVTX nesting / capabilities cache
│   ├── nvtx-tree.parquet
│   └── nvtx-parent.parquet
├── basics.json                     run-level basics, four keys:
│                                     trace_meta   — trace_span_ns, capabilities
│                                     hardware     — GPU / CPU / NIC inventory
│                                     steps        — step_* slices + uniformity (CV)
│                                     typical_step — which step was picked (median) + reason
├── step_<N>/
│   ├── stream_<id>/                one folder per CUDA stream (compute / comm / copy):
│   │   ├── timeline_bucket.json      this stream in 100 time segments (full detail)
│   │   ├── util.json                 just the 100 per-segment util values (quick scan)
│   │   └── source_attribution.json   whole-stream GPU time by source line (desc)
│   ├── cpu_thread_<i>/             CPU side — one folder per thread:
│   │   └── torch_function_util.json  torch-function util per segment + role (fwd/bwd)
│   ├── gpu_util.json               device-level util: union across ALL streams,
│   │                                 bucketed + per-stream means — the aggregate
│   │                                 the per-stream util.json files can't show
│   ├── memcpy.json                 by_path: nvtx-path + source-line attribution
│   ├── sync.json                   by_path: hidden .item()/.cpu() stalls by source
│   ├── idle.json                   idle gaps > 100µs within the step
│   ├── exposed_idle.json           hotspots ranked by step-time impact — each row
│   │                                 carries both levers (exposed_idle_ms + fusible_ms
│   │                                 + lever); by_region / by_source / by_sync_source
│   └── fusion_candidates.json      source lines ranked by fusible (memory-bound
│                                     elementwise/reduction) GPU time — tech4 targets
(perf_analysis_report.md is not emitted for now — analyze the JSON directly.)
```

### Capability gating

- `basics.json` → `trace_meta.capabilities.has_nvtx == false` → user didn't run `talos_profiler` + `--trace=nvtx,cuda`. Tell them to recapture.
- `trace_meta.capabilities.has_cuda_contexts == false` → kernel-to-NVTX attribution is unavailable; `by_source` lines fall back to `(unattributed)`. Ask the user to recapture with full CUDA tracing.

---

## Step 2: Analyze

`stats.py` produces a JSON workspace (no markdown report for now — read the JSON). What you get and what it means:

- **`basics.json`** — trace span + capabilities, hardware, per-step durations + uniformity (CV), and which step was picked as *typical* (warmup / outlier steps are skipped). Sanity-check here first: very few steps, high CV, or a missing capability bit weakens everything downstream.
- **`step_<N>/stream_<id>/`** — one folder per CUDA stream (compute / comm / copy):
  - **`util.json`** — the stream's per-segment GPU util + min/mean/max. The quick scan. `util` is keyed by bucket index so a low value names its bucket directly: `{"stream", "bucket_ns", "start_ns", "mean_util", "util": {"0": <util>, "1": <util>, …, "99": <util>}}` — the key matches the bucket `i` in `timeline_bucket.json`.
  - **`timeline_bucket.json`** — full per-segment detail (clipped so `util` ≤ 100% even when one kernel spans several segments):
    ```json
    {"i": 10, "start_ns": ..., "end_ns": ...,
     "kernel_count": 25, "h2d_count": 0, "d2h_count": 0, "d2d_count": 0, "memset_count": 3,
     "busy_ms": 11.26, "kernel_ms": 11.256, "memcpy_ms": 0.0, "idle_ms": 0.05, "util": 0.9956,
     "by_source": [{"source": "<file>:<line>", "gpu_ms": <ms>, "count": <n>}, ...]}
    ```
    `by_source` attributes the segment's GPU time to the **source line** (`file:line`) that launched it (forward and backward alike; `(unattributed)` = launched outside any `talos::` range, e.g. NCCL internals).
  - **`source_attribution.json`** — the same attribution rolled up over the **whole stream**, largest first: `[{"source": "<file>:<line>", "gpu_ms": <ms>, "count": <n>, "pct_of_step": <pct>}, ...]`.
- **`step_<N>/cpu_thread_<i>/torch_function_util.json`** — CPU side, one folder per **active** thread (mirrors `stream_<id>/`): that thread's **torch-function utilisation** per segment (`util` keyed by bucket, same shape as a stream's `util.json`), plus `role`, `cover_ms` (torch-function time) and `cuda_api_cover_ms` (CUDA-API time).

  This is a **C++ / dispatch-utilisation** metric: the fraction of the thread's wall time spent *inside* `talos::` ranges — i.e. inside a `torch.*` call, from the moment it crosses from Python into the C++ dispatcher. Read it two ways:
  1. **The gaps (`1 − util`) are Python overhead.** Time on the thread *outside* any `talos::` range is the Python-side glue between torch ops — the interpreter, building arg/kwarg lists, control flow, pure-Python helpers. Wide gaps ⇒ the thread is spending its time in Python, not getting work to the GPU.
  2. **The covered time (`util`) is itself host-side C++ cost, not GPU compute.** A `talos::` range spans dispatch (operator resolution / kernel selection) plus the autograd graph-building (recording the `grad_fn` / `next_edges`) plus the kernel launch — all CPU work that merely *issues* GPU work. So even at high util this is launch/build cost on the host; it tells you the thread is dispatch-bound (busy feeding the GPU), which is only good insofar as the GPU stays busy (see Principle).

  So **high util ⇒ ops issued back-to-back, little Python glue** (host cost is the unavoidable C++ dispatch/build); **low util ⇒ the gaps dominate** — **Python** between ops on a `forward` thread, the **C++ autograd engine** between ops on the `backward` worker (its ranges are the synthesized backward ranges). A thread with **`role: other`** runs no tracked torch ops (so `util` ≈ 0) but is busy launching raw CUDA — comm / copy / CUDA-graph (see `cuda_api_cover_ms`); it's listed because that activity is significant. Near-idle helper threads are filtered out. Note: a forward thread reads ~0 through the backward window because it's *blocked* there (not overhead) — cross-ref the backward thread, which is active then.
- **`step_<N>/gpu_util.json`** — **device-level** utilisation: the union of busy time (kernel/memcpy/memset) across **all** streams, in the same 100-bucket format as a stream's `util.json`, plus `per_stream_mean_util`. Read this **before** the per-stream files: streams can each look moderate while the device is nearly saturated (busy windows interleave) — compare `sum(per_stream_mean_util)` against `gpu.util`: a large gap means heavy stream overlap (don't double-count "idle" that another stream fills); ≈ equal means no overlap, so a low-util bucket here is genuinely dead time. Totals match `exposed_idle.json`'s `gpu` block.
- **`step_<N>/memcpy.json` / `sync.json` / `idle.json`** — data movement, hidden host-sync stalls (`.item()`/`.cpu()`), and idle gaps, each with `by_path` carrying a **`source`** (`file:line`) + `total_ms` / `pct_of_step`. These are **self-time** views (how long the op itself ran / waited).
- **`step_<N>/exposed_idle.json`** — the **step-time-impact** view: lines / regions ranked by how much they slow the step. Its core is *the GPU idle each is responsible for*. Device idle (window − union of kernel/memcpy/memset) is charged to the launch that ends each gap, and the host-sync-exposed slice is split out. Three cuts: `by_region` (NVTX-marker rollup, so one logical unit isn't split across lines), `by_source` (`file:line` + `op` + `launch_count`, for the Optimizer to act on), and `by_sync_source` (the slice actually caused by `.item()`/`.cpu()` waits). `idle_split` gives the headline `host_sync_ms` vs `launch_bound_ms`. Each `by_region` / `by_source` row **also carries the busy lever** — `busy_ms` and `fusible_ms` (memory-bound time reducible by fusion) — plus a per-row `lever` hint, so the lever is chosen **per hotspot** (a model mixes idle-bound regions like the optimizer / masking and busy-bound ones like embedding / layernorm). A big self-time `.item()` whose host wait overlaps GPU-busy work contributes ~0 exposed idle — removing it moves nothing. Caveats: the "next launch" attribution folds the Python glue *before* a launch into it (a reasonable "what is the GPU waiting for", not exact causation); `fusible_ms` counts only fusible kinds, so a compute-bound region (large `busy_ms`, small `fusible_ms`) correctly stays on the idle lever.
- **`step_<N>/fusion_candidates.json`** — source-line regions ranked by **fusible GPU time** (the kernel-fusion / tech4 angle). Each kernel is classified by its demangled name; only memory-bound families count toward `fusible_gpu_ms`: **`elementwise`** (pointwise ops — `nan_to_num`, `clamp`, `where`, activations…, fuses directly) and **`reduction`** (`sum` / `mean` / `norm` / `softmax` — fuses *with its surrounding pointwise*, not in isolation). Compute-bound (`gemm` / `conv` / `attention`) and non-fusible (`comm` / `sort`) kernels are shown in `by_kind` but excluded. CUPTI has no per-kernel byte count, so `gpu_ms` is the memory-traffic proxy: a region ranks high whether it is **launch-bound** (many tiny kernels, low util) or **bandwidth-bound** (few large kernels at high util — invisible to the util timeline). Each row: `{source, fusible_gpu_ms, fusible_kernel_count, total_gpu_ms, total_kernel_count, pct_of_step, by_kind}`. This is the **busy-lever detail**: `exposed_idle.json` already flags a hotspot as `busy` via its `fusible_ms`; come here for the per-kind breakdown of what is actually fusible.

### Principle

The analysis is **GPU-centric**. The goal is keeping the GPU busy, so findings are anchored in GPU under-utilization, stalls, or work on the critical path. The per-stream `util` timeline is the source of truth.

CPU-side costs are relevant when they block GPU progress. Cross-reference the same time window on the compute stream's `util.json` / `timeline_bucket.json`: low utilization identifies a real stall, while sustained utilization indicates overlapped work. Each finding should tie back to the GPU idle or critical-path work it explains.

For launch-bound regions, report launch evidence: kernel count, CUDA runtime launch API count/time (`veloq stats --type runtime --nvtx ...`), and the enclosing CPU NVTX range or local CPU timer (`veloq slices ...`). Use GPU util, idle gaps, and step duration to judge whether that launch cost is on the critical path.

### Analysis directions

There's no fixed sequence here — these are complementary angles, and you decide how to follow them. The general idea: first read the coarse, whole-step stats to build a mental picture of the timeline; then zoom into whatever looks off (a low-util stretch, a hot source line) and drill the root cause with `veloq`. The rest is yours to drive — follow the evidence.

**Build the picture first.** Pick the **compute stream** (highest busy_ms / kernel_count; comm is D2D/NCCL-heavy, copy is H2D/D2H-heavy). Then look from either angle:
- **GPU utilization** — start with `gpu_util.json` (device level, union across all streams): its `gpu.util` is the single headline number, and a low-util bucket there is *genuinely* dead time (no stream is filling it). Then drill the compute stream's `util.json` for the same bucket window; the comm / copy streams' `util.json` at that index hints at why (comm busy while compute idles → exposed communication; nothing busy → a CPU/launch or host-sync gap).
- **Rank hotspots on `exposed_idle.json`** (detailed under the artifacts above) — rank each by the larger of its `exposed_idle_ms` and `fusible_ms`, then follow its `lever`: `idle` → read `idle_split` (`host_sync` vs `launch_bound`); `busy` → drill `fusion_candidates.json` for the fusible `by_kind` breakdown. `gpu.idle_pct` is overall orientation only.
- **CPU / dispatch utilisation** — when the compute stream has a low-util stretch and nothing on comm/copy explains it, read `cpu_thread_<i>/torch_function_util.json` over that bucket window. Low torch-function util there means the gap is **Python overhead** (the thread is between torch ops, not feeding the GPU); high util means the host is **dispatch/launch-bound** (busy in C++ issuing work). Either way the cost is host-side, so it only matters when it lines up with GPU idle — cross-ref the compute stream's `util.json` at the same bucket.

**Then drill into the suspicious / hot regions.** For a bucket `i` worth a closer look, open `timeline_bucket.json` at `i` (its `by_source`, counts, `kernel_ms` vs `memcpy_ms`) and query that `start_ns`/`end_ns` window with `veloq` to pin the root cause. Things the evidence often points at: a hidden host-sync stall (a big `cudaStreamSynchronize` from a `.item()` / `.cpu()` / host-side print line), a pageable `HtoD` copy, a line of many tiny kernels, exposed (non-overlapped) communication, or a gap between steps. `memcpy.json` / `sync.json` `source` is null when the event sits under a non-`talos::` region — read its raw `nvtx_path` or `veloq correlate <key>`. (veloq mechanics: companion `nsys-profile-analysis` skill.)

Report each finding with the evidence that matches its bottleneck: GPU ms / % for compute or memory-traffic findings; launch count, runtime API time, and CPU-range time for launch-bound findings.

---

## Drilling deeper with veloq

The stats phase extracts a fixed evidence set, sufficient for most diagnoses. For sharper queries, run `veloq` directly — after stats.py's parquet synthesis, NVTX_EVENTS carries `talos::` ranges on both forward and backward ops (backward gets the forward label verbatim), so an nvtx-path glob matches a module in both directions:

- **Per-step bucket analysis** (100 bins, with per-kind breakdown):
  ```bash
  START=...; END=...; BUCKET=$(((END - START) / 100))
  veloq timeline "$TRACE" --from @$START --to @$END --bucket "${BUCKET}"ns \
    | python3 -c 'import json,sys
for r in json.load(sys.stdin)["data"]["rows"]:
    print(r["key"], r["busy_ns"] / r["total_ns"], r["breakdown"])'
  ```
- **Cross-step regression diff**: key both timeline outputs by `.data.rows[].key` and compare the buckets that moved.
- **Drilling into one class's kernels**: `veloq stats T --type kernel --nvtx 'talos::*@Linear*' --group-by demangled --limit 10` (matches forward + backward; to isolate backward, intersect with the `autograd::engine::evaluate_function` range).
- **CPU launch site for one event**: `veloq inspect "$TRACE" <row_id>` / `veloq correlate "$TRACE" <row_id>`.

For full veloq mechanics, defer to the companion `nsys-profile-analysis` skill.

---

## Error handling

- `veloq: command not found` → `pip install --force-reinstall talos` (the binary ships bundled with the wheel).
- `nsys: command not found` on the first export call → install `nsys ≥ 2024.6` or supply a pre-exported `_pqtdir/` directly.
- `pyarrow required for parquet synthesis` → `pip install pyarrow` (usually already present in PyTorch envs).
- `basics.json` → `trace_meta.capabilities.has_nvtx == false` → user didn't run `talos_profiler` + `--trace=nvtx,cuda`.
- `trace_meta.capabilities.has_cuda_contexts == false` → kernel-to-NVTX attribution unavailable; timeline `by_source` falls back to `(unattributed)`. Ask user to recapture with full CUDA tracing.
- `basics.json` → `steps.count == 0` / `typical_step == null` → no `step_*` ranges; either the user didn't set `TORCH_PERF_ANALYSIS_ENABLE=1` (most common — check stderr for "disabled; set TORCH_PERF_ANALYSIS_ENABLE=1"), or `talos_profiler.start() / .stop()` didn't bracket the captured steps, or the steps weren't marked with `talos_profiler.step()`.
- timeline `by_source` is all forward lines (no backward) → bwd synthesis didn't find any seq matches; check that seq-carrying events exist in the trace (`talos::` ranges / `talos_seq:` marks with an int64 payload — emitted automatically between `talos_profiler.start()` and `.stop()`).
- `jq: command not found` → only the `nsys-profile-analysis` / `ncu-profile-analysis` example commands use it. Install it (`apt install jq` / `brew install jq`) or read the JSON with `python3 -c 'import json,sys; ...'`.
- Deep veloq-specific issues → consult the `nsys-profile-analysis` skill.
