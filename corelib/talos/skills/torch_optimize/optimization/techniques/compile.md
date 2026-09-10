# Technique 5 — Compile stable PyTorch regions with `torch.compile` / Inductor

Shared steps — semantic contract, code layout, artifacts, verification, and reporting — are in [`_common.md`](../_common.md). This file covers only what is specific to this technique.

## Purpose

Use `torch.compile` / Inductor to cut Python-frame, ATen-dispatch, and
kernel-launch overhead in a stable PyTorch tensor region. Inductor can also
fuse pointwise/reduction subgraphs, so it is the first try for a fusible
memory-bound region before hand-written kernel fusion.

This is the **lightweight path to try before Technique 3**: if the hot region is compile-friendly, try T5 first; if compile is unsafe, unstable, too fragmented, or ineffective, fall back to T3's C++ host ATen lowering. (T5 ≠ T3: T3 moves region orchestration into a C++ host ATen path; T5 keeps the source in PyTorch.)

## Scope

Applies when a hot source line/module is a stable PyTorch tensor subgraph and
profiling shows launch/dispatch-bound behavior (`lever: idle`), or a
memory-bound fusible subgraph (`lever: busy`).

**Good candidates:** dense tensor regions with many small PyTorch ops, and
inputs whose dtype/device/layout/shape variants are finite or normalizable.

**Poor candidates:** regions with I/O or CPU materialization, data-dependent
Python control flow, unbounded shapes that cannot be normalized, or persistent
recompiles/graph breaks that leave too little compiled work.

## Compile-specific constraints (beyond [`_common.md`](../_common.md)'s equivalence)

The generic semantic-equivalence contract lives in [`_common.md`](../_common.md). Compile adds:

- Compile / autotune time must **not** enter measured latency — warm up first.
- No persistent recompile or material graph break after representative warmup.
- Training gradients and all configured distributed behavior must be unchanged.

## Procedure

1. Resolve the hot boundary from timeline / nsys / torch-perf-analysis evidence: module, source line, NVTX path, launch count, CPU range time, GPU idle/busy window, representative input shape / dtype / device / layout.
2. Exclude I/O, CPU materialization, and unstable Python control flow from the
   compile region.
3. Add a guarded lazy compile path behind this optimization's flag, eager as the default-safe fallback (see [`_common.md`](../_common.md) → Code layout). Typical shape:

```python
def forward(self, *args):
    if enable_compile:
        if self._compiled is None:
            self._compiled = torch.compile(self._forward_impl, mode=..., fullgraph=..., dynamic=...)
        return self._compiled(*args)
    return self._forward_impl(*args)
```

Record `mode`, `dynamic`, `fullgraph`, cudagraph setting, and inference-only vs
training-supported. Prefer a small stable inner region over a whole module.

4. Warm up with representative real inputs before any measurement.
5. Run with recompile/graph-break logs on; resolve churn before benchmarking.
6. For a finite variant set, warm up every variant. Otherwise normalize the
   input contract (`dynamic=True`, padding, bucketing, tensorizing the branch,
   or splitting the region); revert if churn remains.

## Local check (iterate signal, non-authoritative)

Run `TORCH_LOGS=recompiles,graph_breaks <run-workload>` (optionally `TORCHDYNAMO_VERBOSE=1`) after representative warmup. You are on track when:

- the target's launch / API / dispatch-bound evidence improves — the outcome that matters is its `exposed_idle.json` share dropping (`launch_bound` idle for a dispatch-bound region, or `fusible_ms` for a fused memory-bound region);
- there is no persistent recompile after warmup, graph breaks are fixed and
  understood, and the remaining compiled islands are large enough to matter;
- the timing you look at excludes compile / autotune time (warm up → confirm recompile logs clean → only then time).

Accuracy: compare eager vs compiled outputs — forward and, for training,
backward/gradients — across representative real inputs. This becomes the
persistent local checker ([`_common.md`](../_common.md) → Artifacts).

This is a scratch signal; accepted round evidence comes from [`_common.md`](../_common.md) → Verification.

