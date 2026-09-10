# Technique 2 — Batch structurally identical parallel submodules

Shared steps — semantic contract, code layout, artifacts, verification, and reporting — are in [`_common.md`](../_common.md). This file covers only what is specific to this technique.

## Purpose

Replace many small per-submodule ops with a few grouped ops so the module stops being kernel-launch bound.

## Scope

Applies when a hot source line or module is caused by independently runnable sibling work under the same high-level parent. Many launches at one source line are not enough; resolve the owner boundary before batching:

- Roll the finding up to the nearest high-level parent that owns the repeated sibling work — via `nvtx_path` / the Torch model source structure, aggregating with the window's `timeline_bucket.json` rows (per-line kernel counts inside the low-util buckets) and `source_attribution.json` for the GPU time and launch `count` sitting under that parent. (Not `fusion_candidates.json` — that is the fusion / busy side, Technique 4.)
- Batch only within one high-level parent whose siblings are on the same topology level and have no data dependency on each other.
- Partition siblings by structural signature: op sequence, parameter layout, input contract, dtype, flags, training/eval behavior, and output contract.
- A group of size 1 stays as-is.
- **Exact** — identical per-element shapes, differing only along the batch dimension: stack and run one batched op.
- **Isomorphic** — shapes differ only along declared batchable dimensions: reconcile shapes, then batch.

## Procedure

1. Start from the dispatched hot source/module and build a parent rollup using `nvtx_path` or the Torch model source structure: kernel launch `count` and the low-util window it sits in (the window's buckets in `timeline_bucket.json` / `util.json`), total `gpu_ms` (`source_attribution.json`), and child modules/ranges under each high-level parent that contributes to the finding.
2. Select a parent whose cost is concentrated in repeated sibling work. Record the child count, structural signatures, input shapes, output shapes, and whether the siblings share the same input tensors or receive shape-compatible inputs.
3. Group members by structural signature; within each signature, split into shape-exact buckets first.
4. For each batchable group above the break-even point (grouped fixed overhead < the per-module launches it replaces), batch it with the cheapest valid reconcile rung, then split the output back per member:
   - **shape-bucket** (first, pure win) — members already same-shape: stack along a leading group dim and run one batched op — `matmul` / `bmm` for linears, `F.layer_norm` and elementwise for the rest. No wasted compute.
   - **pad + mask** — pad the varying dim to the group max, run the batched op, then mask/slice the padding out. **Padding-waste guard:** keep this rung only when the padded extra FLOPs stay below the per-module launch savings.
   - **grouped / segment GEMM** — when stock ATen offers one (a single kernel over differently-shaped GEMMs, no padding waste).
5. Inside the grouped forward, read each member's live parameters with `torch.stack` (or per-member slices), so members remain the parameter owners and gradients, `state_dict`, and optimizer state are unchanged.

A reconcile that would need a custom kernel (no stock grouped-GEMM) belongs to Technique 3/4, not here — Technique 2 only recomposes with existing `bmm` / grouped ops / pad+mask.

## Local check (iterate signal, non-authoritative)

Re-profile with this optimization's flag on (via the `torch-perf-analysis` skill). You are on track when the dispatched target's launch-bound evidence improves: kernel launch count drops, CUDA launch/runtime API count or time drops, and the target's CPU NVTX range or local CPU timer drops. The outcome that matters is the target's low-util window filling in — its buckets' util rising in the compute stream's `util.json` and the device-level `gpu_util.json` — the launch-count drop is the mechanism, the window recovering is the win.

Accuracy: compare grouped vs per-module output in float64 to a tight tolerance, and confirm gradients reach every member parameter. For the pad + mask rung, verify the padding positions contribute nothing to any member's output or grad (this becomes the persistent local checker; see [`_common.md`](../_common.md) → Artifacts).

This is a scratch signal; accepted round evidence comes from [`_common.md`](../_common.md) → Verification.
