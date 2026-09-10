# Technique 4 — Fuse elementwise/reduction subgraphs (ATen rewrite → Triton → CUDA)

Shared steps — semantic contract, code layout, artifacts, verification, and reporting — are in [`_common.md`](../_common.md). This file covers only what is specific to this technique.

## Purpose

Cut kernel count and memory traffic by fusing a region's elementwise and reduction subgraphs, keeping the remainder unchanged. Three rungs, cheapest first — take the lowest one that achieves the fusion:

- **(A) equivalent ATen/PyTorch rewrite** — collapse the subgraph into fewer stock kernels with fewer intermediates.
- **(B) hand-written Triton kernel** — when no stock rewrite collapses it (or A still leaves HBM round-trips), fuse the subgraph into one Triton kernel bound as a custom op. This is the default custom-kernel path: simpler to write / autotune / maintain than raw CUDA.
- **(C) raw CUDA / C++ custom op** — only when Triton can't express the pattern or the last bit of performance needs hand control.

Applies to a region in any form — original Python/ATen or an already-lowered C++ path.

## Scope

Memory-bound elementwise/reduction subgraphs within the region (the `fusion_candidates.json` target — its `by_kind` shows the `elementwise` / `reduction` mix), whether the region is launch-bound (many tiny kernels at low util) or bandwidth-bound (few large kernels at high util). Three shapes to fuse:

- **Cross-boundary** — elementwise ops separated only by control flow.
- **Horizontal** — the same elementwise op applied to several independent inputs.
- **Reduction** — a reduction (`sum` / `mean` / `norm` / `softmax`) fused with the elementwise feeding or following it (e.g. `(x*x).sum()`), so the intermediate never round-trips HBM. Fuse the reduction *with its surrounding pointwise*; do not hand-rewrite the cross-thread reduction in isolation.

## Procedure

1. From the region's compute graph, list the elementwise and reduction fusion candidates.
2. For each candidate, take the lowest rung that achieves the fusion:
   - **Rung A — ATen/PyTorch rewrite (try first).** Replace the subgraph with an equivalent built from stock PyTorch ops that launches fewer kernels and materializes fewer intermediates — e.g. a prefix-sum (`cumsum`) for a sliding pool, `einsum` / broadcasting for a multiply-then-reduce, a single functional call for a chain of pointwise ops. No `.cu` / `.cpp` / Triton / `cpp_extension`; the win is fewer stock CUDA kernels and less HBM traffic. (Distinct from Technique 3, which keeps the op sequence and only moves issuance to C++ — Rung A changes the op graph itself.)
   - **Rung B — hand-written Triton kernel.** When Rung A cannot collapse the subgraph (no stock equivalent), or to remove the round-trips Rung A still leaves, write one Triton kernel that fuses the candidate into a single pass and bind it as a custom op, leaving the rest of the region as its existing ops. Triton autotunes block sizes and is the default custom-kernel path (simpler to write and maintain than CUDA). For a reduction candidate, the fused kernel absorbs the adjacent pointwise (producer or consumer) into the reduction.
   - **Rung C — raw CUDA / C++ custom op.** Only when Triton can't express the pattern (exotic memory access, warp-level primitives, layouts Triton handles poorly) or the remaining gap justifies hand-tuned CUDA. Same binding as Rung B; the rest of the region stays as existing ops.
3. For a training region, wrap each fused island (any rung) in an autograd Function when autograd cannot already differentiate the rewrite: save the tensors and scalar/config its backward needs, and reduce any broadcast input's gradient back to the input shape.

## Local check (iterate signal, non-authoritative)

Time the target locally with the flag on (forward + backward), and profile it to read kernel count and memory traffic. You are on track when the fused path (any rung) replaced the subgraph — kernel count / memory traffic dropped — and the weighted CPU-timer is no worse than the baseline.

Accuracy: matches the baseline path within tolerance across every captured config and dtype, forward and backward (this becomes the persistent local checker; see [`_common.md`](../_common.md) → Artifacts).

This is a scratch signal; accepted round evidence comes from [`_common.md`](../_common.md) → Verification.
