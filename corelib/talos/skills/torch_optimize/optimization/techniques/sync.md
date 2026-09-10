# Technique 1 — Eliminate or relocate host-device synchronization

Shared steps — semantic contract, code layout, artifacts, verification, and reporting — are in [`_common.md`](../_common.md). This file covers only what is specific to this technique.

## Purpose

Remove or relocate D2H/H2D synchronization in a module so the step stops stalling on host-device round-trips.

## Scope

Two sync directions:

- **D2H** (device → host): `.item()`, `.cpu()`, `.tolist()`, `bool(tensor)` / `int(tensor)`, printing a tensor, and data-dependent-shape ops (`nonzero`, `unique`, `masked_select`, boolean-mask indexing `x[mask]`).
- **H2D** (host → device): moving a Python/CPU value onto the device inside the step — `torch.tensor(...)` then `.cuda()` / `.to(device)`, or a constant tensor rebuilt on the host each step.

## Procedure

For the dispatched target sync, apply the first option that holds:

1. **Remove** — rewrite it as a shape-static, branchless equivalent that stays on device, keep host scalars as Python scalars so they fold into kernels instead of becoming device tensors, and compute or transfer any repeated quantity once.

2. **Relocate** — when the transfer is genuinely required (a host value drives Python control flow, or a host input must reach the device), move the producing computation so the stall overlaps other work: onto the data-prefetch / side stream, or early enough that independent work follows it. A relocated computation reads only inputs available at the new site and produces its result before its consumer runs.

## Local check (iterate signal, non-authoritative)

Re-profile with this optimization's flag on (via the `torch-perf-analysis` skill). **A sync only matters to step time when it exposes GPU idle** — so the signal that counts is the idle it caused dropping, not merely the sync disappearing. You are on track when:

- **main stream/thread target:** the **GPU idle the sync caused drops** — the affected low-util window recovers: its buckets' util rise in the compute stream's `util.json` and in the device-level `gpu_util.json`. The sync going away in `sync.json` (and its `memcpy.json` row dropping) confirms the *mechanism* but is not the win by itself: a sync whose host wait overlapped GPU-busy work exposes ~0 idle, so removing it moves nothing.
- **side stream/thread target:** the recorded blocking evidence improves — same-window main-stream idle shrinks, same-window main-stream `util.json` recovers, or the explicit consumer wait drops.

Accuracy: results are bit-identical — assert equality (this becomes the persistent local checker; see [`_common.md`](../_common.md) → Artifacts).

This is a scratch signal; accepted round evidence comes from [`_common.md`](../_common.md) → Verification.
