# Technique 3 — Lower a region to a C++ host ATen path (cpp_only)

Shared steps — semantic contract, code layout, artifacts, verification, and reporting — are in [`_common.md`](../_common.md). This file covers only what is specific to this technique.

## Purpose

Collapse a marked region's per-op Python frames and ATen dispatch into one C++ call that issues the same native ATen ops, removing host-side kernel-issuance overhead. This is the guaranteed baseline that every later path falls back to.

## Scope

A region that maps cleanly to native ATen: `nn.Module` / functional calls with Python control flow that translates to C++ host code. A composite module with its own multi-step forward, a non-torch third-party call, or control flow that does not map to C++ stays on the original path.

## Procedure

1. Capture the region's real inputs, dtypes, scalar args, and config variants from the workload.
2. Unpack the captured args dict in the C++ entry, names matching the source 1:1.
3. Map each source-level op to one native ATen call — do not lower below ATen granularity. Keep region orchestration and any config / shape / dtype branches in C++ host code; cover dynamic shapes via ATen and dtype via `AT_DISPATCH_*`.
4. Return the same output structure the original region produces.

## Local check (iterate signal, non-authoritative)

Time the region locally with the flag on (in-process CPU-timer + CUDA-event), comparing the C++ path against the original region. The C++ path issues the same native ATen ops, so the win is the host-side issuance time (the CPU-timer); the CUDA-event guards that GPU work is unchanged. You are on track when the weighted CPU-timer is ≥ 0.95× the original region — below that the path is likely miscompiled (typically from lowering an op below native ATen granularity).

Accuracy: matches the original region within tolerance across every captured config and dtype (this becomes the persistent local checker; see [`_common.md`](../_common.md) → Artifacts).

This is a scratch signal; accepted round evidence comes from [`_common.md`](../_common.md) → Verification.
