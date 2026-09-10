# Verification

Inputs: current flag, `RUN_CMD`, and `WARMUP=20`, `STEPS=30`,
`MIN_GAIN_PCT=1.0` unless overridden.

1. Set the current flag to `compare` and run `RUN_CMD`. Compare runs origin and
   optimize on equal input, weights, buffers, and RNG; it returns origin values
   to training. Record forward and VJP/gradient parity in `verification.json`.
   Restore the flag even if the run fails.
2. If the change batches, reorders, caches, or otherwise alters optimizer
   updates, run a warmed full-step origin/optimize check over relevant alias
   cases. Compare post-step parameters and optimizer state; record
   `real.update`.
3. Run profiler-off E2E once for each condition. Baseline is the accepted
   pre-Round flag map with the current flag `origin`; candidate changes only
   that flag to `optimize`. Each run starts from the same deterministic
   initialization and data position. Use identical hardware, process shape, and
   runtime settings; discard warmup and average `STEPS` steps. Set `valid=true`
   only when these conditions, the step window, and flag maps match; otherwise
   set it to `false`.
4. **Keep the per-step times of both runs — one run still tells you its own
   noise.** For each condition record `step_spread_ms` = p90 − p10 over its
   `STEPS` measured steps. Set `noise_ms` to the larger of the two.
5. With `b = baseline.avg_step_ms` and `c = candidate.avg_step_ms`, compute
   `pct = (b - c) / b * 100` and `gain_ms = b - c`. Set `meets_min_gain=true` when
   `pct >= MIN_GAIN_PCT`, and `gain_exceeds_noise = gain_ms > noise_ms`.
   Write it all under `e2e` in `verification.json`.
6. **Sanity-check the noise before you trust the margin.** If
   `gain_exceeds_noise` is false, or `noise_ms` is itself a large fraction of
   `baseline.avg_step_ms` (a jittery workload — background load, thermal drift, a
   shared GPU, dataloader stalls), the single run is not a sound basis for the
   `pct` you measured. Do not silently keep the number: report it with the
   noise stated, and if the run is cheap, re-run both conditions once on a
   quiet machine before writing the record. Never widen `MIN_GAIN_PCT` to make
   a noisy gain pass.

`verification.json` contains measurements only. Its minimal format is in
[`record_schema.md`](record_schema.md). Never use profiled Nsys step time as
E2E evidence.
