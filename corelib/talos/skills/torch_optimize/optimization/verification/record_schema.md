# Verification records

```json
{
  "real": {
    "skipped": "<reason; empty when verified>",
    "forward":    { "max_abs": 0.0, "max_rel": 0.0 },
    "vjp":        { "max_abs": 0.0, "max_rel": 0.0 },
    "update":     { "max_abs": 0.0, "max_rel": 0.0 }
  },
  "e2e": {
    "baseline": {
      "flags": {
        "prior_loops": { "<PRIOR_FLAG>": "optimize" },
        "prior_rounds": { "<KEPT_FLAG>": "optimize" },
        "current_round": { "<CURRENT_FLAG>": "origin" }
      },
      "avg_step_ms": 0.0,
      "step_spread_ms": 0.0
    },
    "candidate": {
      "flags": {
        "prior_loops": { "<PRIOR_FLAG>": "optimize" },
        "prior_rounds": { "<KEPT_FLAG>": "optimize" },
        "current_round": { "<CURRENT_FLAG>": "optimize" }
      },
      "avg_step_ms": 0.0,
      "step_spread_ms": 0.0
    },
    "warmup": 20,
    "steps": 30,
    "pct": 0.0,
    "min_gain_pct": 1.0,
    "meets_min_gain": false,
    "gain_ms": 0.0,
    "noise_ms": 0.0,
    "gain_exceeds_noise": true,
    "metric": "avg_step_ms",
    "valid": true
  }
}
```

`verification.json` stores parity and one profiler-off baseline/candidate
comparison. For backward-affecting patches, `real.vjp` stores gradient parity.
For patches that alter optimizer updates, `real.update` stores post-step
parameter/state parity; otherwise omit it.
`e2e.valid` requires matching conditions; `e2e.meets_min_gain` applies the
gain threshold. It contains measurements, not a verdict.

Each condition is measured **once**; `step_spread_ms` is that run's own
step-to-step noise (p90 − p10 over the `steps` measured steps), so the
single run still carries a noise estimate. `noise_ms` is the larger of the
two `step_spread_ms`; `gain_ms` is `baseline.avg_step_ms −
candidate.avg_step_ms`; `gain_exceeds_noise` is `gain_ms > noise_ms`. These
three are **advisory context, not a gate** — `meets_min_gain` remains the
threshold. A `meets_min_gain: true` with `gain_exceeds_noise: false` means
the run was too noisy to trust at that margin; say so rather than dropping
the number.
