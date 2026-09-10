# Workspace Layout

Shared by all three roles — the Main Loop Agent, the Optimizer, and the Judge all read this to know where things live and who is responsible for writing what. This describes `.torch_optimize/`, the run-local workspace inside the code root being optimized — not the skill's own files.

```
.torch_optimize/
└── loop_<n>/                          one per "profile → work until PASS" cycle; n = 1, 2, 3, ...
    ├── loop_baseline.patch            git checkpoint of the code at loop start — written by: Main Loop Agent
    ├── loop.patch                     the loop's single cumulative optimization patch (every kept optimization so far + this loop's work, each behind its flag); refined across this loop's rounds; the last kept loop's `loop.patch` is the run's one merge artifact — written by: Optimizer
    ├── loop_baseline.nsys-rep         nsys trace — written by: Main Loop Agent
    ├── <torch-perf-analysis outputs>  basics.json, stream_<id>/, cpu_thread_<i>/, ... — written by: Main Loop Agent
    ├── loop_baseline_analysis.md      performance analysis + hotspot call — written by: Main Loop Agent
    ├── round_1/
    │   ├── <nsys-rep + torch-perf-analysis outputs>   written by: Optimizer (validating a change) or Judge (re-measuring)
    │   ├── verification.json          real-input parity + end-to-end perf measurements — written by: Optimizer (verification step)
    │   ├── worklog.md                 written by: Optimizer
    │   ├── report.md                  written by: Optimizer
    │   └── verdict.md                 written by: Judge
    ├── round_2/                       only if round_1's verdict was not PASS
    │   └── ... (same files)
    └── round_<k>/                     continues until a verdict is PASS
```

## Naming

- `loop_<n>`: sequential integer starting at 1, incrementing by 1 for each new loop (`loop_1`, `loop_2`, `loop_3`, ...). One per baseline-profile-to-`PASS` cycle. The sequence *is* the run's history — there is no separate ledger file; the Main Loop Agent reviews prior `loop_<*>/round_<*>/verdict.md` at the start of each loop.
- `round_<k>`: `k` starts at 1 within a loop and increments each time that loop's verdict is `ITERATE`; it resets to 1 in each new loop.