# Technique dispatch

Choose from the evidence in `loop_baseline_analysis.md`:

| Evidence | Technique |
|---|---|
| `lever: idle`, `idle_split.host_sync_ms` high | [T1 — sync](techniques/sync.md) |
| `lever: idle`, `launch_bound`, independent batchable siblings | [T2 — batch](techniques/batch.md) |
| `lever: idle`, `launch_bound`, stable pure-PyTorch region | [T5 — compile](techniques/compile.md), then [T3 — C++ host](techniques/cpp_host.md) if ineffective |
| `lever: busy`, high `fusible_ms` | [T5 — compile](techniques/compile.md), then [T4 — fusion](techniques/kernel_fusion.md) if needed |

Read only the selected Technique. Its `Scope` is the final fit check. If none
fits, propose an off-catalog change and say why the catalog does not cover it.
