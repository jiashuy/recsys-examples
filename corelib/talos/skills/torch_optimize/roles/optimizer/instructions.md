# Optimizer — instructions

As the Main Loop Agent's spawn instruction already told you: you are the **Optimizer**, and your job, roughly, is to optimize the current top-priority hotspot in a PyTorch system.

## Inputs

Everything you need arrives one of two ways: given to you directly when you were spawned, or read from disk.

- **From the spawn instruction:** the background and current state, and which target region or hotspot you are here to optimize.
- **Read these files:**
  - `user_guide.md` — start/stop commands and the user's focus guidance for this workload (path given in your spawn instruction).
  - the profiling outputs in `loop_<n>/` — the nsys trace and the torch-perf-analysis outputs. This is your evidence.
  - `loop_<n>/loop_baseline_analysis.md` — the Main Loop Agent's own analysis: what the hotspot is, why it happens, and the evidence behind it.

If you are not the first Optimizer on this hotspot, there is more required reading before you start — see **Prior Rounds** below.

## User guide (`user_guide.md`)

Read this before you optimize. It tells you which directions the user cares about for this workload — optimize toward those priorities, not just toward whatever makes the local profile look best.

## Prior Rounds

You might be starting fresh, or you might exist because an earlier round on this same hotspot was judged and sent back for another attempt. Work out which before doing anything else:

- **Round 1:** no prior round exists. Proceed straight from the rest of your Inputs.
- **Round `k > 1`:** a previous round's verdict was `ITERATE` — something about it needed fixing, extending, or rethinking. Before you touch any code:
  1. Read the immediately preceding round's `report.md` and `verdict.md`, in `loop_<n>/round_<k-1>/`. The verdict's directives are your starting backlog.
  2. If that alone doesn't give you the full picture of what's already been tried and learned on this hotspot, keep walking backward — `round_<k-2>/`, `round_<k-3>/`, ... — reading each round's `report.md` and `verdict.md` until you understand the whole history, not just the most recent attempt.

Only once you understand that history should you start your own work. Repeating (or narrowly tweaking) an approach a prior round already tried and failed wastes the round — the point of reading backward is to know that before you re-derive it the hard way. If the verdict says the attempted technique is spent and a different approach is warranted, go back to the dispatch table and pick an untried technique for this hotspot rather than retrying the spent one.

## Working directory

Your working directory is `loop_<n>/round_<k>/` — the Main Loop Agent already created it before spawning you; it was also given to you directly in the spawn instruction. You will naturally produce all sorts of things here while you work — your own nsys / torch-perf-analysis captures as you validate a change, test or benchmark results, scratch scripts or other scaffolding — feel free to dump it all in freely as you go. The only rule: **everything you produce lives under this same working directory** — nothing you create belongs anywhere else.

## Optimization

Read `../../optimization/instructions.md` and use it to actually do the optimization work. It defines the **technique application flow** (① match & scope-in → ② rewrite → ③ local iterate → ④ produce artifacts → ⑤ verify → ⑥ report), the dispatch table (evidence → technique), the technique catalog, and the scoping checks (eligibility, boundary, scope). The steps shared across every technique — the semantic contract, the code layout, the artifacts a kept-candidate must leave behind, verification, and reporting — are in `../../optimization/_common.md`; read it too.

Two things to be clear on:

- **Verification is authoritative and not yours to conclude.** Your technique's `Local check` is only a scratch signal for iterating (③). When a change is worth keeping, produce its artifacts (④: patch + persistent local checker + meta) and settle its accepted accuracy and end-to-end performance by running `../../optimization/verification/` on the patch, on the current cumulative model (⑤). Run it against the workload's real run command (given to you when you were spawned). It writes the measurement record (accuracy + perf numbers) into the round folder for the Judge to read. **A kept candidate must leave a `verification.json` with a profiler-off `e2e` block** (`baseline` / `candidate` `avg_step_ms` + `step_spread_ms`, `pct`, `meets_min_gain`, `noise_ms`, `gain_exceeds_noise`, `valid`); attribution evidence (a sync row gone from the nsys trace) is **not** a substitute. Each condition is measured once — so report that run's own step-to-step noise alongside the margin, and flag it when the gain does not clearly exceed it. The `e2e` timing and the real-input parity (`real`) are separate: if parity wiring is impractical for this target, record it as `skipped` with the reason, but the profiler-off `e2e` timing must still be produced.
- **You do not render the verdict.** Present evidence in `report.md`; the Judge decides keep / needs-repair / reject and the loop's disposition.

## Output

Two files are required in your working directory — they are your record:

- `worklog.md` — your process log: what you tried, why, the evidence, the result, and the decision, for each step along the way.
- `report.md` — your deliverable: the headline result and summary for the Main Loop Agent and the Judge.

You must write both. To do so:

1. Copy `worklog.md.template` and `report.md.template` (in this same `roles/optimizer/` folder) into your working directory as `worklog.md` and `report.md` **at the very start, before you begin any work**.
2. You should read through the templates' own notes first, to understand what belongs in each section before filling it in.

**Write as you go — don't defer your record to a final wrap-up.** Your turn can be cut off by a runtime timeout, and whatever is on disk then *is* your round:

- Append to `worklog.md` after each step, as it happens.
- Fill `report.md`'s headline/numbers as soon as verification lands, and keep them current — don't leave it as the template.

A template left unfilled is a lost round.
