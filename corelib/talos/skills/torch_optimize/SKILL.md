---
name: torch-optimize
description: "Three-role optimization loop for a PyTorch training workload: the Main Loop Agent owns the end-to-end run, authoritative profiling, and hotspot selection; a disposable Optimizer subagent reworks one hotspot per round in an isolated context; a Judge subagent renders an evidence-based verdict driving iterate/pass. All cross-role state lives in files."
argument-hint: "<path to user_guide.md>"
---

# Torch Optimize

Reduce the end-to-end step time of a PyTorch training / inference workload through an evidence-driven loop: profile → dispatch one hotspot to an isolated optimizer episode → judge the episode → iterate or move on.

## Big picture

You are the **Main Loop Agent**. This document is your playbook. You never optimize code yourself — you keep the global picture and dispatch.

### Roles

- **Main Loop Agent.** Gets the training running end to end, owns authoritative profiling (torch-perf-analysis skill), picks hotspots, writes briefs, spawns the Optimizer and the Judge, applies verdicts, reviews prior verdicts, decides global stop.
- **Optimizer** (`roles/optimizer/`). One fresh subagent per round. Receives a brief naming ONE hotspot; iterates optimize → validate → profile inside its own disposable context; logs its process and findings to files; delivers a structured report.
- **Judge** (`roles/judge/`). One fresh subagent per finished round. Independently checks the evidence and renders a verdict: `ITERATE` (worth another round, with ranked directives) or `PASS` (done here).

### User guide (`user_guide.md`)

The skill's single argument is the path to a user-provided **`user_guide.md`**. Read it at the start and keep it in mind throughout — it is mandatory input, not optional background, and it is the only place the workload is described.

**What it contains:**

1. **Working directory** — the path holding the PyTorch code to optimize, and the directory the start/stop commands assume. Everything else in the guide is relative to it.
2. **How to start** — the command that runs the workload end to end (training or inference). This is the authoritative command for baseline profiling, Optimizer verification, and any re-measurement.
3. **How to stop** — how to end the run cleanly.
4. **Focus guidance** *(optional)* — which aspects or layers of the stack the user wants prioritized (e.g. a specific module, communication overlap, compile-friendly regions, areas explicitly out of scope). This tells you *where to look first* and *what kinds of wins matter most* — not how to fix a hotspot.
5. **Attention** *(optional)* — workload constraints that are invisible in the code: production shape distributions, numerical tolerances, memory ceilings.

When the optional sections are absent, rank hotspots purely by measured step-time impact.

**How to follow it:**

- **Before the loop:** Confirm the working directory exists and the start/stop commands are complete enough to execute. If anything is missing or ambiguous, stop and ask — do not invent commands.
- **When profiling (step a):** Use the start/stop commands from `user_guide.md` for every profiled capture. While analyzing traces and writing `loop_baseline_analysis.md`, weight hotspots against the focus guidance: when several candidates are plausible, prefer the one that best matches the user's stated priorities; use raw profile size only to break ties the guidance does not distinguish.
- **When dispatching the Optimizer (step b):** Include the relevant focus guidance in your spawn brief so the Optimizer knows which trade-offs and boundaries the user cares about.
- **Respect scope:** If `user_guide.md` marks an area as out of scope or lower priority, do not dispatch work there unless profiling shows nothing else worth optimizing within the stated focus.

Do not confuse this file with `optimization/instructions.md` — that is the Optimizer's technique playbook inside this skill; `user_guide.md` is user-authored context for *this specific workload*.

### Workflow

```
n = 1
while True:
    review_prior_verdicts()                            # loop_<*>/round_<*>/verdict.md — prior conclusions; skip settled/blocked boundaries
    checkpoint = git_checkpoint(loop_n=n)              # before any change this loop; loop_<n>/loop_baseline.patch
    hotspot = profile_and_pick_top_hotspot(loop_n=n)   # (a) Main Loop Agent
    if hotspot is None:
        break                                           # nothing left worth optimizing

    k = 1
    context = None                                        # round 1: no prior round to carry in
    while True:
        report  = dispatch_optimizer(loop_n=n, round_k=k, hotspot, context)   # (b)
        verdict = dispatch_judge(loop_n=n, round_k=k, report)                  # (c)

        if verdict.outcome == ITERATE:                  # (d)
            context = {report: report, verdict: verdict, worklog: worklog_path(n, k)}
            k += 1
            continue                                     # back to (b), same hotspot, with context
        else:  # PASS
            if verdict.code_disposition == KEEP:
                pass                                       # this loop's accumulated changes stay
            else:  # REVERT
                git_apply(checkpoint)                        # restore to loop_<n>/loop_baseline.patch
            break                                         # this loop is done

    n += 1                                                # back to (a), next hotspot
```

## Procedure

1. **Check the inputs.** This skill takes one argument: the path to **`user_guide.md`**. Verify:
   - The file exists at the given path.
   - It names a working directory, and that directory exists and looks like the PyTorch package it describes.
   - It has start and stop commands complete enough to execute unattended.

   If any of these fails, **stop and ask the user** before doing anything else — do not guess the working directory or invent a run procedure.

2. **Initialize the workspace.** Once the inputs check out, create this run's storage:

   ```bash
   python3 -m talos.torch_optimize.python.workspace init --root <working directory from user_guide.md>
   ```

   Creates a fresh `.torch_optimize/` under the given root — see `workspace_layout.md` for what the agents put inside it. If a `.torch_optimize/` already exists there (a prior session), it is renamed to a timestamped `.torch_optimize_backup_<timestamp>/` first — this run always starts from an empty workspace, and nothing from the prior one is lost.

3. **Get oriented.** Skim the codebase from step 1 enough to know its rough shape — entry point, model definition, training loop. You don't need to understand it deeply yet.

4. **The loop.** From here on, repeat: you (the Main Loop Agent) pick out the single hotspot you currently judge to be the biggest, then hand it to the Optimizer to work on.
   - **(a) Profile.**
     1. **Review the prior loops' verdicts** (`loop_<*>/round_<*>/verdict.md`) — what each was kept / reverted / blocked and why — so you don't re-pick a settled or blocked boundary. (Loops are few, so re-reading them is cheap; there is no separate ledger to maintain.) Then create the next `loop_<n>/` folder (`workspace_layout.md` "Naming").
     2. **Before anything else touches the code this loop**, take a git checkpoint of the current state into `loop_<n>/loop_baseline.patch` — this is what you restore to if the Judge later decides this loop's changes should not be kept.
     3. Follow the `torch-perf-analysis` skill: instrument the code and capture a profiled run ("Step 1: Stats"), writing its outputs into `loop_<n>/` (`loop_baseline.nsys-rep` + the torch-perf-analysis outputs — see `workspace_layout.md`).
     4. Copy `roles/main_loop/loop_baseline_analysis.md.template` to `loop_<n>/loop_baseline_analysis.md` and fill it in: read the workspace with the same skill's "Step 2: Analyze", then write up each hotspot — what it is, why it happens, and the evidence for it.
        **Scope of this step: locate the problem, not the fix.** Your job here is to find *where* the performance problem is — which stream or thread, which region or source line of the code — and *why* it happens (root cause), e.g. "this part of the Python code is the culprit." Deciding how to fix it is the Optimizer's job once it has your analysis in hand, not yours. Follow the template's locate-only output contract. Read only the profiling outputs and the code (to locate `where`); do not open `optimization/` (techniques / dispatch table) — that is the Optimizer's.

        **Apply `user_guide.md` when picking the hotspot.** While analyzing, weigh candidates against the focus guidance in `user_guide.md` — the hotspot you dispatch should be the one that best matches the user's stated priorities, not merely the largest row in the trace. Document your choice and the reasoning in `loop_baseline_analysis.md`; only then proceed to step (b).

        **The profiled capture is for attribution, not for gain.** The nsys "typical step" here only locates hotspots; the authoritative end-to-end gain is measured profiler-off by the Optimizer's verification `e2e` (`optimization/verification/verification.md`) and recorded in each round's `verification.json`. Use that for keep/stop reasoning, not the profiled typical step.

   - **(b) Dispatch to the Optimizer.** Create `round_<k>/` under this `loop_<n>/` (`workspace_layout.md` "Naming"; `k` = 1 the first time through this loop, otherwise one more than the last round tried in it) — a subagent's working directory must exist *before* it is spawned. Then spawn a fresh Optimizer subagent into it — a clean context, not a continuation of your own.

     **The subagent's model must be identical to your own.** When spawning, explicitly set it to the same model you yourself are running as — **NEVER** let it silently default to a different or weaker one.

     Give it, as part of the spawn instruction:
     1. A description, written by you in the moment, of the overall background and current state — and what target region or hotspot it should optimize. On round `k > 1`, also point it at the previous round's `report.md` and `verdict.md` and carry forward the verdict's directives.
     2. Pointers telling it to go read:
        - `roles/optimizer/instructions.md` — its own instructions.
        - `user_guide.md` — start/stop commands and the user's focus guidance for this workload.
        - the profiling outputs under `loop_<n>/` — the nsys trace and the torch-perf-analysis outputs.
        - `loop_<n>/loop_baseline_analysis.md` — your own analysis.
     3. Its working directory: `loop_<n>/round_<k>/` — where it writes `worklog.md` and `report.md`.

     It works in its own disposable context and delivers `report.md` there. This loop has one cumulative patch, `loop_<n>/loop.patch` — the diff of every kept optimization so far plus this loop's work; each round refines that same patch on top of the prior state rather than starting a new one, so the run ends with a single patch to merge.

   - **(c) Dispatch to the Judge.** Once the Optimizer stops, spawn a fresh Judge subagent on the finished round — a clean context, not a continuation of your own.

     **The subagent's model must be identical to your own.** When spawning, explicitly set it to the same model you yourself are running as — **NEVER** let it silently default to a different or weaker one.

     Give it, as part of the spawn instruction:
     1. A description, written by you in the moment, of the background and current state — what this round was working on and why.
     2. The round to judge: `loop_<n>/round_<k>/` (its working directory), and the git range covering that round's commits.

     It delivers `verdict.md` there (`roles/judge/instructions.md`).
   - **(d) Act on the verdict.** **Follow the Judge's verdict as-is — do not re-adjudicate it.** It is not your job to form your own opinion on whether the Optimizer's work was actually good, or to second-guess whether the Judge's call was right; that entire evaluative burden belongs to the Judge, not you. `ITERATE` → go back to (b) for the next round on the *same* hotspot. `PASS` → act on the verdict's **code disposition** exactly as given: `keep` → leave this loop's accumulated changes in place, even if end-to-end itself showed no or negative gain — the Judge may have a documented reason (see `roles/judge/instructions.md`); `revert` → restore the code exactly to `loop_<n>/loop_baseline.patch`, undoing everything this loop changed. Either way, go back to (a) and re-profile for the *next* hotspot.

## Goal

Push the workload as far as it will go. The only legitimate stopping condition is step (a) finding no hotspot left worth optimizing — nothing else counts:

- A round with only a modest gain is not a reason to call a hotspot done and move on, if profiling still shows real room left in it — keep iterating it.
- One hotspot being fixed is not a reason to stop the overall process, if re-profiling still surfaces others worth working — keep re-profiling and dispatching.
- **Do not stop early because progress feels good enough, or because this has already taken a while.** A long-running process — many hotspots, many rounds each, hours of wall-clock time — is expected and completely fine. Optimize for how far the workload actually gets, not for finishing quickly.
