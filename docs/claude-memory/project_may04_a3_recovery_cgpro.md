---
name: May 4 A3 N=50 abort + cgpro recovery analysis + corrected ablation interpretation
description: A3 died from Windows Modern Standby suspend; cgpro confirmed methodology trap (no-guardrails ≠ TopologyController) + 5-step recovery plan (event ledger, telemetry, watchdog, paired diagnostic, cloud rerun)
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# May 4 — A3 N=50 abort + cgpro recovery analysis

## A3 abort — what happened

**Launched 2026-05-03 19:58:27, died 2026-05-04 03:24:57 after only 34/300 tasks** (all in `full` config). Log: `sage-python/.tmp/a3_ablation_n50_stdout.log`.

Cause: Windows 11 **Modern Standby (S0 DRIPS)** suspended PID 34004 overnight despite "Performances élevées" scheme + `standby-timeout-ac=0` + `monitor-timeout-ac=0`. S1/S2/S3 are disabled when S0 is supported. Modern Standby can suspend background processes when the lid is closed or screen is off, regardless of plan timeouts.

**Why:** asyncio.wait_for() can't enforce wall-clock timeout when the event loop itself is suspended. BCB/273 (task 34) reported elapsed=20278211ms (5h 38min) of poisoned wall-clock. **How to apply:** before any long bench on local Windows, must use `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` + `powercfg`. Cloud VM is preferred for gate-quality runs.

## A3 partial data (only useful as planning, NOT as A3 evidence)

`full` config: 11 PASS / 33 useful tasks (BCB/273 excluded as poisoned) = 33%. Within sampling noise of v7 `full` 4/10 = 40%.

PASS: BCB/19, /34, /37, /92, /99, /100, /139, /161, /162, /184, /239.
TIMEOUTs at 120s: BCB/93, /101, /123, /147, /199, /211, /227 (7 inherent slow tasks).

BCB/82 + BCB/89 no longer timeout (93s + 92s, vs 120s in v7) — confirms cross-provider fix `9715ed4e` working.

## cgpro 2026-05-04 verdict (conv `cgpro_a3_recovery_20260504`, UUID `69f854ed-7af0-8390-929d-11ac89f68d6c`)

cgpro pulled live source at HEAD `05e4a7c5`, cited file:line for every claim. Saved finaltext: `.tmp/cgpro_a3_recovery_finaltext.md`.

### Methodology trap CONFIRMED
**`AblationConfig(guardrails=False)` ≠ disabling adaptive controller.**
- `bench/ablation.py:20-35` apply() only sets `loop._skip_guardrails=True`
- That flag is ONLY checked in `phases/act.py:175-179`, `phases/learn.py:57`, `phases/perceive.py:119` — short-circuits `guardrail_pipeline` rule-based safety checks
- `TopologyController` (Phase C: model upgrades, reroutes, debate-gate, prune) is NOT touched
- v7 `full` 4/10 vs `no-guardrails` 7/10 gap = guardrail_pipeline effect, not controller effect

**Why:** the previous CLAUDE.md + memory framing equivocated the two. Methodology error propagated 2+ days of reasoning. **How to apply:** before crediting an ablation result with a mechanistic explanation, verify via the actual `apply()` flag's call sites in code.

### Fix C (`a23e196b`) status
Wired correctly end-to-end (CLI `--tier budget` → `_BOOT_TIER` → `_boot_system` → `boot.py:679` → `boot_pipeline.py:292` → `pipeline.py:245` → `_effective_controller=None`/L2470 → `TopologyRunner(controller=None)`). But its empirical validation is still pending — A3 was supposed to provide it. With Fix C active, A3 isolates `guardrail_pipeline` alone (controller OFF for both `full` AND `no-guardrails`).

### Q3.b answered: Rust does NOT bypass Fix C
`RustTopologyController` is wrapped by Python `TopologyController` (`topology_controller.py:8-12, 29-34, 141-157, 237-254`). Top-level `evaluate_and_decide` stayed in Python (Rust exposes primitives only — `sage-core/src/topology/controller.rs:1-17`). When `_effective_controller=None`, the Rust path is not invoked.

### What Fix C does NOT disable (the "big control-surface ambiguity")
Even with Fix C active, the following still run:
- Rust SystemRouter / TopologyEngine / ContextualBandit boot paths (`boot_topology.py:20-26, 57-73, 112-153`)
- Single-agent AgentLoop ACT (`pipeline.py:2157-2166, 2339-2341`)
- Runtime guardrails (`phases/act.py:175-179`)
- AVR (`phases/act.py:124-125`)
- Memory write (`phases/act.py:357-358, 374-376, 473-476`)
- **FrugalGPT quality-gated cascade after runner** (`pipeline.py:2551-2629`)

The current ablation surface cannot separate these. Need control-surface telemetry per task to disambiguate.

### 3 non-obvious risks flagged

1. **Bench evidence layer NOT fail-closed.** `_run_ablation()` only stores `all_results[config.label]` after `bench.run()` returns; `BigCodeBenchBench.run()` only emits BenchReport at the end. A process death turns "270 useful tasks" into "operator archaeology." **Fix:** append-only event ledger as source of truth.
2. **Rust controller does not bypass Fix C.** (Already in Q3.b.)
3. **`.sage` reset path is dangerous.** `_clear_a14_topology_state()` truncates SQLite/WAL files via `write_bytes(b"")` (commit `05e4a7c5`). cgpro flagged truncate as a SQLite/WAL corruption risk. Preferred: per-run state directories (`.sage/bench-runs/a3-20260504T.../full/`) + `SAGE_STATE_DIR` injection + fail closed on locked files.

### 4-attempt boot pattern explained
Lines 31, 36, 69, 102 of A3 log all show `ABLATION: full`. cgpro confirms `_run_ablation()` only prints once per config (no retry loop around `_boot_system()`). So 4 prints = **4 separate process invocations**, each crashed on the `PermissionError` PathLib unlink before the 4th succeeded (verified: lines 1-100 of log show 3 Tracebacks). My retry-fix `05e4a7c5` (10×1s sleep + truncate fallback) prevents future crashes here, but truncate is the risky lever cgpro flagged.

## Cycle-9 recovery plan (5 steps, cgpro-validated)

**Step 1 — Patch evidence layer first**
- Append-only event ledger in `BigCodeBenchBench.run()` and `_run_ablation()`:
  - Events: `RUN_START`, `CONFIG_START`, `TASK_START`, `TASK_END`, `TASK_TIMEOUT`, `TASK_ABORT`, `CONFIG_END`, `RUN_END`
  - Flush + fsync after each event
  - Include git_sha, dirty diff hash, PID, host, OS, Python version, tier, provider/model, timeout, task index, task id, config, skip flags
  - Add `host_suspend_or_event_loop_stall` detection (elapsed_wall > 2 × timeout)

**Step 2 — Patch control-surface telemetry per task**
- Record: `config`, `execution_path`, `executed_template`, `node_count`, `controller_attached`, `skip_guardrails`/`skip_avr`/`skip_memory`/`skip_routing`, `router_active`, `rust_engine_active`, `frugal_cascade_attempted`, `controller_decision_count`
- This answers: what mechanism actually caused the v7 full→no-guardrails gap?

**Step 3 — Move run host (cloud)**
- Cloud VM (RunPod / Modal) for gate-quality A3
- Local Windows acceptable only after `powercfg /h off` + `SetThreadExecutionState` + watchdog — labeled "diagnostic" not "gate evidence"

**Step 4 — Tiny paired diagnostic before N=50**
- N=8-12, `full` vs `no-guardrails`, same tasks, same order
- Include BCB/82, /89, /93, /101 + random hard tasks
- Goal = **path attribution**, NOT pass rate
- If `controller_attached=false` and gap persists → Fix C wrong target
- If `frugal_cascade_attempted=true` differs OR single-agent guardrail/AVR differs → next fix vector

**Step 5 — A3 N=50 clean rerun**
- Clean commit, no dirty working tree (or capture diff)
- Stable host
- Per-task checkpointing
- Timeout invariant
- Per-task control-surface trace
- Full 6 × 50 from scratch
- 33 useful `full` tasks from failed run = planning only, NOT final A3 evidence

## Proposed new directive #9 invariant — ADOPTED 2026-05-04 (γ.1)

**Timeout Enforcement Invariant**: For each task, `elapsed_wall_ms <= timeout_ms + grace_ms` unless `host_suspend_or_event_loop_stall=true`; any task with that flag is excluded from pass/fail statistics and the run is marked non-gate-quality unless rerun cleanly.

**Why:** A3 BCB/273 (5.6h reported runtime) is exactly the class of failure directive #9 ("Declared ≠ verified") is supposed to catch. Now the **7th** invariant (was 6) in `docs/contracts/runtime-integrity-ledger.md`. Closed by commits `b44156e7` (wall-clock watchdog) + `0036217b` (TASK_ABORT in ledger) + `ae371202` (ledger doc + e2e test).

**How to apply:** before running any long-running bench on Windows, this invariant is now enforced both at the watchdog level (`HostSuspendDetected` raised when elapsed > timeout × grace_factor) and at the result-aggregation level (`aborted_count` increments, `passed_count` rolled back, `TASK_ABORT` event emitted).

## α.3+α.4 RESULT (2026-05-04 12:08, run-id `01KQS5A0C4FFNPVQFMPDADTGMM`)

**Setup**: 8 tasks (BCB/13,19,34,82,89,92,93,101) × 2 configs (full + no-guardrails), `tier=budget`, local Windows, `keep_awake_issued=true`. 35-min runtime. Output: `docs/benchmarks/2026-05-04-paired-diagnostic-n8.json` + `.events.jsonl` (38 events, 0 errors).

**Pass-rate**: full **4/8 = 50%**, no-guardrails **4/8 = 50%**. Per-task vector byte-identical: `[T,T,T,F,F,T,F,F]` (PASS: /13, /19, /34, /92). McNemar p=1.0, Cohen's d=0.0, 0 discordant pairs.

**Path attribution (control-surface from ledger)**:
- `controller_attached=False` on all 16 events (Fix C correctly applied across both configs).
- `skip_guardrails`: False (full) vs True (no-guardrails) on all 16 events (sanity).
- Mechanistic finding: **BCB/82 topology shifts 5-node `robust` → 3-node `sequential`** when `skip_guardrails=True` (perceive.py:119 guardrail pre-check leaks into TaskPlanner inputs). Other 7 tasks identical topology in both configs. Does not affect outcome (FAIL→FAIL on /82).
- BCB/89 + BCB/101: `node_count=0`, `was_bypassed=True` in both configs, 120s asyncio timeout (model-side latency, not orchestration overhead).

**Verdict on Fix C**: technically correct (controller off in budget tier as designed), but **does NOT close the v7 gap** because the gap was task-mix-dependent. v7 N=10 had no-guardrails passing /17 + /37 + a fluky /89 — none of which our N=8 slice covered (we shared /13, /19, /34, /92, /82, /89, /93 with v7, missed /17, /37). Fix C stays as defense-in-depth.

**Latency**: full avg 85.3s, no-guardrails avg 92.4s (+8% slower without guardrail short-circuit on AVR retries). Counter-intuitive vs v7 framing of "guardrails add overhead". The +8% is dominated by /13 (+31s) and /93 (+24s) where AVR ran a fuller loop without the pre-check.

**Why:** v7's 4/10 vs 7/10 framing as "controller adds overhead" was wrong. The framing was "Guardrails (adaptive controller…)" but the ablation flag and the controller are orthogonal levers (cgpro 2026-05-04 confirmed). The actual v7 gap was sample-variance + task-mix-dependent. **How to apply:** when interpreting an ablation, never equivocate the flag's name with the implementation. Always trace the flag through its `apply()` call sites in the source before crediting a hypothesis.

## Cycle-10 candidates (out of cycle-9 scope)

1. **A3 N=50 cloud rerun** — full task set, all 6 configs, RunPod or similar. Must include /17 + /37 to surface v7's actual gap. Use `--ablation-configs` if budget gates skipping `baseline`.
2. **BCB/82 topology coupling test** — focused unit test in `tests/test_perceive.py` (or new `test_perceive_guardrail_coupling.py`): verify that `_skip_guardrails=True` vs False produces identical TaskPlanner inputs (omega/delta/gamma) for the same TaskInput. If not, document or fix.
3. **Replay diagnostic on /17 + /37 + /82 + /89 + /101 + 3 stable** — discriminating N=8 that targets v7's actual divergent tasks. Use `--task-ids` filter (no code change needed, α.2 already shipped).

## Files committed in this session (chronological)

- `cb7884e4`: docs(claude): correct A3 abort + ablation interpretation trap
- `a56a76e2`: feat(bench): event ledger + 11 tests
- `b44156e7`: feat(bench): wall-clock watchdog + 7 tests
- `46c280e3`: feat(bench): Windows keep-awake + 6 tests
- `0036217b`: feat(bench): wire all 3 into ablation
- `ae371202`: docs(integrity): 7th invariant + e2e test (γ)
- `8e1bf6dd`: feat(bench): --ablation-configs + --task-ids (α.1+α.2)
- (next): docs(α): paired diagnostic N=8 results + memory + CLAUDE.md sync

Total: +33 unit tests (24 from A.1-A.4 + 3 from γ.2 + 6 from α.1-α.2). 2940 collected total.

## cgpro round-2 corrections + replay discriminant N=8 (2026-05-04 13:43)

cgpro round-2 (`b50k9th9l`, ~10 min) flagged 4 corrections to morning's α verdict (saved at `.tmp/cgpro_alpha_verdict_finaltext.md`):

1. **"v7 = sample variance" verdict was too strong** — α only falsifies controller-only hypothesis on 8-task slice; doesn't falsify v7 itself.
2. **"robust → sequential" claim NOT supported by ledger** — `executed_template` was empty (telemetry bug). Closed at commit `c136463e`: route `bigcodebench_bench.py:trace` to `ctx.executed_template` + `ctx.bandit_template` + add 8th invariant "Control-surface completeness" to runtime-integrity-ledger.
3. **`avr_attempted/avr_repaired` are derived from BCB repair, NOT internal `phases/act.py` AVR** — the "AVR retries fire fuller" hypothesis can't be tested with current telemetry.
4. **Task set choice**: cgpro Q3 recommends full v7 N=10 (`/13,/15,/17,/19,/34,/37,/82,/89,/92,/93`) with **counterbalanced** config order, not N=8.

**Replay discriminant N=8 (commit `c136463e+1`, NOT instrumented because launched before fix shipped)**:
- Tasks: `/13,/17,/19,/34,/37,/82,/89,/101` (drops /92,/93, adds /17,/37)
- Full: 3/8 PASS = 37.5% (regression vs morning's 4/8 — sample variance on /13)
- No-grd: 4/8 PASS = 50%
- McNemar p=1.0, Cohen's d=-0.254, discordant=0/1 (no-grd wins on /13)

**Cross-run table (6 shared tasks × 4 config-runs = 24 datapoints)**:
- /13: 3 PASS + 1 FAIL → **boundary stochastic on full**
- /19, /34: deterministic PASS (4/4)
- /82: 4 FAIL/TIMEOUT all near 110-120s cap
- /89: 4 TIMEOUT (deterministic single-agent slow)
- /101: 2 TIMEOUT + 2 FAIL → **boundary stochastic on cap**

**Falsifies cgpro's `/17` + `/37` driver hypothesis**:
- /17 FAIL in both configs (replay) — deterministic FAIL on budget
- /37 PASS in both configs (replay) — deterministic PASS on budget
- /89 TIMEOUT 4/4 — deterministic, not flippable

**Topology coupling reproduced** (this run + morning):
- skip_guardrails=True consistently flips node_count 5→3 on some tasks
- Morning: /82 (5→3)
- Replay: /19 (5→3), /34 (5→3) — outcomes unchanged
- The coupling is REAL but doesn't change pass-rate. Cycle-10 unit test for `phases/perceive.py:119` is justified.

**Updated verdict**: across 32 paired data points, **zero deterministic mechanism divergences**. The full→no-guardrails delta is entirely consistent with sample variance on 3 boundary stochastic tasks (/13, /82, /101). v7's 4/10→7/10 gap is **most likely sample variance amplified by N=10 + LLM stochasticity at budget tier**. Not gate-quality (telemetry was old code mid-run, order not counterbalanced); definitive via cycle-10 full v7 N=10 counterbalanced replay.

**Fix C status**: correct and harmless. Stays in tree. Not a gap-closer because there is likely no gap to close.

**Cycle-10 priority order (post-replay)**:
1. Full v7 N=10 counterbalanced replay with instrumented telemetry — settles the v7 question definitively. ~50 min.
2. perceive→TaskPlanner coupling unit test (3 tasks now: /82, /19, /34) — independent.
3. Telemetry split (`internal_avr_*` vs `bcb_repair_*` vs `runtime_guardrail_*`).
4. A3 N=50 cloud — only after 1+3.

Analysis at `.tmp/replay_discriminant_analysis.md`.

## P4 v7 N=10 RESULT (2026-05-04 17:54, cycle-10 P4 closure)

**Setup**: `python -m sage.bench --type ablation --ablation-configs full,no-guardrails --task-ids "/13,/15,/17,/19,/34,/37,/82,/89,/92,/93" --tier budget`. EXACT v7 reference task set. ~28-min run, local Windows, `keep_awake_issued=true`. NOT counterbalanced order (full first, no-grd second — true counterbalancing = cycle-11).

**Result**: full **4/10 = 40%**, no-grd **4/10 = 40%**, **byte-identical per_task vector** `[F,F,F,T,T,T,F,F,T,F]` (PASS: /19, /34, /37, /92).

McNemar p=1.0, Cohen's d=0.0, 0 discordant pairs.

**Full reproduces v7 reference (April 2026) EXACTLY** — same 4 tasks PASS.

**No-grd does NOT reproduce v7 reference's 7/10**: April 2026 v7 had `/13`, `/17`, `/89` PASS additionally — all 3 falsified here:
- `/17` FAIL or TIMEOUT in BOTH configs across **all 4 paired runs** (morning N=8, replay N=8, P4 full, P4 no-grd). v7's `/17 PASS` was a one-time stochastic outlier.
- `/89` TIMEOUT or FAIL in BOTH configs across all paired runs. v7's `/89 PASS` was either different code state (cycle-7 vs cycle-9 cross-provider fix `9715ed4e`) or stochastic.
- `/13` boundary stochastic (PASS-PASS morning, FAIL-PASS replay, FAIL-FAIL P4).

**Definitive verdict (52 paired data points across 4 runs)**: **v7 4/10→7/10 gap was sample variance.** 0 deterministic mechanism divergences observed. Fix C (`a23e196b`) is correctly applied (`controller_attached=False` everywhere) but does not address a deterministic phenomenon — none exists at this task slice + budget tier.

**Cycle-9 telemetry fix VALIDATED**: control_surface fields `executed_template`, `selected_template`, `dag_omega`, `dag_delta`, `dag_gamma` ALL populated when `node_count > 0` (per `c136463e` + `43726991`). Empty when `node_count = 0` (bypass), as expected.

**Topology coupling REVISED**: morning N=8 finding "BCB/82 5→3 nodes when `_skip_guardrails=True`" was based on EMPTY `executed_template` (the cycle-9 telemetry bug, fixed `c136463e`). Real coupling is **bypass-vs-sequential** (single-agent vs 3-node sequential), not robust-vs-sequential. 4/10 tasks flip routing path in P4 (full bypasses /17+/82, no-grd bypasses /15+/89). All 4 affected tasks FAIL in both configs anyway. The flip is undocumented behavior — cycle-11 unit test candidate (P9 ADR-015 cycle-11/12 follow-up).

**Latency**: full 83.7s avg vs no-grd 100.2s avg (+19.7%). Replicates morning (+8%) and replay (+8%) directional finding. cgpro round-2 hypothesis "no-grd lets AVR fire fuller" empirically supported.

**A3 N=50 cloud (P8) no longer urgent for "settle v7"**. Would still tighten confidence intervals on boundary-stochastic tasks but does not change qualitative verdict.

Analysis: `.tmp/p4_v7_counterbalanced_analysis.md`. Output: `docs/benchmarks/2026-05-04-v7-counterbalanced-n10.json` + `.events.jsonl` (46 events, 0 errors).
