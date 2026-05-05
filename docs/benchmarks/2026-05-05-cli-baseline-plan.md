# Cycle-13 CLI baseline benchmark plan — SWE-bench Pro 4-arm ablation

**Status**: Plan only. Run is Cycle-13 work (after `clients/pi-ygn-sage/` npm package ships and B4 wheels CI matrix is green).
**Author**: cycle-12 prelude (2026-05-05).
**Hypothesis under test**: YGN-SAGE harness beats Claude Code on hard SWE-bench Pro tasks by ≥5pp at the cost of ≤2× p50 latency.

---

## Why this benchmark exists

The cycle-12 prelude (this commit chain) ships `sage run --jsonl` per `docs/contracts/SAGE_CLI_PROTOCOL.md`. The cycle-13 work wraps it in a pi-mono adapter. Before any of that ships externally, **we need to prove the value-prop is real** — that a smarter harness (verification + topology evolution + bandit learning) lifts pass-rate enough to justify shipping a slower CLI.

**Why this is the right benchmark to spend money on**: SWE-bench Pro public set (1865 tasks, 41 professional repos, contamination-resistant) is the **only** widely-respected coding-agent benchmark in 2026 where top frontier models score ~23% on public set vs 70%+ on Verified. The huge gap means **harness lift has room to operate** — Claude Code at 64.3% on Verified vs ~23% on Pro is exactly the regime where multi-agent + verification + retry-with-evolution can move the number.

The empirical evidence the harness effect exists:

> Matt Mayer ran an independent test comparing the same model inside different harnesses. Claude Opus: 77% in Claude Code, 93% in Cursor. Across multiple independent studies the harness effect ranges from 5 to 40 percentage points depending on model and task type.
> — *thoughts.jock.pl/p/ai-coding-harness-agents-2026*

If the YGN-SAGE harness can deliver even half of the upper bound (20pp → 10pp lift), it's a defensible product story. If we can't beat Claude Code by ≥5pp on SWE-bench Pro, the pivot to CLI is not justified — we'd be shipping a slower clone.

---

## Hypothesis

**H1 (primary)**: YGN-SAGE backend (with topology evolution, formal verification on applicable tasks, bandit Thompson sampling) achieves higher pass@1 on SWE-bench Pro than Claude Code on the same task set, when both use comparable underlying models.

**H1a (lift)**: Δpass@1 ≥ 5 percentage points.
**H1b (cost)**: p50 latency ≤ 2× Claude Code's, p95 latency ≤ 3× Claude Code's.
**H1c (token economy)**: $cost-per-resolved-task ≤ 1.5× Claude Code's, even though raw tokens may be 2-3× higher (multi-agent runs use more tokens but resolve fewer tasks via retries).

**H2 (secondary)**: YGN-SAGE's evidence-gated learning produces a measurable improvement curve over the 50-task run (later tasks benefit from earlier-task bandit updates). Concretely: if we split the 50-task run into halves of 25 and re-aggregate, the second half's pass@1 ≥ first half's pass@1.

---

## 4-arm ablation

| Arm | What | Purpose | Cost estimate (50 tasks) |
|---|---|---|---|
| **A** | Claude Code direct | Industry baseline | ~$50-100 |
| **B** | pi-mono coding-agent direct (no YGN-SAGE) | Minimal-harness control | ~$30-60 |
| **C** | YGN-SAGE via pi-mono CLI (the pivot product) | Shipping configuration | ~$80-150 |
| **D** | YGN-SAGE direct via `sage run --jsonl` (no UI overhead) | Pure orchestration delta — measures harness effect without TUI rendering cost | ~$80-150 |

**Total cost estimate: $240-460** for the first run. Re-runs for variance estimation: 2 additional runs of arms C+D (~$320-600). Grand total budget: **~$600-1000** for a publishable result.

**Cost reduction lever**: if budget is tight, run on a 25-task subset first ($120-230). The smaller sample widens confidence intervals but the directional signal is the same.

---

## Models per arm

To isolate the harness effect from the model effect, every arm uses the **same underlying model** for code generation: **Claude Opus 4.7** (the current SOTA on SWE-bench Pro per Scale leaderboard, May 2026). This is non-trivial:

- **Arm A**: Claude Code's own Opus integration. No model choice — Claude Code picks.
- **Arm B**: pi-mono coding-agent configured with `--model claude-opus-4-7`.
- **Arm C/D**: YGN-SAGE's bandit may select non-Opus models for some tasks (kNN router decides). To control: force tier=`reasoner` via `SAGE_LLM_TIER=reasoner`, which floors all tasks to Opus.

This reduces statistical noise — any pass@1 difference must come from harness mechanics, not model choice.

**Caveat we surface in the report**: the bandit-learning effect (H2) is partly suppressed by forcing a fixed model. A separate run with bandit unconstrained tests whether learning helps when model selection is part of the loop. That's a second experiment.

---

## Metrics

### Primary (the H1 test)

- **pass@1** (per arm, per task subset). Numerator: tasks resolved under SWE-bench Pro grader. Denominator: 50.
- **p50, p95 latency** (per arm, per task). Wall-clock from task submission to final output.
- **Total $ cost** (per arm). Tracked via existing `CostTracker` (arms C/D) or pi-mono / Claude Code's own metering (arms A/B).

### Secondary (the runtime-integrity sanity checks for arms C/D only)

- **`oracle.trainable` rate** — % of runs where the oracle gate fired with `trainable=True`. Sanity: should be > 50% for a well-functioning evidence pipeline.
- **`bandit_attribution_mismatch` rate** — % of runs where Stage 0's predicted (model_id, template) didn't match Stage 4's actual execution. Per ledger invariant 6, should be < 5% (any higher signals a bug in attribution wiring).
- **`controller_decision` distribution** — count of `upgrade_model` / `prune_node` / `reroute_topology` / `spawn_subagent` / `open_gate` / `continue` actions. Tells us how active the runtime adaptation layer was.
- **`failure(reason="cancelled")` rate** — must be 0 in this benchmark (no human cancels). Above 0 = bug in the runner.

### Tertiary (UX)

- **Tool approval interventions per task** (arm C only — arms A/B/D don't have approval prompts). If this is high (>2 per task), it signals UX friction.
- **`cli_progress` heartbeat ratio** (arm C only) — frames-per-second on stdout. If consistently >5 frames/s, frontend rendering is overheating.

---

## Acceptance / rejection criteria

**Acceptance (ship the pivot)**:
- H1a: pass@1 lift ≥ 5pp for arm C OR arm D vs arm A.
- H1b: p50 latency ≤ 2× arm A.
- Secondary checks all in healthy ranges (`bandit_attribution_mismatch` < 5%, `oracle.trainable` > 50%, `failure(cancelled)` = 0).

**Rejection (revisit pivot)**:
- pass@1 lift < 5pp on either arm C or D. Possible interpretations:
  - The harness effect doesn't reach a 5pp lift on this benchmark → niche claim is too strong. Revisit positioning to a smaller niche (e.g. "topology-aware orchestration for SWE-bench Pro tasks > 1000 LOC").
  - The bandit/MAP-Elites priors are cold (trained on BCB-Hard, not SWE-bench-shaped tasks) → run a warm-up phase, then re-bench.

**Inconclusive**:
- pass@1 lift between 0 and 5pp. We ship cycle-13 anyway as a research preview, run a 200-task follow-up for tighter CI, and update positioning.

---

## Methodology

### Task selection

- Source: SWE-bench Pro **public** set (1865 tasks across 41 repos). Use the public split, NOT private — for reproducibility.
- Subset: 50 tasks, stratified by difficulty bucket (small / medium / large per `task_size` field) and language (Python / Java / C++ where Pro has multi-language tasks). Random-seeded for reproducibility.
- Single-shot. No best-of-N. (Best-of-N could blur the harness-effect signal; can be a second experiment.)

### Run procedure

For each arm × task:
1. Reset working directory to a clean clone of the task's base commit.
2. Apply the task's `problem_statement` as the agent input.
3. Cap wall-clock at 30 minutes per task.
4. Cap $ at $5 per task (tighten via `--budget-usd 5.0` for arms C/D).
5. Capture: final patch, all model calls (token in/out per call), wall-clock, terminal output.
6. Submit final patch to SWE-bench Pro grader (`evalplus`-style harness) for pass/fail.

### Replication

- Run all 4 arms in a single session on a single host (no cross-host hardware variance).
- Total run time estimate: ~12-30 hours wall-clock (4 arms × 50 tasks × ~10-30 min each).
- Output: structured JSON per task per arm at `docs/benchmarks/2026-XX-XX-cli-baseline-N50.json`, plus per-arm aggregate at `docs/benchmarks/2026-XX-XX-cli-baseline-N50-summary.md`.

### Statistical treatment

- Pass@1 differences: report point estimates + Wilson 95% CI. With N=50, a 5pp lift has ~70% power vs noise; consider N=200 for tighter inference if we ship and the marketing claim is "≥5pp lift" specifically.
- Latency: report p50 / p95 per arm; compare via Mann-Whitney U on per-task latencies.

---

## Out of scope (deferred to later experiments)

- **Multi-language coverage breakdown** (which Pro languages benefit most from YGN-SAGE harness vs which don't). Save for the 200-task follow-up.
- **Best-of-N** (run each arm 3× per task and report best). Confounds the harness-effect signal.
- **Partial-pass scoring** (Pro reports partial fixes as 0; some downstream users care). Stick with strict pass@1.
- **Cross-model harness effect** (does YGN-SAGE help Sonnet 4.6 more than Opus 4.7?). Save for cycle-14+.
- **Runtime adaptation under cost pressure** (does Fix C / `tier="budget"` controller-disable hold up under SWE-bench Pro?). Cycle-14+.

---

## Existing benchmark infrastructure to reuse

| Need | Existing asset | File |
|---|---|---|
| SWE-bench harness | `sage.bench.swebench_bench` + `swebench_diff_verifier` | `sage-python/src/sage/bench/swebench_*.py` |
| Per-task budget enforcement | `CostTracker` | `sage-python/src/sage/contracts/cost_tracker.py:13-76` |
| Result JSON schema | Existing benchmark JSONs at `docs/benchmarks/2026-04-17-swebench-lite-ablation-f1.json` | Same shape: configs_run + results array |
| Telemetry capture | `RuntimeEventLog` v0 + `_diff_verifier_outcome` annotation | Cycle-9 + cycle-10 P5 |
| Diff verifier | `SAGE_DIFF_VERIFIER_MODE=observe` (cycle-9 default) | `sage-python/src/sage/bench/swebench_diff_verifier.py` |

The Cycle-13 work mostly wires these together with the new `sage run --jsonl` + the pi-mono adapter. No new harness needed.

---

## Cost-aware framing in the public report

Even if YGN-SAGE is slower per task, **the right marketing metric is "$ per resolved task"**, not "$ per task". On SWE-bench Pro at ~23% pass@1, an arm A run that costs $1 per task with 23% pass-rate spends $4.35 per resolved task. An arm C run that costs $2.50 per task with 35% pass-rate spends $7.14 per resolved task — slower AND more expensive per resolved task. That would be a rejection.

But: an arm C run that costs $2.00 per task with 35% pass-rate spends $5.71 per resolved task — only 31% more $ per resolved task with a 12pp lift. That's a defensible product story.

The benchmark report MUST surface both per-task and per-resolved-task economics so the reader can pick their decision metric.

---

## Risk: we discover the lift is NOT there

What if arm C/D doesn't beat arm A by 5pp?

Then the cycle-13 work doesn't ship as a standalone CLI product. Options:
1. **Reposition as a research framework** — drop the "competitor to Claude Code" claim, double down on "verified adaptive orchestration runtime for research". cgpro 2026-04-30 architect review preferred this framing originally.
2. **Niche down** — find the task subset where YGN-SAGE DOES win (large multi-file refactors? formal-verification-applicable tasks?), ship a CLI for that niche only.
3. **Halt the pivot** — keep YGN-SAGE as a Python framework imported via `from sage import ...`, ship the protocol doc as a permanent contract for any future front-end (research lab, custom IDE plugin, etc).

Each option is recoverable from the cycle-12 prelude work (the `sage run --jsonl` + protocol doc are reusable in any of them).

---

## References

- SWE-bench Pro leaderboard: labs.scale.com/leaderboard/swe_bench_pro_public
- SWE-bench Pro paper / dataset: github.com/scaleapi/swe-bench-pro (May 2026 public release)
- Matt Mayer harness-effect study: thoughts.jock.pl/p/ai-coding-harness-agents-2026
- pi-mono coding agent: github.com/badlogic/pi-mono/tree/main/packages/coding-agent
- Claude Code SWE-bench scores: anthropic.com/news/claude-code-may-2026
- `docs/contracts/SAGE_CLI_PROTOCOL.md` — the protocol arms C/D speak.
- Cycle-12 prelude plan: `C:\Users\yann.abadie\.claude\plans\abstract-finding-pixel.md`.

## Status changes

- 2026-05-05: Plan proposed (cycle-12 prelude, this document).
- TBD (cycle-13 dry-run): N=10 smoke on Linux runner — verify all 4 arms boot + emit telemetry; ~$50.
- TBD (cycle-13 main run): N=50 stratified sample. Cost ~$240-460.
- TBD (cycle-13 analysis): Markdown summary + decision (acceptance / rejection / inconclusive).
- TBD (cycle-14): If accepted, N=200 follow-up for tighter CI. If rejected, repositioning ADR.
