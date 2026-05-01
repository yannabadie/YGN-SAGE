---
name: April 29-30 cycle-7 default-on flip + 3-round cgpro VERIFY APPROVED
description: SAGE_ORACLE default-on flip 2026-04-29 evening + 3-round cgpro VERIFY closed APPROVED 2026-04-30 at commit 4b8af448. BCB-Hard N=50 official Docker pass@1=32%. T1-T6 + R6.1b parser + T4 allowlist + Q1 disable/disabled + A14 reset.
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# Cycle 7 default-on flip + 3-round cgpro VERIFY (29 Apr eve + 30 Apr AM)

**Final state**: cycle-7 APPROVED by cgpro round-3 at commit `4b8af448` (2026-04-30 morning). 29 ship commits (29 Apr 11:49 AM → 22:18 PM) + 5 closure commits (30 Apr AM, includes 3 hotfix + 2 evidence-orphan-cleanup) = 34 cycle-7 commits total.

**cgpro VERIFY trail**:
- **Round 1** (commit `87daf89a`+`f3a89631`): PUSH BACK with 2 blockers + Q1 nuance.
  - Blocker 1: T4 forced `controller_decision.payload` was NOT actually safe — `reason` free-form string was force-surfaced under default-on; redaction layer was credential-only, no allowlist/PII ban/stack-trace ban/4 KiB cap.
  - Blocker 2: public docs/contracts contradicted default-on (runtime-event-log.md, README.md, N=50 validation.md header still pre-flip framing).
  - Q1: kill-switch missed `disable`/`disabled` (operators type these).
- **Round 2** (commit `4b8af448`): Blocker 1 + Q1 APPROVED. Blocker 2 partially closed → 4 sub-blockers + 1 trap-fix all docs/spec only.
  - Sub-A: oracle_verdict.json golden still said "ONLY emitted when SAGE_ORACLE=1".
  - Sub-B: roadmap.md still said "All 4 strategic feature flags ship default-OFF".
  - Sub-C: CLAUDE.md + Dashboard kill-switch list still 4 tokens (no disable/disabled).
  - Sub-D: validation.md A14 path wrong (`docs/ops/runbooks/...` doesn't exist; correct is `docs/operations/...`) + headline imprecise ("every committed run" vs criteria 49/50).
  - Trap-fix: contract fixtures are prose; prose lint missing → added `test_no_stale_cycle7_oracle_contract_phrases` scoped to current-state docs only (not historical evidence).
- **Round 3** (no new commits, all closed in `4b8af448`): **APPROVED**. Cycle-7 closes cleanly.

## The flip itself

**Commit `128e1b89`** — `feat(runtime/oracle,pipeline): cycle-7 default-on flip — SAGE_ORACLE unset = ON`.

- Centralized predicate `oracle_enabled()` in `sage/runtime/oracle/env.py`
- **Default-on contract**: unset `SAGE_ORACLE` → oracle path active (was opt-in `=1` before)
- **Kill-switch values**: `0` / `false` / `off` / `no` (case-insensitive) → oracle silent — operator escape hatch
- Bandit / MAP-Elites / online-evolution / training-memory **only update** when `verdict.trainable=True`
- Posterior epoch=1 (A14 reset done as part of the flip per cycle-6 closure gate)

## Pre-flip prep (mandatory before flip)

1. `f6711385` **fix(runtime/oracle): block raw bench_result.reason leak into oracle_verdict.reason_codes** — closed cgpro PUSH BACK. Raw bench `reason` strings (e.g., `"AssertionError: expected 5, got 3"`) were getting copied into `oracle_verdict.reason_codes`. Now SHA-256-hashed into `EvidenceRef.evidence_hash`.
2. `f9305d74` N=5 smoke validates leak-fix — **0 leaks across 106 events**.
3. `162e82ea` BCB-Hard N=50 evidence + Phase 2bis Docker re-grade.
4. `1dac2d11` Phase 1 official BCB Docker re-grade pass@1 = 0.900 (9/10).
5. `8c8a1c27` runbook: A14 reset operation + tonight cycle-7 evidence runbook + T1-T6 tickets.

## BCB-Hard N=50 evidence (the headline number for cycle 7)

- **Internal pass@1 = 30%** (15/50 `exact/pass` + 34 `exact/fail` + 1 `None/abstain`)
- **Official Docker pass@1 = 32%** (16/50)
- **Per-task agreement = 49/50 = 98%**
- Single `None/None` task = `BigCodeBench/227` (seam abstain, bench fail — agreed direction)
- Cross-check totals: agree=49, diverged_likely_escalation=0, unknown=1
- Files in `docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-*` (UNTRACKED at session end)

**Honest framing locks** (apply when reporting): NOT a leaderboard submission, official `bigcodebench.evaluate` `untrusted_check` path fails on Windows (`os.killpg` → coerces all tasks to `timeout`), seam evaluator falls back to `BigCodeBenchBench._evaluate_solution_with_stderr` (matplotlib-headless subprocess) and tags `bench_result.verifier_id` accordingly. Per AUDIT2 2026-04-24 framing rule: no "above SOTA" or leaderboard-style claims attached.

## Post-flip smoke validations

| Commit | Smoke | Result |
|---|---|---|
| `a5f916ea` | N=5 `SAGE_ORACLE` UNSET | **5/5 oracle_verdicts emitted, 0 raw leaks** — flip works default-on |
| `8b4b34b6` | N=2 `SAGE_ORACLE=0` kill-switch | **0 oracle_verdicts emitted** — escape hatch works |
| `4b257d3e` | N=2 post-5item smoke | validates 5 cycle-7 post-flip commits |

## R6.1b cleanups (post-cycle-6 follow-ups, shipped during cycle-7)

- `b4392042` **feat(runtime/evidence): R6.1b pytest parser anchored to pytest summary delimiter** — only matches `=== N passed, M failed in T.Ts ===` line, not docstring text containing `"1 passed in 0.1s"`.
- `aacc364e` **fix(runtime/oracle): R6.1b ToolOracle parser-pass guard uses scoped fatals only** — incidental fatal + parser pass = trainable pass (not abstain). Compute `scoped_fatals` BEFORE scope filter; non-blocking, safe direction.

## T-tickets shipped (cycle-7 minor flips)

| Ticket | Commit(s) | Purpose |
|---|---|---|
| **T1** topology engine-first diagnostic flags | `db1afced` | Diagnostic visibility for topology routing |
| **T2** memory write_gate skip-reason telemetry | `b71f4897` (telemetry split) + `b6820f2b` (phase 0/1 wiring memory→AgentLoops) | Per-node AgentLoops now have memory backends; skip-reason telemetry split into discrete enum values |
| **T4** unredact controller_decision payload | `0fbe8e79` (unredact safe fields) + `b6042383` (analyzer reconciliation test) | Diagnostic visibility into controller decisions |
| **T5** top-3 candidate logging | `276d6ce5` (initial) + `87be56fe` (attribution + finite guard) | Provider attribution debugging — guard against non-finite scores |
| **T6** SAGE_BENCH_DISABLE_REPAIR | `515739bf` | Clean first-attempt measurement flag for benches |

**T2 status**: phase 0/1 (memory backend wiring) shipped. Phase 2/3 (full memory write paths) NOT yet shipped. New test `sage-python/tests/test_memory_write_gate_telemetry.py` UNTRACKED at session end.

## Path E step 3 (cycle 6 R6.1a verify, ran during cycle-7 prep)

- `c1a45213` **R6.1a Path E — bench-result feedback seam for synchronous-eval benches** (Path E step 1+2)
- `e74289fd` BCB Hard Instruct N=10 with feedback seam — PASS (4 cgpro Path E B' criteria all PASS)
- `3bdbcaaa` Gate D `SAGE_ORACLE=1` SWE-bench Lite N=10 throwaway bandit — PASS (paired bandit DB swap, prod posteriors not polluted)

## A14 reset (paired with flip)

Per CLAUDE.md "Posterior epoch=1 (post A14 reset 2026-04-29)" — the cycle-6 R6.1a closure gate said "Operator A14 reset checkpoint paired with the flip operation (NOT prerequisite for R6.1a closure; paired with the actual default-on operation)".

Reset done as part of the flip evening. Old off-policy posteriors discarded; production starts fresh with epoch=1 posteriors that will accumulate ON-policy now that bandit causality is correct (roadmap-A14 + A14b shipped 2026-04-27).

## Untracked at session end (TO COMMIT before continuing)

```
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-canonical-predictions.jsonl
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-jsonl/  (49 trace files)
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-manifest.json
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-official-grade.json
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-official-grade.log
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-official-pass-at-k.json
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-validation.md
docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50.json
docs/benchmarks/2026-04-29-smoke-killswitch-post5item-N1.json
docs/benchmarks/2026-04-29-smoke-post-t2-N3.json
sage-python/docs/benchmarks/2026-04-29-predictions-hard-instruct.jsonl
sage-python/tests/test_memory_write_gate_telemetry.py
```

## Cycle-8 ordering (cgpro round-3 recommendation)

cgpro APPROVED with explicit ordering:

1. **R6.1c + T4/A16 generalization** — per-(producer, delta_kind, event_type) allowlist schemas + max string lengths + schema-versioned historical validation. **This subsumes the deferred `--allow-legacy-controller-reason` flag** (the right home is payload schema versioning: `event_type=controller_decision schema=v1_pre_allowlist_reason | v2_allowlist_only`).
2. **A14 epoch fail-closed guard** — `boot_topology.py` reads `~/.sage/posterior_epoch.json`; raise if `epoch != 1` AND state files exist. Plus poison-pill marker `{"epoch": 0, "contaminated": true, "do_not_restore_without_manual_override": true}` in moved-aside backups under `~/.sage/contaminated_pre_a14_20260429/`. Prevents accidental restore from silently reintroducing off-policy posteriors.
3. **A22 follow-ups** — bucket analysis script that aggregates `_diff_verifier_outcome` first / `_diff_verifier_reasons` second; off-mode regression test that JSONL contains NONE of the 3 verifier keys; deletion-side `/dev/null` test for `file_creation_or_deletion`.
4. **T2 phase 2/3** — full memory write paths beyond the per-node AgentLoop wiring shipped in `b6820f2b`. NOTE: cgpro explicitly said do this AFTER payload schema gates are tight (R6.1c).
5. **planner producer live integration** — structural-only facts (`topology_selected` / `decomposition_applied`). LAST per cgpro because it adds evidence flow before payload contracts are tight.

Non-blocking polish from round-3: validation.md criteria table row 3 still says "on every run" with PASS (49/50). cgpro suggests changing to "on every verdict-emitting run" in cycle 8 — not a blocker.

Other open items unchanged from prior cycles:
- **roadmap-A14 rollout** — extend `output_schema` to more tools opportunistically.
- **roadmap-A3** — N=50 paired observe-vs-repair smoke (API-budget gated).

## Methodology insights from cycle 7 (apply to cycle 8)

1. **A14 reset paired with flip, not before**: cycle-6 closure note explicitly said "do NOT reset persisted production posteriors as part of R6.1a. That belongs to the default-on operation." Followed correctly — reset epoch=1 timestamped at flip moment.
2. **Pre-flip leak audit caught real bug**: cgpro round 1 `f6711385` push-back found raw `bench_result.reason` was bypassing the SHA-256 hash channel in `oracle_verdict.reason_codes`. Without the audit this would have leaked production assertion strings into trainable telemetry. Validated by N=5 smoke (`f9305d74`, 106 events 0 leaks) BEFORE flipping.
3. **N=50 evidence is the right granularity for headline numbers**: N=10 is noise-dominated (±10pp per task flip — see April 21 lesson). N=50 gives 30% pass@1 with stable distribution; per-task agreement 98% across internal vs Docker grade.
4. **T-tickets are sized to ship same evening as flip**: T1-T6 are minor diagnostic / telemetry / wiring flips. None require cgpro VERIFY individually. Bundled smoke (`4b257d3e`) validates 5 of them at once.
5. **cgpro VERIFY of substantial cycles is non-negotiable**: cycle-7 was Claude-led from a runbook, NOT cgpro DESIGN-locked. Going into VERIFY without a DESIGN spec meant cgpro found 2 round-1 blockers + 4 round-2 sub-blockers that cleaner DESIGN-first would have prevented. Lesson: even when ship is internal-runbook-driven, run cgpro VERIFY before declaring done. Cycles >1000 LOC touching runner/oracle deserve cgpro VERIFY regardless of methodology shortcuts.
6. **Contract drift is a class, not an instance**: cgpro round-2 trap was "contract fixtures are prose, prose is not tested deeply enough". Field-presence tests passed even though the golden invariants directly contradicted the contract docs. Fix: `test_no_stale_cycle7_oracle_contract_phrases` scoped to current-state docs (NOT historical evidence). Apply this pattern to future contract migrations: when flipping a contract default, add a regression test of stale phrase strings in the docs scoped set.
7. **Historical evidence is frozen, current-state docs are policed**: when adding contract lint tests, scope must exclude historical evidence (`docs/benchmarks/`) and operation runbooks (`docs/operations/`). A pre-flip evidence file saying `SAGE_ORACLE=1` is not stale — that was the actual run config. Lint should rewrite history NEVER, current state YES.
8. **Schema versioning > one-off flag**: cgpro deferred `--allow-legacy-controller-reason` to cycle-8 R6.1c on the principle that one-off compat flags solve today's convenience but don't generalize. Schema versioning is the architectural answer (controller_decision payload `schema=v1_pre_allowlist_reason` vs `v2_allowlist_only`).
