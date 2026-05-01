---
name: April 30 cycle-8 closeout + cgpro architect review + cycle-9 strategy lock
description: Closeout phase before cycle-9 (commit 86681ac8). cgpro architect review accepted 6 reproches méthodologiques + locked cycle-9 strategy via Q1/Q2/Q3. Status JSON, runtime-integrity-ledger, rust-python-boundary docs shipped. Cycle-9 = budget tier paired ablation A14b route_integrated() repair + A2 N=10 + decision gate. A22/A8/A9/A10/A11/A13 confirmed shipped/stale. SWE-bench-Live = cycle-11 reproducibility lane.
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# Cycle-8 closeout + cgpro architect review + cycle-9 lock

**Final state**: Closeout shipped at commit `86681ac8`. cgpro architect review (saved at `.tmp/cgpro_architect_review_finaltext.md`, conv `cgpro_architect_review`) and cycle-9 strategy locks (`.tmp/cgpro_cycle9_strategy_finaltext.md`) drive cycle-9+ ordering.

## Stack cycle-8 final

```
86681ac8 docs(closeout,architecture): cycle-8 closeout + cgpro architect review locks
f9521616 fix(topology,boot,docs): cycle-8 step 2 A14 VERIFY round-1 fixes
6b2ebcbe feat(topology,boot): cycle-8 step 2 — A14 epoch fail-closed guard
49648263 docs(cycle7): cycle-8 R6.1c re-validation disclosure on N=50 evidence
9944674e fix(runtime/event_log,bench): R6.1c VERIFY round-1 fixes #1+#2
78565578 feat(runtime/event_log,bench): cycle-8 R6.1c — payload schema versioning
```

## Closeout deliverables (`86681ac8`)

| Artifact | Purpose |
|---|---|
| `scripts/status_snapshot.py` | Single source of truth for test counts. Runs `pytest --collect-only` + `cargo test --list`, writes `docs/status/current.json` with schema_version=v1 + git SHA + UTC timestamp. Modes: default write / `--print` / `--check` (CI tripwire). |
| `docs/status/current.json` | First canonical snapshot. **Reveals 3-doc inconsistency** cgpro Q-C flagged: 2887 Python (vs 2484-2501 claims, +400) / 544 Rust (vs 501-522, +43) / 100 sage-discover (vs 95, +5). |
| `docs/contracts/runtime-integrity-ledger.md` | 5-invariant cross-reference per cgpro Q-A (payload schema, oracle evidence, posterior epoch, contaminated backup, RunFrame summary). 4-column table per invariant: declared label / verified content / side-effect blocked / tests. **NO code refactor** — documentary only. |
| `docs/contracts/rust-python-boundary.md` | Rust↔Python ownership matrix per cgpro Q-B. Three "shim accretion" zones documented: graph_get_predecessors (Windows PyO3 workaround), TopologyController split-brain (Rust+Python orchestration), observability bridge (Rust spans → Python OTel). |
| `ALIRE3.md` | External advisory snapshot, committed with frontmatter explaining advisory-only status. Pointers to cgpro architect review as primary verdict. |
| **CLAUDE.md directive #9** | "Declared ≠ verified — runtime integrity principle". Crystallizes cross-cycle pattern cgpro identified across 3 cycle traps. Mandates registration in runtime-integrity-ledger + regression test before shipping any new "label gates side-effect" code path. |

## cgpro architect review key verdicts (33 KB, conv `cgpro_architect_review`)

### Methodological — 6 reproches ACCEPTED

| # | Reproche | New protocol |
|---|---|---|
| 1 | DESIGN misses principal trap | Add "Adversarial threat models" section: 3 scenarios + 1 test per scenario + expected pre-fix failure |
| 2 | Spec inflation | Scope budget per cycle, "no new subsystem unless..." gate |
| 3 | Don't challenge mis-framed prompts | cgpro will signal explicitly: "question mis-framed / stale premise / missing dependency" |
| 4 | Roundtrip cost ignored | All PUSH BACK classified `[blocking-now]` / `[parallel-safe]` / `[defer-to-round-N]` |
| 5 | No cross-cycle pattern matching | New principle: **Declared ≠ verified. Label ≠ provenance. Docs ≠ contract.** |
| 6 | Verbosity buries signal | Executive summary 5 lines max + then detail |

### Architectural — Q-A through Q-H

- **Q-A**: 4 invariant concepts ARE coherent (Runtime Integrity), but **NO physical refactor** in cycle-9 (would create churn without benchmark gain). Phase 1 = ledger doc only ✓ shipped. Phase 2 (v0.2) = re-export aliases. Phase 3 = Rust-first consolidation for stateful hot paths.
- **Q-B**: Direction healthy (Python → Rust, no reverse). Three shim zones to monitor.
- **Q-C**: Test counts inconsistent across docs. Need single status JSON ✓ shipped.
- **Q-D**: A22/A8/A9/A10/A11/A13 = **CLOSED/STALE in roadmap main**. Cycle-8 step 3 "A22 follow-ups" is WRONG. Real Tier 1: A14b, A2/A3, T2 minimal, B3+A18+A16+A19, B9, B4.
- **Q-E**: 6 cycles produced quality but NOT benchmark gains. Missing X = closed-loop benchmark-grounded learning attribution. Path 6 checkpoint stale, don't unpark yet.
- **Q-F**: 90% there: runtime spine. Last 10% hard: docs single-source, trace/replay, stable APIs. 30% there (multi-month): learned policy, full memory, public deployment, replication-grade harness.
- **Q-G**: Don't do mega `runtime/integrity/` refactor. Defer OTel luxuries. Defer Pareto Path B. Don't retrain Path 6 without evidence freeze.
- **Q-H**: New DESIGN template (9 sections), VERIFY format (executive summary first), Tier 1/2/3 classification, roundtrip budget.

### Strategic — Q1/Q2/Q3 (cycle-9 locks)

- **Q1**: option (c) staged. **Cycle-9 = budget tier paired ablation** (DeepSeek/Gemini Flash). Premium frontier (Opus 4.6 / current frontier at eval date) reporté Cycle-12+. **Don't hardcode any specific model** — baseline = "frontier current at eval date".
- **Q2**: **(γ) plain `route_integrated()` for A14b round-1**. NO Stage-0 scope creep (don't move kNN embedding). NO `_contextual` with empty context (untested implicit). `_contextual` = A14b.2 / Cycle-10 candidate IF cycle-9 N=10 shows contextual routing is the next bottleneck.
- **Q3**: A31 S-MMU Tier 2, A32 AdaptiveMutator Tier 2, **SWE-bench-Live Tier 1 strategic but Cycle-11 reproducibility lane** (NOT cycle-9). Cycle-9 must stay learning-loop/attribution focused.

## Cycle-9 DESIGN ask DISPATCHED 2026-04-30 (post-closeout)

Conv `cgpro_cycle9_a14b_design` (NEW conv per "fresh per ticket" pattern). BG ID `bbq1rozzs`. Prompt structure follows new 9-section template per cgpro Q-H methodology:
- §0 Executive summary (5 lines)
- §1 Goal (concrete pipeline.py:975 fix)
- §2 Scope budget (max 4 files, max 1 new concept, stop conditions)
- §3 Runtime invariant per directive #9 (declared decision_id ↔ verified executed arm)
- §4 **3 adversarial threat models** with required pre-fix failing tests : Stage-0 fallback bypass, reroute mid-run via quality cascade, multi-agent topology divergence
- §5 Contract matrix (4 specific Q to lock)
- §6 Tests-first plan
- §7 Ops/docs sync (ledger + routing README + roadmap)
- §8 Rollback/crash semantics (no bypass — A14b is non-bypassable)
- §9 Roundtrip budget (1+1max per cgpro Q-H)

**Critical codebase finding while drafting**: commit `1011b3ae` (2026-04-27) **already shipped Rust `route_integrated_contextual()`** with constraint-aware bandit cancellation, fresh decision_id semantics, and 3 unit tests. cgpro Q2 verdict (γ — plain `route_integrated()`) was given without knowing this. The plain `route_integrated()` (line 384 in system_router.rs) ALSO already uses bandit selection internally and returns decision_id, so cgpro's verdict still holds for round-1 closure. Just noting for cycle-10+ : `_contextual` is fully shipped, ready to wire when round-2 cycle-10 needs it.

## Cycle-9+ ordering (locked)

```
Cycle-9 — Learning attribution + benchmark loop (TIER 1)
  - A14b route_integrated() Stage-0 repair (γ option)
  - Minimal T2 memory write paths (top 2 paths affecting BCB/SWE)
  - A2 N=10 BCB-Hard budget-tier paired smoke
  - Decision gate: ≥35% pass@1 vs cycle-7 30% baseline → A3 N=50
                   [25%-35%] → diagnostic via trace evidence
                   <25% → rollback A14b

Cycle-10 — Tool policy / runtime safety bundle (TIER 1)
  - B3 ToolPolicy capability manifest
  - A18 dynamic tool validation fail-closed
  - A16 centralized redaction
  - A19 MCP/A2A authentication
  - (A14b.2 _contextual routing IF cycle-9 shows it's the bottleneck)

Cycle-11 — v0.2 public release lane + reproducibility (TIER 1)
  - B4 platform wheels
  - SWE-bench-Live smoke (replaces SWE-bench Lite as primary public gauge)
  - CI full matrix with wheel install
  - B2 trace+replay design (impl optional)
  - GAIA / AgentBench / τ-bench IF UI/tools/general-agent story is ready

Cycle-12+ — Path 6 / learned topology
  - Eval existing checkpoint vs current runtime
  - Compare template baseline vs learned policy
  - Retrain ONLY with clean post-A14/post-OracleStack evidence
  - Premium frontier challenge eval ("SAGE + frontier-at-eval-date" vs "frontier alone")
```

## Open follow-ups from architect review

- Tier 2 backlog (after Cycle-9 closes): A31 S-MMU cold-start gap, A32-followup AdaptiveMutator wiring, B2 durable trace+replay design, A14 output_schema risk-based rollout, B1.d FastAPI auto-instrument
- Tier 3 (defer): A12 Pareto Path B, B1.b.7 logfire-mode Rust export, B1.b.9 OTLP batch exporter, B1.c sage-discover instrumentation, B1.e sampler tuning
- Closed/Stale (REMOVED from active queue): A22/A8/A9/A10/A11/A13 (confirmed shipped in roadmap.md main)

## Web SOTA April 2026 (researched independently of cgpro)

Findings that informed Q1 decision:

- **SWE-bench Lite top**: Claude Opus 4.6 alone = **62.7%** (62 models evaluated; pricepertoken.com leaderboard 2026)
- **SWE-bench Pro Python**: SageAgent + Gemini 3 Flash = **59.0%** (Berkeley RDI March 2026 paper). Memory ablation = **+2.8pp only** (vs NoMem 56.2%) — memory is NOT the dominant lever; agent system / scaffold is.
- **The Conductor** (ICLR 2026, arXiv 2512.04388): 7B RL-trained orchestrator, SOTA LiveCodeBench + GPQA — direct Path 6 territory.
- **AgentConductor** (ICLR 2026, arXiv 2602.17100): Qwen2.5-3B + GRPO + Verl + vLLM topology evolution — direct Path 6 v2 territory (notre `sage-topology-policy-v2` Nemotron-Orchestrator-8B = même approche).
- **SWE-bench-Live** (NeurIPS 2025 D&B, microsoft/SWE-bench-Live): monthly contamination-free updates, 1565 tasks / 164 repos. Cgpro: "exactement aligné avec replication-grade benchmark suite".

**Strategic implication**: 6 cycles d'invariants runtime ont produit qualité/sécurité/auditabilité mais pas de gain benchmark. Le "X" missing = closed-loop benchmark-grounded learning attribution. Cycle-9 A14b is the FIRST step of that closure.

## Methodology lessons from this closeout phase

1. **Architect review > more cycles** when 3 cycles in a row have produced same-class trap. cgpro stepped back, named the pattern (Declared ≠ verified), classified the backlog, killed scope creep proactively. Worth ~2-3 cycles of avoided cycle-loop overhead.

2. **Codebase + web research in parallel** is the Claude+cgpro complementary force-multiplier. Claude found 2887 actual tests vs 2501 claim (codebase live), web confirmed Opus 4.6 = 62.7% SWE-bench Lite + memory = +3pp not +30pp. cgpro had to take both as input to lock cycle-9 strategy correctly.

3. **Mis-framed prompts cost roundtrips**. My own cycle-9 architect review prompt listed A22 as "open follow-up" — it had been shipped 4 days earlier. cgpro caught it (Reproche 3 in action). Lesson: verify backlog status against roadmap.md BEFORE asking cgpro.

4. **Stale documentation is an architectural hazard, not just doc debt**. Three docs (Dashboard, README, CLAUDE.md) had three different test counts. The +400 actual delta vs claims indicates the dashboards ARE NOT being maintained as single source of truth. status_snapshot.py + current.json + lint test (future cycle) closes this class.

5. **External advisory docs need explicit disposition**. ALIRE3.md was open in user's IDE, untracked, referenced as advisory but not committed. Without disposition note, mixing it with the primary cgpro architect review confused priorities. Now committed with `status: advisory` frontmatter.
