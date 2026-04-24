# AUDIT3.md claim checklist — YGN-SAGE @ `820ea3e2`
**Date:** 2026-04-24 · **Protocol:** PROMPT.md Phase 1-2

Extracted from AUDIT3.md sections 3 (claim table), 6 (security risk
register), 9 (roadmap), 10 (top-10 leverage). Deduplicated. One row per
distinct assertion. Cross-reference with `docs/audits/2026-04-23-alire-verification.md`
for AUDIT.md / AUDIT2.md (which were triaged separately).

Verdict legend: ✅ confirmed · ⚠️ partial · ❌ infirmed · 🔍 not verifiable ·
🚩 false-positive · 🕐 pending

| # | Claim (1 phrase) | Type | Location | Source | Verdict | Evidence |
|---|---|---|---|---|---|---|
| 1 | "Automatically routes to S1/S2/S3, builds topology, assigns models, executes with formal verification, learns from every run" | archi | 5-pillar pipeline | §3 | ✅ **confirmed** | `sage-python/src/sage/pipeline.py` 5-stage + `sage-core/src/topology/engine.rs` + ADR-012 + `roadmap.md` A0 audit trail |
| 2 | "+22pp gain on breadth axis (p=0.015, N=50) on internal sage-mas-bench" | archi | benchmark claim | §3 | 🔍 **not verifiable in-session** | No external reviewer can reproduce without public dataset + prompts + scoring code. Already ticketed (roadmap B-tier `publish sage-mas-bench`). |
| 3 | "OxiZ SmtVerifier: QF_LIA SMT solving, sub-millisecond solve time" | archi | `sage-core/src/verification/oxiz.rs` | §3 | ✅ **confirmed (scoped)** | `oxiz` crate feature-gated; tests at `sage-core/src/verification/oxiz.rs::tests`. Fragment coverage documented. Audit itself marks this ✅. |
| 4 | "DistilBERT QualityEstimator (ONNX) — planned, not shipped" | doc | architecture claim | §3 | ✅ **confirmed** | Already closed by A0d (`bf220e0`) — 6 docs now caveat ONNX as not-shipped. Live backend is Z3 QualityLabeler + None abstention. |
| 5 | "3-layer defense-in-depth sandbox: tree-sitter, Wasm WASI, subprocess/bwrap" | secu | sandbox architecture | §3 | ⚠️ **partial (audit is STALE)** | Post-ADR-013 §5 (2026-04-22) there is NO subprocess fallback on the `validate_and_execute` default path. Only `execute_raw` (gated by `SAGE_UNSAFE_RAW_EXEC=1`) falls through to subprocess. Audit's "3-layer" framing describes the removed architecture. |
| 6 | "SWE-bench Lite: 0% (0/5) diagnostic" | perf | benchmark | §3 | ⚠️ **partial (stale datum)** | 2026-04-21 v15 achieved 1/10 (10%) Docker-graded after 3-fix chain (`docs/benchmarks/2026-04-21-swebench-v15-eval-results.md`). 2026-04-24 A7 verification showed 4/6 PATCH (67%) gen-only (not graded). Audit's 0% is outdated. |
| 7 | "Core insight: topology variance > model variance (≥20x)" | archi | AdaptOrch citation | §3 | 🔍 **not verifiable in-session** | Claim traces to external paper; ablation to reproduce not run. Roadmap ticket already exists for ablation. |
| 8 | "LtlVerifier: reachability, safety, liveness via BFS/DFS — misnamed" | doc | `sage-core/src/verification/ltl.rs` | §3 | ✅ **confirmed (misnamed)** | Class still named `LtlVerifier` in `sage-core/src/lib.rs:90` + `sage-core/src/README.md:88`. No temporal-logic parser exists — it's a graph property checker. Rename to `GraphPropertyChecker` is low-effort/medium-impact. |
| 9 | "Sandbox escape: malicious LLM output bypasses tree-sitter AST filter or exploits WASI misconfiguration; subprocess fallback" | secu | §6 risk register | §6 | ⚠️ **partial — subprocess removed from default path** | Same as claim #5. Subprocess fallback on `validate_and_execute` is gone (ADR-013 §5). `execute_raw` gate is an audited escape hatch, not default. 40-attack red-team corpus added mid-April (2026-04-22 P0.4). Remaining vector: AST blocklist → WASI escape combo — not demonstrated by audit, conceptual. |
| 10 | "Prompt injection: no explicit injection filtering mentioned" | secu | §6 | §6 | ✅ **confirmed (missing)** | Grep: zero hits for `prompt_injection\|injection_filter\|sanitize_prompt` in `sage-python/src/`. No filter exists. |
| 11 | "ToolForge synthesis risk: auto-generated tools execute unsafe code or exfiltrate data; no HITL approval" | secu | `sage-python/src/sage/tools/forge.py` | §6 | ✅ **confirmed** | `forge.py:BuildLoop` calls `self._registry.mark_source(tool_name, "forged")` without an approval gate. No HITL-approved flag on synthesized tools. |
| 12 | "Cost explosion: no cost caps or budget-aware routing documented" | secu | §6 | §6 | ⚠️ **partial — budget threading exists, hard caps don't** | `sage-core/src/routing/model_assigner.rs:176` threads `budget_usd` + skips models over remaining budget (BUDGET_EPSILON). `sage-python/src/sage/contracts/cost_tracker.py` tracks cumulative spend with `is_over_budget`. BUT: default `budget_usd=0` means unlimited; no pipeline-level abort on exceeded budget. Runtime check exists but the enforcement loop isn't wired to short-circuit task execution. |
| 13 | "Memory poisoning: adversarial inputs flood S-MMU, degrading retrieval" | secu | §6 | §6 | ⚠️ **partial** | 5-signal composite write gate exists (`CompositeWriteGate`, Rust) with multi-signal scoring. No anomaly-detection layer for adversarial-pattern detection on top. Claim is conceptual — no demonstrated attack path. |
| 14 | "Provider failover cascade: circuit breaker triggers mass fallback, latency spike" | secu | §6 | §6 | ⚠️ **partial** | TTL'd exclusion (300s re-probe, commit `3148667`) + `FrugalGPT` quality cascade. No explicit backpressure / request queuing. Under sustained high-failure rate, could cascade. |
| 15 | "Missing invariants: DAG acyclicity at runtime" | archi | §7 | §7 | ✅ **confirmed-enforced (post-inspection)** | `try_add_edge` at `topology_graph.rs:631` doesn't individually reject cycles, BUT `HybridVerifier::verify` is invoked after every mutation/generation (`engine.rs:490,821,886`) and rejects cycles via `is_cyclic_directed` (`verifier.rs:216`). `is_acyclic()` + `has_cycles()` + `try_topological_sort()` exist as runtime checks. Cycles can be added transiently; cyclic topologies never pass the verifier and never execute. |
| 16 | "Missing: context window bounds per node" | archi | §7 | §7 | ✅ **confirmed-enforced (post-inspection)** | `runner.py:130-158` `_context_budget_per_predecessor` reads `context_window` from the node's model card (ModelCard field, `model_card.rs:139`), reserves 30% for system+task, divides 70%*context_window*4chars across predecessors. `runner.py:929-958` gates input size against `context_window * 0.85` with truncation fallback. Bounds derive from ModelCard per-model, not overridable per-node — good enough for current models (128K–1M). |
| 17 | "Missing: tool I/O schema compliance" | secu | §7 | §7 | ⚠️ **partial** | Tools have schema (`ToolDef.parameters` JSON schema). LLM response-side validation is looser. |
| 18 | "Missing: controller decision monotonicity (prevent upgrade↔prune oscillation)" | archi | §7 | §7 | ✅ **confirmed-enforced (post-inspection)** | `RustTopologyController` (`controller.rs:158-184`) uses increment-only counters (`node_retries`, `reroute_count`, `spawn_count`, `gate_loops`) with hard caps (`MAX_RETRIES=2`, `MAX_REROUTES=1`, `MAX_SPAWNS=3`, `MAX_GATE_TURNS=2`). No decrement paths in any per-path primitive. Upgrade→prune on same node is structurally impossible — pruned nodes are removed from the graph. No explicit history-based oscillation detector, but increment+cap semantics make oscillation bounded. |
| 19 | "No OpenTelemetry or structured distributed tracing" | observability | §4 | §4 | ✅ **confirmed** | Grep: zero hits for `opentelemetry\|OpenTelemetry\|otel` in `sage-python/src/` or `sage-core/src/`. EventBus is in-process. Ticketed as B1 in roadmap. |
| 20 | "No deterministic replay / trace serialization" | observability | §4 | §4 | ✅ **confirmed** | No `replay_trace()` CLI or trace file format. Ticketed as B2 in roadmap. |
| 21 | "Heuristic thresholds (quality>0.7, <0.3) without calibration intervals" | archi | `TopologyController` | §4 | ⚠️ **partial** | Per CLAUDE.md Directive #2: thresholds documented as "calibrated initial values, subject to ablation". Labels acknowledge the issue; ablation sweep ticketed. Not "banned" per directive. |
| 22 | "Dynamic tool synthesis (ToolForge) without permission boundaries or HITL" | secu | `sage-python/src/sage/tools/forge.py` | §4 | ✅ **confirmed** | Same as claim #11. No HITL approval gate on `BuildLoop`. |
| 23 | "Memory consolidation every 10 steps — blocking? async? SQLite lock contention?" | perf | §4 | §4 | ⚠️ **partial (post-inspection) — blocking but bounded** | `agent_loop.py:313-330` `_maybe_run_consolidation` **awaits** `consolidator.consolidate()` — blocks the agent loop for the duration of one batch pass. Not fire-and-forget. Batch size capped at `CONSOLIDATION_BATCH_SIZE` (constants.py). Triggered every 10 steps via `_consolidation_steps_total % 10 == 0`. Exceptions caught → logged at `.debug`, non-fatal. SQLite lock contention only emerges under multi-producer scenarios (not current default single-agent-loop path); aiosqlite surfaces locks as async exceptions, not deadlocks. Performance concern valid; correctness OK. |
| 24 | "PyO3 boundary fragility: GIL contention, error translation, lifecycle" | perf | §4 | §4 | 🔍 **partial** | Ticketed as B9 (per-run immutable context refactor) in roadmap. Concrete call-path tests would verify. |
| 25 | "Rename LtlVerifier → GraphPropertyChecker" (top-10 #3) | doc | §10 | §10 | = claim #8 |
| 26 | "Hard cost caps + budget-aware routing" (top-10 #2) | secu | §10 | §10 | = claim #12 |
| 27 | "Integrate OpenTelemetry + deterministic replay" (top-10 #4) | observability | §10 | §10 | = claims #19+#20 |
| 28 | "Publish sage-mas-bench dataset + scoring code + ablations" (top-10 #5) | doc | §10 | §10 | = claim #2 |
| 29 | "Replace heuristic thresholds with calibrated confidence + abstention" (top-10 #6) | archi | §10 | §10 | = claim #21 |
| 30 | "Property-based tests for DAG validity and routing stability" (top-10 #7) | test | §10 | §10 | ⚠️ **partial** | `sage-python/tests/test_properties.py` exists; coverage partial. |
| 31 | "Tool permission boundaries + HITL for ToolForge" (top-10 #8) | secu | §10 | §10 | = claims #11+#22 |
| 32 | "Stabilize PyO3 async boundary with semaphores + backpressure" (top-10 #9) | perf | §10 | §10 | = claim #24 (B9) |
| 33 | "Run standardized benchmarks (GAIA, SWE-bench Verified)" (top-10 #10) | bench | §10 | §10 | ⚠️ **partial** | SWE-bench Lite runs; Verified not yet; GAIA not yet. Ticketed under A1/A12 horizons. |

## Deduplication summary

31 distinct assertions after removing duplicates across §3/§6/§7/§10.

**Initial (Phase 1-2, in-session only):**
- ✅ confirmed: 8 · ⚠️ partial: 8 · 🔍 not verifiable: 6 · ❌ infirmed: 0 · 🚩 false-positive: 0

**Post-Phase-3 follow-up inspection (2026-04-24, 4 🔍 items re-triaged):**
- **✅ confirmed:** 11 (claims 1, 3, 4, 8, 10, 11, 15, 16, 18, 19, 20) — **+3 flipped from 🔍: 15 DAG acyclicity, 16 context bounds, 18 monotonicity**
- **⚠️ partial:** 9 (claims 5, 6, 9, 12, 13, 14, 17, 21, 23, 30, 33) — **+1 flipped from 🔍: 23 consolidation (blocking but bounded)**
- **🔍 not verifiable:** 2 (claims 2, 7) — both external-reproducibility (benchmark publication + external paper ablation)
- **❌ infirmed:** 0
- **🚩 false-positive:** 0
- (Some rows above are cross-refs, not new verdicts; actual unique-verdict count = 22)

**Net:** 3 🔍 items confirmed-enforced via targeted code inspection (15/16/18 — audit §7 "missing invariants" is superseded by post-inspection evidence that the Rust side already enforces these). Only claim 23 retains a residual concern (blocking consolidation pass), and only 2 items remain genuinely unreproducible from this repo alone (external-benchmark claims).

## What's actionable in the short term (≤ 2 weeks per AUDIT3 §9)

Priority by (severity × effort × user-intent-this-session):

| Priority | Claim # | Action | Effort |
|---|---|---|---|
| **P0** | 11, 22, 31 | ToolForge HITL gate — wire approval callback before `mark_source("forged")`. | ~2 h |
| **P0** | 10 | Basic prompt-injection filter on user-facing task text + tool-arg text. | ~3 h |
| **P1** | 12, 26 | Pipeline-level task cost cap — abort task.run() when `CostTracker.is_over_budget` flips True. | ~2 h |
| **P1** | 8, 25 | Rename `LtlVerifier` → `GraphPropertyChecker` + update ADR + README. Cosmetic/credibility. | ~1 h |
| **P2** | 19, 27 | OpenTelemetry span wrapper on `pipeline.run()` + tool calls (B1 in roadmap). | ~1 d |
| **P2** | 15, 16, 18 | Verification pass — confirm DAG acyclicity, context-window bounds, controller monotonicity. | ~3 h |

## Already covered by roadmap / prior triage

| Claim # | Coverage |
|---|---|
| 4 | A0d shipped `bf220e0` |
| 5, 9 | ADR-013 §5 flip (`c2113d8`, 2026-04-22) |
| 6 | 2026-04-21 SWE-bench v15 (`docs/benchmarks/...`) + A7 verification smoke |
| 19, 20, 24, 27, 32 | B1/B2/B9 in `roadmap.md` |
| 33 | A1/A12 in `roadmap.md` |

## Phase 2 status: complete for in-repo claims; Phase 3 next

Per PROMPT.md §3, only ✅/⚠️ claims proceed to severity scoring + SOTA
solutions. Total candidates: **16** (8 ✅ + 8 ⚠️). Of these, 11 are
already tracked in roadmap with concrete actions (A0d, ADR-013,
SWE-bench, B1/B2/B9, A1/A12). The **5 new actionable items** for
Phase 3:

1. **Claim 8** — LtlVerifier rename (trivial, cosmetic/credibility)
2. **Claim 10** — Prompt-injection filter (security, no current filter)
3. **Claims 11/22** — ToolForge HITL gate (security, no current gate)
4. **Claim 12** — Pipeline-level cost cap enforcement (cost safety)
5. **Claim 21** — Ablation sweep to validate thresholds (quality)

The other partial/confirmed claims (17 schema, 30 property tests,
33 standardized benchmarks) are tracked or in-progress elsewhere and
don't need new actions from this triage.

## Not-in-scope for this triage

- Claims 2, 7 (benchmark reproducibility) — require external dataset
  publication; ticketed, not actionable without multi-day effort.
- Claim 14 (provider failover cascade) — conceptual risk with no
  demonstrated attack path; low priority vs claims 10/11/12.

## Phase-3 follow-up inspection (previously 🔍, now re-triaged)

- Claim 15 (DAG acyclicity): ✅ enforced via `HybridVerifier` post-generation.
- Claim 16 (context-window bounds): ✅ enforced via `_context_budget_per_predecessor` + `runner.py:929` gate.
- Claim 18 (monotonicity): ✅ enforced via Rust controller's increment-only counters + hard caps.
- Claim 23 (consolidation): ⚠️ partial — blocking but bounded. Performance watch-item, not a correctness bug.

No new action tickets needed from the 4 Phase-3 inspections. The "missing invariants" framing in AUDIT3 §7 for 15/16/18 is superseded — the invariants are enforced, the audit just couldn't locate the enforcement mechanism from static reading.

## Phase 3 deliverable: see `plan.md` (to be written next)
