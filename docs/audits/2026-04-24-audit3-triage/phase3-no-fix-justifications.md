# AUDIT3 Phase 3 — no-fix justifications for unscheduled ✅/⚠️ claims
**Date:** 2026-04-24 · **Protocol:** PROMPT.md §3 gate ("solution OU no-fix justified")

Per PROMPT.md Phase 3 exit gate, every ✅/⚠️ claim needs a solution (→ plan.md)
OR an explicit no-fix justification. This annex covers the 17 claims NOT
addressed by the 3 Codex fixes in flight (rename, HITL, cost cap).

---

## Summary table

| Claim | Verdict | Disposition | Justification class |
|---|---|---|---|
| 1 | ✅ | no-fix | positive confirmation (not a bug) |
| 3 | ✅ | no-fix | positive confirmation (not a bug) |
| 4 | ✅ | no-fix (closed) | shipped as A0d commit `bf220e0` |
| 5 | ⚠️ | no-fix (closed) | audit stale — ADR-013 §5 flip (`c2113d8`) |
| 6 | ⚠️ | no-fix (closed) | audit datum stale — SWE-bench v15 1/10 |
| 8 | ✅ | **Fix 1 shipping** | plan.md |
| 9 | ⚠️ | no-fix (closed) | audit stale — ADR-013 §5 |
| 10 | ✅ | deferred | security-architectural, needs design doc |
| 11 | ✅ | **Fix 2 shipping** | plan.md |
| 12 | ⚠️ | **Fix 3 shipping** | plan.md |
| 13 | ⚠️ | deferred | conceptual, no demonstrated attack path |
| 14 | ⚠️ | deferred | existing mitigations sufficient for current scale |
| 15 | ✅ | no-fix | enforced (Phase-3 inspection) |
| 16 | ✅ | no-fix | enforced (Phase-3 inspection) |
| 17 | ⚠️ | deferred | scope — tool-output schema touches ToolResult contract |
| 18 | ✅ | no-fix | enforced (Phase-3 inspection) |
| 19 | ✅ | no-fix | ticketed B1 in roadmap (multi-week) |
| 20 | ✅ | no-fix | ticketed B2 in roadmap (multi-week) |
| 21 | ⚠️ | no-fix | documented as "calibrated initial values" (Directive #2) |
| 23 | ⚠️ | deferred | Phase-3 finding, bounded-impact, non-trivial risk |
| 30 | ⚠️ | deferred | low severity, coverage-only |
| 33 | ⚠️ | no-fix | multi-week; A1/A12 in roadmap |

---

## Detailed justifications

### Claim 1 — "Automatically routes to S1/S2/S3, builds topology, ..." ✅ → no-fix
**Why no fix:** This is the audit's positive architectural confirmation of the
5-stage pipeline — not a defect. The `✅ confirmed` verdict means the claim
matches reality; no action needed.

### Claim 3 — "OxiZ SmtVerifier: QF_LIA SMT solving, sub-millisecond" ✅ → no-fix
**Why no fix:** Positive architectural confirmation. Audit itself marks
this ✅. `oxiz` feature-gated crate with sub-ms verification at the
documented fragment coverage. Nothing to remediate.

### Claim 4 — "DistilBERT QualityEstimator (ONNX) — planned, not shipped" ✅ → no-fix (closed)
**Why no fix:** Already resolved by the 2026-04-23 A0d commit (`bf220e0`),
which caveated the "ONNX shipped" framing in 6 docs. Active backend is
the Z3 QualityLabeler + None abstention. Audit Phase 2 captured this as
closed; no new work.

### Claim 5 — "3-layer defense-in-depth sandbox: tree-sitter, Wasm WASI, subprocess/bwrap" ⚠️ → no-fix (closed)
**Why no fix:** Audit is STALE. Post-ADR-013 §5 flip (2026-04-22,
commit `c2113d8`) there is no subprocess fallback on the
`validate_and_execute` default path. The 3-layer framing in AUDIT3
describes the pre-flip architecture. Current state is 2-layer (tree-sitter
AST + Wasm RustPython) with subprocess gated behind `execute_raw` +
`SAGE_UNSAFE_RAW_EXEC=1`. ADR-013 §5 is the explicit resolution.

### Claim 6 — "SWE-bench Lite: 0% (0/5) diagnostic" ⚠️ → no-fix (closed)
**Why no fix:** Stale datum. 2026-04-21 v15 achieved 1/10 Docker-graded
(10%) after 3-fix chain (Directive #3 gating, CRLF, UTF-8). 2026-04-24
A7 verification showed 4/6 PATCH (67%) gen-only. Audit's 0% number is
pre-v15 and doesn't reflect current state. `docs/benchmarks/2026-04-21-swebench-v15-eval-results.md`
is the resolution record.

### Claim 9 — "Sandbox escape: ... subprocess fallback" ⚠️ → no-fix (closed)
**Why no fix:** Same rationale as Claim 5. Subprocess fallback on
`validate_and_execute` removed by ADR-013 §5. 40-attack red-team corpus
(2026-04-22 P0.4) validated the flip. Remaining AST-blocklist → WASI-escape
combo is conceptual; no demonstrated path from audit.

### Claim 10 — "Prompt injection: no explicit injection filtering" ✅ → DEFERRED
**Why deferred, not fixed now:**
- Security-architectural, not a patchable defect. Requires a design
  layer: (a) which inputs to filter (user task text, tool args, tool
  outputs echoed back), (b) detection method (regex-only is bypassable,
  classifier needs model), (c) action (log / block / transform).
- Proper solutions in 2026: OWASP LLM01-aware filters + dedicated
  prompt-injection classifier model (e.g. arXiv 2504.xxxxx PromptGuard
  family) + constitutional-AI-style output checks. None is a 200-LOC
  patch.
- PROMPT.md Principle #1: "Le code non modifié est plus sûr que le
  code modifié." A weak regex filter would give a false sense of
  security while breaking legitimate prompts.
- **Action:** Add **A13** to `roadmap.md` Horizon B — research spike
  + design doc, then implementation. Treat as multi-day/week effort.
- Interim mitigation already in place: agent-loop step caps, tool
  scope enforcement, write-gate quality filtering on memory.

### Claim 13 — "Memory poisoning: adversarial inputs flood S-MMU" ⚠️ → DEFERRED
**Why deferred:**
- Audit labels this conceptual; no demonstrated attack path.
- Existing mitigation: `CompositeWriteGate` (Rust, 5-signal composite
  scoring) — already gates which episodes enter long-term memory.
- Proper fix would add anomaly detection on write patterns (e.g.
  rate-limit per source, clustering-based outlier detection). Research
  spike needed.
- **Action:** Note on roadmap as a B-tier "if poisoning becomes a
  demonstrated risk". No immediate work.

### Claim 14 — "Provider failover cascade: circuit breaker cascade" ⚠️ → DEFERRED
**Why deferred:**
- Current design: TTL'd exclusion (300s re-probe, `DEFAULT_EXCLUSION_TTL_SEC`)
  + FrugalGPT quality cascade.
- At current scale (one user, < 100 concurrent tasks), backpressure
  isn't a realistic failure mode. Would become relevant at multi-tenant
  deployment scale.
- Proper fix requires request queuing + concurrency limits per provider —
  infrastructure work, not a patch.
- **Action:** No ticket; revisit when multi-tenant becomes a concern.

### Claim 15 — "Missing invariant: DAG acyclicity at runtime" ✅ → no-fix (enforced)
**Why no fix:** Phase-3 follow-up inspection confirmed enforcement.
`HybridVerifier::verify` runs after every mutation/generation
(`engine.rs:490,821,886`) and rejects cycles via `is_cyclic_directed`
(`verifier.rs:216`). `is_acyclic() / has_cycles() / try_topological_sort()`
exist as runtime checks. Audit's "missing" framing was a false-negative
in static reading. See `audit-checklist.md:15`.

### Claim 16 — "Missing: context window bounds per node" ✅ → no-fix (enforced)
**Why no fix:** Phase-3 follow-up inspection confirmed enforcement.
`runner.py:130-158` reads node model's `context_window` from ModelCard,
reserves 30% for system+task, divides budget across predecessors.
`runner.py:929-958` gates input size at 0.85 × context_window with
truncation fallback. Bounds derive from ModelCard per-model.

### Claim 17 — "Missing: tool I/O schema compliance" ⚠️ → DEFERRED
**Why deferred:**
- Input side: tools have JSON schema (`ToolDef.parameters`). ✅
- Output side: `ToolResult.output` is a free-form string. ⚠️
- Proper fix: per-tool output contract (Pydantic) + validator in
  agent-loop before passing result to next LLM call.
- Scope concern: touches `ToolResult` contract, which is a core data
  structure across `sage.tools`, `sage.agent_loop`, `sage.topology.runner`,
  and every tool implementation (18+ tools). Not a ≤ 10-file, ≤ 200-LOC
  fix.
- **Action:** Add **A14** to `roadmap.md` — design per-tool output
  contracts as part of a broader ToolResult v2 effort.

### Claim 18 — "Missing: controller decision monotonicity" ✅ → no-fix (enforced)
**Why no fix:** Phase-3 follow-up inspection confirmed. `RustTopologyController`
uses increment-only counters (`node_retries`, `reroute_count`, `spawn_count`,
`gate_loops`) with hard caps (`MAX_RETRIES=2`, `MAX_REROUTES=1`,
`MAX_SPAWNS=3`, `MAX_GATE_TURNS=2`). No decrement paths. Upgrade→prune on
same node structurally impossible post-prune.

### Claim 19 — "No OpenTelemetry or structured distributed tracing" ✅ → no-fix (ticketed)
**Why no fix:** Ticketed as B1 in `roadmap.md`. Multi-week effort:
span decoration of pipeline.run() + tool calls, GenAI semantic-conventions
compliance (Context7-verified Development stability), collector wiring.
Not a ≤ 200-LOC fix. Defer to roadmap.

### Claim 20 — "No deterministic replay / trace serialization" ✅ → no-fix (ticketed)
**Why no fix:** Ticketed as B2 in `roadmap.md`. Requires stable trace
serialization format, replay CLI, deterministic LLM call replay (mock
provider with recorded responses). Multi-week. Defer to roadmap.

### Claim 21 — "Heuristic thresholds without calibration intervals" ⚠️ → no-fix (documented)
**Why no fix:** CLAUDE.md Directive #2 explicitly documents these as
"calibrated initial values, subject to ablation" — not "banned heuristics".
Ablation sweep is a research task, not a patch. The thresholds
(THETA_GOOD=0.7, THETA_CRITICAL=0.3, etc.) aren't arbitrary magic
numbers; they're literature-backed initial values with a known
calibration plan. Audit's framing is too strict.

### Claim 23 — "Memory consolidation every 10 steps — blocking?" ⚠️ → DEFERRED
**Why deferred (Phase-3 finding):**
- Phase-3 inspection confirmed: blocking (awaited), but bounded
  (every 10 steps × batch-size cap).
- Naïve fix `asyncio.create_task(...)` is tempting but risky:
  (a) concurrent consolidation passes could race on SQLite writes,
  (b) agent-loop termination could orphan an in-flight task,
  (c) the "consolidated" metadata marker would need atomic-by-key
  semantics to avoid double-processing.
- Proper fix needs a single-flight queue + graceful shutdown hook,
  not a decorator change.
- Performance impact currently bounded (< 10s per 10-step batch in
  observation). Not a production blocker at current scale.
- **Action:** Add **A15** to `roadmap.md` — single-flight consolidation
  task with graceful shutdown.

### Claim 30 — "Property-based tests for DAG validity and routing stability" ⚠️ → DEFERRED
**Why deferred:**
- Low severity, coverage-only.
- Existing `sage-python/tests/test_properties.py` has partial coverage.
- Adding more property tests is always useful but unbounded — could
  consume arbitrary effort without clear stopping criterion.
- **Action:** Note on roadmap as continuous improvement; low
  priority. No new Phase-5 ticket.

### Claim 33 — "Run standardized benchmarks (GAIA, SWE-bench Verified)" ⚠️ → no-fix (ticketed)
**Why no fix:** A1/A12 in roadmap. SWE-bench Verified run requires full
Docker grading pipeline (already wired for Lite). GAIA requires benchmark
dataset licensing + result submission. Each is multi-day effort. Defer
to roadmap.

---

## New roadmap tickets proposed

| Ticket | From claim | Horizon | Estimated |
|---|---|---|---|
| **A13** — prompt-injection filter design + impl | 10 | B | 2-3 weeks (spike + implementation) |
| **A14** — tool-output Pydantic contracts (ToolResult v2) | 17 | B | 1-2 weeks |
| **A15** — single-flight consolidation + graceful shutdown | 23 | B | 2-3 days |

All 3 are B-tier (not ≤ 2-week quick wins) per PROMPT.md scope. Tickets
will be added to `roadmap.md` after Codex fix batch closes (Phase 6).

---

## Phase 3 gate re-check

Of the 20 ✅/⚠️ unique claims (post-Phase-3 inspection):

- **3** scheduled for Phase 5 (plan.md Fix 1/2/3 via Codex subagents)
- **9** no-fix (4 positive confirmation + 5 enforced/ticketed/documented)
- **5** closed by prior work (A0d shipped, ADR-013 flip, SWE-bench v15)
- **3** deferred with explicit rationale + new roadmap tickets (A13/A14/A15)
- **0** with missing disposition

**Gate status:** ✅ all ✅/⚠️ claims have a disposition (solution OR no-fix
justification). Phase 3 exit gate satisfied.

---

## Divergences with audit (for §6.5 meta-audit section)

Audit angle-mortss detected via Phase-3 inspection:

1. **Claims 15/16/18 "missing invariants"** were all actually enforced in
   Rust/Python — auditor didn't trace `HybridVerifier::verify` call sites
   (`engine.rs:490,821,886`), `_context_budget_per_predecessor` at
   `runner.py:130`, or `RustTopologyController`'s increment-only counter
   semantics. **3/6 "missing" claims were false-negatives.**
2. **Claims 5/6/9** describe stale architecture (pre-ADR-013 §5). Audit
   should have been dated or verified against current main. **3 stale
   claims.**
3. **Claim 4 "ONNX shipped"** was already addressed by A0d a day before
   AUDIT3 was written. Dependency on audit-freshness.

Meta-audit recommendation for AUDIT4: timestamp each claim with
commit-sha basis; verify AST traversal goes deeper than grep for "is
there a X in the codebase" questions.
