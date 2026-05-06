# ADR-015 — `pipeline.py` decomposition into stage modules

**Status:** Proposed — 2026-05-04 (cycle-10 P9). Implementation deferred to cycle-11/12.
**Related:** cgpro_kimi_audit_response_20260504, cycle-10 plan
(`.claude/plans/2026-05-04-cycle-10-verified-runtime-release-preview.md`),
runtime-integrity-ledger.md (8 invariants).

## Context

`sage-python/src/sage/pipeline.py` is the canonical 5-stage cognitive
orchestration pipeline (CLASSIFY → DECOMPOSE → SELECT_TOPOLOGY → ASSIGN
→ EXECUTE → LEARN). At HEAD `f22a77a0` it has accumulated:

- **2983 lines** (144 KB).
- **44 `Any` type annotations** in the `CognitiveOrchestrationPipeline.__init__`
  signature alone — effectively constructor-injection-hell.
- **53 callables** (45+ private methods + 2 classes + 3 module-level
  functions).
- **At least 6 distinct concern surfaces**: routing, decomposition,
  topology, model assignment, execution, learning, plus runtime-event
  emission, bandit attribution, RunFrame coordination, FrugalGPT cascade,
  and a snapshot/restore singleton-mutation block flagged "B9 deferred"
  in code comments.

External audits (Kimi 2026-05-04 RAPPORT_FINAL, RAPPORT_AUDIT_ARCHITECTURAL)
correctly flagged this as a god-object pattern. cgpro 2026-05-04 round-2
review concurred that the file is "high-risk monolith" but recommended
deferring the actual decomposition past cycle-10 because the cycle-10
work focus is **release truth + adoption hardening**, and a multi-week
churn on the most sensitive runtime path should not happen alongside
those concerns.

This ADR captures the proposed module boundaries, the contracts that
must be preserved across the decomposition, and the characterization
tests required **before any code extraction begins**, so the cycle-11/12
implementation has a clear and reviewable target.

## Decision

Decompose `pipeline.py` into **six stage modules + three coordinator
modules** under a new package `sage.pipeline_v2/` (cycle-11) or as
direct replacements in `sage.pipeline/` (cycle-12, breaking import
path). The implementing cycle decides between additive (v2 alongside)
and replacing layout based on import-graph audit results.

### Proposed module boundaries

| Module | Source range (approx.) | Responsibility |
|---|---|---|
| `sage/pipeline_v2/classify.py` | `pipeline.py:1025-1138` (`_stage_classify`) | Stage 0: kNN router primary, Rust SystemRouter integrated, ContextualBandit attribution decision_id, ComplexityRouter Priority-3 fallback. |
| `sage/pipeline_v2/decompose.py` | `pipeline.py:1139-1180` (`_stage_decompose`, `_build_topology_from_hint`) | Stage 1: TaskPlanner.plan_auto → TaskDAG; compute_dag_features (omega/delta/gamma). |
| `sage/pipeline_v2/select_topology.py` | `pipeline.py:1181-1699` (`_stage_select_topology` + 12 helpers) | Stage 2: Rust TopologyEngine 6-path dispatch; macro-topology selection from DAGFeatures; cache + budget guard. |
| `sage/pipeline_v2/assign_models.py` | `pipeline.py:1700-1971` (`_stage_assign_models`, `_verify_assignment_formal`, log helpers) | Stage 3: Rust ModelAssigner per-node assignment + bandit quality override + Z3 verify (non-blocking). |
| `sage/pipeline_v2/execute.py` | `pipeline.py:2120-2695` (`_stage_execute` + AgentLoop bypass + topology runner glue) | Stage 4: bandit decision → AgentLoop bypass OR TopologyRunner; Fix C `_effective_controller` guard; FrugalGPT cascade. **HIGHEST CHURN** — must preserve B9 (per-run AgentLoop factory) work-around. |
| `sage/pipeline_v2/learn.py` | `pipeline.py:2696-end` (`_stage_learn`) | Stage 5: QualityEstimator → OracleStack → trainable gate → bandit.record_outcome_checked / MAP-Elites / online-evolution / training-memory. |
| `sage/pipeline_v2/context.py` | `pipeline.py:117-150` (`PipelineContext`) | Mutable state object passed between stages. **Becomes immutable per-stage clone** (cycle-12). |
| `sage/pipeline_v2/runtime_events.py` | `pipeline.py:413-646` (`_runtime_*` helpers, ~12 functions) | Topology-selected / model-assigned / final-status / RunFrame / control-surface event emission. |
| `sage/pipeline_v2/bandit_attribution.py` | `pipeline.py:430-473, 1972-2057` (`_emit_bandit_attribution_mismatch`, `_record_bandit_outcome_checked`, `_clear_bandit_decision`) | Stage-0 → Stage-5 bandit decision_id lifecycle (invariant 6 in runtime-integrity-ledger.md). |
| `sage/pipeline_v2/__init__.py` | reconstruction of `CognitiveOrchestrationPipeline` thin façade | Wires the 6 stages + 3 coordinators; preserves the public `run()` / `run_with_frame()` / `run_with_bench_evaluator()` entry points byte-identical. |

### Contracts that MUST be preserved

The decomposition must preserve **every** invariant in
`docs/contracts/runtime-integrity-ledger.md`. The 4 invariants directly
touched by `pipeline.py`:

1. **Event payload schema** (invariant 1): all `_runtime_emit_*` helpers
   route to the same `payload_schemas.py` validator. Cannot be split
   across modules without losing schema-version coupling.
2. **Oracle evidence** (invariant 2): `_stage_learn` consumes the
   OracleStack `verdict.trainable` flag and gates **all four**
   downstream side-effects (bandit / MAP-Elites / online-evolution /
   training-memory). The decomposition cannot let any of those four
   side-effects be reachable from a stage that did not see the gate.
3. **Bandit attribution** (invariant 6): the `decision_id` issued in
   `_stage_classify` must survive to `_stage_learn` and call
   `record_outcome_checked` (or `cancel`) exactly once per run. The
   `bandit_attribution.py` module above is the single owner of this
   lifecycle.
4. **Control-surface completeness** (invariant 8, cycle-9 P-cgpro-round-2):
   `_capture_control_surface` (in `bench/bigcodebench_bench.py`)
   reads from `pipeline.last_context` fields. The decomposition
   must keep these fields populated identically (`executed_template`,
   `selected_template`, `topology_id`, `dag_features`, `node_count`,
   `was_bypassed`).

Additional contracts:

- **`PipelineContext` field set** is a stable surface (consumed by
  bench, dashboards, observability). The split into `context.py` must
  preserve all 30+ fields verbatim. A field rename is a breaking change
  and requires a separate ADR.
- **`run()` / `run_with_frame()` / `run_with_bench_evaluator()`**
  signatures remain byte-identical. The thin façade `__init__.py`
  routes to the new modules.
- **`AgentLoop` singleton mutation block** (cycle-10 P6 will refactor
  this to a per-run factory). The decomposition must not regress
  whatever P6 ships. **P6 lands first**; P9 implementation cannot start
  until P6 is closed.
- **Fix C `_effective_controller`** (`pipeline.py:2467-2479`,
  `controller=None` when `tier=="budget"`). Must be preserved in
  `execute.py`.
- **FrugalGPT cascade** (`pipeline.py:2551-2629`). Stays in
  `execute.py`; cgpro flagged it as one of the surfaces Fix C does NOT
  disable, so the ablation matrix expects it reachable when guardrails
  are off.

## Characterization tests required BEFORE extraction

Before any code is moved, **lock the current behavior with golden
characterization tests**. These run against the un-refactored
`pipeline.py` and must pass byte-identically against `pipeline_v2/`
after extraction.

### Required golden tests (cycle-11 acceptance gate)

1. **`test_pipeline_v2_run_byte_identical.py`**: parametrize over a
   small fixture set (10 budget-tier BCB tasks, 5 reasoner-tier),
   compare:
   - Final `result` text (string-equal)
   - `PipelineContext.system / domain / topology_id /
     selected_template / executed_template / node_count / dag_features /
     assignments / pipeline_cost / pipeline_latency_ms`
   - Event ledger sequence (event_type list + ordering)
   - bandit decision_id (set comparison; one ID per run)
   Mock providers + AgentLoop with deterministic stubs so the test is
   reproducible.

2. **`test_pipeline_v2_oracle_gate_invariant.py`**: assert every
   side-effect (bandit.record_outcome_checked, MAP-Elites archive
   insert, online_evolution, training_memory.store) is **only** called
   when `OracleVerdict.trainable=True`. Test by injecting fixtures with
   `trainable=False` and asserting zero calls to the side-effect mocks.

3. **`test_pipeline_v2_bandit_attribution_invariant.py`**: per the
   ledger invariant 6, every `decision_id` issued in Stage 0 must
   produce exactly one `record_outcome_checked` OR `cancel` call by
   end of run, regardless of execution path (single-agent bypass,
   topology runner, FrugalGPT cascade, error fallback).

4. **`test_pipeline_v2_fix_c_budget_tier_no_controller.py`**:
   `_effective_controller` is `None` when `tier=="budget"`, regardless
   of whether `controller` was passed at construction.

5. **`test_pipeline_v2_control_surface_fields.py`**: post-run, the
   following fields are populated and non-empty when `node_count > 0`:
   `executed_template`, `selected_template`, `topology_id`,
   `dag_features.{omega,delta,gamma}`. (This is invariant 8 from
   cycle-9 cgpro round-2.)

These tests are themselves part of the cycle-11 PR. The current
`pipeline.py` should pass all 5 before the move begins.

### Implementation order (cycle-11 / cycle-12)

1. **Cycle-11 prep (1-2 days):**
   - Write the 5 characterization tests against current `pipeline.py`.
   - Verify all 5 pass at HEAD.
   - Run `cargo test` + `pytest` full suites green at HEAD.

2. **Cycle-11 phase 1 (additive, 3-5 days):**
   - Create `sage/pipeline_v2/` with the 9 modules above as
     **wrappers** that internally delegate to `pipeline.py` private
     methods (no logic move yet, just imports).
   - All 5 characterization tests pass against `pipeline_v2/`.

3. **Cycle-12 phase 2 (move, 5-7 days):**
   - Move private method bodies one stage at a time from `pipeline.py`
     into `pipeline_v2/<stage>.py`.
   - Each move is its own commit. Run characterization + Rust tests
     after each.
   - When all 6 stages moved, delete the original methods from
     `pipeline.py`.

4. **Cycle-12 phase 3 (cleanup, 1-2 days):**
   - Reduce 44 `Any` annotations in `__init__` by typing the actual
     dependencies. Each `Any` → real type triggers downstream type
     fixes.
   - Update `AI-ARCHITECTURE.md`, root README capability table, and
     `runtime-integrity-ledger.md` module cross-references.

Total budget: ~10-15 days across cycle-11 + cycle-12. **Do NOT do this
in cycle-10.** The cycle-10 budget went to P0/P1/P2/P3/P7 (release
truth + small crash hardening + state-truth alignment) — `pipeline.py`
churn alongside that would have created exactly the kind of compound
risk this ADR is written to avoid.

## Consequences

### Positive

- The ledger invariants (1, 2, 6, 8) become **explicit module
  boundaries** instead of implicit conventions in a 2983-line file.
  Each is unit-testable in isolation.
- The 44 `Any` annotations in `__init__` are forced to surface real
  types, which finds whatever silent contract drift has accumulated
  since cycle-7.
- New contributors can read one stage at a time. Currently the entry
  cost is "read 2983 lines or get nothing".
- Future ADRs (e.g. an immutable per-run context, ADR-016 candidate)
  become tractable because there's somewhere clean to put them.

### Negative

- **2-3 weeks of churn on the most sensitive runtime path.** Bench
  numbers must be re-validated post-refactor; if any drifts, the
  decomposition is the first suspect.
- The thin-façade `__init__.py` re-introduces an indirection layer.
  Stack traces gain one frame per stage. Acceptable given the
  testability gain.
- `bench/bigcodebench_bench.py:_capture_control_surface` reads private
  pipeline state (`pipeline.last_context.executed_template` etc.).
  The decomposition needs to keep that field set populated identically
  or the cycle-9 control-surface invariant breaks downstream. Mitigation:
  characterization test #5 above catches drift.

### Mitigations

- Cycle-11 phase 1 (additive) is reversible. If golden tests fail
  unexpectedly, revert is one commit.
- Each stage move is independent. Worst-case rollback is one stage.
- The 5 characterization tests are the gate. They run in CI on every
  cycle-11/12 PR. No PR merges if any of the 5 regresses.

## Alternatives Considered

### A. Burn-down rewrite (Sage-Lite, ~2000 LOC)
**Rejected.** Recommended by Kimi audit. Discards the cycle 5-9 runtime
integrity work (8 invariants, OracleStack, EvidenceProducers) which is
the differentiating asset per cgpro round-2 and ALIRE2.md. The
2983-line monolith is a problem; the architecture it implements is not.

### B. Leave `pipeline.py` as-is
**Rejected.** External audit visibility is now permanently elevated; a
3000-line god-object is the easiest finding for any future audit. The
maintenance cost of "2983 lines and 44 `Any`s" rises every cycle as
new invariants accrete.

### C. Decomposition in cycle-10 alongside truth-sync + B4 wheels
**Rejected (already, by this ADR's deferral).** Compound risk: any
post-decomposition bench drift would be confounded with claim-truth
realignment + adoption-path changes. Cycle-10 is "small, disciplined,
conservative" by cgpro+plan agreement. Cycle-11 is the right place.

## References

- cgpro consultation `cgpro_kimi_audit_response_20260504`, finaltext
  at `.tmp/cgpro_kimi_audit_finaltext.md`
- Cycle-10 plan: `.claude/plans/2026-05-04-cycle-10-verified-runtime-release-preview.md`
- Runtime integrity ledger: `docs/contracts/runtime-integrity-ledger.md`
- AI-ARCHITECTURE.md (rewritten 2026-05-04 at commit `ea9f7837`)
- Kimi audit corpus: `kimi/RAPPORT_FINAL_YGN_SAGE.md`,
  `kimi/RAPPORT_AUDIT_ARCHITECTURAL_YGN_SAGE.md`,
  `kimi/rapport_audit_sage_python.md`
- Advisor (Claude Sonnet 4.7) review notes per cycle-10 P0-P7

## Status changes

- 2026-05-04: Proposed (cycle-10 P9) — this document.
- 2026-05-05 (cycle-11): Accepted with characterization tests landed
  (37 P9 phase-1 tests covering byte-identical run + oracle gate +
  bandit attribution + Fix C + control-surface).
- 2026-05-05 (cycle-12 Phase A + Phase B): pipeline_v2/ scaffold
  shipped + 6 stage bodies (~2050 LOC) moved out of pipeline.py.
- **2026-05-06 (cycle-13 K Phase 2.1): Implemented — facade extraction
  with transition seams retained.** cgpro `cgpro_phase21_facade_rewrite_20260506`
  round-1 DESIGN_LOCKED + round-2 GO_STEP_B + round-3 GO_STEP_C amended +
  **round-4 OPTION_3** (the empirical 27-file `pipeline._stage_*` mock
  contract was incompatible with cgpro round-3's "delete delegators in
  C2" plan). Final Phase 2.1 acceptance amended:
    - pipeline.py 1800 LOC → ~702 LOC (1098 LOC migrated; **landing
      inside cgpro round-4 amended target 650-800 LOC**).
    - 5 stage bodies + orchestrator + 5 helper modules + PipelineContext
      dataclass all moved to pipeline_v2/.
    - 6 `_stage_*` methods + ~22 `_<helper>` delegator methods retained
      as transitional runtime test seams (the 27-file `pipeline._stage_*
      = <fake>` mock contract is unchanged).
    - 37/37 P9 phase-1 tests byte-identical at every commit.
    - PEP 562 `__getattr__` in `pipeline_v2/__init__.py` defers
      `from sage.pipeline import …` to attribute-access time,
      breaking the otherwise-circular dependency once
      `PipelineContext` source moved to `pipeline_v2/context.py`.
    - `PipelineContext.__module__ == "sage.pipeline"` preserved via
      explicit `setattr` in `pipeline_v2/context.py` so existing
      tests / bench / dashboard / observability assertions on the
      legacy module path continue to pass.
- **TBD (Phase 2.2 Proposed)**: rewrite the 27 test files that
  monkeypatch `pipeline._stage_<X> = <fake>`; replace the stage
  monkeypatch contract with module-function patching or
  public-effect assertions; remove the 6 `_stage_*` delegators;
  remove the ~22 `_<helper>` delegators where safe (separate grep
  pass); reach **`pipeline.py < 300 LOC`** (the original Phase 2.1
  cible reclassified to Phase 2.2 acceptance per cgpro round-4
  OPTION_3 verdict).
