# ADR-017 — Pipeline façade compatibility-method retirement

**Status:** Implemented — cycle-13 K Phase 2.2 closed 2026-05-07.
**Related:** ADR-015 (the meta-ADR for the `pipeline.py` decomposition;
shipped + closed in cycle-13 K Phase 2.1 `2026-05-06` then Phase 2.2  <!-- narrative-guard: allow historical-record -->
`2026-05-07`); ADR-016 (the AgentLoop bypass factory; cycle-12 closure);
runtime-integrity-ledger.md (10 invariants); cgpro
`cgpro_phase22_test_rewrite_20260506` (DESIGN_LOCK + Stage A → F VERIFY
rounds, 2026-05-06 / 2026-05-07).

## Context

ADR-015 cycle-13 K Phase 2.1 closure (HEAD `96155232`, 2026-05-06)  <!-- narrative-guard: allow historical-record -->
shipped the public façade with the runtime extracted to
`sage.pipeline_v2`, but kept on the public class
`CognitiveOrchestrationPipeline`:

  - 6 private compatibility methods named `_stage_<X>(ctx)` that
    forwarded to the corresponding module function in
    `sage.pipeline_v2.<X>` (one per pipeline stage).
  - ~22 private compatibility methods named `_<helper>(...)` that
    forwarded to the corresponding helper module function in
    `sage.pipeline_v2.bandit_attribution / runtime_events /
    memory_gate / topology_helpers / costing`.

These were retained for one cycle as a temporary instance patch
surface: 27 existing test files patched the public class with
`pipeline._stage_<X> = <fake>` to intercept stage execution, and a
similar inventory of tests patched the helper compatibility methods.
Phase 2.1 round-4 `OPTION_3` reclassified the seam removal +  <!-- narrative-guard: allow historical-record -->
the original `pipeline.py < 300 LOC` target to a follow-up cycle so
the test rewrite could be DESIGN-locked separately.

## Decision

Cycle-13 K Phase 2.2 retires the six private compatibility methods
named after the pipeline stages, the ~22 helper compatibility methods,
and the inline `__init__` body, and moves all 27 test files onto a
permanent module-function patching contract.

### Scope

  - 6 compatibility methods named `_stage_<X>` removed from
    `CognitiveOrchestrationPipeline` (Stage C atomic, commit
    `10e38931`). Production callsites + 27 test files retargeted to
    call `sage.pipeline_v2.<X>.<X>(pipeline, ctx, ...)` directly.
  - 36 compatibility methods named `_<helper>` removed from the
    public class (Stage D2 atomic, commit `6f0b2606`). Production
    callsites + tests retargeted to the helper modules under
    `sage.pipeline_v2`.
  - The `__init__` body moved to
    `sage.pipeline_v2.constructor.initialize_pipeline()` (Stage D3,
    commit `970c451c`); the public `__init__` signature is
    preserved unchanged.
  - `pipeline.py` reduced to ≤ 293 raw lines at Stage F closure
    (Stage D4 → 290 LOC at HEAD `17ee3c38`; Stage F restored mypy 0
    by adding class-body annotations + LOC discipline). HARD GATE
    `< 300 LOC` met.

### Permanent patching strategy (replaces the retired surface)

  - Stage bodies — patch the owning module function:
    `monkeypatch.setattr("sage.pipeline_v2.<stage>.<stage>", _fake)`
    where `<stage>` is one of `classify`, `decompose`,
    `select_topology`, `assign_models`, `execute`, `learn`.
  - Helper bodies — patch the owning helper module function under
    `sage.pipeline_v2.<helper_module>`. The 36 retired methods map
    1:1 onto module functions in `bandit_attribution.py`,
    `runtime_events.py`, `memory_gate.py`, `topology_helpers.py`,
    or `costing.py`.
  - `_emit` remains on the public class as the EventBus boundary
    (Q3a lock — stateful seam preserved, 1-line method form).
  - `_run_internal` remains as the private façade method into
    `sage.pipeline_v2.orchestrator.run_internal`.
  - `sage.pipeline._new_runtime_run_id` and `sage.pipeline.time.monotonic`
    remain public monkeypatch surfaces (Q6 string-monkeypatch
    contract preserved for run-frame tests; the orchestrator
    resolves these dynamically through `pipeline_mod.<X>` so the
    tests' monkeypatches still fire).

## Outcome

  - 27 test files rewritten to module-function patching or
    public-effect assertions.
  - 6 private stage compatibility methods removed.
  - 36 private helper compatibility methods removed.
  - Constructor initialization moved to
    `sage.pipeline_v2.constructor.initialize_pipeline()`.
  - `pipeline.py` reduced below 300 raw lines (293 LOC at Stage F
    closure).
  - Deletion contracts: 58 PASS (RED at Stage A, GREEN after
    Stage C + Stage D2).
  - P9 invariant set: 42/42 byte-identical preserved at every
    Stage B / C / D / E / F commit. (Phase 2.1 originally reported  <!-- narrative-guard: allow historical-record -->
    37/37; Phase 2.2 canonicalized the current set at 42/42 after
    the test rewrites added 5 invariant tests.)
  - Stage F follow-up: the constructor extraction at Stage D3 was
    found during the E3 final gates to have removed mypy-visible
    instance attribute initialization from `pipeline.py`. Stage F
    restored mypy 0 by adding class-body instance attribute
    declarations for the 31 attributes initialized by
    `pipeline_v2.constructor.initialize_pipeline`. Pure type-system
    change; runtime behavior unchanged.

## Consequences

### Positive

  - Tests now patch explicit module ownership instead of mutating
    private compatibility methods on the public class. The public
    runtime API surface is reduced.
  - The 27-file empirical patching contract is now uniform: one
    module function per stage or helper, patched with
    `monkeypatch.setattr("sage.pipeline_v2.<module>.<function>", _fake)`.
  - `pipeline.py` is now small enough (< 300 LOC) to read end-to-end
    in one screen. The cycle-9 directive #9 "Declared ≠ verified"
    audit lane gains a clean target.

### Negative / Risk

  - Future contributors writing new tests must use the
    module-function patching pattern; the historical `pipeline._stage_*
    = <fake>` shortcut no longer works. The 58 deletion contracts
    in `tests/test_phase22_deletion_contracts.py` enforce this at
    commit time.
  - Runtime semantics depend on mypy-invisible flow through
    `initialize_pipeline()`. Stage F mitigated this by adding
    class-body annotations; future contributors adding new
    attributes to `initialize_pipeline()` must remember to also
    declare them on `CognitiveOrchestrationPipeline`. The mypy gate
    surfaces drift.

### Mitigations

  - 58 deletion contracts in `tests/test_phase22_deletion_contracts.py`
    fail closed if any of the retired methods or module-level
    symbols come back.
  - `narrative_guard_phase22.py` runs as a fail-closed lint  <!-- narrative-guard: allow historical-record -->
    preventing the legacy migration vocabulary from re-entering  <!-- narrative-guard: allow historical-record -->
    the source / test / public-narrative surfaces, with same-line
    `narrative-guard: allow <reason>` markers reserved for
    genuinely historical references in ADR-015 / ADR-016 /
    cycle-history blocks of `CLAUDE.md`.
  - 42/42 P9 byte-identical at every commit; 312 wider Phase 2.2
    regression PASS at Stage D4 closure.

## References

  - Cycle-13 K Phase 2.2 commit chain (HEAD `cd3967c8` E1 →
    `491b8752` E2 → `54a94eb7` Stage F → this commit E3):
    Stage A `b1f6a09d`, Stage B1.1..B2.f (15 commits), Stage C
    `10e38931`, Stage D1.a/b/c, Stage D2 `6f0b2606`, Stage D3
    `970c451c`, Stage D4 `17ee3c38`, Stage E1 `cd3967c8`, Stage E2
    `491b8752`, Stage F `54a94eb7`.
  - cgpro DESIGN_LOCK conv `cgpro_phase22_test_rewrite_20260506`
    (in-project YGN-SAGE per Yann directive 2026-05-06).
  - ADR-015 (the meta-ADR; this ADR is the cycle-13 K Phase 2.2
    closure note).
  - `scripts/narrative_guard_phase22.py` (the line-based fail-closed
    lint that pins the post-Phase-2.2 vocabulary on the touch
    surface).
  - `tests/test_phase22_deletion_contracts.py` (58 deletion
    contracts — 6 retired stage-named methods + 36 retired helper
    methods + 6 stay-on-class methods + 10 module-level symbols).
