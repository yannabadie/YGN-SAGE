---
name: Phase 2.1 façade rewrite SHIPPED — pipeline.py 1800→727 LOC, 22 commits, 10 cgpro VERIFY rounds
description: Cycle-13 K Phase 2.1 closeout 2026-05-06 — facade extraction with transition seams retained per cgpro round-4 OPTION_3. cgpro round-10 SHIP_PHASE_2_1 + GO_PHASE_2_2_DESIGN signed. Phase 2.2 explicit DESIGN_LOCK pending.
type: project
originSessionId: 88857be6-7048-463a-8ee4-cb3b4cca20fd
---
**HEAD**: `96155232` (cycle-13 K Phase 2.1 closure SHIP signed)
**Conv cgpro**: `cgpro_phase21_facade_rewrite_20260506` (id `69fb53c1-5da0-8394-b73a-4c5e61f966af`, in YGN-SAGE project)
**Verdict final**: round-10 = SHIP_PHASE_2_1 + GO_PHASE_2_2_DESIGN

## What shipped

22 commits since Phase 1.5h closure `560731b2`:

| Phase | Last commit | LOC pipeline.py |
|---|---|---|
| 1.5h | `560731b2` | 1800 (start) |
| A1-A4 | `0ed9d32d` | 1745 |
| B1-B5 | `4ff2beab` | 993 |
| C0 | `b3ad09c2` | 993 |
| D0+D1 | `c6fafae0` | 729 |
| E0 | `caf7cdb4` | 729 |
| E1 | `a9a7094d` | 702 |
| F | `9f3732d2` | 727 |
| F2-F2.4 | `96155232` | 727 |

**Final state**: pipeline.py 1800 → 727 LOC (1073 migrated, 60% reduction). 14 modules in `pipeline_v2/` (~3661 LOC):
- 6 stage bodies (classify/decompose/select_topology/assign_models/execute/learn)
- orchestrator.py (`run_internal` body, Step D)
- context.py (`PipelineContext` dataclass body, Step E1)
- 5 helper modules (bandit_attribution, runtime_events, memory_gate, topology_helpers, costing)
- __init__.py (PEP 562 lazy `__getattr__`)

37/37 P9 byte-identical at every commit. ruff clean, mypy 0, claims_audit --strict OK.

## Why Option 3 (cgpro round-4 OPTION_3 reclassification)

cgpro round-3 originally scoped Step C as "delete 6 `_stage_*` delegators + rewrite test_pipeline_v2_phase_a_wrappers.py". My empirical grep before C2 revealed **27 test files** mock `pipeline._stage_<X> = <fake>` as a runtime test seam (fast unit isolation, oracle gate setup, bypass/topology controller scenarios, observability isolation, E2E injection). cgpro round-4 reclassified `_stage_*` deletion + helper delegator purge + `<300 LOC` target to Phase 2.2 with own DESIGN_LOCK covering test rewrite contract change.

Phase 2.1 closes as **"facade extraction with transition seams retained"**. Pipeline.py 727 LOC inside cgpro round-4 amended target 650-800 LOC.

## Critical garde-fous applied

1. **PEP 562 lazy `__getattr__`** in `pipeline_v2/__init__.py` (Step E0) — must land BEFORE PipelineContext move (E1) to avoid circular `pipeline → pipeline_v2.context → pipeline_v2 → pipeline`.

2. **`PipelineContext.__module__ = "sage.pipeline"`** setattr in `pipeline_v2/context.py` — preserves bench/dashboard/observability assertions on legacy module path despite physical move.

3. **`pipeline_mod` indirection** in `orchestrator.py` — `from sage import pipeline as pipeline_mod` INSIDE function (not top-level) preserves `monkeypatch.setattr("sage.pipeline._new_runtime_run_id", ...)` + `monkeypatch.setattr("sage.pipeline.time.monotonic", ...)` from `test_run_frame.py`. Naive top-level `from sage.pipeline import _new_runtime_run_id` would freeze the symbol pre-monkeypatch.

4. **Stage seams stay** in `_run_internal`: orchestrator calls `pipeline._stage_classify(ctx)` (delegator), NOT `classify(self, ctx)` direct, so the 27-file mock contract is unchanged.

5. **Logger `sage.pipeline`** in all carved-out modules per Q7 trap "logger name drift" — trace-grep continuity preserved.

## Test rewrites (Phase 2.1 scope)

- `test_pipeline_v2_phase_a_wrappers.py` — module docstring rewritten "Phase A — additive wrappers" → "pipeline_v2 stage ownership + legacy seam compatibility tests" with 6 enumerated current-state invariants
- `test_pipeline_does_not_top_level_import_pipeline_v2` → `test_pipeline_top_level_pipeline_v2_imports_are_allowlisted` (allows `from sage.pipeline_v2.context import PipelineContext # noqa: E402` only at top level; all other pipeline_v2 imports must be LOCAL inside delegator method bodies)
- `test_pipeline_v2_context_module_reexports_only` → `test_pipeline_v2_context_module_preserves_legacy_identity` (rewritten docstring documenting 3-front backward compat: identity, `__module__`, pickle support)
- `test_pipeline_v2_package_exposes_expected_modules` — section header + docstring + import list updated to enumerate the actual 14 pipeline_v2 modules; "(placeholder)" comments removed

## 10 cgpro VERIFY rounds

| Round | Verdict | Catch type |
|---|---|---|
| 1 | DESIGN_LOCKED | Q1-Q7 architectural traps (PipelineContext + `_run_internal` + helper boundary + backward-compat + Phase A tests + drift checklist + circular-import) |
| 2 | GO_STEP_B | Confirmed delegator pattern; line-budget drift trap warned; cost helper to costing.py NEW (not assign_models.py) |
| 3 | GO_STEP_C amended | Order C0→C4; `<300 LOC` target maintained; D2/E2 helper delegator purge added as new sub-step |
| 4 | OPTION_3 | Empirical 27-file mock contract — Phase 2.1 reclassified to "facade extraction with transition seams" + Phase 2.2 own DESIGN_LOCK for test rewrite |
| 5 | EDIT_REQUIRED | F docs drift (ADR header / 3 stage docstrings / Phase A test docstring) |
| 6 | EDIT_REQUIRED | F2 residual (ADR Related round-5 chain false / context test name) |
| 7 | EDIT_REQUIRED | F2.1 residual (ADR LOC numbers / package-structure stale + placeholder comments) |
| 8 | EDIT_REQUIRED | F2.2 residual (ADR "5 stage bodies" typo) |
| 9 | EDIT_REQUIRED | F2.3 residual (same "5 stage" typo in pipeline.py + pipeline_v2/__init__.py — F2.3 grep was ADR-only, missed code) |
| **10** | **SHIP_PHASE_2_1 + GO_PHASE_2_2_DESIGN** | All gates green |

cgpro non-blocking note for Phase 2.2: add `grep "5-stage|5 stage|six stage|6 stage"` to vocabulary normalization checklist.

## Lesson for Phase 2.2 DESIGN_LOCK

Phase 2.1 docs sweep consumed 5 EDIT_REQUIRED rounds (5-9) catching predictable drifts that a single comprehensive `grep -RnE "<patterns>" docs/ sage-python/src/ sage-python/tests/` would have caught pre-push. **Phase 2.2 DESIGN_LOCK should include a pre-commit narrative-guard-style mechanical lint** — similar to Phase 0 narrative guard (27 docs × 14 patterns) — running on:
- ADR-015 + ADR-016
- pipeline.py + pipeline_v2/*.py
- test_pipeline_v2_phase_a_wrappers.py
- Any test file mocking `pipeline._stage_*` (27 files in scope for Phase 2.2)

Suggested patterns: `Phase A|Phase B|placeholder|do NOT move|helper ownership migration is Phase C|5-stage|5 stage|six stage|6 stage`.

## Phase 2.2 next steps (cgpro authorized scope)

New conv `cgpro_phase22_test_rewrite_20260506` (in YGN-SAGE project, NO `--resume` per Yann directive 2026-05-06):

1. Inventory 27 test files monkeypatching `pipeline._stage_<X>`
2. Rewrite to module-function patching OR public-effect assertions
3. Remove 6 `_stage_*` delegators
4. Inventory + remove ~22 `_<helper>` delegators where safe
5. Reach `pipeline.py < 300 LOC` (architectural target reclassified from Phase 2.1)
6. Preserve 37/37 P9 byte-identical + wider regression
7. Vocabulary normalization sweep ("5-stage" → "6-stage" historical text)

Estimated: ~3-5 day session with own DESIGN_LOCK + characterization-test-first discipline + pre-commit narrative-guard sweep.
