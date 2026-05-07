---
name: May 1 cycle-9 T2 closeout + test sweep
description: T2 + A14b + oracle-off stabilization. Commits f01a9cc6 + e6117e73. 5 pre-existing failures fixed. Baseline stable.
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# Cycle-9 T2 integration + baseline stabilization (2026-05-01)

**Commits**: `f01a9cc6` + `e6117e73` pushed to `origin/main`.

## What shipped

### 1 — test_pipeline.py A14b migration (3 tests fixed)
After A14b, `pipeline._rust_router.record_outcome_checked()` is the learning path. `_MockRouter` was missing `route_integrated` / `record_outcome_checked` / `cancel_bandit_decision`, causing the pipeline to fall back to Threat-1 path → `bandit.recorded` stayed empty → 3 tests failed.

Fix:
- `_MockRouter`: added 3 methods + `checked_recorded: list` attribute
- `_MockBanditDecision`: added `system=1`, `confidence=0.9`, `estimated_cost=0.001`, `selected_template`
- 3 tests: `pipeline._rust_router = router` wiring + assertions moved to `router.checked_recorded`
- All 39/39 `test_pipeline.py` pass.

### 2 — T2 gate_rejected telemetry (`act.py`)
T2 branch added `log_write_gate_skipped(reason="gate_rejected")` inside `_gate_allows()` when gate returns `allowed=False`. This was NOT applied to main. Applied now. Closes the observability gap where explicit gate rejection was silent (only `log.debug` was emitted).

### 3 — test_memory_write_gate_telemetry.py T2 test
`_RejectingGate` helper + `test_gate_rejected_logs_skip_reason_without_persisting` added from T2. Verifies: gate rejects → `memory.write_gate.fired decision=abstain` logged, `memory.write_gate.skipped reason=gate_rejected` logged, episodic_memory.stored stays empty. 50/50 targeted tests pass.

### 4 — Memory/docs files committed
`docs/claude-memory/` (11 files), `CGPRO.md`, `SYMPHONY.md` committed to main (were untracked orphans).

### 5 — oracle-off fixtures (commit e6117e73, 2026-05-01 session 2)
5 pre-existing test failures caused by cycle-7 oracle DEFAULT-ON (cycle-7 flip `87daf89a`).
Root cause: `allow_training_updates = (not oracle_on) or oracle_trainable` = False when oracle is
ON and no `oracle_verdict` in test context → evolution/archive callbacks skipped.

Fixes:
- `test_oracle_stack.py`: `_RustRouterProxy` class + `pipeline._rust_router = _RustRouterProxy(bandit)`
  in `_make_pipeline`. A14b causal path now reachable. 28/28 pass.
- `test_pillar_logging.py`: `_legacy_oracle_off` autouse fixture (`SAGE_ORACLE=0`). 3 tests pass.
- `test_online_evolution.py`: same `_legacy_oracle_off` fixture. 2 tests pass.
All 59 tests across 3 files pass.

## Branch cleanup status
- `codex-t2-memory-write-paths`: changes now integrated into main (`f01a9cc6`). Branch can be deleted.
- `feat/symphony-dev-orchestration`, `symphony/YGN-9`, `YGN-14`, `YGN-16`, `YGN-13`, `YGN-15`: Symphony automation branches. Not merged. YGN-14 says NOT READY for A2 smoke (3 infra blockers: worker can't reach endpoint 127.0.0.1:4059, cgpro can't run in worker sandbox, YGN-16 not in worker checkout).

## Confirmed baseline (post e6117e73, broad sweep 2026-05-01)
- **33 failed** (down from 41), 2589 passed, 41 skipped, 7 errors, 258s
- All 33 are pre-existing: `contaminated_pre_a14_state` RuntimeError (tests booting full pipeline without epoch guard — epoch missing in temp dirs) + test_provider_pool_wiring (3 errors) + test_rust_integration (4 errors)
- Failing test files: test_semantic_wiring (6), test_speculative_routing (6), test_system_hint (4), test_topology_routing (3), test_smmu_e2e (2), test_topology_events (2), test_topology_learn (2), test_pipeline_bypass (2), test_public_api (2), test_rust_integration (1), test_sandbox_safety (1), test_single_control_plane (1)
- Our session net improvement: **−8 failures** (3 pillar_logging + 2 online_evolution + 2 oracle_stack + 1 side-effect)

## Next steps (cycle-9 remaining)
1. A2 N=10 budget-tier paired smoke — needs YGN-14 blockers cleared OR manual smoke bypassing Symphony.
2. Decision gate: ≥35% pass@1 vs cycle-7 baseline 30% → A3 N=50.

## Why: Token economy
Instruction from user: minimize Claude tokens, delegate to cgpro/codex/deepseek. This session used Claude only for orchestration + small targeted fixes. No cgpro/codex calls needed — all changes were ≤50 LOC, within direct-edit threshold.
