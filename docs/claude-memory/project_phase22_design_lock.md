---
name: Phase 2.2 DESIGN_LOCK contract — cgpro signed 2026-05-06
description: Locked Stage A→E plan for Phase 2.2 test rewrite + delegator purge + <300 LOC. cgpro conv cgpro_phase22_test_rewrite_20260506 (id 69fb7f75-06f4-8391-983e-54a3af1fa7a9, in YGN-SAGE project, no --resume per Yann directive 2026-05-06).
type: project
originSessionId: ae15e41f-58ed-438b-8f62-6e3feb79131b
---
**Verdict**: `GO_PHASE_2_2_DESIGN_LOCKED` (cgpro one-shot, single round, 2026-05-06).
**Conv**: `cgpro_phase22_test_rewrite_20260506` (id `69fb7f75-06f4-8391-983e-54a3af1fa7a9`).
**Baseline**: HEAD `96155232` (cycle-13 K Phase 2.1 closure SHIP signed).

## 3 mandatory amendments

1. **Module-function only seam** for `_stage_*`. No compatibility shim post Stage C.
2. **`pipeline.py < 300 raw lines` is HARD GATE** — not amended to `<400`/`<500`. Requires façade-shrink substep (preferred: extract `__init__` body to `sage.pipeline_v2.constructor.initialize_pipeline(self, ...)`).
3. **`_emit` stays in Phase 2.2** — intentionally retained private runtime seam (stateful through `self.event_bus`, called by carved-out runtime helpers).

## Locked Stage A→E plan

- **Stage A** (1 commit): narrative_guard_phase22.py + 6 `_stage_*` deletion contract tests (RED at A) + helper-purge contract tests with final expected absence list.
- **Stage B** (small batches): B1 Pattern A direct-call tests → module-function calls. B2 Pattern B instance-injection → `monkeypatch.setattr("sage.pipeline_v2.X.X", ...)`. B3 string-monkeypatch contracts (`test_run_frame.py` + `test_oracle_stack.py` for `sage.pipeline._new_runtime_run_id` + `time.monotonic`) PRESERVED + verify.
- **Stage C** (1 atomic commit): C1 orchestrator.py `run_internal` local imports of pipeline_v2 stage functions (NOT top-level — Trap 1). C2 delete 6 `_stage_*`. C3 verify deletion contract green + 27 rewritten tests + P9 + wider regression.
- **Stage D**: D1 rewrite `pipeline_v2` internal `self._foo` callsites (select_topology.py / classify.py / learn.py / execute.py / orchestrator.py per Q7 audit list). D2 delete ~32 helpers (KEEP `_emit` + `_run_internal`). D3 codex batch. D4 extract `__init__` body to `pipeline_v2/constructor.py` for LOC margin.
- **Stage E**: E1 `pipeline.py < 300 raw lines`. E2 narrative_guard PASS. E3 ADR-015 closure or ADR-017 new. E4 claims_audit --strict + ruff + mypy + final regression.

## Final acceptance gates

- 37/37 P9 byte-identical at every commit
- rewritten 27-file stage-seam tests pass
- `_stage_*` deletion contract GREEN
- safe-helper deletion contract GREEN
- `pipeline.py < 300 raw lines` (HARD GATE)
- ruff clean
- mypy 0
- claims_audit --strict OK
- narrative guard PASS
- final wider regression per Phase 2.2 baseline

## Rollback discipline

- Every commit preserves 37/37 P9 byte-identical.
- If P9 breaks: revert + retry with smaller scope. **No forward-fixing across commits.**

## Q1-Q7 locked answers

- **Q1** Module-function patching is unique sanctioned strategy. **No mixed instance-level seam preservation** (no `__getattr__`, no dynamic adapter, no temporary compat property). Async stages: `await decompose(pipeline, ctx)` etc.
- **Q2** Stricter purge rule: pure delegator + first-arg-pipeline + ALL production callsites in `src/**` rewritten + ALL test callsites rewritten + no class-level `hasattr(pipeline, "_foo")` / `CognitiveOrchestrationPipeline._foo` / instance monkeypatchability assertion + not lifecycle/state hook. Class-level introspection counts (Trap 5).
- **Q3 Stays**: `__init__`, `run`, `run_with_frame`, `run_with_bench_evaluator`, `_run_internal`, `_emit`, module-level `_new_runtime_run_id` / `time` / `PipelineContext` re-export / `BUDGET_EXCEEDED_RESULT` / `EXECUTE_HALTED_UNVERIFIED` / `EXECUTE_UNVERIFIED` / `_is_strict_governance` / `_resolve_task_budget_usd` / `_BANDIT_ATTRIBUTION_REASON_CODES`. State attrs stay as attrs.
- **Q3 Purge candidates** (38 total): 6 `_stage_*` + 3 memory_gate (`_emit_budget_exceeded`, `_build_write_gate`, `_record_to_memory`) + 13 runtime_events (NOT `_emit`) + 12 topology_helpers + 5 bandit_attribution + 2 model_assignment + 1 provider_fallback (`_pick_fallback_provider`) — **= 6 stage + 32 helpers = 38 purgeable**.
- **Q3a `_emit` STAYS.** Stateful EventBus boundary, mock-popular, called from `pipeline_v2/execute.py`.
- **Q3b** `<300 raw lines` is HARD GATE. Preferred: extract constructor body. Acceptable alternative: extract `run_with_bench_evaluator` body + shrink class/module docstrings.
- **Q5** Order locked: `B test rewrites → C orchestrator rewrite + _stage_* deletion → D helper purge`.
- **Q6** orchestrator.py keep `from sage import pipeline as pipeline_mod` inside `run_internal` for `_new_runtime_run_id` + `time.monotonic` monkeypatch contracts. Stage functions local-imported inside `run_internal`.
- **Q7** HARD GATE — audit ALL `pipeline_v2/*.py` `self._foo()` / `pipeline._foo()` callsites BEFORE helper deletion. Specific list per cgpro:
  - `select_topology.py`: `_build_topology_from_hint`, `_log_topology_structure`, `_apply_topology_budget_and_cache`, `_topology_candidate_items`, `_log_topology_candidates`
  - `classify.py`: `_clear_bandit_decision`, `_cancel_bandit_decision`, `_emit_bandit_attribution_mismatch`
  - `learn.py`: `_record_bandit_outcome_checked`, `_cancel_bandit_decision`, `_clear_bandit_decision`
  - `execute.py`: `_emit_budget_exceeded`, `_is_single_agent_execution`, `_runtime_emit_topology_selected`, `_runtime_emit_model_assigned`, `_estimate_topology_cost`, `_pick_fallback_provider`
  - `orchestrator.py`: `_build_write_gate`, `_runtime_emit_topology_selected`, `_runtime_emit_model_assigned`, `_runtime_final_status`, `_runtime_final_node_count`, `_record_to_memory`
- **Q7-bis** Locked circular-import discipline: pipeline.py MAY import `PipelineContext` from `pipeline_v2.context` only top-level; MUST NOT top-level import any other `pipeline_v2.*`. orchestrator.py MUST keep `pipeline_mod` indirection. pipeline_v2 modules MAY top-level import pipeline.py constants/helpers; SHOULD use TYPE_CHECKING for types; SHOULD local-import for cycle risk.

## 6 traps pre-empted

1. Module-function monkeypatch fails if orchestrator imports too early → use **local imports** in `run_internal`.
2. `_build_write_gate` + `_record_to_memory` mock sites in `observability/test_pipeline_spans.py` → must move to module-function patching (NOT relying on `_emit` exemption).
3. `.claude/rules/architecture.md` says "Pipeline (5-stage)" while listing six boxes — fix in Stage E.
4. Stage numbering inconsistency: code uses 0-5, README uses 1-6. **Locked wording**: "6 runtime stages: CLASSIFY → DECOMPOSE → SELECT_TOPOLOGY → ASSIGN_MODELS → EXECUTE → LEARN". Avoid numeric labels unless file has stable numbering convention.
5. Contract tests check **class-level absence**, not instance-dict: `assert not hasattr(CognitiveOrchestrationPipeline, "_stage_classify")`.
6. **Do NOT delete `_run_internal`** — keep as private façade method (subclass/private override surfaces).

## Narrative guard regex (cgpro-locked)

```
Phase A | Phase B | Phase 2\.1 | placeholder | do\s+NOT\s+move | helper ownership migration is Phase C | 5[-\s]stage | 5\s+stage | six[-\s]stage | 6\s+stage | stage seam | transition seam | delegator
```

Allow marker (per-line, NOT file-level): `# narrative-guard: allow <reason>` or `<!-- narrative-guard: allow <reason> -->`.

Glob list: 32 patterns (4 docs/source + 27 test files + .claude/rules/architecture.md + CLAUDE.md + README.md + AGENTS.md). Fail-closed on missing globs.

Saved as `.tmp/narrative_guard_phase22.py` (Stage A1 commit).

## Commit message format

Prefix: `cycle-13k phase2.2/<stage>: <imperative summary>` (lowercase, no caps).

Body template:
```
Gates:
- P9 byte-identical: 37/37
- targeted pytest: <command/result>
- ruff: <result or deferred until E>
- mypy: <result or deferred until E>
- narrative_guard_phase22: PASS
- pipeline.py raw LOC: <n>
```

## Final lock text (paste back to cgpro VERIFY rounds)

```
GO_PHASE_2_2_DESIGN_LOCKED:
Rewrite the 27 tests from Pipeline instance _stage seams to pipeline_v2 module-function seams; update orchestrator.py to call module functions with local imports; preserve sage.pipeline._new_runtime_run_id and sage.pipeline.time.monotonic monkeypatch contracts through pipeline_mod; delete the six _stage_* methods; purge safe helper delegators only after all production and test callsites are rewritten; keep _emit, _run_internal, module-level constants/helpers, PipelineContext re-export, and public API methods; shrink pipeline.py below 300 raw lines; normalize all stale 5-stage/transition-seam/delegator narrative; require P9 byte-identical at every commit and final ruff/mypy/claims_audit/narrative guard PASS.
```

## Continuity

- Conv `cgpro_phase22_test_rewrite_20260506` is the active thread for VERIFY rounds (resume with `--resume cgpro_phase22_test_rewrite_20260506`).
- Cycle-13 K Phase 2.1 thread `cgpro_phase21_facade_rewrite_20260506` is closed (round-10 GO_PHASE_2_2_DESIGN signed).
- DESIGN_LOCK doc: `.tmp/cgpro_phase22_design_lock.md` (323 lines, 18 KB).
- DESIGN_LOCK response: `.tmp/cgpro_phase22_design_lock_finaltext.md` (508 lines, 23 KB).

## Tooling

- `scripts/narrative_guard_phase22.py` — Stage A1 (commit `b1f6a09d`). Pre-commit lint over 31 globs with cgpro-locked regex + per-line `# narrative-guard: allow <reason>` markers. Will go GREEN at Stage E2 sweep.
- `scripts/phase22_inventory.py` — B1.2-sync.b (commit `20f17f76`). AST-based source-of-truth filter for stage seam usage. Built after the `test_pipeline_adaptation.py:90` oversight per advisor 2026-05-06. Filters by `--stage` / `--kind` / `--sync-only` / `--async-only` / `--helper` / `--file`. Outputs CSV. Runs against `--scope tests` (default) or `--scope src` (Stage D Q7 audit). **Use this for every Stage B sub-batch and Stage C/D scope decisions.**
- `sage-python/tests/test_phase22_deletion_contracts.py` — Stage A2 (commit `b1f6a09d`). 16 PASSED + 42 XFAILED (RED-as-contract). Stage C drops xfail on 6 stage methods; Stage D drops xfail on 36 helpers.

## Progress (commits)

| Commit | Stage | Files | Net diff |
|---|---|---|---|
| `b1f6a09d` | A | 2 (script + test) | +368 |
| `c7870f16` | B1.1 | 3 tests | +13 -10 |
| `d740ddee` | B1.2-sync | 6 tests | +29 -19 |
| `20f17f76` | B1.2-sync.b | test_pipeline_adaptation + scripts/phase22_inventory.py | +393 -1 |
| `b928fb07` | B1.2-async.a | decompose 2 sites in 2 files | +5 -3 |
| `b0050d16` | B1.2-async.b | execute small/core 7 sites in 4 files | +12 -7 |
| `41202b72` | B1.2-async.c | execute invariant-heavy 8 sites in 3 files | +11 -8 |
| `7ac5e884` | B1.2-async.d | learn small/core 3 sites + Trap 3 fix asyncio.run | +5 -3 |
| `d3f860f2` | B1.2-async.e | learn invariant/evolution-heavy 22 sites in 4 files | +27 -22 |
| `00206207` | B2.a | test_pillar_logging.py select_topology Pattern B | +6 -5 |
| `113f4c82` | B2.b | test_system_hint.py mixed (3 sync calls + 10 stage assigns) | +28 -23 |
| `a0865ce4` | B2.c | test_pipeline_v2_phase_a_wrappers.py round-trip retired (Path A) | +143 -257 (-114 LOC) |
| `57ea24f0` | B2.d | test_oracle_stack.py Pattern B (5 stage assigns) | +20 -12 |
| `577c7bb3` | B2.e | observability/test_pipeline_spans.py Pattern B + Trap 2 | +48 -19 |
| `c85007ae` | B2.f | test_run_frame.py Pattern B (Stage B COMPLETE) | +29 -24 |

**Stage B COMPLETE at `c85007ae`** (16 commits Phase 2.2). All 27 test files rewritten; 53 Pattern A direct calls + 31 Pattern B instance assigns migrated to module-function patching; 14+ Q6 string monkeypatches preserved; P9 byte-identical 42/42 at every commit; 42 deletion contracts still XFAILED RED-as-contract.

| `10e38931` | C | atomic orchestrator + 6 _stage_* deletion (pipeline.py 738->642 -96 LOC) | +64 -164 |
| `4c1cb37c` | D1.a | memory-gate helper retarget (no deletion). 7 production callsites + 13 test sites. _emit stays Q3a. Helper xfails STILL XFAIL. | +51 -36 |
| `309136b6` | D1.b | runtime-events helper retarget (orchestrator + execute + classify + bandit_attribution + 3 tests). 13 production callsites + 4 test sites. Module function signature change reason_code arg[1]→arg[2] documented. | +48 -25 |
| `pending` | D1.c | topology/costing/assign/bandit helper retarget (5+ files + tests) |  |
| `pending` | D2 | atomic helper deletion (36 helpers) + xfail drop |  |
| `pending` | D3 | constructor extraction (conditional, only if pipeline.py >= 300 LOC after D2) |  |
| `pending` | D4 | touched-file narrative cleanup |  |
| `pending` | E | full narrative_guard PASS + ADR closure + claims_audit |  |

**Stage C SHIPPED at `10e38931`**: 6 _stage_* methods DELETED, orchestrator retargeted, Q6 + Trap 2 preserved. pipeline.py 738->642 (-96, -13%). 6 stage deletion contracts now PASS, 36 helper still XFAIL.

**Stage D split locked by cgpro post-Stage-C** (5 sub-blocks):
- D1.a memory-gate retarget — SHIPPED `4c1cb37c`
- D1.b runtime-events retarget — PENDING
- D1.c topology/costing/assign/bandit retarget — PENDING
- D2 atomic helper deletion + xfail drop — PENDING
- D3 conditional constructor extraction — PENDING (only if pipeline.py >= 300 LOC after D2)
- D4 touched-file narrative cleanup — PENDING (full narrative sweep still Stage E)

Async batches remaining (cgpro 5-batch split): .b execute small/core (4 files), .c execute invariant-heavy (3 files), .d learn small/core (3 files), .e learn invariant-heavy (4 files). May propose consolidation to .exe + .lrn (2 batches) if cgpro accepts.

Pattern B remaining: 6 files / 31 sites (test_oracle_stack, test_pillar_logging, observability/test_pipeline_spans, test_pipeline_v2_phase_a_wrappers, test_run_frame, test_system_hint).

## AST inventory snapshot (HEAD `20f17f76`)

- Pattern A direct calls (kind=call): **53 sites across 14 files**
  - decompose: 4 (2 in B1.2-async.a in-flight, 2 deferred in test_pipeline_v2_phase_a_wrappers Pattern B file)
  - execute: 16 sites (B1.2-async.b/.c)
  - learn: 26 sites (B1.2-async.d/.e)
  - sync remaining: 3 in mixed Pattern B files (test_system_hint × 3 classify, test_pipeline_v2_phase_a_wrappers × 1 classify + 1 select_topology + 1 assign_models)
- Pattern B instance assigns (kind=assign): **31 sites across 6 files**
  - test_oracle_stack.py (5), test_pillar_logging.py (2), test_pipeline_spans.py (6), test_pipeline_v2_phase_a_wrappers.py (2), test_run_frame.py (6), test_system_hint.py (10)
