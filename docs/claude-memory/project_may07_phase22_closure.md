---
name: Phase 2.2 closure 2026-05-07 — pipeline.py compatibility-method retirement
description: Cycle-13 K Phase 2.2 SHIPPED. 27 test files rewritten + 6 _stage_* + 36 _<helper> deleted + constructor extracted + mypy 0 restored. pipeline.py at 293 LOC.
type: project
originSessionId: ae15e41f-58ed-438b-8f62-6e3feb79131b
---
Cycle-13 K Phase 2.2 fully closed 2026-05-07 in autonomous session continuation. Recipe was cgpro DESIGN_LOCK → claude scaffold → cgpro VERIFY pre-commit → SHIP loop, in single conv `cgpro_phase22_test_rewrite_20260506` (in-project YGN-SAGE per Yann directive 2026-05-06; new convs created without --resume but kept alive cross-question per `feedback_cgpro_project_centralization`).

**Why:** Phase 2.1 (cycle-13 K, 2026-05-06) closed the façade extraction with 6 `_stage_*` + ~22 `_<helper>` private compatibility methods retained as a temporary instance patch surface (27 test files patched `pipeline._stage_<X> = <fake>`). Phase 2.1 round-4 OPTION_3 reclassified the seam removal + the `pipeline.py < 300 LOC` target to Phase 2.2 with its own DESIGN_LOCK.

**How to apply:**
- Active cgpro conv `cgpro_phase22_test_rewrite_20260506` is now closed; future conv on cycle-13 main run / cycle-14 façade rewrite Phase C / CLI protocol gaps starts a new conv (no --resume, auto-routes to YGN-SAGE project).
- Stage A→F→E3 commit chain: `b1f6a09d` → ... → `1b99271e`. 22 commits for Phase 2.2 alone (15 in Stage B + 4 in Stage D + 1 each in C, F, E1, E2, E3).
- Final state: pipeline.py 293 LOC (< 300 HARD GATE), 58 deletion contracts PASS, 42/42 P9 byte-identical, 3163 pytest passed (1 pre-existing pollution flake), mypy 0 / ruff clean / claims_audit OK / narrative_guard PASS.
- 3 ADRs current: ADR-015 (Implemented closed), ADR-016 (Implemented), ADR-017 NEW (Phase 2.2 compatibility-method retirement closure).
- Test counter: 3183 Py / 553 Rust → 3241 Py / 555 Rust / 100 sage-discover.

**Lessons applied during the cycle:**
1. **mypy regression at D3 caught at E3 final-gate** — D3 constructor extraction (`970c451c`) introduced 179 attr-defined errors that pytest didn't surface. Stage F (`54a94eb7`) added bare class-body instance annotations on `CognitiveOrchestrationPipeline` for the 31 attrs initialized by `pipeline_v2.constructor.initialize_pipeline()`. Lesson: when extracting `__init__` body, mypy can't follow the function call to infer attribute creation; explicit class-body declarations are required.
2. **Stage D1.c retarget incompleteness** — `test_pipeline_fallback_provider.py` (8 tests) was missed during the helper retarget pass. Fixed at E3 by adding the file to the retarget set (`from sage.pipeline_v2.execute import pick_fallback_provider` + 9 callsite rewrites). Lesson: D1.c retarget grep should be done with `grep -rln 'pipeline\._<helper_name>(' sage-python/tests/` for each retired helper, not just the production path.
3. **type:ignore ceiling raise pattern** — Phase 2.1/2.2 module split duplicated the `from sage_core import TopologyExecutor # type: ignore[import-not-found]` LOCAL imports across 3 reroute / multi-agent / single-agent fallback paths in `pipeline_v2/execute.py`. Same diagnostic category as the pre-decomposition single import; the split duplicated the import, not the category. Ceiling raised 51 → 54 with per-ignore justification block per the established pattern.
4. **cgpro browser crash recovery** — single conv crashed twice during Stage E3 with `Target page, context or browser has been closed` and `keyboard.type` timeouts. Fix: user runs `cgpro adopt` from terminal to re-import ChatGPT desktop app session. Same incident pattern as 2026-05-06. After adopt, session reports "healthy" and ask succeeds.
5. **Parent-baseline proof for "pre-existing failure" claims** — cgpro round-1 EDIT_REQUIRED on E3 demanded `git checkout <parent_sha> && pytest <same-cmd>` to prove the 2 final-gate flakes were pre-existing. Of the 10 parent-baseline failures, 9 were fixed by E3 (all attributable to Phase 2.2 retirement scope) and 1 (`test_swebench_ca_patch::test_wrapper_copies_bundle_into_build_dir`) reproduced identically at parent. cgpro round-2 returned GO_PUSH. Lesson: don't claim "pre-existing flake" without running the same command at parent — isolation passes only prove order-dependence, not parent-presence.

**Phase 2.2 commit chain (22 commits, all on origin/main):**
- A `b1f6a09d` (narrative guard + 58 RED/GREEN deletion contracts)
- B1.1..B2.f (15 commits, 27 test files rewritten module-function patching)
- C `10e38931` (atomic 6 `_stage_*` deletion + orchestrator retarget)
- D1.a `4c1cb37c` (memory_gate retarget)
- D1.b `309136b6` (runtime_events retarget)
- D1.c `3e98f379` (topology/costing/assign/bandit retarget)
- D2 `6f0b2606` (atomic 36-helper deletion)
- D3 `970c451c` (constructor extraction `pipeline_v2/constructor.py`)
- D4 `17ee3c38` (touched-file narrative cleanup)
- E1 `cd3967c8` (source + test narrative cleanup + README capability row)
- E2 `491b8752` (ADR-015 + ADR-016 + CLAUDE.md historical record)
- F `54a94eb7` (mypy 0 restoration — class-body instance annotations)
- E3 `1b99271e` (ADR-017 closure + status propagation + final gates)

**Open follow-ups (not Phase 2.2 scope):**
- `ALIRE3.md` pre-existing working-tree deletion (separate cleanup commit).
- `test_swebench_ca_patch::test_wrapper_copies_bundle_into_build_dir` pre-existing pollution flake.
- Wider `~/.sage/` test-pollution class still tracked from cycle-11 evening.
- Cycle-13 main run wiring (arms A/B/C/D, $240-460 budget, 3-5 days).
- CLI v0 protocol gaps (`cli_progress`, `set_budget`, `cancel`, `cli_complete.final_seq`) per cgpro DESIGN E trap Q5.
- Patch repair budget for diff_verifier observe → repair mode.
