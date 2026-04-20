# Rust-First Architectural Completion — Design Spec

**Date:** 2026-04-20
**Author:** Claude Opus 4.7 + Yann Abadie (brainstorming)
**Status:** Approved — ready for writing-plans invocation
**Horizon:** 2–4 weeks (6–8 sessions)
**Axis selected:** A — Architectural continuity (Rust-first, close Critical Directive #1 gaps)

---

## Context

Evening of 2026-04-19 closed 6 architectural bypasses ("Rust built, Python doesn't call it" pattern):

| ID | Commit | What |
|---|---|---|
| G-series | `c905d06` | `CompositeWriteGate` wired into `phases/act.py` 3 memory writes |
| Factory guard | `0b00abd` | Regression test — factory must forward write_gate |
| H1 | `2cd840e` | `engine.should_evolve()`/`evolve()` wired into LEARN |
| H4 | `dc51976` | `cache_topology()` added — fixes silent record_outcome bypass caught in H1 |
| H5 | `27a9a4c` | `write_gate` wired onto singleton AgentLoop (bypass path) |
| H6 | `aa348e1` | `_on_drift` wired onto singleton AgentLoop (bypass path) |
| Meta | `aa348e1` | `docs/audits/bypass-patterns.md` — methodology catalog |

Three known-queued items remain (flagged in `bypass-patterns.md` §"Next audit"):
- `max_steps` / `stall_cap` / `tools` on the singleton AgentLoop vs the factory
- Rust `TopologyController` port — 6 decision paths in Python, zero in Rust, explicit Critical Directive #1 violation
- `RustEntityGraph` consolidation with Python `CausalMemory` — refactor, not bypass

MAP-Elites archive growth was proven in the test suite (`TestRealEngineEvolutionLoop`) but never observed on a real benchmark run.

---

## Scope

This plan covers the **first two items** (Phase 1: audits + empirical MAP-Elites validation; Phase 2: Controller Rust port). The `RustEntityGraph` consolidation is out of scope — it is a refactor, not a bypass, and deserves its own design cycle.

**Non-goals:**
- No pass-rate improvements claimed. This plan closes architecture-claim gaps.
- No new Rust crate dependencies beyond existing `regex`.
- No ABI breaks to `sage_core` PyO3 surface. New classes added, existing signatures preserved.

---

## Phase 1 — Audit-Complete (weeks 1–2, 3–4 sessions)

**Exit criterion:** zero unverified bypass in `bypass-patterns.md` §"Next audit". Each fix = 1 commit + 1 empirical regression test (pattern H4-style, using real Rust objects guarded by `@pytest.mark.skipif(not _HAS_SAGE_CORE)`).

### 1.1 `max_steps` singleton audit

- **Suspected bug.** `boot.py:279` configures the singleton with `max_steps=MAX_AGENT_STEPS=20`. The bypass branch in `pipeline.py:941+` never re-scales by `ctx.system`. For S1 tasks, the singleton can spin 20 tool-call turns when the factory path would have capped at 5.
- **Fix pattern (H5-style).** In the bypass branch, after the `validation_level` assignment block (pipeline.py:973–977), add `self._agent_loop.config.max_steps = {1: 5, 2: 10, 3: 20}[ctx.system]` (mirroring `agent_loop_factory.py:130–135`).
- **Regression test.** `test_pipeline_single_agent_scales_max_steps_by_system` — spy AgentLoop; system=1 → `config.max_steps==5`; system=2 → 10; system=3 → 20. Fail if a refactor drops the line.
- **Done-when.** Test green, commit pushed, `bypass-patterns.md` §"Next audit" row flipped.

### 1.2 `stall_cap` singleton audit

- **Suspected bug.** `AgentConfig.stall_after_tool_steps` default is 0, which disables D8 soft-breaker. Factory computes `node_stall_cap = node_max_steps - 1` for S2/S3 (`agent_loop_factory.py:149–152`). On bypass path for S3, the singleton can thrash 20 tool turns without breaking.
- **Fix.** In same bypass branch, after 1.1's fix: `self._agent_loop.config.stall_after_tool_steps = max(0, self._agent_loop.config.max_steps - 1) if self._agent_loop.config.max_steps > 5 else 0`.
- **Regression test.** `test_pipeline_single_agent_sets_stall_cap_matching_factory` — system=3 → stall_cap=19, system=1 → 0.
- **Done-when.** Test green, commit, doc row flipped.

### 1.3 `tools` filter singleton audit

- **Pre-analysis required.** The singleton has no `tools` filter (receives all tools). Factory filters per role (verifier gets only execute_bash+memory, formatter gets memory-only, etc.). On the bypass path, the task has no topology role — single agent is implicitly "actor" → all tools is semantically correct.
- **Action.** Empirically verify by enumerating tool call counts on a single-agent smoke vs a multi-node smoke. Document in `bypass-patterns.md` §"Not a bypass (false positives)" if confirmed intentional, otherwise fix.
- **Done-when.** Either documented false-positive OR fix + test like 1.1/1.2.

### 1.4 MAP-Elites archive growth — empirical smoke

- **Setup.** Run `python -m sage.bench --type swebench --subset lite --limit 5` with `SAGE_LOG_LEVEL=INFO`. Capture full log.
- **Assertions (inspect log after run).**
  - `archive_cell_count` grew from 0 to ≥ 1 over 5 tasks
  - At least 1 "Online evolution fired" log line if cell_count reached `EVOLUTION_MIN_OUTCOMES=5`
  - Final `engine.archive_cell_count()` matches the growth curve from the log
- **Failure modes & diagnostic.**
  - **Cells stay 0:** H4 has a remaining downstream bypass. Open `TestRealEngineEvolutionLoop` output vs real-run output, diff.
  - **Cells grow but should_evolve never fires:** constants mis-calibrated (likely EVOLUTION_MIN_OUTCOMES=5 too high for 5-task smokes). Document in `bypass-patterns.md`, possibly adjust constants OR accept that short smokes won't trigger evolve — document that acceptance.
- **Deliverable.** `docs/benchmarks/2026-04-2x-archive-growth-smoke.md` — curve + verdict. Commit.

### 1.5 PyO3 inventory sweep

- **Goal.** Find bypasses we missed yesterday. Systematic grep of every `#[pyclass]` in `sage-core/src/` vs Python runtime call sites.
- **Script.**
  ```bash
  # Every pyclass export
  grep -rn "#\[pyclass\]\|m\.add_class" sage-core/src --include="*.rs"
  # Per class, count Python runtime call sites (not imports, not tests)
  for cls in $(grep -oP '#\[pyclass(\(name = "\K[^"]+)?' sage-core/src/**/*.rs | awk -F: '{print $NF}'); do
      py_count=$(grep -rn "$cls\b" sage-python/src/sage --include="*.py" | grep -v "^.*/tests/" | wc -l)
      echo "$cls: $py_count runtime refs"
  done
  ```
- **Triage.** For each class with 0 runtime refs, classify: (a) stale/dead — mark for deletion ADR; (b) bypass — add to Phase 1 follow-up; (c) future/planned — document in bypass-patterns.md false-positives.
- **Deliverable.** `docs/audits/2026-04-2x-pyo3-inventory.md` — 3-column table: Rust class / Python runtime refs / Verdict.
- **Done-when.** Report committed. If ≥1 bypass found → extend Phase 1 with matching fix commit before proceeding.

### 1.6 ADR-011: Singleton AgentLoop vs Factory Asymmetry

- **Location.** `YGN-SAGE/Decisions/ADR-011-Singleton-vs-Factory-Asymmetry.md`
- **Content.** Formalize the pattern: boot.py creates a shared singleton, factory creates fresh per-node loops. List every attribute each path configures. Codify the rule: **the bypass path MUST re-configure every attribute the factory sets**. Reference `bypass-patterns.md` §"Two-path check".
- **Written when.** After 1.1, 1.2, 1.3 are committed — references them as evidence.

---

## Phase 2 — TopologyController Rust Port (weeks 3–4, 6 sessions)

**Exit criterion:** Python `topology_controller.py` is a ~50-line wrapper delegating to `RustTopologyController`. All 6 decision paths have both Rust and Python tests. `maturin develop` passes. ADR-012 written.

### 2.1 Rust scaffold + PyO3 + constants (1 session, ~400 LOC Rust)

- **New file.** `sage-core/src/topology/controller.rs`
- **Struct.**
  ```rust
  #[pyclass]
  pub struct RustTopologyController {
      reroute_count: u32,
      spawn_count: u32,
      node_retries: HashMap<usize, u32>,
      abstain_count: u32,
      node_qualities: HashMap<usize, f32>,
  }
  ```
- **Constants (mirror Python).** `THETA_GOOD = 0.7`, `THETA_CRITICAL = 0.3`, `THETA_CONSISTENCY = 0.5`, `THETA_PRUNE = 0.2`, `MAX_RETRIES = 2`, `MAX_REROUTES = 1`, `MAX_GATE_TURNS = 2`, `MAX_SPAWNS = 3`.
- **Methods at scaffold.** `new()`, `evaluate_and_decide()` stub returning `AdaptationDecision::continue_()`. Details populated in 2.2–2.6.
- **Expose in `lib.rs`.** `m.add_class::<topology::controller::RustTopologyController>()?;`
- **Python integration.** `topology_controller.py` gains `try/except` on import, stores `self._rust_ctrl = RustTopologyController() if available else None`. Both Python legacy and Rust coexist until 2.6 flips the default.
- **Test Rust.** `cargo test controller::new_has_expected_initial_state`.
- **Test Python.** `test_rust_controller_imports_successfully` — PyO3 roundtrip.
- **Maturin rebuild required.** `cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor` (~30s on this machine).
- **Commit.** Scaffold only — no logic yet.
- **Done-when.** Both tests pass, commit pushed.

### 2.2 Port path 1: empty/error reroute (1 session)

- **Python ref.** `topology_controller.py:134–149`.
- **Rust method.** `fn check_empty_error_reroute(&mut self, result: &str, node_idx: usize) -> Option<AdaptationDecision>`.
- **Python delegate.** `if self._rust_ctrl: decision = self._rust_ctrl.check_empty_error_reroute(result, node_idx); if decision: return decision` — legacy Python path remains below for paths not yet ported.
- **Equivalence test.** `test_rust_path1_matches_python` — 20+ sample inputs, same AdaptationDecision.action / target_node / reason from both implementations.
- **Done-when.** All 20+ cases match, commit.

### 2.3 Port paths 2–3: quality cascade + debate gate (1 session)

- **Paths.** `topology_controller.py:176–183`.
- **Scope clip.** Port path 2 (quality ≥ THETA_GOOD → continue) fully. For path 3 (debate gate), port the gate check; keep `_open_gate` helper in Python for now (pure Python logic reading topology graph — port later or leave as Python permanent wrapper).
- **Rust method.** `fn check_quality_cascade(&mut self, quality: f32, node_idx: usize) -> Option<AdaptationDecision>`.
- **Done-when.** Equivalence test for both paths.

### 2.4 Port paths 4–5: parallel inconsistency + importance prune (1 session)

- **Paths.** `topology_controller.py:201–222`.
- **Helpers to port.** `compute_consistency_score(outputs: &[&str]) -> f32` and `compute_importance_score(node_idx, result, parallel_outputs) -> f32`.
- **Rust methods.** `fn check_parallel_inconsistency(...)` and `fn check_importance_prune(...)`.
- **Done-when.** Equivalence tests on 20+ cases each.

### 2.5 Port path 6: emergent subtask spawn (1 session)

- **Path.** `topology_controller.py:224–233`.
- **Helper.** `_detect_emergent_subtask(result)` — uses regex. Port with `regex` crate (already a Rust dep).
- **Rust method.** `fn check_emergent_spawn(&mut self, result: &str) -> Option<AdaptationDecision>`.
- **Done-when.** Equivalence test.

### 2.6 Finalize: upgrade_model + fallback + ADR-012 (1 session)

- **Hardest path.** `_resolve_upgrade_model` reads `ModelRegistry` (already Rust) + routing state. Port becomes: Rust method that invokes `rust_model_registry.resolve_upgrade(...)`.
- **Python shrink.** `topology_controller.py` becomes a thin wrapper — ~50 lines, just delegates. Legacy Python logic deleted.
- **ADR-012.** `YGN-SAGE/Decisions/ADR-012-TopologyController-Rust-Port.md` — document the port, delegation pattern, fallback criteria, perf notes.
- **CLAUDE.md update.** Flip "TopologyController Rust primary" in `.claude/rules/architecture.md`.
- **Maturin final rebuild, full suite.** `cargo test` + `python -m pytest sage-python/tests/` — all green.
- **Done-when.** Commit, push, MEMORY.md + Obsidian Pillar-5 (Strategy) + Changelog updated.

---

## Session-Hygiene Protocol (mandatory)

To prevent catastrophic context loss across ~8 sessions:

### Session-start routine (~2 min, ALWAYS)

Before any code:
1. `Read docs/superpowers/plans/2026-04-20-rust-first-plan.md` — current checkpoint
2. `Read docs/audits/bypass-patterns.md` — methodology (in case refs needed)
3. `git log --oneline -10`
4. `git diff --stat HEAD~3..HEAD`
5. Identify the next unchecked item in the plan, state it explicitly to the user.

### Session budget

- **1–2 commits per session maximum.** Each ≤ 300 LOC, each includes a regression test.
- **Proactive stop if conversation > 4 hours.** No "one more fix."
- **Commit the plan file itself** each session when items flip state — the plan file IS the progress tracker.

### Session-close routine (~5 min, MANDATORY)

1. `git push origin main`
2. Update `MEMORY.md` — 1 line pointing to today's commit + statuses flipped
3. Update `docs/superpowers/plans/2026-04-20-rust-first-plan.md` — tick items done, note blockers
4. If Obsidian Changelog / Pillar / ADR needs update → do it NOW
5. Kill zombie shells (`ps -ef` audit)
6. Final commit: `chore: session close NNNN-NN-NN — Phase X.Y done, next = X.Z`

### Golden rule

> "If an important fact is not in a file at the end of the session, it ceases to exist in the next session."

---

## Bootstrap instruction for a fresh session

> "Execute the plan at `docs/superpowers/plans/2026-04-20-rust-first-plan.md`. Start with the session-startup routine in §Session-Hygiene Protocol. Do one item from Phase 1, commit, push, session-close. Do NOT exceed 1 commit unless the plan file explicitly allows."

That is sufficient context for a fresh Claude to resume autonomously.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Context loss between sessions distorts plan | Plan file is primary truth; conversation is ephemeral |
| Maturin rebuild breaks between sessions | Each Rust commit includes `maturin develop` success in the commit message |
| Phase 2 port introduces a bypass (H4-pattern recurrence) | Every port path has Rust-vs-Python equivalence test with 20+ cases before ship |
| Bypass found in Phase 1.5 inventory requires a pivot | Extend Phase 1 with the fix; document in bypass-patterns.md; does not invalidate Phase 2 |
| User priorities shift mid-plan | Plan is a menu, each Phase 1.x is independent. Can pause or re-order without breaking contract |

---

## References

- `docs/audits/2026-04-18-astropy-14995-decision-path.md` — the original audit that started this loop
- `docs/audits/bypass-patterns.md` — the methodology
- `~/.claude/projects/C--Code-YGN-SAGE/memory/MEMORY.md` — auto-loaded pointer for new sessions
- Recent commits: c905d06, 0b00abd, 2cd840e, dc51976, 27a9a4c, aa348e1
- Critical Directive #1 (Rust-First) in `CLAUDE.md`
- Related research (web, 2025–2026): "[Why some agentic AI developers are moving code from Python to Rust](https://developers.redhat.com/articles/2025/09/15/why-some-agentic-ai-developers-are-moving-code-python-rust)" — validates the Rust-owns-control-loop pattern for this port.
