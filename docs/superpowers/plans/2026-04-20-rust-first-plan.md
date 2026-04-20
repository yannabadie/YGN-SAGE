# Rust-First Architectural Completion — Executable Plan

**Status:** ACTIVE — plan of record. Every session updates this file.
**Spec:** [docs/superpowers/specs/2026-04-20-rust-first-plan-design.md](../specs/2026-04-20-rust-first-plan-design.md)
**Started:** 2026-04-20
**Session count:** 1 of ~8 planned

---

## IF YOU ARE A FRESH SESSION, READ THIS FIRST

You were told: "continue / execute the plan / next item / etc."

**Your startup routine** (do all of this, in order, before touching code):

```bash
# 1. Read this plan file top-to-bottom — you are here
# 2. Read the methodology
Read docs/audits/bypass-patterns.md

# 3. State of the repo
git log --oneline -10
git diff --stat HEAD~3..HEAD
git status --short
```

Then find the first `[ ]` item below and begin. One item per session maximum. When done, execute the §Session Close Routine before stopping.

**Do NOT** try to do multiple items. Do NOT try to "get ahead." The plan explicitly forbids it because every item needs its own empirical validation and its own commit. Stick to one.

---

## Progress tracker

| Phase | Item | Status | Commit | Session |
|---|---|---|---|---|
| 1 | 1.1 `max_steps` singleton audit | [x] | `b7ced9d` | 1 |
| 1 | 1.2 `stall_cap` singleton audit | [~] | (commit in flight) | 1 |
| 1 | 1.3 `tools` filter singleton audit | [ ] | — | — |
| 1 | 1.4 MAP-Elites archive growth smoke | [ ] | — | — |
| 1 | 1.5 PyO3 inventory sweep | [ ] | — | — |
| 1 | 1.6 ADR-011 Singleton vs Factory | [ ] | — | — |
| 2 | 2.1 Rust scaffold + PyO3 | [ ] | — | — |
| 2 | 2.2 Port path 1 (empty/error) | [ ] | — | — |
| 2 | 2.3 Port paths 2–3 (quality + debate) | [ ] | — | — |
| 2 | 2.4 Port paths 4–5 (parallel + prune) | [ ] | — | — |
| 2 | 2.5 Port path 6 (emergent spawn) | [ ] | — | — |
| 2 | 2.6 Finalize + ADR-012 | [ ] | — | — |

**Legend:** `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Phase 1 — Audit-Complete

### 1.1 `max_steps` singleton audit

**Suspected bug.** `boot.py:279` configures the singleton with `max_steps=MAX_AGENT_STEPS=20`. The bypass branch at `pipeline.py:941+` never re-scales by `ctx.system`. For S1 tasks this is 4x the factory's max.

**Touch points:**
- Read: `sage-python/src/sage/boot.py:270-285`, `sage-python/src/sage/pipeline.py:941-1020`, `sage-python/src/sage/agent_loop_factory.py:130-135`
- Edit: `sage-python/src/sage/pipeline.py` — after the `validation_level` assignment block (~line 977), add:
  ```python
  # Symmetric with agent_loop_factory.py:130-135 (H5/H6 pattern extension)
  self._agent_loop.config.max_steps = {1: 5, 2: 10, 3: 20}.get(ctx.system, 10)
  ```
- Edit: `sage-python/tests/test_pipeline.py` — new test:
  ```python
  @pytest.mark.asyncio
  async def test_pipeline_single_agent_scales_max_steps_by_system():
      # Same _SpyAgentLoop pattern as H5/H6 tests
      # For system=1 → config.max_steps == 5
      # For system=2 → 10
      # For system=3 → 20
  ```
- Edit: `docs/audits/bypass-patterns.md` §"Next audit" — move row from queued → done with commit SHA.

**Done-when:**
- Test passes in isolation
- Full suite `python -m pytest sage-python/tests/ --ignore=sage-python/tests/test_e2e_live_providers.py` still green
- Commit message references this plan item
- Plan file updated: `[ ]` → `[x]` with commit SHA

**Commit template:**
```
fix(agent): scale singleton max_steps by system — close H5-class bypass

Bypass path at pipeline.py:941 left singleton max_steps=20 regardless
of ctx.system, while factory scales 5/10/20 per system tier
(agent_loop_factory.py:130-135). S1 tasks on bypass path spun 4x the
intended budget before the loop could exit.

Extends the H5/H6 singleton-vs-factory asymmetry pattern documented in
bypass-patterns.md. Regression test mirrors H5's _SpyAgentLoop
approach.

Plan item 1.1 of docs/superpowers/plans/2026-04-20-rust-first-plan.md.
```

---

### 1.2 `stall_cap` singleton audit

**Suspected bug.** `AgentConfig.stall_after_tool_steps` default=0 disables D8 soft-breaker on the singleton. Factory computes `node_stall_cap = node_max_steps - 1` for S2/S3 (`agent_loop_factory.py:149-152`).

**Depends on 1.1** (needs max_steps set first to compute cap).

**Touch points:**
- Edit: `pipeline.py` — immediately after 1.1's edit:
  ```python
  new_max = self._agent_loop.config.max_steps
  self._agent_loop.config.stall_after_tool_steps = (new_max - 1) if new_max > 5 else 0
  ```
- Edit: `tests/test_pipeline.py` — new test:
  ```python
  @pytest.mark.asyncio
  async def test_pipeline_single_agent_sets_stall_cap_matching_factory():
      # system=3 → stall_cap == 19
      # system=1 → stall_cap == 0 (D8 off, budget too tight)
  ```
- Edit: `bypass-patterns.md` — flip queued row.

**Done-when / commit template:** same pattern as 1.1.

---

### 1.3 `tools` filter singleton audit

**Pre-analysis question.** Is this actually a bug? The singleton has no role (it's the default "actor"). Factory filters because nodes have roles. Single-agent bypass has no role concept → all tools is arguably correct.

**Procedure:**
1. Run a single-agent smoke (system=1) and a sequential-template smoke (system=2), grep logs for tool call invocations.
2. Compare: does the single-agent path invoke tools the factory would filter out (e.g., a sink-role node invoking execute_bash)?
3. **If no**: document in `bypass-patterns.md` §"Not a bypass (common false positives)" with evidence. No code change.
4. **If yes**: fix following H5 pattern — detect a "role hint" from the topology (if single-node topology has a `role` attribute, respect it).

**Done-when:**
- Either empirical evidence + doc update committed (false-positive branch)
- OR fix + regression test committed (real-bug branch)
- Plan updated, `bypass-patterns.md` updated.

---

### 1.4 MAP-Elites archive growth — empirical smoke

**Setup.**
```bash
export SAGE_LOG_LEVEL=INFO
export PYTHONUNBUFFERED=1
python -m sage.bench --type swebench --subset lite --limit 5 2>&1 | tee /tmp/archive-growth-smoke.log
```

**Assertions to grep from the log:**
```bash
# Cell count progression
grep -E "archive_entry_inserted|archive_cell_count" /tmp/archive-growth-smoke.log
# Evolution fires (H1)
grep -c "Online evolution fired" /tmp/archive-growth-smoke.log
```

**Deliverable.** `docs/benchmarks/2026-04-2x-archive-growth-smoke.md` containing:
- Run command + seed + date
- `cell_count` at each task completion (extracted from log)
- Total `evolve()` fires observed
- Verdict: ✅ growing / ❌ stuck at 0 (diagnose H4 downstream) / ⚠️ grows but never triggers evolve (diagnose constants)

**Done-when:**
- Benchmark doc committed
- If verdict is ❌ or ⚠️: Phase 1 gains an additional fix item before moving on (add row 1.4a).

---

### 1.5 PyO3 inventory sweep

**Script to run:**
```bash
# Print every #[pyclass] name
grep -hoE "#\[pyclass(\(name = \"[^\"]+\"\))?\]" sage-core/src/**/*.rs | head -50
# Cross-ref: for each class, count runtime Python refs (exclude tests + imports)
for cls in RustCompositeWriteGate RustRelevanceGate RustKnnRouter RustEntityGraph RustQualityEstimator ModelAssigner ModelRegistry SystemRouter TopologyEngine TopologyGraph TopologyNode ToolExecutor WasmSandbox ContextualBandit SmtVerifier RagCache HardwareProfile WorkingMemory; do
    py=$(grep -rln "$cls\b" sage-python/src/sage --include="*.py" 2>/dev/null | grep -v "^.*/tests/" | wc -l)
    echo "$cls: $py files"
done
```

**Deliverable.** `docs/audits/2026-04-2x-pyo3-inventory.md` — 3-column table:
- Rust class | Python runtime references (file:line) | Verdict (wired / stale / bypass)

**Triage rule per class:**
- 0 refs + in architecture.md as wired → **bypass**, add Phase 1 fix
- 0 refs + not claimed as wired → **stale/dead**, add to `bypass-patterns.md` false-positives
- ≥ 1 ref but only imports (no method calls) → inspect manually

**Done-when:**
- Inventory doc committed
- Any discovered bypass added to Phase 1 as a new item BEFORE moving to 1.6 or Phase 2.

---

### 1.6 ADR-011: Singleton vs Factory Asymmetry

**Prerequisite.** 1.1–1.3 committed so you have concrete evidence to cite.

**File.** `YGN-SAGE/Decisions/ADR-011-Singleton-vs-Factory-Asymmetry.md`

**Structure:**
- **Context** — 6 bypasses in 2026-04-19 evening; the three singleton bypasses (1.1, 1.2, 1.3 if applicable) extend the pattern.
- **Decision** — formalize the rule: the bypass branch at `pipeline.py:941+` MUST re-configure every attribute the factory sets in `create_node_agent_loop` (`agent_loop_factory.py`). Single source of truth for that list: the ADR body.
- **Consequences** — every future factory change MUST be mirrored in the bypass branch. Regression tests MUST cover both paths.
- **References** — H5, H6, 1.1, 1.2, 1.3 commits; `bypass-patterns.md`.

**Done-when:**
- ADR committed
- Phase 1 row in Progress tracker above all `[x]`
- Plan file commit: `docs: Phase 1 complete — ready for Phase 2`

---

## Phase 2 — TopologyController Rust Port

### 2.1 Rust scaffold + PyO3 + constants

**Create:** `sage-core/src/topology/controller.rs`

**Minimum scaffold:**
```rust
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct RustTopologyController {
    reroute_count: u32,
    spawn_count: u32,
    node_retries: HashMap<usize, u32>,
    abstain_count: u32,
    node_qualities: HashMap<usize, f32>,
}

// Thresholds — calibrated initial values, subject to ablation (CLAUDE.md directive).
// Mirror of topology_controller.py constants.
const THETA_GOOD: f32 = 0.7;
const THETA_CRITICAL: f32 = 0.3;
const THETA_CONSISTENCY: f32 = 0.5;
const THETA_PRUNE: f32 = 0.2;
const MAX_RETRIES: u32 = 2;
const MAX_REROUTES: u32 = 1;
const MAX_GATE_TURNS: u32 = 2;
const MAX_SPAWNS: u32 = 3;

#[pymethods]
impl RustTopologyController {
    #[new]
    fn new() -> Self { /* init all to 0 / empty */ }

    // Stub — populated in 2.2–2.6
    fn evaluate_and_decide(
        &mut self,
        node_idx: usize,
        result: String,
        task: String,
    ) -> Option<AdaptationDecision> {
        None  // triggers Python fallback
    }
}
```

**Edit `sage-core/src/lib.rs`** — add `m.add_class::<topology::controller::RustTopologyController>()?;` in module init.

**Edit `sage-python/src/sage/topology_controller.py`** — add import guard at top:
```python
try:
    from sage_core import RustTopologyController
    _HAS_RUST_CTRL = True
except ImportError:
    _HAS_RUST_CTRL = False
```
In `TopologyController.__init__`, conditionally create `self._rust_ctrl = RustTopologyController() if _HAS_RUST_CTRL else None`. No behavior change yet — logic continues through Python.

**Maturin rebuild (required after every 2.x edit that touches Rust):**
```bash
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor
```

**Tests:**
- Rust: `cargo test controller::new_has_expected_initial_state` in `sage-core/src/topology/controller.rs`
- Python: `test_rust_topology_controller_imports_successfully` in new `sage-python/tests/test_rust_controller.py`

**Done-when:** both tests green, commit pushed, maturin rebuild verified in commit message.

---

### 2.2 Port path 1 — empty/error reroute

**Python ref.** `topology_controller.py:134-149`.

**Rust method.**
```rust
fn check_empty_error_reroute(
    &mut self,
    result: &str,
    node_idx: usize,
) -> Option<AdaptationDecision> {
    if !is_empty_or_error(result) { return None; }
    if self.reroute_count >= MAX_REROUTES {
        return Some(AdaptationDecision::continue_at(node_idx, "reroute budget exhausted"));
    }
    self.reroute_count += 1;
    let reason = if result.trim().is_empty() { "empty output" } else { "error-like output" };
    Some(AdaptationDecision::reroute(node_idx, reason))
}
```

**Python delegate.** Early in `evaluate_and_decide`, after state init:
```python
if self._rust_ctrl is not None:
    decision = self._rust_ctrl.check_empty_error_reroute(result or "", node_idx)
    if decision is not None:
        return decision
# else fall through to Python legacy
```

**Equivalence test.** `sage-python/tests/test_rust_controller.py::test_path1_matches_python_on_20_samples` — 20 (result, node_idx) tuples; both controllers produce same `(action, target_node, reason)`.

**Done-when:** all 20 cases match, commit, plan updated.

---

### 2.3 Port paths 2–3 — quality cascade + debate gate

**Python ref.** `topology_controller.py:176-183`.

**Scope.** Port quality-cascade fully. For debate gate, port the threshold check; `_open_gate` helper stays Python for this commit (port in a future sprint if needed).

**Rust method.** `check_quality_cascade(&mut self, quality: f32, node_idx: usize) -> Option<AdaptationDecision>`.

**Test.** `test_path2_3_matches_python_on_20_samples`.

**Done-when:** equivalence across 20 samples, commit, plan updated.

---

### 2.4 Port paths 4–5 — parallel inconsistency + importance prune

**Python ref.** `topology_controller.py:201-222`.

**Helpers to port (new Rust functions):**
- `compute_consistency_score(outputs: &[String]) -> f32`
- `compute_importance_score(node_idx: usize, result: &str, parallel_outputs: &[String]) -> f32`

**Rust methods:**
- `check_parallel_inconsistency(&mut self, node_idx, parallel_outputs) -> Option<AdaptationDecision>`
- `check_importance_prune(&self, node_idx, result, parallel_outputs, quality_is_known) -> Option<AdaptationDecision>`

**Done-when:** equivalence tests, commit, plan updated.

---

### 2.5 Port path 6 — emergent subtask spawn

**Python ref.** `topology_controller.py:224-233`.

**Helper.** `detect_emergent_subtask(result: &str) -> Option<String>` using `regex` crate (already a Rust dep).

**Rust method.** `check_emergent_spawn(&mut self, result: &str) -> Option<AdaptationDecision>`.

**Done-when:** equivalence test, commit, plan updated.

---

### 2.6 Finalize + ADR-012

**Port `_resolve_upgrade_model`.** Reads Rust `ModelRegistry` (already PyO3-exposed). Rust method delegates to registry lookup.

**Shrink Python `topology_controller.py`.** With all 6 paths ported, the Python class becomes a thin wrapper:
```python
class TopologyController:
    def __init__(self, ...):
        self._rust_ctrl = RustTopologyController() if _HAS_RUST_CTRL else None
        # ... (legacy Python state for fallback only)

    def evaluate_and_decide(self, node_idx, result, task, topology, ctx, parallel_outputs, *, output=None):
        if self._rust_ctrl:
            return self._rust_ctrl.evaluate_and_decide(node_idx, result or output or "", task, ...)
        return self._legacy_python_evaluate(...)
```

**Full suite.**
```bash
cd sage-python && python -m pytest tests/ --ignore=tests/test_e2e_live_providers.py --ignore=tests/integration
cd sage-core && cargo test
```
Both must be fully green.

**ADR-012.** `YGN-SAGE/Decisions/ADR-012-TopologyController-Rust-Port.md`
- Context — Critical Directive #1 violation catalogued in `bypass-patterns.md`
- Decision — port, delegate from Python
- Consequences — Python `topology_controller.py` is now a ~50-line wrapper; legacy Python logic deleted. Performance: controller invocations (~100/run) now Rust-native.
- References — commits 2.1–2.6.

**Update `.claude/rules/architecture.md`** — flip "TopologyController: Python" mention in Pillar 5 to "TopologyController: Rust primary + Python fallback".

**Update `CLAUDE.md`** — tests count row if it changed.

**Commit messages** must always reference this plan item (`2.6 of 2026-04-20-rust-first-plan.md`).

**Done-when:**
- All 12 Progress tracker rows `[x]`
- ADR-012 committed
- `CLAUDE.md` + `architecture.md` updated
- Plan file: add "COMPLETE — 2026-05-XX, N sessions, Z commits" banner at top

---

## Session Close Routine (run every session, in order)

```bash
# 1. Push (never skip)
git push origin main

# 2. Update plan file — tick boxes, log surprises
$EDITOR docs/superpowers/plans/2026-04-20-rust-first-plan.md

# 3. Update MEMORY.md — 1 line per session
$EDITOR ~/.claude/projects/C--Code-YGN-SAGE/memory/MEMORY.md

# 4. Obsidian touch (only when a major milestone lands)
#    - New ADR?           → YGN-SAGE/Decisions/ + Decisions-MOC update
#    - Phase complete?    → YGN-SAGE/Architecture/Changelog-Apr9-XX.md
#    - Pillar changed?    → YGN-SAGE/Architecture/Pillar-N-*.md

# 5. Kill zombie shells
ps -ef | awk '$5 ~ /^Apr/ {print $2}' | xargs kill 2>/dev/null

# 6. Final commit if plan/memory/obsidian changed
git add . && git commit -m "chore: session close YYYY-MM-DD — Phase X.Y done, next = X.Z"
git push origin main
```

---

## Risks (also in spec §Risks)

- **Context loss:** mitigated by this plan file being the primary truth, not conversation
- **Maturin rebuild drift:** every Rust commit message must state `maturin develop` success
- **H4-pattern recurrence:** every port path has Rust-vs-Python equivalence test with 20+ cases
- **Bypass found in 1.5:** extend Phase 1, do not skip
- **User priority shift:** plan is menu — each item is independent

---

## Log of session updates (append-only)

Every session that touches this file appends a line here.

- 2026-04-20 [plan written, session 0] — spec written, plan checked in, no code yet
- 2026-04-20 [session 1] — 1.1 max_steps singleton audit: wired `self._agent_loop.config.max_steps = {1:5, 2:10, 3:20}.get(ctx.system, 10)` in `pipeline.py` after the validation_level block; added regression test `test_pipeline_single_agent_scales_max_steps_by_system` using `SAGE_ABLATION_NO_TOPOLOGY=1` to force bypass across all three tiers; updated `docs/audits/bypass-patterns.md`. 36/36 `test_pipeline.py` green; full suite: 1927 passed, 5 errors + 1 failed all pre-existing asyncio-fixture pollution (pass in isolation).
