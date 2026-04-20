# Post-Rust-First Phase 1 Stabilization — Design

**Date:** 2026-04-20
**Status:** Spec — awaiting user review → writing-plans
**Follows:** `docs/superpowers/plans/2026-04-20-rust-first-plan.md` (completed same day, commits `152fe5e..e26cd7b` + H11 fix `55f5086`)
**Triage evidence:** `docs/audits/2026-04-20-alire-docs-triage.md` (empirical grep) + Codex verdict (session `019daafc-f624-7e81-b774-1deae1b8ab59`, 2026-04-20, converged on all ALIRE.md hallucinations)

## 1. Context

The Rust-First plan left two architectural warts and one interrupted cleanup:

1. **`__setattr__` mirror magic** (commit `e26cd7b`, `sage-python/src/sage/topology_controller.py`): the Python TopologyController uses a `__setattr__` hook to auto-propagate `_reroute_count` / `_spawn_count` mutations to the Rust controller. This was explicitly transitional (ADR-012).
2. **Duplicated state across the boundary**: Python holds `self._reroute_count`, `self._spawn_count`, `self._node_retries`, `self._abstain_count`; Rust holds the same fields. The `__setattr__` mirror keeps them aligned at write time, but reads are inconsistent (some code reads Python, some reads Rust getters).
3. **Path 6 emergent-subtask regex detection** (`sage-core/src/topology/controller.rs:40-66`, three regex patterns + `detect_emergent_subtask` + `check_emergent_spawn`): violates Directive #2 (minimal heuristics). Functionally superseded by the `sage_recurse` tool (commit `13463fb`, Sprint 4) which is registered by default in `boot.py:520`. The user asked to remove this ("H12", interrupted in the prior session).

Two user-opened proposals framed the direction:
- `ALIRE.md` — "Path 6c embedding-guided evolution": rejected in triage. Every proposed Rust API is invented (`SmmuContext::retrieve_similar_topologies`, `MapElitesArchive::get_elite`, `HybridVerifier::light_verify`, `TopologySchema` type, Rust named arguments). Contradicts itself on "model-free" vs QualityEstimator dependency and on "not heuristic" vs `omega > 0.65` branches. Not this spec. A clean Path 6c against real APIs is a separate future spec.
- `ALIRE2.md` §1.1 — explicit sync methods replacing `__setattr__`: good direction, but the bidirectional `sync_from_python` / `sync_to_python` shape it proposes is overreach. Codex flagged: "Rust is authoritative — don't maintain duplicated Python counters." This spec adopts the **Rust-authoritative-façade** variant instead.
- `ALIRE2.md` §1.2 — TOML-configured path-6 regex + optional ONNX fallback: rejected. Dresses a heuristic as configuration. The right answer (per Directive #2 + `sage_recurse` existence) is to **delete** the regex.

## 2. Goal

Close the post-Rust-First architectural debt in one cleanup pass:

1. Make Rust the single source of truth for `TopologyController` runtime state.
2. Turn Python `TopologyController` into a thin read-only façade with an explicit test-seeding back door.
3. Remove the path-6 regex detection entirely and route emergent subtasks exclusively through the `sage_recurse` tool, with a Rust-side budget gate.
4. Delete the `_evaluate_and_decide_legacy` fallback and the `sage_core`-optional code path.

Out of scope (future specs):
- New Path 6c policy (re-derived against real APIs).
- Sprint 5 ablation execution.
- OxiZ v0.2.0 upgrade.
- Real SWE-bench smoke validation.

Success criteria:
- `__setattr__` magic removed from `TopologyController`.
- Zero Python-side shadow copies of Rust state (`self._reroute_count` etc. deleted).
- Path-6 regex patterns + `detect_emergent_subtask` + `check_emergent_spawn` deleted from Rust and Python.
- `sage_recurse` tool gated by `should_trigger_emergent_spawn` + `record_emergent_spawn` at dispatch time.
- Importing `TopologyController` without `sage_core` raises a clear `ImportError` at `__init__`, not a silent fallback.
- Python test suite ≥ 1944 passed (vs 1939 baseline; net ≈ +5-6 after migrations).
- Rust test suite ≥ 473 passed (vs 478 baseline; net ≈ -3 to -5 after dead-test deletion, +7 for new scaffolds).
- No regression on BCB Hard / SWE-bench smoke (benchmarks not re-run — this is architectural, not scoring-path).

## 3. Architecture

```
┌── Python sage.topology_controller ──────────────────┐
│  class TopologyController:                          │
│    __init__: creates self._rust_ctrl                │
│             raises ImportError if no sage_core      │
│    @property reroute_count → self._rust_ctrl.reroute_count │
│    @property spawn_count   → self._rust_ctrl.spawn_count   │
│    @property abstain_count → self._rust_ctrl.abstain_count │
│    @property node_retries  → dict(self._rust_ctrl.node_retries_view()) │
│    evaluate_and_decide() → delegates to Rust directly │
│    _seed_for_tests(reroute=0, spawn=0, retries=None, │
│                    abstain=0) → Rust seed method    │
│                                                     │
│  DELETED: _evaluate_and_decide_legacy (l.333, ~140 lines) │
│  DELETED: __setattr__ mirror (l.~80-95, ~15 lines)  │
│  DELETED: self._reroute_count, self._spawn_count,   │
│           self._node_retries, self._abstain_count (shadow) │
└───────────────────────┬─────────────────────────────┘
                        │ PyO3 (read-only getters + seed + mutate-via-methods)
┌───────────────────────▼─────────────────────────────┐
│  Rust RustTopologyController (authoritative)        │
│    state: reroute_count, spawn_count, node_retries, │
│           abstain_count, node_qualities, gate_loops │
│    constants: THETA_* (unchanged), MAX_* (unchanged │
│               including MAX_SPAWNS — user requested │
│               keeping spawn_count tracking)         │
│                                                     │
│  NEW getters: reroute_count, spawn_count,           │
│               abstain_count, node_retries_view(),   │
│               node_qualities_view()                 │
│  NEW methods: should_trigger_emergent_spawn(node) → bool │
│               record_emergent_spawn(node) → Result  │
│               seed_state_for_legacy_tests(...)      │
│                                                     │
│  DELETED: detect_emergent_subtask (module fn, 14 L) │
│  DELETED: emergent_subtask_res (3 regex patterns, 12 L) │
│  DELETED: check_emergent_spawn (compound detector+budget method) │
└─────────────────────────────────────────────────────┘
                        │
                        │ agent calls sage_recurse tool
┌───────────────────────▼─────────────────────────────┐
│  sage_recurse tool (sage-python/src/sage/tools/     │
│    sage_recurse.py)                                 │
│  NEW: controller param at build time                │
│  NEW: sage_recurse_origin_node ContextVar           │
│  NEW: budget gate (should_trigger + record before   │
│       dispatch)                                     │
│  Unchanged: sage_recurse_depth ContextVar,          │
│             MAX_DEPTH guard                         │
└─────────────────────────────────────────────────────┘
                        ▲
                        │ sets origin_node ContextVar
┌───────────────────────┴─────────────────────────────┐
│  TopologyRunner (sage-python/src/sage/execution/)   │
│  Around agent.run(): sets sage_recurse_origin_node  │
│  to node.index                                      │
└─────────────────────────────────────────────────────┘
```

## 4. Components

### 4.1 Rust additions — `sage-core/src/topology/controller.rs`

```rust
#[pymethods]
impl RustTopologyController {
    // ── Getters ────────────────────────────────────────
    #[getter] fn reroute_count(&self) -> u32 { self.reroute_count }
    #[getter] fn spawn_count(&self) -> u32 { self.spawn_count }
    #[getter] fn abstain_count(&self) -> u32 { self.abstain_count }

    // Dict-shaped state exposed as list-of-tuples (PyO3 dict requires GIL dance)
    fn node_retries_view(&self) -> Vec<(usize, u32)> {
        self.node_retries.iter().map(|(k, v)| (*k, *v)).collect()
    }
    fn node_qualities_view(&self) -> Vec<(usize, f32)> {
        self.node_qualities.iter().map(|(k, v)| (*k, *v)).collect()
    }

    // ── Emergent spawn gating (replaces check_emergent_spawn) ──
    fn should_trigger_emergent_spawn(&self, _node_idx: usize) -> bool {
        self.spawn_count < MAX_SPAWNS
    }

    fn record_emergent_spawn(&mut self, _node_idx: usize) -> PyResult<()> {
        if self.spawn_count >= MAX_SPAWNS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "spawn budget exhausted ({}/{} reached)",
                self.spawn_count, MAX_SPAWNS
            )));
        }
        self.spawn_count += 1;
        Ok(())
    }

    // ── Test seed (replaces __setattr__ magic) ────────
    fn seed_state_for_legacy_tests(
        &mut self,
        reroute_count: u32,
        spawn_count: u32,
        node_retries: Vec<(usize, u32)>,
        abstain_count: u32,
    ) -> PyResult<()> {
        if reroute_count > MAX_REROUTES + 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reroute_count={} implausible (MAX_REROUTES={})",
                reroute_count, MAX_REROUTES
            )));
        }
        self.reroute_count = reroute_count;
        self.spawn_count = spawn_count;
        self.node_retries = node_retries.into_iter().collect();
        self.abstain_count = abstain_count;
        Ok(())
    }
}
```

### 4.2 Rust removals — `sage-core/src/topology/controller.rs`

| Lines | Symbol | Why |
|---|---|---|
| ~40-52 | `fn emergent_subtask_res() -> &'static [Regex]` | 3 regex patterns (Directive #2) |
| ~54-66 | `fn detect_emergent_subtask(&str) -> Option<String>` | module-level detector, 14 lines |
| ~295-395 (approx) | `fn check_emergent_spawn(&mut self, result, node_idx) -> Option<RustAdaptationDecision>` | compound method combining detector + budget |
| test suite | `detect_emergent_subtask_picks_up_{todo,additionally,prerequisite}_pattern` + `_none_on_plain_content` | 4 tests on deleted detector |
| test suite | `check_emergent_spawn_{fires_and_increments,respects_max_spawns,returns_none_when_no_match}` | 3 tests on deleted method |
| test consts | `LLM_OUTPUT_WITH_TODO`, `LLM_OUTPUT_WITH_ADDITIONALLY`, `LLM_OUTPUT_WITH_PREREQUISITE`, `LLM_OUTPUT_EMERGENT_FOR_SPAWN`, `LLM_OUTPUT_EMERGENT_BURST` | unused after test deletion |

`use regex::Regex` remains — `error_output_re()` (path 1 empty/error detection) still uses regex. Its pattern vocabulary is enumerable and bounded, legitimate under Directive #2 as a detector for a closed set of error/traceback surfaces.

### 4.3 Python changes — `sage-python/src/sage/topology_controller.py`

**Removed (~200+ lines net):**
- `__setattr__` mirror method (15 lines, commit `e26cd7b`).
- `_evaluate_and_decide_legacy` method (~140 lines, starts at l.333).
- Its call site at l.196 (`if not _HAS_SAGE_CORE: return self._evaluate_and_decide_legacy(...)`).
- Shadow state init in `__init__`: `self._reroute_count = 0`, `self._spawn_count = 0`, `self._node_retries = {}`, `self._abstain_count = 0`.
- All 30 references to `self._{reroute,spawn,abstain}_count` and `self._node_retries[i]` — rewritten to `self._rust_ctrl.*` (reads) or appropriate setter methods (mutations).

**Added (~30 lines):**
- `ImportError` raise at `__init__` if `not _HAS_SAGE_CORE`.
- Properties: `reroute_count`, `spawn_count`, `abstain_count`, `node_retries` (all `@property`, no setters — assigning raises `AttributeError`).
- `_seed_for_tests(reroute=0, spawn=0, retries=None, abstain=0)` helper that calls `self._rust_ctrl.seed_state_for_legacy_tests(...)`.

**Kept:**
- `set_node_retries(node_idx, value)` individual setter on Rust (exists, used by H11 cross-path test + pipeline arithmetic-verify branch).
- All six path methods (`check_empty_error_reroute`, `check_quality_cascade`, etc.) — unchanged.

### 4.4 `sage_recurse` changes — `sage-python/src/sage/tools/sage_recurse.py`

**Added:**
```python
from contextvars import ContextVar

# NEW — set by TopologyRunner before node dispatch
sage_recurse_origin_node: ContextVar[int | None] = ContextVar(
    "sage_recurse_origin_node", default=None
)

def build_sage_recurse_tool(
    run_fn: Callable[..., Awaitable[Any]],
    controller: "TopologyController | None" = None,  # NEW
) -> Tool:
    async def sage_recurse(sub_task: str, budget_usd: float | None = None,
                           hint: str | None = None) -> str:
        # 1. Input sanity (unchanged)
        if not sub_task or not sub_task.strip():
            return "Error: sage_recurse requires a non-empty sub_task."

        # 2. Depth guard (unchanged)
        depth = sage_recurse_depth.get()
        if depth >= MAX_DEPTH:
            _log.warning("sage_recurse refused: depth %d >= MAX %d", depth, MAX_DEPTH)
            return f"Error: sage_recurse refused — max recursion depth {MAX_DEPTH} reached"

        # 3-4. NEW — budget gate
        origin_node = sage_recurse_origin_node.get()
        if controller is not None and origin_node is not None:
            if not controller._rust_ctrl.should_trigger_emergent_spawn(origin_node):
                _log.info("sage_recurse refused: spawn budget exhausted "
                          "(node=%d, spawn_count=%d)",
                          origin_node, controller._rust_ctrl.spawn_count)
                return "Error: sage_recurse refused — spawn budget exhausted for this execution"
            try:
                controller._rust_ctrl.record_emergent_spawn(origin_node)
            except Exception as exc:
                _log.error("sage_recurse record failed: %s", exc)
                return f"Error: sage_recurse refused — {exc}"

        # 5. Dispatch (unchanged body, wrapped in try)
        try:
            token = sage_recurse_depth.set(depth + 1)
            try:
                result = await run_fn(sub_task, budget_usd=budget_usd, hint=hint)
                return str(result)
            finally:
                sage_recurse_depth.reset(token)
        except Exception as exc:
            _log.exception("sage_recurse dispatch failed")
            return f"Error: sage_recurse dispatch failed: {type(exc).__name__}: {exc}"

    return Tool(name="sage_recurse", ...)
```

**Rationale for gating invariants:**
- Budget-record **before** dispatch: a failed spawn still counts. Prevents an agent from looping on `sage_recurse(sub_task="retry")` after errors.
- Gate no-op if `controller is None`: preserves standalone tool testability.
- Gate no-op if `origin_node is None`: allows sage_recurse usage from contexts outside the topology runner (scripts, nested calls).
- Budget is **global per TopologyController instance** (single `u32`), not per-node. User-confirmed simpler design; per-node HashMap is future evolution if observed hotspots.

### 4.5 `boot.py` wiring — `sage-python/src/sage/boot.py`

Resolved accessor path: `boot_pipeline.py:297` already wires `_pipeline.controller = _controller`, so at boot.py tool-registration time (l.520) the controller is reachable via `system.pipeline.controller`. No new accessor needed.

```python
# sage-python/src/sage/boot.py  (l.518-523 becomes:)
try:
    from sage.tools.sage_recurse import build_sage_recurse_tool
    controller = getattr(system.pipeline, "controller", None)
    tool_registry.register(build_sage_recurse_tool(system.run, controller=controller))
    _log.info("Core tools: sage_recurse registered (max depth 3, budget-gated=%s)",
              controller is not None)
except (ImportError, RuntimeError) as exc:
    _log.debug("sage_recurse not available: %s", exc)
```

`getattr(..., None)` keeps graceful-degrade: if a deployment builds the system without `TopologyController` (e.g. `model_assigner=None` per `boot_pipeline.py:266` gate), the tool still registers without a gate (equivalent to pre-refactor behavior). Logged so operators can see the gate status.

### 4.6 `TopologyRunner` hook

Exact seam: `sage-python/src/sage/topology/runner.py:840` — `TopologyRunner._execute_node(self, node_idx, task, context_override)`. This method is the single entry point that dispatches to `_execute_code_node`, `_execute_solver_node`, or `_execute_node_via_agent_loop` (code-node, solver-node, and LLM-node respectively). Wrapping here covers all three node types.

```python
# sage-python/src/sage/topology/runner.py
from sage.tools.sage_recurse import sage_recurse_origin_node

async def _execute_node(
    self, node_idx: int, task: str, context_override: str | None = None,
) -> str:
    """Execute a single topology node — LLM call or code sandbox."""
    token = sage_recurse_origin_node.set(node_idx)
    try:
        node = self.graph.get_node(node_idx)
        node_type = getattr(node, "node_type", "llm")
        if node_type == "code":
            return await self._execute_code_node(node_idx, task, context_override)
        if node_type == "solver" or getattr(node, "role", "") == "solver":
            return await self._execute_solver_node(node_idx, task, context_override)
        if self._agent_loop_factory:
            return await self._execute_node_via_agent_loop(node_idx, task, context_override)
        # ...existing fallback path for LLM nodes without factory...
    finally:
        sage_recurse_origin_node.reset(token)
```

Note: the fallback path after `_agent_loop_factory` check (lines ~870+ of the current file) must also sit inside the try block — the existing code falls through without an explicit return in one branch. Plan step will verify by running the LLM-fallback tests.

Parallel branches (`asyncio.gather` at l.1154, l.1331): each concurrent call to `_execute_node` gets its own ContextVar token via `.set()` — ContextVars are copied per-task in asyncio, so parallel nodes see their own `origin_node` without cross-pollination. No additional locking needed.

## 5. Data flow (summary)

Detailed in brainstorm session. Four canonical flows:

- **Flow A — Mutation during `evaluate_and_decide`:** Python → Rust method call → Rust mutates its own state → Python wrapper enriches optional fields (model_id, invariant_feedback, gate_source/target). No shadow mirror.
- **Flow B — State reads:** Python accesses `@property` → getter crosses PyO3 once → returns Rust value. No caching.
- **Flow C — Emergent spawn:** Agent calls `sage_recurse` tool → depth guard → budget guard (should_trigger + record) → dispatch `run_fn` (recursive pipeline call) → always returns a string. Budget debits on both success and failure.
- **Flow D — Test seed:** `ctrl._seed_for_tests(reroute=1, retries={0: 2})` → `_rust_ctrl.seed_state_for_legacy_tests(1, 0, [(0, 2)], 0)`. Bypasses the normal mutation path; test-only.

## 6. Error handling

### 6.1 Missing `sage_core` at init
Raises `ImportError` at `TopologyController.__init__` with actionable message pointing to `maturin develop`. Changes contract vs prior silent fallback to `_evaluate_and_decide_legacy`.

### 6.2 Direct Python attribute mutation
`ctrl.reroute_count = 5` now raises `AttributeError` (no `.setter` on the `@property`). Callers must use `_seed_for_tests` or go through `evaluate_and_decide`.

### 6.3 `record_emergent_spawn` at cap
Returns `PyValueError("spawn budget exhausted ...")`. `sage_recurse` catches and returns an error-string (tool contract: never raise).

### 6.4 `seed_state_for_legacy_tests` implausible input
Returns `PyValueError` if `reroute_count > MAX_REROUTES + 10` (slack for test scenarios that probe "just past the cap" — tighter bound rejects legitimate edge-case tests). No lower bound (u32 cannot be negative).

### 6.5 Tool dispatch failure
`sage_recurse` catches any exception from `run_fn`, logs it, returns `"Error: sage_recurse dispatch failed: ..."`. Budget already decremented (intentional DoS guard).

## 7. Testing strategy

### 7.1 Deletions (dead code after H12)
- Rust: 4 detect tests, 3 check_emergent_spawn tests, 5-6 unused `LLM_OUTPUT_*` consts. ≈ 7-10 tests.
- Python: 2-3 path 6 tests in `test_rust_controller.py`, `test_max_spawns_respected` in `test_topology_controller.py` (migrated to budget-gate test instead).

### 7.2 Migrations (seed refactor)
Each direct mutation of `_reroute_count` / `_spawn_count` / `_node_retries[i]` / `_abstain_count` in tests → call to `_seed_for_tests(...)`. 7-8 call sites across `test_topology_controller.py`, `test_pipeline_adaptation.py`, `test_rust_controller.py`.

### 7.3 New Rust tests (+7)
```rust
#[test] fn should_trigger_emergent_spawn_allows_within_budget()
#[test] fn should_trigger_emergent_spawn_refuses_at_cap()
#[test] fn record_emergent_spawn_increments_counter()
#[test] fn record_emergent_spawn_errors_at_cap()
#[test] fn seed_state_for_legacy_tests_populates_all_fields()
#[test] fn seed_state_for_legacy_tests_rejects_implausible_reroute()
#[test] fn getters_match_internal_state_after_decisions()
```

### 7.4 New Python tests (+4 unit + 5 integration = 9)
Unit (`test_topology_controller.py`):
- `test_python_facade_reads_rust_state`
- `test_python_facade_rejects_direct_mutation` (expects `AttributeError`)
- `test_missing_sage_core_raises_importerror` (monkeypatch `_HAS_SAGE_CORE=False`)
- `test_seed_for_tests_is_observable`

Integration (`test_sage_recurse_spawn_gate.py`, new file):
- `test_sage_recurse_gate_allows_first_spawn`
- `test_sage_recurse_gate_refuses_over_budget`
- `test_sage_recurse_no_gate_without_controller` (backwards compat)
- `test_sage_recurse_records_budget_before_dispatch` (dispatch fail still decrements)
- `test_sage_recurse_origin_node_contextvar_set_by_runner`

### 7.5 Equivalence anti-regression
One synthetic cascade test that drives `evaluate_and_decide` through a realistic quality-cascade-then-reroute scenario and asserts `controller.X == controller._rust_ctrl.X` for every exposed property. Protects against silent drift between façade and source-of-truth.

### 7.6 Ablation flag
`SAGE_ABLATION_NO_RECURSE=1` must keep working — adds a unit test loading `boot.py` with the flag on and verifying `TopologyController` still constructs without the tool registered.

### 7.7 Expected test counts post-change
- Rust: 478 → ~473-475 (net -3 to -5).
- Python: 1939 → ~1944-1945 (net +5-6).

## 8. Commit sequencing (proposed, writing-plans refines)

Candidate ordering — each commit compiles + passes its own scope tests:

1. **Commit A — Rust getters + seed.** Additive. Tests: getters + seed_state. Python still uses shadow state + `__setattr__`. No regressions.
2. **Commit B — Python façade conversion.** Replace shadow state + `__setattr__` with `@property` + `_seed_for_tests`. Migrate all test sites. Delete `_evaluate_and_decide_legacy`. Add `ImportError`. Delete `_HAS_SAGE_CORE` fallback branches.
3. **Commit C — H12 removal.** Delete `detect_emergent_subtask`, `check_emergent_spawn`, path-6 regex, associated tests + consts. Add `should_trigger_emergent_spawn` + `record_emergent_spawn`. Delete `MAX_SPAWNS` test that's been superseded.
4. **Commit D — `sage_recurse` gate.** Add `sage_recurse_origin_node` ContextVar, `controller` build param, gate logic. Wire `boot.py`. Add integration tests.
5. **Commit E — `TopologyRunner` hook.** Set/reset `sage_recurse_origin_node` around node dispatch.
6. **Commit F — Docs + ADR finalization.** Update ADR-012 (mark "Python wrapper legacy path" as fully removed). Update `CLAUDE.md` test counts + architecture summary. Update `.claude/rules/architecture.md` for S1/S2/S3/Strategy pillar description.

Plan skill will produce the detailed step-by-step executing-plans version.

## 9. Risks

| Risk | Mitigation |
|---|---|
| `@property` read adds PyO3 overhead per access | Controller not hot-path; measured negligible. If proved hot in profiling, snapshot once per `evaluate_and_decide` call into a local var. |
| Deleting `_evaluate_and_decide_legacy` breaks sage_core-less environments | All test configs and CI require sage_core. Documented in CLAUDE.md. Explicit ImportError gives clear failure mode. |
| `sage_recurse` backwards-compat for standalone callers | `controller` param is `Optional`, defaults to `None`. Existing callers unchanged. |
| Test migrations miss a shadow-state reference and runtime fails | Grep-guided audit + the equivalence test in §7.5 would catch drift. Commit B lands tests and production code together. |
| `record_emergent_spawn` debits on failure too — DoS guard may feel "unfair" | Documented behavior in tool docstring + integration test. Alternative (debit only on success) reopens the looping-agent attack surface. |
| Parallel node execution via `asyncio.gather` (l.1154, l.1331 of runner.py) — ContextVar coherence | Python 3.7+ asyncio copies ContextVar on `asyncio.create_task` / `asyncio.gather`, so each concurrent `_execute_node` call has its own `sage_recurse_origin_node`. Verified behavior; add an explicit parallel test (origin_node differs per concurrent node) for regression. |

## 10. References

- `docs/audits/2026-04-20-alire-docs-triage.md` — empirical ALIRE.md triage (converged with Codex verdict).
- `docs/superpowers/plans/2026-04-20-rust-first-plan.md` — parent plan (completed).
- `docs/superpowers/specs/2026-04-20-rust-first-plan-design.md` — parent spec.
- `YGN-SAGE/Decisions/ADR-012-TopologyController-Rust-Port.md` — port decision that flagged this as follow-up.
- `docs/audits/bypass-patterns.md` — methodology for validating bypass fixes (apply §4 empirical check at plan completion).
- PyO3 guidance (via Context7 `/pyo3/pyo3`): `#[pyclass]` with `&mut self` methods is the canonical mutable-state pattern; `#[pyclass(frozen)]` + `Mutex` is for thread-safe alternatives (not needed here — controller is single-threaded per pipeline).
- Codex consultation session `019daafc-f624-7e81-b774-1deae1b8ab59` — resumable via `/consult --continue` if reconcile needed during implementation.
