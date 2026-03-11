# Cognitive Engine Gap Fill — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill all 8 gaps between the Cognitive Engine design doc and the actual codebase — make routing end-to-end, wire the topology executor, improve quality signals, and complete all missing integration points.

**Architecture:** Modify Rust `system_router.rs` to integrate bandit + telemetry scoring. Add `topology_id` to `RoutingDecision`. Wire `TopologyExecutor` into Python `agent_loop.py`. Improve quality estimation in `boot.py`. Add `set_decay_factor` to bandit. All changes are additive — no breaking API changes.

**Tech Stack:** Rust (PyO3 0.25, petgraph 0.6, serde), Python 3.12 (sage-python)

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `sage-core/src/routing/system_router.rs` | Modify | Add bandit integration, telemetry scoring, topology_id in RoutingDecision |
| `sage-core/src/routing/bandit.rs` | Modify | Add `set_decay_factor()`, `warm_start_from_cards()` |
| `sage-core/src/routing/model_registry.rs` | Modify | Add telemetry calibration (observation-based affinity adjustment) |
| `sage-python/src/sage/boot.py` | Modify | Wire integrated routing, richer quality signal |
| `sage-python/src/sage/agent_loop.py` | Modify | Wire TopologyExecutor for node scheduling |
| `sage-python/src/sage/quality_estimator.py` | Create | Multi-signal quality estimation (replaces `len > 10`) |
| `tests/e2e_comprehensive.py` | Modify | Add tests for new gaps |

---

### Task 1: Add `topology_id` to `RoutingDecision` + `set_decay_factor` to Bandit (Gap #2, #7)

**Files:**
- Modify: `sage-core/src/routing/system_router.rs:104-128`
- Modify: `sage-core/src/routing/bandit.rs:498-567`

- [ ] **Step 1: Add `topology_id` field to `RoutingDecision`**

Add an optional `topology_id` field (empty string = not yet assigned). This prepares for the integrated routing pipeline.

```rust
// In RoutingDecision struct:
#[pyo3(get)]
pub topology_id: String,  // empty = not yet assigned, filled by integrated route
```

Update all construction sites to include `topology_id: String::new()`.

- [ ] **Step 2: Add `set_decay_factor()` and `warm_start_from_cards()` to ContextualBandit**

```rust
// In PyO3 methods:
#[pyo3(name = "set_decay_factor")]
pub fn py_set_decay_factor(&mut self, factor: f64) {
    self.decay_factor = factor.clamp(0.9, 1.0);
}

#[pyo3(name = "warm_start_from_cards")]
pub fn py_warm_start_from_cards(&mut self, cards: Vec<ModelCard>, templates: Vec<String>) {
    // Register arms from card x template combinations
    // Set initial quality prior from card's best affinity score
}
```

- [ ] **Step 3: Run Rust tests**

```bash
cd sage-core && cargo test --no-default-features --lib
```

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/routing/system_router.rs sage-core/src/routing/bandit.rs
git commit -m "feat(routing): add topology_id to RoutingDecision, set_decay_factor + warm_start to bandit"
```

---

### Task 2: Telemetry-Calibrated Scoring in SystemRouter (Gap #1, #4)

**Files:**
- Modify: `sage-core/src/routing/system_router.rs`
- Modify: `sage-core/src/routing/model_registry.rs`

- [ ] **Step 1: Add telemetry calibration to ModelRegistry**

```rust
// In ModelRegistry, add:
telemetry_adjustments: HashMap<String, TelemetryAdjustment>,

pub struct TelemetryAdjustment {
    quality_observed: f32,  // Running average of observed quality
    cost_observed: f32,     // Running average of observed cost
    observation_count: u32,
    // Blended affinity = (1-w)*card_prior + w*observed, where w = min(obs/50, 0.8)
}
```

Add methods:
- `record_telemetry(model_id, quality, cost)` — updates running averages
- `calibrated_affinity(model_id, system) -> f32` — blends card prior with telemetry

- [ ] **Step 2: Integrate bandit + telemetry into SystemRouter**

Add optional bandit and registry-telemetry to `SystemRouter`:

```rust
pub struct SystemRouter {
    registry: ModelRegistry,
    bandit: Option<ContextualBandit>,
}

// New method:
pub fn route_integrated(
    &mut self,
    task: &str,
    constraints: &RoutingConstraints,
    topology_id: &str,
) -> RoutingDecision {
    // 1. Hard constraints
    // 2. Structural analysis -> decide system
    // 3. Telemetry-calibrated scoring (via registry.calibrated_affinity)
    // 4. If bandit available: bandit.choose() for model selection
    // 5. Fallback: budget-constrained from registry
    // 6. Return decision with topology_id
}

// New: record_outcome feeds both registry telemetry and bandit
pub fn record_outcome(&mut self, decision_id: &str, quality: f32, cost: f32, latency_ms: f32) { ... }
```

- [ ] **Step 3: Run Rust tests**

```bash
cd sage-core && cargo test --no-default-features --lib
```

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/routing/
git commit -m "feat(routing): integrate telemetry-calibrated scoring + bandit into SystemRouter"
```

---

### Task 3: Quality Estimator (Gap #6)

**Files:**
- Create: `sage-python/src/sage/quality_estimator.py`
- Modify: `sage-python/src/sage/boot.py:281-309`

- [ ] **Step 1: Create QualityEstimator**

Multi-signal quality estimation replacing `len(result) > 10`:

```python
class QualityEstimator:
    """Estimate result quality from multiple signals (0.0-1.0)."""

    @staticmethod
    def estimate(task: str, result: str, latency_ms: float = 0.0,
                 had_errors: bool = False, avr_iterations: int = 0) -> float:
        if not result or not result.strip():
            return 0.0

        score = 0.0
        # Signal 1: Non-empty response (baseline 0.3)
        score += 0.3

        # Signal 2: Length adequacy (0.0-0.2)
        # Short tasks expect short answers, long tasks expect longer
        task_words = len(task.split())
        result_words = len(result.split())
        if task_words < 10:
            length_score = min(result_words / 20, 1.0) * 0.2
        else:
            length_score = min(result_words / 50, 1.0) * 0.2
        score += length_score

        # Signal 3: Code task detection + code presence (0.0-0.2)
        code_keywords = {"def ", "class ", "function ", "import ", "```"}
        task_wants_code = any(kw in task.lower() for kw in
                             ["write", "code", "implement", "function", "class", "fix"])
        result_has_code = any(kw in result for kw in code_keywords)
        if task_wants_code and result_has_code:
            score += 0.2
        elif not task_wants_code:
            score += 0.1  # Non-code tasks get partial credit

        # Signal 4: No error indicators (0.0-0.15)
        error_indicators = ["error", "exception", "traceback", "failed"]
        if not had_errors and not any(e in result.lower() for e in error_indicators):
            score += 0.15

        # Signal 5: AVR convergence bonus (0.0-0.15)
        if avr_iterations > 0:
            if avr_iterations <= 2:
                score += 0.15  # Quick convergence = confident
            elif avr_iterations <= 4:
                score += 0.10
            else:
                score += 0.05  # Many iterations = struggling

        return min(score, 1.0)
```

- [ ] **Step 2: Wire into boot.py `_record_topology_outcome`**

Replace:
```python
quality = 1.0 if result and len(result) > 10 else 0.3
```
With:
```python
from sage.quality_estimator import QualityEstimator
quality = QualityEstimator.estimate(
    task, result, latency_ms=latency_ms,
    had_errors=bool(getattr(self.agent_loop, '_last_error', None)),
    avr_iterations=getattr(self.agent_loop, '_last_avr_iterations', 0),
)
```

- [ ] **Step 3: Run Python tests**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/quality_estimator.py sage-python/src/sage/boot.py
git commit -m "feat(quality): multi-signal quality estimator replaces len>10 heuristic"
```

---

### Task 4: Wire TopologyExecutor in agent_loop.py (Gap #8)

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py`
- Modify: `sage-python/src/sage/boot.py`

- [ ] **Step 1: Add topology execution scheduling to agent_loop**

In `agent_loop.py`, add a method that uses the Rust `TopologyExecutor` to determine execution order when a topology is available:

```python
def _schedule_from_topology(self, topology) -> list[dict]:
    """Use Rust TopologyExecutor to get node execution order.

    Returns list of node specs: [{"role": str, "model_id": str, "system": int}, ...]
    Falls back to single-node if executor unavailable.
    """
    try:
        from sage_core import PyTopologyExecutor
        executor = PyTopologyExecutor(topology)
        schedule = []
        while not executor.is_done():
            ready = executor.next_ready(topology)
            if not ready:
                break
            for idx in ready:
                node = topology.get_node(idx)
                schedule.append({
                    "index": idx,
                    "role": node.role,
                    "model_id": node.model_id,
                    "system": node.system,
                })
                executor.mark_completed(idx)
        return schedule
    except Exception as e:
        _log.debug("TopologyExecutor scheduling failed (%s), using default", e)
        return []
```

In the `run()` method's THINK phase, after topology is available, use the schedule to inform the system prompt with role context:

```python
# If topology engine provided a multi-node topology, inject role context
if self.topology_engine and hasattr(self, '_current_topology'):
    schedule = self._schedule_from_topology(self._current_topology)
    if len(schedule) > 1:
        roles_desc = ", ".join(f"{n['role']}(S{n['system']})" for n in schedule)
        # Inject as system context for the LLM
        topology_context = f"[Topology: {len(schedule)} agents — {roles_desc}]"
        # Prepend to messages
```

- [ ] **Step 2: Wire topology from boot.py into agent_loop**

In `boot.py` `AgentSystem.run()`, after topology generation, store the topology on the agent_loop:

```python
if topology_result:
    self.agent_loop._current_topology = topology_result.topology
else:
    self.agent_loop._current_topology = None
```

- [ ] **Step 3: Run Python tests**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/src/sage/boot.py
git commit -m "feat(topology): wire TopologyExecutor scheduling into agent_loop"
```

---

### Task 5: Integrated Routing in boot.py (Gap #4 wiring)

**Files:**
- Modify: `sage-python/src/sage/boot.py`

- [ ] **Step 1: Use `route_integrated` when available**

In `AgentSystem.run()`, replace the separate routing + bandit calls with the integrated path:

```python
# When rust_router has bandit integration:
if hasattr(self.rust_router, 'route_integrated'):
    topology_id = topology_result.topology.id if topology_result else ""
    decision = self.rust_router.route_integrated(task, constraints, topology_id)
    system_num = int(decision.system)
    model_id = decision.model_id
    # Bandit selection happens inside route_integrated
    bandit_decision = None  # No longer separate
```

And in `_record_topology_outcome`:
```python
if hasattr(self.rust_router, 'record_outcome'):
    self.rust_router.record_outcome(
        decision.decision_id, quality, cost, latency_ms,
    )
```

- [ ] **Step 2: Run Python tests**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/boot.py
git commit -m "feat(routing): use integrated route_integrated path in boot.py"
```

---

### Task 6: Update E2E Tests (all gaps)

**Files:**
- Modify: `tests/e2e_comprehensive.py`

- [ ] **Step 1: Add tests for new gaps**

```python
def test_routing_decision_has_topology_id():
    """Gap #2: RoutingDecision has topology_id field."""
    from sage_core import SystemRouter, ModelRegistry
    # ... construct and verify decision.topology_id exists

def test_bandit_set_decay_factor():
    """Gap #7: decay_factor is configurable."""
    from sage_core import ContextualBandit
    bandit = ContextualBandit()
    bandit.set_decay_factor(0.99)
    # verify no error

def test_quality_estimator_signals():
    """Gap #6: Multi-signal quality estimation."""
    from sage.quality_estimator import QualityEstimator
    # Empty result
    assert QualityEstimator.estimate("task", "") == 0.0
    # Code task with code result
    score = QualityEstimator.estimate("write a function", "def foo(): return 42")
    assert score > 0.5
    # Non-code task
    score2 = QualityEstimator.estimate("what is 2+2", "4")
    assert score2 > 0.3

def test_telemetry_calibration():
    """Gap #1: Registry has telemetry calibration."""
    from sage_core import ModelRegistry
    # ... verify record_telemetry and calibrated_affinity methods exist

def test_router_route_integrated():
    """Gap #4: SystemRouter has route_integrated with bandit."""
    from sage_core import SystemRouter, ModelRegistry, RoutingConstraints, ContextualBandit
    # ... verify route_integrated method exists and works
```

- [ ] **Step 2: Run E2E tests**

```bash
python tests/e2e_comprehensive.py
```

- [ ] **Step 3: Commit**

```bash
git add tests/e2e_comprehensive.py
git commit -m "test(e2e): add tests for all 8 cognitive engine gaps"
```
