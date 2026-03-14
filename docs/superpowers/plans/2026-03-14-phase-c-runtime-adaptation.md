# Phase C: Runtime Adaptation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime adaptation to the cognitive orchestration pipeline — TopologyController evaluates each node's output and decides to continue, upgrade model, prune, reroute, or spawn sub-agents. Wire OxiZ formal verification into pipeline stages.

**Architecture:** TopologyController (Python) orchestrates 4 adaptation actions post-node. OxiZ (Rust, existing) provides formal verification in Stage 3 (assignment) and Stage 5 (learn). Rust `batch_cosine_similarity` (new, on existing RustEmbedder) powers ConsistencyScore for parallel output comparison. All additions are backward compatible — controller=None preserves Phase B behavior.

**Tech Stack:** Rust (PyO3, ort ONNX — existing), Python 3.12+ (asyncio, dataclasses), OxiZ SmtVerifier (existing), ProcessRewardModel (existing), AgentTool (existing)

**Spec:** `docs/superpowers/specs/2026-03-14-phase-c-runtime-adaptation-design.md`

---

## Chunk 1: Rust batch_cosine_similarity + ConsistencyScore

### Task 1: Rust batch_cosine_similarity on RustEmbedder

**Files:**
- Modify: `sage-core/src/memory/embedder.rs`

- [ ] **Step 1: Add cosine_sim helper function**

Add outside the `impl RustEmbedder` block (module-level helper):
```rust
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 { return 0.0; }
    dot / (norm_a * norm_b)
}
```

- [ ] **Step 2: Add batch_cosine_similarity to `#[pymethods]` block**

Add inside the `#[pymethods] impl RustEmbedder` block (near `embed_batch`):
```rust
/// Compute pairwise cosine similarity for a batch of texts.
/// Returns flattened upper-triangle: [(0,1), (0,2), (1,2), ...].
#[pyo3(name = "batch_cosine_similarity")]
pub fn py_batch_cosine_similarity(&mut self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<f32>> {
    let embeddings = self.embed_batch(py, texts)?;
    let n = embeddings.len();
    let mut sims = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i+1)..n {
            sims.push(cosine_sim(&embeddings[i], &embeddings[j]));
        }
    }
    Ok(sims)
}
```

NOTE: `embed_batch` takes `&mut self` — so `batch_cosine_similarity` must also take `&mut self`. Read the actual signature before implementing.

- [ ] **Step 3: Add Rust unit test**

```rust
#[test]
fn test_cosine_sim_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_sim(&a, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_sim_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!(cosine_sim(&a, &b).abs() < 1e-6);
}

#[test]
fn test_cosine_sim_zero_norm() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!(cosine_sim(&a, &b).abs() < 1e-6);
}
```

- [ ] **Step 4: Build and test**

```bash
cd sage-core && cargo test --no-default-features --lib cosine_sim
cd sage-core && cargo test --no-default-features --lib 2>&1 | tail -3
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Rust batch_cosine_similarity on RustEmbedder (SIMD 768-dim)"
```

### Task 2: Python ConsistencyScore wrapper

**Files:**
- Create: `sage-python/src/sage/consistency.py`
- Create: `sage-python/tests/test_consistency_score.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for ConsistencyScore — pairwise cosine similarity."""
from __future__ import annotations
import pytest

def test_identical_texts_high_score():
    from sage.consistency import consistency_score
    score = consistency_score(["hello world", "hello world"])
    assert score > 0.99

def test_different_texts_lower_score():
    from sage.consistency import consistency_score
    score = consistency_score(["the cat sat on the mat", "quantum computing advances"])
    assert score < 0.9

def test_single_text_returns_one():
    from sage.consistency import consistency_score
    assert consistency_score(["just one"]) == 1.0

def test_empty_returns_one():
    from sage.consistency import consistency_score
    assert consistency_score([]) == 1.0

def test_fallback_without_rust():
    from sage.consistency import consistency_score
    # Force Python fallback
    score = consistency_score(["a", "b"], embedder=None)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Implement consistency.py**

```python
"""ConsistencyScore — mean pairwise cosine similarity of text embeddings."""
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger(__name__)


def consistency_score(texts: list[str], embedder: Any = None) -> float:
    """Mean pairwise cosine similarity. 1.0=identical, 0.0=orthogonal.

    Uses Rust batch_cosine_similarity if embedder has it.
    Falls back to sentence-transformers or returns 1.0 if only hash available.
    """
    if len(texts) <= 1:
        return 1.0

    # Try Rust SIMD path
    if embedder and hasattr(embedder, 'batch_cosine_similarity'):
        try:
            sims = embedder.batch_cosine_similarity(texts)
            return sum(sims) / len(sims) if sims else 1.0
        except Exception as exc:
            log.debug("Rust batch_cosine_similarity failed: %s", exc)

    # Try sentence-transformers fallback
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, normalize_embeddings=True)
        n = len(embeddings)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(float(np.dot(embeddings[i], embeddings[j])))
        return sum(sims) / len(sims) if sims else 1.0
    except ImportError:
        pass

    # Hash embeddings = meaningless cosine. Return 1.0 to avoid spurious reroutes.
    log.debug("No semantic embedder available, returning 1.0 (no consistency check)")
    return 1.0
```

- [ ] **Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_consistency_score.py -v
git add -A && git commit -m "feat: ConsistencyScore — pairwise cosine similarity with Rust/Python fallback"
```

---

## Chunk 2: OxiZ wiring into pipeline

### Task 3: Wire verify_provider_assignment into Stage 3

**Files:**
- Modify: `sage-python/src/sage/pipeline.py`
- Create: `sage-python/tests/test_oxiz_pipeline.py`

- [ ] **Step 1: Write test**

Test that verify_provider_assignment is called after assign_models. Mock both.

- [ ] **Step 2: Add adapter + wiring in pipeline.py `_stage_assign_models()`**

After `self.assigner.assign_models(...)`, add:
```python
# OxiZ formal verification (Phase C)
self._verify_assignment_formal(ctx)
```

Create `_verify_assignment_formal(ctx)` method that:
1. Builds adapter dicts from TopologyGraph nodes (capabilities, model_ids, security labels)
2. Calls `verify_provider_assignment()` from z3_verify
3. Logs warning + emits EventBus on failure
4. Catches all exceptions gracefully

Read `contracts/z3_verify.py:verify_provider_assignment()` to understand what `TaskDAG` and `ProviderSpec` fields it actually accesses — build minimal adapter dicts.

- [ ] **Step 3: Run test, commit**

```bash
git add -A && git commit -m "feat: wire verify_provider_assignment (OxiZ) into Pipeline Stage 3"
```

### Task 4: PolicyVerifier.verify_node() staticmethod

**Files:**
- Modify: `sage-python/src/sage/contracts/policy.py`
- Add to: `sage-python/tests/test_oxiz_pipeline.py`

- [ ] **Step 1: Write test**

```python
def test_verify_node_passes_valid():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=1, max_cost_usd=2.0)
    preds = [MockNode(security_label=0)]
    assert PolicyVerifier.verify_node(node, preds, budget_remaining=5.0) is True

def test_verify_node_fails_info_flow():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=0)  # public
    preds = [MockNode(security_label=2)]  # confidential predecessor
    assert PolicyVerifier.verify_node(node, preds, budget_remaining=5.0) is False

def test_verify_node_fails_budget():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=0, max_cost_usd=10.0)
    assert PolicyVerifier.verify_node(node, [], budget_remaining=1.0) is False
```

- [ ] **Step 2: Implement verify_node as @staticmethod**

Add to `PolicyVerifier` class in `contracts/policy.py`:
```python
@staticmethod
def verify_node(node: Any, predecessors: list[Any], budget_remaining: float,
                max_fan_in: int = 5) -> bool:
    """Node-scoped policy check. Duck-typed for TopologyNode/TaskNode."""
    label = getattr(node, 'security_label', 0)
    cost = getattr(node, 'max_cost_usd', 0.0)
    pred_labels = [getattr(p, 'security_label', 0) for p in predecessors]
    if pred_labels and label < max(pred_labels):
        return False  # info-flow violation
    if cost > budget_remaining:
        return False  # budget violation
    if len(predecessors) > max_fan_in:
        return False  # fan-in violation
    return True
```

- [ ] **Step 3: Run test, commit**

```bash
git add -A && git commit -m "feat: PolicyVerifier.verify_node() — node-scoped security/budget check"
```

### Task 5: PRM lightweight in Stage 5 + latency fix

**Files:**
- Modify: `sage-python/src/sage/pipeline.py` (`_stage_learn`)
- Add to: `sage-python/tests/test_oxiz_pipeline.py`

- [ ] **Step 1: Write test for PRM blending**

Test that when PRM is available and result has `<think>` tags, quality blends 80/20.

- [ ] **Step 2: Fix latency_ms bug + add PRM in _stage_learn**

In `_stage_learn()`:
- Fix: remove `/ 1000.0` from latency_ms (it's already in ms)
- Add: detect structured content (`<think>` or `assert` in result)
- Add: if structured, call `self.prm.calculate_r_path(result)`, blend at 80/20
- Catch PRM exceptions and log (not silent pass)

- [ ] **Step 3: Run test, commit**

```bash
git add -A && git commit -m "feat: PRM lightweight scoring in Stage 5 LEARN + fix latency_ms units"
```

### Task 6: Topology pre-validation

**Files:**
- Modify: `sage-python/src/sage/pipeline.py` (`_stage_select_topology`)

- [ ] **Step 1: Add budget feasibility check after topology selection**

In `_stage_select_topology()`, after topology is selected:
```python
# Pre-validate budget feasibility
if ctx.topology and hasattr(ctx.topology, 'node_count'):
    total_node_cost = sum(
        getattr(ctx.topology.get_node(i), 'max_cost_usd', 0)
        for i in range(ctx.topology.node_count())
    )
    if total_node_cost > ctx.budget:
        log.warning("Topology budget %.2f > pipeline budget %.2f", total_node_cost, ctx.budget)
        self._emit("TOPOLOGY_BUDGET_WARNING", {"total": total_node_cost, "budget": ctx.budget})
```

Simple Python arithmetic — OxiZ `verify_arithmetic_expr` is overkill for a sum comparison.

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "feat: topology pre-validation budget check in Stage 2"
```

---

## Chunk 3: TopologyController

### Task 7: TopologyController with 4 adaptation actions

**Files:**
- Create: `sage-python/src/sage/topology_controller.py`
- Create: `sage-python/tests/test_topology_controller.py`

- [ ] **Step 1: Write tests**

Tests for all 4 actions + edge cases:
```python
def test_continue_on_good_quality():
    """quality >= 0.7 → CONTINUE"""

def test_upgrade_model_on_critical_quality():
    """quality < 0.3, retries < 2 → UPGRADE_MODEL"""

def test_prune_on_low_importance():
    """importance < 0.2 → PRUNE_NODE"""

def test_reroute_on_inconsistency():
    """parallel outputs, consistency < 0.5 → REROUTE_TOPOLOGY"""

def test_max_reroute_forces_continue():
    """_reroute_count >= 1 → force CONTINUE"""

def test_spawn_on_emergent_subtask():
    """result contains 'need to also' → SPAWN_SUBAGENT"""

def test_max_spawns_respected():
    """_spawn_count >= 3 → skip spawn, CONTINUE"""

def test_quality_blends_prm():
    """80% heuristic + 20% PRM when structured content"""

def test_no_prm_on_plain_text():
    """PRM not called when no <think> tags"""
```

- [ ] **Step 2: Implement topology_controller.py**

Create `sage-python/src/sage/topology_controller.py` with:
- `AdaptationDecision` dataclass
- `TopologyController` class with `evaluate_and_decide()`, `compute_consistency_score()`, `compute_importance_score()`
- All thresholds as named class constants
- `_reroute_count`, `_spawn_count`, `_node_retries` as instance state
- PRM guard: check `<think>` or `assert` before calling `calculate_r_path()`
- Import `consistency_score` from `sage.consistency`

- [ ] **Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_topology_controller.py -v
git add -A && git commit -m "feat: TopologyController — 4 adaptation actions with quality blending"
```

---

## Chunk 4: Runner adaptation loop + Pipeline integration + Boot wiring

### Task 8: TopologyRunner adaptation loop

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py`

- [ ] **Step 1: Add controller parameter to __init__**

```python
def __init__(self, graph, executor, llm_provider, llm_config=None,
             *, provider_pool=None, controller=None):
    ...
    self._controller = controller
```

- [ ] **Step 2: Add adaptation methods**

```python
async def _retry_with_upgrade(self, node_idx, decision, task):
    """Model upgrade: re-resolve provider and retry node."""

async def _spawn_sub(self, node_idx, decision, task):
    """Sub-agent spawn via AgentTool.from_agent() or direct LLM."""

async def _reroute_and_restart(self, task, ctx):
    """Reroute: regenerate topology and restart from scratch."""
```

- [ ] **Step 3: Modify run() loop to call controller after each node**

After `result = await self._execute_node(node_idx, task)`, add:
```python
if self._controller:
    decision = self._controller.evaluate_and_decide(
        node_idx, result, task, self.graph, ctx
    )
    if decision.action == "upgrade_model":
        result = await self._retry_with_upgrade(node_idx, decision, task)
    elif decision.action == "reroute_topology":
        return await self._reroute_and_restart(task, ctx)
    elif decision.action == "spawn_subagent":
        await self._spawn_sub(node_idx, decision, task)
    # prune_node and continue: no special handling needed
```

- [ ] **Step 4: Verify existing tests pass (controller=None)**

```bash
cd sage-python && python -m pytest tests/ -k "runner" -v
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: TopologyRunner adaptation loop with controller integration"
```

### Task 9: Pipeline integration + Boot wiring

**Files:**
- Modify: `sage-python/src/sage/pipeline.py`
- Modify: `sage-python/src/sage/boot.py`

- [ ] **Step 1: Add controller to pipeline**

In `CognitiveOrchestrationPipeline.__init__`, add `controller=None` param. Store as `self.controller`.

Pass `controller=self.controller` to TopologyRunner in `_stage_execute()`.

- [ ] **Step 2: Wire in boot.py**

After pipeline instantiation, add controller creation (wrapped in try/except).

- [ ] **Step 3: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -x -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: wire TopologyController into pipeline + boot.py"
```

### Task 10: Integration test + CLAUDE.md

**Files:**
- Create: `sage-python/tests/test_pipeline_adaptation.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write integration tests**

```python
def test_adaptation_upgrade_on_failure():
    """Pipeline with mock node that returns low quality → controller upgrades model → retry succeeds."""

def test_adaptation_budget_exhausted_graceful():
    """Pipeline where budget runs out → controller continues without upgrade."""

def test_oxiz_verification_fail_proceeds():
    """verify_provider_assignment fails → pipeline proceeds (non-blocking warning)."""

def test_prm_skipped_on_plain_text():
    """Stage 5 LEARN doesn't call PRM when result has no <think> tags."""

def test_controller_none_preserves_phase_b():
    """controller=None → pipeline behaves identically to Phase B."""
```

- [ ] **Step 2: Update CLAUDE.md**

Add `topology_controller.py` and `consistency.py` to Key Python Modules. Update pipeline.py description to mention Phase C adaptation. Add OxiZ wiring notes.

- [ ] **Step 3: Run all tests, commit**

```bash
cd sage-python && python -m pytest tests/ -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -10
cd sage-core && cargo test --no-default-features --lib 2>&1 | tail -3
git add -A && git commit -m "test: Phase C integration tests + CLAUDE.md update"
```
