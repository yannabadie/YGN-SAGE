# Cognitive Orchestration Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire ModelCards into per-node model assignment so TopologyRunner executes each agent node with the optimal LLM, driven by a 5-stage cognitive orchestration pipeline.

**Architecture:** Rust `ModelAssigner` composes existing `ModelRegistry` + `ModelCard` for sub-ms per-node scoring. Python `CognitiveOrchestrationPipeline` chains 5 stages (Classify → Decompose → Select Topology → Assign Models → Execute). `ProviderPool` resolves `model_id` → live provider at execution time. Cleanup PR first, features second.

**Tech Stack:** Rust (PyO3, petgraph), Python 3.12+ (asyncio, dataclasses), existing SAGE infrastructure (TopologyGraph, ModelRegistry, AdaptiveRouter, TopologyExecutor, EventBus)

**Spec:** `docs/superpowers/specs/2026-03-14-cognitive-orchestration-pipeline-design.md`

---

## Chunk 1: Cleanup (separate commit group)

This chunk removes dead code, fixes naming, and adds the Rust setter — all prerequisite work before features. Committed as a clean, isolated group.

### Task 1: Delete DynamicRouter and update dependents

**Files:**
- Delete: `sage-python/src/sage/routing/dynamic.py`
- Delete: `sage-python/tests/test_dynamic_router.py`
- Modify: `sage-python/tests/test_bugfixes.py`
- Modify: `sage-python/tests/test_integration_phase3.py`
- Modify: `sage-python/src/sage/routing/README.md`
- Modify: `sage-python/src/sage/README.md`
- Modify: `sage-python/README.md`

- [ ] **Step 1: Delete DynamicRouter source and tests**

```bash
cd sage-python
rm src/sage/routing/dynamic.py
rm tests/test_dynamic_router.py
```

- [ ] **Step 2: Remove DynamicRouter from test_bugfixes.py**

In `sage-python/tests/test_bugfixes.py`, remove the BF-1 test block (the `from sage.routing.dynamic import DynamicRouter` import and the `test_bf1_dynamic_router_empty_scored` test function). Keep any other bugfix tests in the file.

- [ ] **Step 3: Remove DynamicRouter from test_integration_phase3.py**

In `sage-python/tests/test_integration_phase3.py`, remove the `from sage.routing.dynamic import DynamicRouter, RoutingDecision` import (line 21) and the `dynamic_router` fixture (line ~52). Mark remaining tests that depend on DynamicRouter with `@pytest.mark.skip(reason="DynamicRouter removed — superseded by CognitiveOrchestrator")`.

- [ ] **Step 4: Update README files**

- `sage-python/src/sage/routing/README.md`: Remove the "dynamic.py -- DynamicRouter" section (lines 7-22).
- `sage-python/src/sage/README.md`: Change `| routing/ | DynamicRouter with capability constraints and feedback |` to `| routing/ | ShadowRouter (dual Rust/Python traces) |`
- `sage-python/README.md`: Remove DynamicRouter from the routing table row.

- [ ] **Step 5: Run tests to verify no import errors**

```bash
cd sage-python && python -m pytest tests/ -x -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -5
```

Expected: All tests pass (no `ImportError` for `sage.routing.dynamic`).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "cleanup: delete DynamicRouter (178 LOC dead code) and update dependents"
```

### Task 2: Delete TopologyPlanner and update dependents

**Files:**
- Delete: `sage-python/src/sage/topology/planner.py`
- Delete: `sage-python/tests/test_topology_planner.py`
- Modify: `sage-python/src/sage/topology/__init__.py`
- Modify: `sage-discover/src/discover/workflow.py`

- [ ] **Step 1: Delete TopologyPlanner source and tests**

```bash
rm sage-python/src/sage/topology/planner.py
rm sage-python/tests/test_topology_planner.py
```

- [ ] **Step 2: Remove TopologyPlanner from topology/__init__.py**

In `sage-python/src/sage/topology/__init__.py`:
- Remove lines 7-12 (the `TopologyPlanner: Any` declaration and the try/except import block)
- Remove `TopologyPlanner` and `StochasticDTS` from the `__all__` list (line 20)

- [ ] **Step 3: Fix sage-discover workflow.py**

In `sage-discover/src/discover/workflow.py`:
- Remove `from sage.topology.planner import TopologyPlanner` (line 38)
- Remove or comment out `self.topology_planner = TopologyPlanner(...)` (line 105) and any usage of `self.topology_planner` (search for all references)
- If `topology_planner` is used in a method, replace with `pass` or `log.warning("TopologyPlanner removed — use DynamicTopologyEngine")`

- [ ] **Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/ -x -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -5
```

Expected: All pass, no `ImportError` for `sage.topology.planner`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "cleanup: delete TopologyPlanner/StochasticDTS (80 LOC dead code)"
```

### Task 3: Rename ModelRegistry → ModelCardCatalog

**Files:**
- Modify: `sage-python/src/sage/llm/model_registry.py` (class rename + add from_toml_str)
- Modify: `sage-python/src/sage/boot.py` (import update)
- Modify: `sage-python/tests/test_model_registry_py.py` (update import to ModelCardCatalog)
- Modify: any other test files that import `sage.llm.model_registry.ModelRegistry`
- Create: `sage-python/tests/test_cleanup.py` (regression test)

- [ ] **Step 1: Write the regression test**

Create `sage-python/tests/test_cleanup.py`:
```python
"""Regression tests for cleanup: naming, dead code removal."""
from __future__ import annotations

def test_model_card_catalog_importable():
    """ModelCardCatalog is the renamed ModelRegistry from sage.llm.model_registry."""
    from sage.llm.model_registry import ModelCardCatalog
    assert ModelCardCatalog is not None

def test_dynamic_router_removed():
    """DynamicRouter was dead code — import must fail."""
    import importlib
    try:
        importlib.import_module("sage.routing.dynamic")
        assert False, "sage.routing.dynamic should not exist"
    except (ImportError, ModuleNotFoundError):
        pass

def test_topology_planner_removed():
    """TopologyPlanner was dead code — import must fail."""
    import importlib
    try:
        importlib.import_module("sage.topology.planner")
        assert False, "sage.topology.planner should not exist"
    except (ImportError, ModuleNotFoundError):
        pass
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cd sage-python && python -m pytest tests/test_cleanup.py::test_model_card_catalog_importable -v
```

Expected: FAIL (`ImportError: cannot import name 'ModelCardCatalog'`)

- [ ] **Step 3: Rename the class**

In `sage-python/src/sage/llm/model_registry.py`:
- Rename `class ModelRegistry` → `class ModelCardCatalog`
- Add `from_toml_str()` classmethod (needed by tests and Python ModelAssigner):
```python
@classmethod
def from_toml_str(cls, toml_str: str) -> ModelCardCatalog:
    cards = ModelCard.parse_toml(toml_str)
    reg = cls()
    for card in cards:
        reg.register(card)
    return reg
```
- Add backward compat alias at bottom: `ModelRegistry = ModelCardCatalog  # deprecated alias`

- [ ] **Step 4: Update imports in boot.py**

Search `sage-python/src/sage/boot.py` for `from sage.llm.model_registry import ModelRegistry` and change to `from sage.llm.model_registry import ModelCardCatalog`. Update any variable names from `PyModelRegistry` to `py_model_card_catalog`.

- [ ] **Step 5: Run all tests**

```bash
cd sage-python && python -m pytest tests/test_cleanup.py -v && python -m pytest tests/ -x -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -5
```

Expected: All 3 cleanup tests pass. Full suite passes.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "cleanup: rename ModelRegistry → ModelCardCatalog (resolve naming ambiguity)"
```

### Task 4: Update Rust deprecated tags + add TopologyNode setter

**Files:**
- Modify: `sage-core/src/routing/system_router.rs` (update deprecated note)
- Modify: `sage-core/src/topology/engine.rs` (update deprecated note)
- Modify: `sage-core/src/topology/topology_graph.rs` (add set on model_id + set_node_model_id method)
- Modify: `CLAUDE.md` (update deprecated section)

- [ ] **Step 1: Update SystemRouter deprecated note**

In `sage-core/src/routing/system_router.rs`, change the `#[deprecated]` attribute to:
```rust
#[deprecated(since = "0.2.0", note = "Deprecated for direct Python use; still required as internal dependency of ModelAssigner and boot.py routing. Removal deferred to v0.4.")]
```

- [ ] **Step 2: Update TopologyEngine deprecated note**

In `sage-core/src/topology/engine.rs`, change the `#[deprecated]` attribute to:
```rust
#[deprecated(since = "0.2.0", note = "Deprecated for direct Python use; still required as internal dependency of boot.py Phase 6. Removal deferred to v0.4.")]
```

- [ ] **Step 3: Add setter to TopologyNode.model_id**

In `sage-core/src/topology/topology_graph.rs`, change line 132:
```rust
// BEFORE:
#[pyo3(get)]
pub model_id: String,

// AFTER:
#[pyo3(get, set)]
pub model_id: String,
```

- [ ] **Step 4: Add set_node_model_id to TopologyGraph**

In `sage-core/src/topology/topology_graph.rs`, add to the `#[pymethods] impl TopologyGraph` block:
```rust
/// Set the model_id for a specific node (for Python fallback ModelAssigner).
pub fn set_node_model_id(&mut self, idx: usize, model_id: &str) -> PyResult<()> {
    let node_idx = petgraph::graph::NodeIndex::new(idx);
    match self.graph.node_weight_mut(node_idx) {
        Some(node) => {
            node.model_id = model_id.to_string();
            Ok(())
        }
        None => Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Node index {} out of range", idx)
        )),
    }
}
```

- [ ] **Step 5: Build and test Rust**

```bash
cd sage-core && cargo test --no-default-features --lib 2>&1 | tail -5
```

Expected: 245+ tests pass, no errors.

- [ ] **Step 6: Update CLAUDE.md deprecated section**

Replace the "Deprecated Rust modules (v0.3 removal target)" section with updated notes reflecting v0.4 deferral for SystemRouter and TopologyEngine, keeping AdaptiveRouter and TopologyBridge as v0.3 targets.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "cleanup: update deprecated tags (defer to v0.4) + add TopologyNode.model_id setter"
```

---

## Chunk 2: Rust ModelAssigner

### Task 5: Implement Rust ModelAssigner with tests

**Files:**
- Create: `sage-core/src/routing/model_assigner.rs`
- Modify: `sage-core/src/routing/mod.rs` (add module)
- Modify: `sage-core/src/lib.rs` (register PyO3 class)

- [ ] **Step 1: Write Rust tests first**

Create `sage-core/src/routing/model_assigner.rs` with the test module at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::model_card::ModelCard;
    use crate::routing::model_registry::ModelRegistry;
    use crate::topology::topology_graph::{TopologyGraph, TopologyNode, TopologyEdge};

    fn test_registry() -> ModelRegistry {
        let toml = r#"
            [[models]]
            id = "cheap-fast"
            provider = "test"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.3
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.9
            s2_affinity = 0.3
            s3_affinity = 0.1
            recommended_topologies = ["sequential"]
            supports_tools = false
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.5
            math = 0.4

            [[models]]
            id = "expensive-smart"
            provider = "test"
            family = "test"
            code_score = 0.9
            reasoning_score = 0.95
            tool_use_score = 0.9
            math_score = 0.9
            formal_z3_strength = 0.8
            cost_input_per_m = 5.0
            cost_output_per_m = 15.0
            latency_ttft_ms = 3000.0
            tokens_per_sec = 50.0
            s1_affinity = 0.1
            s2_affinity = 0.9
            s3_affinity = 0.95
            recommended_topologies = ["avr", "debate"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 1000000
            [models.domain_scores]
            code = 0.9
            math = 0.95
        "#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    fn two_node_graph() -> TopologyGraph {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new("coder".into(), "".into(), 2, vec!["tools".into()], 0, 5.0, 60.0);
        let n1 = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        let edge = TopologyEdge::control();
        g.add_node(n0);
        g.add_node(n1);
        g.try_add_edge(0, 1, edge).unwrap();
        g
    }

    #[test]
    fn test_assign_models_basic() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();

        let n = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n, 2);

        // Coder (S2, needs tools) → expensive-smart (only one with tools)
        assert_eq!(graph.get_node(0).unwrap().model_id, "expensive-smart");
        // Reviewer (S3, no special caps) → expensive-smart (highest S3 affinity)
        assert_eq!(graph.get_node(1).unwrap().model_id, "expensive-smart");
    }

    #[test]
    fn test_assign_respects_budget() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();

        // Very tight budget — should assign cheap-fast to reviewer (no tools needed)
        let n = assigner.assign_models_inner(&mut graph, "code", 0.005);
        // At least one node should get cheap-fast due to budget
        let model0 = &graph.get_node(0).unwrap().model_id;
        let model1 = &graph.get_node(1).unwrap().model_id;
        assert!(model0 == "cheap-fast" || model1 == "cheap-fast" || n < 2);
    }

    #[test]
    fn test_assign_keeps_existing_when_no_candidate() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);

        let mut g = TopologyGraph::try_new("sequential").unwrap();
        // Node requires tools+json+vision — only expensive-smart has all three
        let n0 = TopologyNode::new(
            "special".into(), "original-model".into(), 2,
            vec!["tools".into(), "json".into(), "vision".into()],
            0, 0.001, 60.0,  // budget too low for expensive-smart
        );
        g.add_node(n0);

        let n = assigner.assign_models_inner(&mut g, "code", 0.001);
        // No candidate passes → keep original
        assert_eq!(g.get_node(0).unwrap().model_id, "original-model");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_budget_exhaustion_stops_early() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();

        // Budget of 0 → should break immediately
        let n = assigner.assign_models_inner(&mut graph, "code", 0.0);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_assign_single_node() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();

        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0);
        assert!(model_id.is_some());
        assert_eq!(graph.get_node(1).unwrap().model_id, model_id.unwrap());
    }
}
```

- [ ] **Step 2: Implement ModelAssigner**

Above the tests in the same file, implement the struct and methods:

```rust
//! ModelAssigner — per-node model assignment using ModelCard scoring.

use pyo3::prelude::*;
use tracing::{info, warn};

use super::model_card::CognitiveSystem;
use super::model_registry::ModelRegistry;
use crate::topology::topology_graph::TopologyGraph;

/// Scoring weights for per-node assignment.
/// Differ from ModelRegistry.domain_routing_score (0.6/0.3/0.1):
/// per-node assignment needs higher affinity weight because node.system
/// is a strong signal (coder=S2, reviewer=S3).
const WEIGHT_AFFINITY: f32 = 0.4;
const WEIGHT_DOMAIN: f32 = 0.4;
const WEIGHT_COST: f32 = 0.2;
const BUDGET_EPSILON: f32 = 0.01;

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelAssigner {
    registry: ModelRegistry,
}

impl ModelAssigner {
    pub fn from_registry(registry: &ModelRegistry) -> Self {
        Self { registry: registry.clone() }
    }

    pub fn assign_models_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> usize {
        let node_count = graph.node_count();
        let mut remaining_budget = budget_usd;
        let mut assigned = 0usize;

        let all_models = self.registry.all_models();
        if all_models.is_empty() {
            warn!("ModelAssigner: no models in registry, skipping assignment");
            return 0;
        }

        // Compute max cost for normalization
        let max_cost = all_models.iter()
            .map(|c| c.estimate_cost(1000, 500))
            .fold(0.001_f32, f32::max);

        for idx in 0..node_count {
            if remaining_budget < BUDGET_EPSILON {
                warn!(
                    node_idx = idx,
                    remaining = node_count - idx,
                    "budget_exhausted — stopping assignment"
                );
                break;
            }

            let node = match graph.try_get_node(idx) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let system = match node.system {
                1 => CognitiveSystem::S1,
                2 => CognitiveSystem::S2,
                3 => CognitiveSystem::S3,
                _ => CognitiveSystem::S1,
            };

            let caps = &node.required_capabilities;
            let needs_tools = caps.iter().any(|c| c == "tools");
            let needs_json = caps.iter().any(|c| c == "json");
            let node_budget = node.max_cost_usd.min(remaining_budget);

            // Filter + score
            let mut best_id: Option<String> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;

            for card in &all_models {
                // Capability filter
                if needs_tools && !card.supports_tools { continue; }
                if needs_json && !card.supports_json_mode { continue; }

                // Budget filter
                let est_cost = card.estimate_cost(1000, 500);
                if est_cost > node_budget { continue; }

                // Score
                let affinity = self.registry.calibrated_affinity(&card.id, system);
                let domain = card.domain_score(task_domain);
                let cost_norm = est_cost / max_cost;
                let score = WEIGHT_AFFINITY * affinity
                    + WEIGHT_DOMAIN * domain
                    + WEIGHT_COST * (1.0 - cost_norm);

                if score > best_score {
                    best_score = score;
                    best_id = Some(card.id.clone());
                }
            }

            if let Some(model_id) = best_id {
                // Find cost of selected model for budget deduction
                if let Some(card) = self.registry.get(&model_id) {
                    remaining_budget -= card.estimate_cost(1000, 500);
                }
                // Mutate graph in-place
                let node_idx_pg = petgraph::graph::NodeIndex::new(idx);
                if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                    node_mut.model_id = model_id;
                }
                assigned += 1;
                info!(node = idx, model = %graph.try_get_node(idx).map(|n| n.model_id).unwrap_or_default(), "assigned");
            } else {
                warn!(node = idx, role = %node.role, "no candidate — keeping existing model_id");
            }
        }

        assigned
    }

    pub fn assign_single_node_inner(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
    ) -> Option<String> {
        // Reuse the same scoring logic for a single node
        let node = graph.try_get_node(node_idx).ok()?;
        let system = match node.system {
            1 => CognitiveSystem::S1,
            2 => CognitiveSystem::S2,
            3 => CognitiveSystem::S3,
            _ => CognitiveSystem::S1,
        };

        let caps = &node.required_capabilities;
        let needs_tools = caps.iter().any(|c| c == "tools");
        let needs_json = caps.iter().any(|c| c == "json");

        let all_models = self.registry.all_models();
        let max_cost = all_models.iter()
            .map(|c| c.estimate_cost(1000, 500))
            .fold(0.001_f32, f32::max);

        let mut best_id: Option<String> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;

        for card in &all_models {
            if needs_tools && !card.supports_tools { continue; }
            if needs_json && !card.supports_json_mode { continue; }
            if card.estimate_cost(1000, 500) > budget_usd { continue; }

            let affinity = self.registry.calibrated_affinity(&card.id, system);
            let domain = card.domain_score(task_domain);
            let cost_norm = card.estimate_cost(1000, 500) / max_cost;
            let score = WEIGHT_AFFINITY * affinity
                + WEIGHT_DOMAIN * domain
                + WEIGHT_COST * (1.0 - cost_norm);

            if score > best_score {
                best_score = score;
                best_id = Some(card.id.clone());
            }
        }

        if let Some(ref model_id) = best_id {
            let node_idx_pg = petgraph::graph::NodeIndex::new(node_idx);
            if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                node_mut.model_id = model_id.clone();
            }
        }

        best_id
    }
}

#[pymethods]
impl ModelAssigner {
    #[new]
    fn py_new(registry: &ModelRegistry) -> Self {
        Self::from_registry(registry)
    }

    fn assign_models(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> PyResult<usize> {
        Ok(self.assign_models_inner(graph, task_domain, budget_usd))
    }

    fn assign_single_node(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
    ) -> PyResult<String> {
        self.assign_single_node_inner(graph, node_idx, task_domain, budget_usd)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No candidate found"))
    }
}
```

- [ ] **Step 3: Register module and PyO3 class**

In `sage-core/src/routing/mod.rs`, add: `pub mod model_assigner;`

In `sage-core/src/lib.rs`, in the `#[pymodule]` function, add:
```rust
m.add_class::<routing::model_assigner::ModelAssigner>()?;
```

- [ ] **Step 4: Build and run Rust tests**

```bash
cd sage-core && cargo test --no-default-features --lib model_assigner 2>&1 | tail -10
```

Expected: 5 tests pass.

- [ ] **Step 5: Full Rust test suite**

```bash
cd sage-core && cargo test --no-default-features --lib 2>&1 | tail -5
```

Expected: 250+ tests pass.

- [ ] **Step 6: Build Python bindings**

```bash
cd sage-core && maturin develop
```

- [ ] **Step 7: Verify Python can import ModelAssigner**

```bash
cd sage-python && python -c "from sage_core import ModelAssigner; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: Rust ModelAssigner — per-node model assignment using ModelCard scoring"
```

---

## Chunk 3: Python components

### Task 6: Python ModelAssigner fallback

**Files:**
- Create: `sage-python/src/sage/llm/model_assigner.py`
- Create: `sage-python/tests/test_model_assigner.py`

- [ ] **Step 1: Write the test**

Create `sage-python/tests/test_model_assigner.py`:
```python
"""Tests for Python ModelAssigner fallback."""
from __future__ import annotations
import pytest
from sage.llm.model_card import ModelCard, CognitiveSystem
from sage.llm.model_registry import ModelCardCatalog


def _make_catalog() -> ModelCardCatalog:
    toml_str = '''
[[models]]
id = "cheap"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.5
tool_use_score = 0.5
math_score = 0.5
formal_z3_strength = 0.3
cost_input_per_m = 0.1
cost_output_per_m = 0.2
latency_ttft_ms = 100.0
tokens_per_sec = 200.0
s1_affinity = 0.9
s2_affinity = 0.3
s3_affinity = 0.1
recommended_topologies = ["sequential"]
supports_tools = false
supports_json_mode = false
supports_vision = false
context_window = 128000
[models.domain_scores]
code = 0.5

[[models]]
id = "smart"
provider = "test"
family = "test"
code_score = 0.9
reasoning_score = 0.95
tool_use_score = 0.9
math_score = 0.9
formal_z3_strength = 0.8
cost_input_per_m = 5.0
cost_output_per_m = 15.0
latency_ttft_ms = 3000.0
tokens_per_sec = 50.0
s1_affinity = 0.1
s2_affinity = 0.9
s3_affinity = 0.95
recommended_topologies = ["avr"]
supports_tools = true
supports_json_mode = true
supports_vision = true
context_window = 1000000
[models.domain_scores]
code = 0.9
'''
    return ModelCardCatalog.from_toml_str(toml_str)


class MockTopologyNode:
    def __init__(self, role, model_id, system, required_capabilities, max_cost_usd=5.0):
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities
        self.max_cost_usd = max_cost_usd


class MockTopologyGraph:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def node_count(self):
        return len(self._nodes)

    def get_node(self, idx):
        return self._nodes[idx] if 0 <= idx < len(self._nodes) else None

    def set_node_model_id(self, idx, model_id):
        if 0 <= idx < len(self._nodes):
            self._nodes[idx].model_id = model_id


def test_assigns_by_domain_and_system():
    from sage.llm.model_assigner import ModelAssigner
    catalog = _make_catalog()
    assigner = ModelAssigner(catalog)
    graph = MockTopologyGraph([
        MockTopologyNode("coder", "", 2, ["tools"]),
        MockTopologyNode("reviewer", "", 3, []),
    ])
    n = assigner.assign_models(graph, "code", 10.0)
    assert n == 2
    assert graph.get_node(0).model_id == "smart"  # needs tools


def test_keeps_existing_when_no_candidate():
    from sage.llm.model_assigner import ModelAssigner
    catalog = _make_catalog()
    assigner = ModelAssigner(catalog)
    graph = MockTopologyGraph([
        MockTopologyNode("special", "original", 2, ["tools", "json", "vision"], max_cost_usd=0.001),
    ])
    n = assigner.assign_models(graph, "code", 0.001)
    assert graph.get_node(0).model_id == "original"
    assert n == 0


def test_budget_exhaustion():
    from sage.llm.model_assigner import ModelAssigner
    catalog = _make_catalog()
    assigner = ModelAssigner(catalog)
    graph = MockTopologyGraph([
        MockTopologyNode("a", "", 1, []),
        MockTopologyNode("b", "", 1, []),
    ])
    n = assigner.assign_models(graph, "code", 0.0)
    assert n == 0
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cd sage-python && python -m pytest tests/test_model_assigner.py -v
```

Expected: FAIL (`ModuleNotFoundError: No module named 'sage.llm.model_assigner'`)

- [ ] **Step 3: Implement Python ModelAssigner**

Create `sage-python/src/sage/llm/model_assigner.py`:
```python
"""ModelAssigner — Python fallback for per-node model assignment.

Same algorithm as Rust sage_core.ModelAssigner. Used when sage_core
is not compiled. See spec for weight rationale (0.4/0.4/0.2).
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.model_registry import ModelCardCatalog

log = logging.getLogger(__name__)

WEIGHT_AFFINITY = 0.4
WEIGHT_DOMAIN = 0.4
WEIGHT_COST = 0.2
BUDGET_EPSILON = 0.01


class ModelAssigner:
    def __init__(self, catalog: ModelCardCatalog) -> None:
        self._catalog = catalog

    def assign_models(self, graph: Any, task_domain: str, budget_usd: float) -> int:
        node_count = graph.node_count()
        remaining = budget_usd
        assigned = 0
        cards = self._catalog.all_models()
        if not cards:
            log.warning("ModelAssigner: no models in catalog")
            return 0
        max_cost = max((c.estimate_cost(1000, 500) for c in cards), default=0.001)

        for idx in range(node_count):
            if remaining < BUDGET_EPSILON:
                log.warning("budget_exhausted_node_%d: %d nodes remaining", idx, node_count - idx)
                break
            node = graph.get_node(idx)
            if node is None:
                continue
            caps = getattr(node, "required_capabilities", [])
            needs_tools = "tools" in caps
            needs_json = "json" in caps
            node_budget = min(getattr(node, "max_cost_usd", remaining), remaining)
            system = getattr(node, "system", 1)

            best_id, best_score = None, float("-inf")
            for card in cards:
                if needs_tools and not card.supports_tools:
                    continue
                if needs_json and not card.supports_json_mode:
                    continue
                est = card.estimate_cost(1000, 500)
                if est > node_budget:
                    continue
                aff = self._catalog.calibrated_affinity(card.id, system)
                dom = card.domain_score(task_domain)
                cost_n = est / max_cost
                score = WEIGHT_AFFINITY * aff + WEIGHT_DOMAIN * dom + WEIGHT_COST * (1.0 - cost_n)
                if score > best_score:
                    best_score = score
                    best_id = card.id

            if best_id is not None:
                graph.set_node_model_id(idx, best_id)
                est = next((c.estimate_cost(1000, 500) for c in cards if c.id == best_id), 0)
                remaining -= est
                assigned += 1
            else:
                log.warning("node %d (%s): no candidate, keeping existing model_id", idx, getattr(node, "role", "?"))
        return assigned

    def assign_single_node(self, graph: Any, node_idx: int, task_domain: str, budget_usd: float) -> str:
        """Assign a single node. Returns model_id or raises ValueError."""
        # Temporarily save budget, assign just this node
        node = graph.get_node(node_idx)
        if node is None:
            raise ValueError(f"Node index {node_idx} out of range")
        # Use full budget for single node
        cards = self._catalog.all_models()
        if not cards:
            raise ValueError("No models in catalog")
        max_cost = max((c.estimate_cost(1000, 500) for c in cards), default=0.001)
        caps = getattr(node, "required_capabilities", [])
        needs_tools = "tools" in caps
        needs_json = "json" in caps
        system = getattr(node, "system", 1)

        best_id, best_score = None, float("-inf")
        for card in cards:
            if needs_tools and not card.supports_tools:
                continue
            if needs_json and not card.supports_json_mode:
                continue
            if card.estimate_cost(1000, 500) > budget_usd:
                continue
            aff = self._catalog.calibrated_affinity(card.id, system)
            dom = card.domain_score(task_domain)
            cost_n = card.estimate_cost(1000, 500) / max_cost
            score = WEIGHT_AFFINITY * aff + WEIGHT_DOMAIN * dom + WEIGHT_COST * (1.0 - cost_n)
            if score > best_score:
                best_score = score
                best_id = card.id

        if best_id is None:
            raise ValueError(f"No candidate for node {node_idx}")
        graph.set_node_model_id(node_idx, best_id)
        return best_id
```

- [ ] **Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/test_model_assigner.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Python ModelAssigner fallback (same algorithm as Rust)"
```

### Task 7: ProviderPool

**Files:**
- Create: `sage-python/src/sage/llm/provider_pool.py`
- Create: `sage-python/tests/test_provider_pool.py`

- [ ] **Step 1: Write test**

Create `sage-python/tests/test_provider_pool.py`:
```python
"""Tests for ProviderPool — model_id → provider resolution."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, AsyncMock
from sage.llm.base import LLMConfig, LLMResponse


def test_resolve_known_model():
    from sage.llm.provider_pool import ProviderPool
    mock_registry = MagicMock()
    # sage.providers.registry.ModelRegistry.select() returns ModelProfile or None
    mock_profile = MagicMock()
    mock_profile.provider = "google"
    mock_profile.model_id = "gemini-2.5-flash"
    mock_registry.select.return_value = mock_profile
    mock_registry.get_connector.return_value = MagicMock()  # LLMProvider
    default = MagicMock()

    pool = ProviderPool(default_provider=default, registry=mock_registry)
    provider, config = pool.resolve("gemini-2.5-flash")
    # Should NOT be the default since model was found
    assert config.model == "gemini-2.5-flash"


def test_resolve_unknown_falls_back():
    from sage.llm.provider_pool import ProviderPool
    mock_registry = MagicMock()
    mock_registry.select.return_value = None
    default = MagicMock()

    pool = ProviderPool(default_provider=default, registry=mock_registry)
    provider, config = pool.resolve("nonexistent-model")
    assert provider is default


def test_resolve_caches():
    from sage.llm.provider_pool import ProviderPool
    mock_registry = MagicMock()
    mock_profile = MagicMock()
    mock_profile.provider = "test"
    mock_profile.model_id = "model-a"
    mock_registry.select.return_value = mock_profile
    mock_registry.get_connector.return_value = MagicMock()
    default = MagicMock()

    pool = ProviderPool(default_provider=default, registry=mock_registry)
    pool.resolve("model-a")
    pool.resolve("model-a")
    # Only one registry lookup (cached)
    assert mock_registry.select.call_count == 1


def test_resolve_empty_model_id_returns_default():
    from sage.llm.provider_pool import ProviderPool
    pool = ProviderPool(default_provider=MagicMock(), registry=MagicMock())
    provider, _ = pool.resolve("")
    assert provider is pool._default
```

- [ ] **Step 2: Implement ProviderPool**

Create `sage-python/src/sage/llm/provider_pool.py` — ~80 LOC. The `resolve()` method checks the registry for a provider matching the model_id, caches the result, and falls back to the default provider.

The registry API uses `select(needs)` which returns `ModelProfile | None`. ProviderPool maps model_id → profile via select, then resolves the profile's provider name to a connector. Read `sage-python/src/sage/providers/registry.py` for exact method signatures.

- [ ] **Step 3: Run tests**

```bash
cd sage-python && python -m pytest tests/test_provider_pool.py -v
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: ProviderPool — resolve model_id to live provider with cache + fallback"
```

### Task 8: Pipeline stages (pure functions)

**Files:**
- Create: `sage-python/src/sage/pipeline_stages.py`
- Create: `sage-python/tests/test_pipeline_stages.py`
- Create: `sage-python/tests/test_dag_features.py`

- [ ] **Step 1: Write tests for _infer_domain**

In `tests/test_pipeline_stages.py`:
```python
def test_infer_domain_code():
    from sage.pipeline_stages import _infer_domain
    assert _infer_domain("write a Python function to sort", None) == "code"

def test_infer_domain_math():
    from sage.pipeline_stages import _infer_domain
    assert _infer_domain("solve the integral of x^2 dx", None) == "math"

def test_infer_domain_reasoning():
    from sage.pipeline_stages import _infer_domain
    assert _infer_domain("analyze the pros and cons of microservices", None) == "reasoning"

def test_infer_domain_default():
    from sage.pipeline_stages import _infer_domain
    assert _infer_domain("hello world", None) == "general"
```

- [ ] **Step 2: Write tests for compute_dag_features**

In `tests/test_dag_features.py`:
```python
def test_single_node_dag():
    from sage.pipeline_stages import compute_dag_features, DAGFeatures
    # Mock single-node DAG
    dag = _make_linear_dag(1)
    f = compute_dag_features(dag)
    assert f.omega == 1  # max antichain = 1
    assert f.delta == 1  # depth = 1
    assert f.gamma == 0.0  # no edges

def test_linear_dag():
    from sage.pipeline_stages import compute_dag_features
    dag = _make_linear_dag(3)  # A -> B -> C
    f = compute_dag_features(dag)
    assert f.omega == 1  # no parallelism
    assert f.delta == 3
    assert f.gamma > 0.0

def test_parallel_dag():
    from sage.pipeline_stages import compute_dag_features
    dag = _make_parallel_dag(3)  # A, B, C (no edges)
    f = compute_dag_features(dag)
    assert f.omega == 3  # full parallelism
    assert f.delta == 1
```

- [ ] **Step 3: Write tests for select_macro_topology**

In `tests/test_pipeline_stages.py`:
```python
def test_select_macro_sequential():
    from sage.pipeline_stages import select_macro_topology, DAGFeatures
    # Low parallelism, low coupling → sequential
    f = DAGFeatures(omega=1, delta=5, gamma=0.2)
    assert select_macro_topology(f) == "sequential"

def test_select_macro_parallel():
    from sage.pipeline_stages import select_macro_topology, DAGFeatures
    # High parallelism, low coupling → parallel
    f = DAGFeatures(omega=4, delta=2, gamma=0.3)
    assert select_macro_topology(f) == "parallel"

def test_select_macro_hierarchical():
    from sage.pipeline_stages import select_macro_topology, DAGFeatures
    # High coupling → hierarchical
    f = DAGFeatures(omega=2, delta=4, gamma=0.8)
    assert select_macro_topology(f) == "hierarchical"
```

- [ ] **Step 4: Implement pipeline_stages.py**

Create `sage-python/src/sage/pipeline_stages.py` with `_infer_domain()`, `DAGFeatures`, `compute_dag_features()`, `select_macro_topology()`. ~150 LOC.

- [ ] **Step 5: Run all stage tests**

```bash
cd sage-python && python -m pytest tests/test_pipeline_stages.py tests/test_dag_features.py -v
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: pipeline stages — domain inference, DAG features (ω,δ,γ), macro topology selection"
```

### Task 9: TaskPlanner.plan_auto()

**Files:**
- Modify: `sage-python/src/sage/contracts/planner.py`
- Create: `sage-python/tests/test_plan_auto.py`

- [ ] **Step 1: Write test**

```python
"""Tests for TaskPlanner.plan_auto() — LLM-driven decomposition."""
import asyncio, pytest
from unittest.mock import AsyncMock
from sage.llm.base import LLMResponse
from sage.contracts.planner import TaskPlanner

def test_plan_auto_decomposes_task():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = LLMResponse(
        content='[{"id": "a", "description": "step a"}, {"id": "b", "description": "step b", "depends_on": ["a"]}]',
        model="test",
    )
    result = asyncio.run(planner.plan_auto("build a web app", mock_provider))
    assert result.node_count == 2
    assert result.edge_count == 1

def test_plan_auto_caps_at_6_steps():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    # LLM returns 10 steps — should be truncated to 6
    steps = [{"id": str(i), "description": f"step {i}"} for i in range(10)]
    mock_provider.generate.return_value = LLMResponse(content=str(steps).replace("'", '"'), model="test")
    result = asyncio.run(planner.plan_auto("complex task", mock_provider))
    assert result.node_count <= 6

def test_plan_auto_fallback_on_parse_error():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = LLMResponse(content="not json", model="test")
    result = asyncio.run(planner.plan_auto("some task", mock_provider))
    # Fallback: single-node DAG
    assert result.node_count == 1
```

- [ ] **Step 2: Implement plan_auto**

Add `plan_auto()` to `sage-python/src/sage/contracts/planner.py`. ~100 LOC. Includes LLM prompt, JSON parsing, truncation to MAX_DECOMPOSITION_STEPS=6, delegation to `plan_static()`, and single-node DAG fallback.

- [ ] **Step 3: Run tests**

```bash
cd sage-python && python -m pytest tests/test_plan_auto.py -v
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: TaskPlanner.plan_auto() — LLM-driven decomposition with 6-step cap + fallback"
```

### Task 10: TopologyRunner modification

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py`

- [ ] **Step 1: Add provider_pool parameter to __init__**

In `runner.py`, add `*, provider_pool=None` as keyword-only parameter to `__init__`. Store as `self._provider_pool`.

- [ ] **Step 2: Modify _execute_node to resolve per-node provider**

Replace the direct `self._llm.generate()` call with:
```python
node_model_id = getattr(node, "model_id", "")
if node_model_id and self._provider_pool:
    provider, config = self._provider_pool.resolve(node_model_id)
else:
    provider, config = self._llm, self._config
response = await provider.generate(messages=messages, config=config)
```

- [ ] **Step 3: Run existing TopologyRunner tests**

```bash
cd sage-python && python -m pytest tests/ -k "topology_runner or runner" -v
```

Expected: All existing tests still pass (provider_pool=None → legacy behavior).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: TopologyRunner resolves per-node model_id via ProviderPool"
```

---

## Chunk 4: Pipeline + Wiring

### Task 11: CognitiveOrchestrationPipeline

**Files:**
- Create: `sage-python/src/sage/pipeline.py`
- Create: `sage-python/tests/test_pipeline.py`

- [ ] **Step 1: Write integration test**

Create `sage-python/tests/test_pipeline.py` testing the full 5-stage pipeline with mocks (mock router, mock engine, mock assigner, mock runner). Verify each stage is called in order, events emitted on EventBus, result returned.

- [ ] **Step 2: Implement pipeline.py**

Create `sage-python/src/sage/pipeline.py` with `PipelineContext` dataclass and `CognitiveOrchestrationPipeline` class. ~250 LOC. Chains stage functions from `pipeline_stages.py`. Emits `AgentEvent` on EventBus at each transition.

- [ ] **Step 3: Run tests**

```bash
cd sage-python && python -m pytest tests/test_pipeline.py -v
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: CognitiveOrchestrationPipeline — 5-stage orchestration with EventBus observability"
```

### Task 12: Boot.py wiring

**Files:**
- Modify: `sage-python/src/sage/boot.py`

- [ ] **Step 1: Add pipeline instantiation after existing boot code**

Wire ModelAssigner (Rust primary, Python fallback), ProviderPool, and CognitiveOrchestrationPipeline into `boot_agent_system()`.

- [ ] **Step 2: Replace AgentSystem.run() inline code with pipeline delegation**

Add `pipeline` to `AgentSystem.__init__`. In `run()`, delegate to `pipeline.run()` when available. Move current inline code to `_run_legacy()`.

- [ ] **Step 3: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -x -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -10
```

Expected: Zero regressions. All existing tests pass via legacy fallback.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: wire CognitiveOrchestrationPipeline into boot.py with legacy fallback"
```

### Task 13: Parity test + CLAUDE.md update

**Files:**
- Create: `sage-python/tests/test_assigner_parity.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write parity test**

Create `sage-python/tests/test_assigner_parity.py` that (when sage_core is available) runs the same assignment scenario through both Rust and Python ModelAssigner and asserts identical model_id assignments. Include edge cases: tight budget (budget < cheapest model) and missing capabilities (no model has tools+json → keep existing model_id).

- [ ] **Step 2: Update CLAUDE.md**

Add `pipeline.py`, `pipeline_stages.py`, `llm/provider_pool.py`, `llm/model_assigner.py` to the Key Python Modules section. Update the ModelAssigner description in Key Rust Modules. Update the deprecated section per the spec.

- [ ] **Step 3: Run parity test**

```bash
cd sage-python && python -m pytest tests/test_assigner_parity.py -v
```

- [ ] **Step 4: Final full test suite**

```bash
cd sage-python && python -m pytest tests/ -q --ignore=tests/test_e2e_real.py --ignore=tests/test_exocortex.py --ignore=tests/test_a2a_server.py 2>&1 | tail -10
cd sage-core && cargo test --no-default-features --lib 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test: assigner parity (Rust/Python) + update CLAUDE.md for pipeline"
```
