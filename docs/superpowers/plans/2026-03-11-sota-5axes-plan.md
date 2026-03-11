# SOTA 5-Axes Upgrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 SOTA improvements (LTL verification, invariant synthesis, CMA-ME evolution, MCTS topology search, learned routing) plus quick wins, all Rust-first.

**Architecture:** New Rust modules behind feature flags, exposed via PyO3. Minimal Python changes (wiring only). Each axe is a self-contained module that integrates into existing topology/verification/routing subsystems.

**Tech Stack:** Rust 1.94 + OxiZ 0.1.3 + petgraph + PyO3 0.25 + tracing. No new external crates.

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `sage-core/src/verification/ltl.rs` | LTL temporal property checks on TopologyGraph (BFS reachability, safety, liveness) |
| `sage-core/src/topology/cma_me.rs` | CMA-ME emitter for continuous topology parameter optimization |
| `sage-core/src/topology/mcts.rs` | MCTS tree search for topology space exploration |

### Modified Files
| File | Changes |
|------|---------|
| `sage-core/src/verification/mod.rs` | Add `verify_invariant_with_feedback()`, `synthesize_invariant()`, tracing spans |
| `sage-core/src/verification/mod.rs` | Re-export `ltl` module |
| `sage-core/src/topology/mod.rs` | Re-export `cma_me`, `mcts` modules |
| `sage-core/src/topology/engine.rs` | Wire CMA-ME into `evolve()`, MCTS as 6th path in `generate()` |
| `sage-core/src/topology/map_elites.rs` | Add `all_entries_mut()` for CMA fitness feedback |
| `sage-core/src/topology/pyo3_wrappers.rs` | Return opaque topology_id from generate(), expose LtlVerifier |
| `sage-core/src/topology/verifier.rs` | Wire LTL checks into HybridVerifier |
| `sage-core/src/routing/router.rs` | Add shadow trace JSONL writer, `retrain_thresholds()` |
| `sage-core/src/lib.rs` | Register new PyO3 classes |
| `sage-python/src/sage/topology/kg_rlvr.py` | Wire `verify_invariant_with_feedback()` |
| `sage-python/src/sage/agent_loop.py` | Inject CEGAR feedback into S3 escalation |
| `sage-python/src/sage/memory/episodic.py` | Add WAL pragma |
| `sage-python/src/sage/memory/semantic.py` | Add WAL pragma |
| `sage-python/src/sage/memory/causal.py` | Add WAL pragma |
| `sage-python/src/sage/routing/shadow.py` | 2-tier gate (soft 500/10%, hard 1000/5%) |

---

## Chunk 1: Quick Wins (SQLite WAL + Tracing + PyO3 Opaque ID)

### Task 1: SQLite WAL Mode

**Files:**
- Modify: `sage-python/src/sage/memory/episodic.py:39-50`
- Modify: `sage-python/src/sage/memory/semantic.py:25-35`
- Modify: `sage-python/src/sage/memory/causal.py:40-55`

- [ ] **Step 1: Add WAL pragma to episodic.py**

In `episodic.py`, after `CREATE TABLE IF NOT EXISTS`, add:
```python
await db.execute("PRAGMA journal_mode=WAL")
await db.execute("PRAGMA synchronous=NORMAL")
await db.execute("PRAGMA busy_timeout=5000")
```

- [ ] **Step 2: Add WAL pragma to semantic.py**

In `semantic.py`, after SQLite connection init, add same 3 PRAGMA lines.

- [ ] **Step 3: Add WAL pragma to causal.py**

In `causal.py`, after SQLite connection init, add same 3 PRAGMA lines.

- [ ] **Step 4: Run Python tests**

Run: `cd sage-python && python -m pytest tests/ -q`
Expected: 1143+ passed, 0 failures

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/memory/episodic.py sage-python/src/sage/memory/semantic.py sage-python/src/sage/memory/causal.py
git commit -m "fix(memory): enable SQLite WAL mode for concurrent write safety"
```

### Task 2: Tracing Spans on SmtVerifier

**Files:**
- Modify: `sage-core/src/verification/mod.rs:1-25`

- [ ] **Step 1: Add tracing import**

At line 13, after `use oxiz::{...}`, add:
```rust
use tracing::{info, instrument};
```

- [ ] **Step 2: Add `#[instrument]` to public methods**

Annotate each `pub fn` in `#[pymethods] impl SmtVerifier` with:
```rust
#[instrument(skip(self), fields(method = "prove_memory_safety"))]
```
Use appropriate method name for each. Add `info!("result" = %result)` before return.

- [ ] **Step 3: Run Rust tests**

Run: `cargo test --features smt`
Expected: 351+ passed

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/verification/mod.rs
git commit -m "feat(tracing): add structured spans to SmtVerifier methods"
```

### Task 3: PyO3 Opaque Topology ID

**Files:**
- Modify: `sage-core/src/topology/pyo3_wrappers.rs:25-55`

- [ ] **Step 1: Add `topology_id()` getter to PyGenerateResult**

```rust
/// Return only the topology ID (opaque handle). Use get_topology() for lazy-load.
pub fn topology_id(&self) -> String {
    // Generate a ULID or hash from the topology
    format!("{:x}", self.inner.topology.hash())
}
```

- [ ] **Step 2: Verify existing `topology()` still works (backward compat)**

The existing `topology()` method stays. New code should prefer `topology_id()` + `get_topology()`.

- [ ] **Step 3: Run Rust tests**

Run: `cargo test`
Expected: 351+ passed

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/topology/pyo3_wrappers.rs
git commit -m "feat(pyo3): add opaque topology_id getter for lazy-load pattern"
```

---

## Chunk 2: Axe 4 — LTL Model Checking (`ltl.rs`)

### Task 4: Create LTL Verifier Module

**Files:**
- Create: `sage-core/src/verification/ltl.rs`
- Modify: `sage-core/src/verification/mod.rs` (add `pub mod ltl;`)

- [ ] **Step 1: Write failing test for reachability**

In `ltl.rs`, add test module:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::topology_graph::TopologyGraph;
    use crate::topology::templates::TemplateStore;

    #[test]
    fn test_reachability_sequential() {
        let store = TemplateStore::new();
        let g = store.create("sequential").unwrap();
        let v = LtlVerifier::new();
        // In sequential A→B→C, A can reach C
        assert!(v.check_reachability(&g, 0, 2));
        // C cannot reach A (DAG)
        assert!(!v.check_reachability(&g, 2, 0));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features smt test_reachability_sequential`
Expected: FAIL — module/struct not defined

- [ ] **Step 3: Implement LtlVerifier struct with reachability**

```rust
use crate::topology::topology_graph::TopologyGraph;
use petgraph::visit::Bfs;
use tracing::instrument;

pub struct LtlVerifier;

impl LtlVerifier {
    pub fn new() -> Self { Self }

    /// BFS reachability: can we reach `to` from `from`?
    #[instrument(skip(self, graph))]
    pub fn check_reachability(&self, graph: &TopologyGraph, from: usize, to: usize) -> bool {
        let pg = graph.inner();  // petgraph::DiGraph access
        let from_idx = petgraph::graph::NodeIndex::new(from);
        let to_idx = petgraph::graph::NodeIndex::new(to);
        let mut bfs = Bfs::new(pg, from_idx);
        while let Some(node) = bfs.next(pg) {
            if node == to_idx { return true; }
        }
        false
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features smt test_reachability_sequential`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/verification/ltl.rs sage-core/src/verification/mod.rs
git commit -m "feat(ltl): add LtlVerifier with BFS reachability check"
```

### Task 5: LTL Safety Check (no HIGH→LOW paths)

**Files:**
- Modify: `sage-core/src/verification/ltl.rs`

- [ ] **Step 1: Write failing test for safety**

```rust
#[test]
fn test_safety_no_high_to_low() {
    let v = LtlVerifier::new();
    // Build graph: A(label=3) → B(label=1) — HIGH flows to LOW = UNSAFE
    let mut g = TopologyGraph::new();
    g.add_node("A", "analyst", "model-a", 2, vec![], 3, 1.0, 30.0);
    g.add_node("B", "reporter", "model-b", 1, vec![], 1, 0.5, 15.0);
    g.add_edge(0, 1, "control", None, "open", None, 1.0).unwrap();
    let result = v.check_safety(&g);
    assert!(!result.passed);
    assert!(result.violations.iter().any(|v| v.contains("HIGH") && v.contains("LOW")));
}

#[test]
fn test_safety_same_level_ok() {
    let v = LtlVerifier::new();
    let mut g = TopologyGraph::new();
    g.add_node("A", "coder", "model-a", 2, vec![], 2, 1.0, 30.0);
    g.add_node("B", "reviewer", "model-b", 2, vec![], 2, 0.5, 15.0);
    g.add_edge(0, 1, "control", None, "open", None, 1.0).unwrap();
    let result = v.check_safety(&g);
    assert!(result.passed);
}
```

- [ ] **Step 2: Implement check_safety**

```rust
pub struct LtlResult {
    pub passed: bool,
    pub violations: Vec<String>,
}

/// Safety: no edge from higher security_label to lower.
#[instrument(skip(self, graph))]
pub fn check_safety(&self, graph: &TopologyGraph) -> LtlResult {
    let mut violations = Vec::new();
    for edge in graph.inner().edge_indices() {
        let (src, dst) = graph.inner().edge_endpoints(edge).unwrap();
        let src_label = graph.inner()[src].security_label;
        let dst_label = graph.inner()[dst].security_label;
        if src_label > dst_label {
            violations.push(format!(
                "Unsafe flow: node {} (label=HIGH:{}) → node {} (label=LOW:{})",
                src.index(), src_label, dst.index(), dst_label
            ));
        }
    }
    LtlResult { passed: violations.is_empty(), violations }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --features smt test_safety`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/verification/ltl.rs
git commit -m "feat(ltl): add safety check — no HIGH→LOW info flow"
```

### Task 6: LTL Liveness + Bounded Liveness

**Files:**
- Modify: `sage-core/src/verification/ltl.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn test_liveness_all_entries_reach_exits() {
    let store = TemplateStore::new();
    let g = store.create("parallel").unwrap();
    let v = LtlVerifier::new();
    let result = v.check_liveness(&g);
    assert!(result.passed);
}

#[test]
fn test_bounded_liveness_depth_limit() {
    let store = TemplateStore::new();
    let g = store.create("sequential").unwrap();
    let v = LtlVerifier::new();
    // Sequential 3-node: depth=2, limit=5 should pass
    assert!(v.check_bounded_liveness(&g, 5).passed);
    // limit=1 should fail (depth=2 > 1)
    assert!(!v.check_bounded_liveness(&g, 1).passed);
}
```

- [ ] **Step 2: Implement liveness (BFS from each entry, must reach an exit)**

```rust
/// Liveness: every entry node can reach at least one exit node.
#[instrument(skip(self, graph))]
pub fn check_liveness(&self, graph: &TopologyGraph) -> LtlResult {
    let entries = graph.entry_nodes();
    let exits: HashSet<usize> = graph.exit_nodes().into_iter().collect();
    let mut violations = Vec::new();
    for entry_idx in &entries {
        let mut found_exit = false;
        let idx = petgraph::graph::NodeIndex::new(*entry_idx);
        let mut bfs = Bfs::new(graph.inner(), idx);
        while let Some(node) = bfs.next(graph.inner()) {
            if exits.contains(&node.index()) { found_exit = true; break; }
        }
        if !found_exit {
            violations.push(format!("Deadlock: entry node {} cannot reach any exit", entry_idx));
        }
    }
    LtlResult { passed: violations.is_empty(), violations }
}
```

- [ ] **Step 3: Implement bounded liveness (longest path ≤ K)**

```rust
/// Bounded liveness: longest path from any entry to any exit ≤ depth_limit.
#[instrument(skip(self, graph), fields(depth_limit))]
pub fn check_bounded_liveness(&self, graph: &TopologyGraph, depth_limit: usize) -> LtlResult {
    let entries = graph.entry_nodes();
    let exits: HashSet<usize> = graph.exit_nodes().into_iter().collect();
    let mut violations = Vec::new();
    // DFS with depth tracking
    for entry_idx in &entries {
        let max_depth = self.dfs_max_depth(graph, *entry_idx, &exits);
        if max_depth > depth_limit {
            violations.push(format!(
                "Path from entry {} has depth {} > limit {}",
                entry_idx, max_depth, depth_limit
            ));
        }
    }
    LtlResult { passed: violations.is_empty(), violations }
}

fn dfs_max_depth(&self, graph: &TopologyGraph, start: usize, exits: &HashSet<usize>) -> usize {
    let mut max_depth = 0;
    let mut stack = vec![(start, 0usize)];
    while let Some((node, depth)) = stack.pop() {
        if exits.contains(&node) { max_depth = max_depth.max(depth); continue; }
        let idx = petgraph::graph::NodeIndex::new(node);
        for neighbor in graph.inner().neighbors(idx) {
            stack.push((neighbor.index(), depth + 1));
        }
    }
    max_depth
}
```

- [ ] **Step 4: Run all LTL tests**

Run: `cargo test --features smt ltl`
Expected: 4+ PASS

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/verification/ltl.rs
git commit -m "feat(ltl): add liveness and bounded liveness checks"
```

### Task 7: Wire LTL into HybridVerifier + PyO3

**Files:**
- Modify: `sage-core/src/topology/verifier.rs`
- Modify: `sage-core/src/topology/pyo3_wrappers.rs`
- Modify: `sage-core/src/lib.rs`

- [ ] **Step 1: Add LTL checks to HybridVerifier**

In `verifier.rs`, import `LtlVerifier` and add calls in `verify()`:
```rust
let ltl = LtlVerifier::new();
let safety = ltl.check_safety(graph);
if !safety.passed { result.errors.extend(safety.violations); }
let liveness = ltl.check_liveness(graph);
if !liveness.passed { result.warnings.extend(liveness.violations); }
```

- [ ] **Step 2: Expose LtlVerifier as PyO3 class**

Add `#[pyclass]` + `#[pymethods]` to `LtlVerifier` and `LtlResult`.

- [ ] **Step 3: Register in lib.rs**

Add `m.add_class::<LtlVerifier>()?;` and `m.add_class::<LtlResult>()?;`

- [ ] **Step 4: Run all tests (Rust + Python)**

Run: `cargo test --features smt && cd sage-python && python -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/verification/ltl.rs sage-core/src/topology/verifier.rs sage-core/src/topology/pyo3_wrappers.rs sage-core/src/lib.rs
git commit -m "feat(ltl): wire LtlVerifier into HybridVerifier + PyO3 bindings"
```

---

## Chunk 3: Axe 2 — Invariant Synthesis Loop (CEGAR)

### Task 8: SmtVerifier.verify_invariant_with_feedback()

**Files:**
- Modify: `sage-core/src/verification/mod.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn test_invariant_feedback_success() {
    let v = SmtVerifier::new();
    let r = v.verify_invariant_with_feedback("x > 5", "x > 3");
    assert!(r.safe);
    assert!(r.violations.is_empty());
}

#[test]
fn test_invariant_feedback_failure_has_diagnostics() {
    let v = SmtVerifier::new();
    let r = v.verify_invariant_with_feedback("x > 0", "x > 10");
    assert!(!r.safe);
    assert!(!r.violations.is_empty());
    // Should explain which clause fails
    assert!(r.violations[0].contains("x > 10"));
}
```

- [ ] **Step 2: Implement verify_invariant_with_feedback**

Returns `SmtVerificationResult` instead of `bool`. On failure, analyses which sub-clause of `post` is the weakest link by testing each clause individually against `pre`.

```rust
#[instrument(skip(self))]
pub fn verify_invariant_with_feedback(&self, pre: &str, post: &str) -> SmtVerificationResult {
    let start = Instant::now();
    // Try full implication first
    if self.verify_invariant(pre, post) {
        return SmtVerificationResult {
            safe: true, violations: vec![],
            proof_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }
    // Diagnose: which clause(s) of post fail?
    let mut violations = Vec::new();
    // Split post on " and " to find individual clauses
    let clauses: Vec<&str> = post.split(" and ").collect();
    if clauses.len() > 1 {
        for clause in &clauses {
            if !self.verify_invariant(pre, clause.trim()) {
                violations.push(format!("Failed clause: {}", clause.trim()));
            }
        }
    }
    if violations.is_empty() {
        violations.push(format!("Implication {} → {} does not hold", pre, post));
    }
    SmtVerificationResult {
        safe: false, violations,
        proof_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --features smt test_invariant_feedback`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/verification/mod.rs
git commit -m "feat(smt): add verify_invariant_with_feedback with clause diagnostics"
```

### Task 9: synthesize_invariant() CEGAR loop

**Files:**
- Modify: `sage-core/src/verification/mod.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn test_synthesize_finds_valid_invariant() {
    let v = SmtVerifier::new();
    // Given candidates that are too strong, synthesizer should find working one
    let result = v.synthesize_invariant(
        "x > 5",                              // pre
        &["x > 100", "x > 10", "x > 3"],     // post candidates (strong → weak)
        5                                      // max rounds
    );
    assert!(result.is_some());
    let found = result.unwrap();
    assert!(v.verify_invariant("x > 5", &found));
}

#[test]
fn test_synthesize_none_when_impossible() {
    let v = SmtVerifier::new();
    let result = v.synthesize_invariant(
        "x > 0",
        &["x > 100", "x > 50"],  // all too strong
        2
    );
    // May or may not find via weakening — but should not panic
    assert!(result.is_none() || v.verify_invariant("x > 0", &result.unwrap()));
}
```

- [ ] **Step 2: Implement synthesize_invariant**

Iterates candidates strongest→weakest, returns first that verifies. If none work, tries weakening the weakest candidate by replacing strict comparisons with non-strict.

```rust
#[instrument(skip(self))]
pub fn synthesize_invariant(
    &self, pre: &str, post_candidates: &[&str], max_rounds: u32
) -> Option<String> {
    // Round 1: try each candidate as-is
    for candidate in post_candidates {
        if self.verify_invariant(pre, candidate) {
            return Some(candidate.to_string());
        }
    }
    // Round 2+: weaken last candidate
    if let Some(weakest) = post_candidates.last() {
        let weakened = weakest
            .replace(" > ", " >= ")
            .replace(" == ", " >= ");
        if self.verify_invariant(pre, &weakened) {
            return Some(weakened);
        }
    }
    None
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --features smt test_synthesize`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/verification/mod.rs
git commit -m "feat(smt): add synthesize_invariant CEGAR loop"
```

### Task 10: Wire Python kg_rlvr.py + agent_loop.py

**Files:**
- Modify: `sage-python/src/sage/topology/kg_rlvr.py:143-165`
- Modify: `sage-python/src/sage/agent_loop.py:609-625`

- [ ] **Step 1: Wire verify_invariant_with_feedback in kg_rlvr.py**

Replace `verify_invariant()` call in `verify_step()` (line ~204) to use feedback variant:
```python
if self._rust:
    r = self._rust.verify_invariant_with_feedback(pre, post)
    if r.safe:
        return 1.0
    # Inject feedback into step for S3 retry
    self._last_invariant_feedback = r.violations
    return -1.0
```

- [ ] **Step 2: Wire feedback into S3 escalation in agent_loop.py**

In the S3 escalation block (lines 617-624), append invariant feedback if available:
```python
if hasattr(self, 'prm') and hasattr(self.prm.kg, '_last_invariant_feedback'):
    feedback = self.prm.kg._last_invariant_feedback
    if feedback:
        escalation_msg += f"\n\nPrevious invariant failures:\n"
        for f in feedback:
            escalation_msg += f"- {f}\n"
```

- [ ] **Step 3: Run Python tests**

Run: `cd sage-python && python -m pytest tests/test_kg_rlvr.py tests/test_agent_loop.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/topology/kg_rlvr.py sage-python/src/sage/agent_loop.py
git commit -m "feat(s3): wire invariant synthesis feedback into S3 CEGAR loop"
```

---

## Chunk 4: Axe 1 — CMA-ME for MAP-Elites

### Task 11: Create CmaEmitter Module

**Files:**
- Create: `sage-core/src/topology/cma_me.rs`
- Modify: `sage-core/src/topology/mod.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_emitter_new() {
        let emitter = CmaEmitter::new(3, 0.3);
        assert_eq!(emitter.dimension(), 3);
        assert_eq!(emitter.population_size(), 0);
    }

    #[test]
    fn test_cma_ask_returns_samples() {
        let mut emitter = CmaEmitter::new(3, 0.3);
        let samples = emitter.ask(5);
        assert_eq!(samples.len(), 5);
        for s in &samples { assert_eq!(s.len(), 3); }
    }

    #[test]
    fn test_cma_tell_updates_mean() {
        let mut emitter = CmaEmitter::new(3, 0.3);
        let samples = emitter.ask(5);
        let fitnesses: Vec<f64> = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let old_mean = emitter.mean().to_vec();
        emitter.tell(&samples, &fitnesses);
        // Mean should shift toward high-fitness samples
        assert_ne!(emitter.mean(), &old_mean[..]);
    }
}
```

- [ ] **Step 2: Implement CmaEmitter**

Pure Rust, no BLAS. Small-dimension CMA (3D: cost, timeout, edge_weight).

```rust
/// CMA-ME Emitter for continuous topology parameter optimization.
/// Adapts a multivariate Gaussian (mean + covariance) toward high-fitness regions.
pub struct CmaEmitter {
    dim: usize,
    mean: Vec<f64>,
    sigma: f64,           // step size
    cov: Vec<Vec<f64>>,   // dim x dim covariance (identity init)
    generation: u32,
}

impl CmaEmitter {
    pub fn new(dim: usize, initial_sigma: f64) -> Self { ... }
    pub fn dimension(&self) -> usize { self.dim }
    pub fn population_size(&self) -> usize { 0 /* stateless */ }
    pub fn mean(&self) -> &[f64] { &self.mean }

    /// Sample `n` parameter vectors from current distribution.
    pub fn ask(&mut self, n: usize) -> Vec<Vec<f64>> { ... }

    /// Update distribution based on fitnesses (higher = better).
    /// Sorts by fitness, shifts mean toward top-μ samples, updates covariance.
    pub fn tell(&mut self, samples: &[Vec<f64>], fitnesses: &[f64]) { ... }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test cma_emitter`
Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/topology/cma_me.rs sage-core/src/topology/mod.rs
git commit -m "feat(evo): add CmaEmitter for continuous parameter optimization"
```

### Task 12: Wire CMA-ME into evolve()

**Files:**
- Modify: `sage-core/src/topology/engine.rs:455-564`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn test_evolve_uses_cma_for_continuous_params() {
    let mut engine = DynamicTopologyEngine::new();
    // Insert a base topology into archive
    let store = TemplateStore::new();
    let topo = store.create("sequential").unwrap();
    engine.cache_topology(topo.clone());
    engine.record_outcome("test-id", "test-task", 0.8, 0.05, 100.0, 2);
    // Evolve should produce mutated topologies with adapted params
    engine.evolve(1, 5);
    assert!(engine.archive_cell_count() >= 1);
}
```

- [ ] **Step 2: Integrate CmaEmitter in evolve()**

In `evolve()`, after sampling population from archive:
```rust
// Extract continuous params from each candidate
let params: Vec<Vec<f64>> = population.iter().map(|e| {
    vec![
        e.graph.node(0).max_cost_usd as f64,
        e.graph.node(0).max_wall_time_s as f64,
        1.0, // edge weight placeholder
    ]
}).collect();

// CMA step
let new_params = self.cma_emitter.ask(population.len());
// Apply CMA-sampled params to mutated graphs
for (i, candidate) in mutated.iter_mut().enumerate() {
    if let Some(p) = new_params.get(i) {
        candidate.set_node_budget(0, p[0] as f32, p[1] as f32);
    }
}
```

After evaluation:
```rust
let fitnesses: Vec<f64> = evaluated.iter().map(|e| e.quality as f64).collect();
self.cma_emitter.tell(&new_params, &fitnesses);
```

- [ ] **Step 3: Run tests**

Run: `cargo test test_evolve`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/topology/engine.rs sage-core/src/topology/cma_me.rs
git commit -m "feat(evo): wire CMA-ME into evolve() for continuous param adaptation"
```

---

## Chunk 5: Axe 5 — MCTS Topology Search

### Task 13: Create MCTS Module

**Files:**
- Create: `sage-core/src/topology/mcts.rs`
- Modify: `sage-core/src/topology/mod.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates::TemplateStore;

    #[test]
    fn test_mcts_node_ucb1() {
        let mut node = MctsNode::new_root();
        node.visit_count = 10;
        node.total_quality = 7.0;
        let ucb = node.ucb1(100, 1.41);
        assert!(ucb > 0.7); // exploitation = 0.7
        assert!(ucb < 2.0); // exploration bounded
    }

    #[test]
    fn test_mcts_search_returns_topology() {
        let store = TemplateStore::new();
        let root = store.create("sequential").unwrap();
        let mut searcher = MctsSearcher::new(50, 100);
        let result = searcher.search(root);
        assert!(result.is_some());
    }
}
```

- [ ] **Step 2: Implement MctsSearcher**

```rust
use crate::topology::topology_graph::TopologyGraph;
use crate::topology::mutations;
use std::time::Instant;

pub struct MctsNode {
    pub topology: TopologyGraph,
    pub visit_count: u32,
    pub total_quality: f64,
    pub children: Vec<MctsNode>,
}

impl MctsNode {
    pub fn ucb1(&self, parent_visits: u32, c: f64) -> f64 {
        if self.visit_count == 0 { return f64::INFINITY; }
        let exploit = self.total_quality / self.visit_count as f64;
        let explore = c * ((parent_visits as f64).ln() / self.visit_count as f64).sqrt();
        exploit + explore
    }
}

pub struct MctsSearcher {
    max_simulations: u32,
    max_time_ms: u64,
}

impl MctsSearcher {
    pub fn new(max_simulations: u32, max_time_ms: u64) -> Self { ... }

    pub fn search(&mut self, root: TopologyGraph) -> Option<TopologyGraph> {
        let start = Instant::now();
        let mut tree = MctsNode::new_root_with(root);
        for _ in 0..self.max_simulations {
            if start.elapsed().as_millis() as u64 > self.max_time_ms { break; }
            self.simulate(&mut tree);
        }
        tree.best_child().map(|c| c.topology.clone())
    }

    fn simulate(&self, node: &mut MctsNode) { ... }
    fn expand(&self, node: &mut MctsNode) { ... }
    fn rollout(&self, topology: &TopologyGraph) -> f64 { ... }
    fn backpropagate(path: &mut [&mut MctsNode], quality: f64) { ... }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test mcts`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/topology/mcts.rs sage-core/src/topology/mod.rs
git commit -m "feat(mcts): add MctsSearcher with UCB1 topology space exploration"
```

### Task 14: Wire MCTS as 6th Path in generate()

**Files:**
- Modify: `sage-core/src/topology/engine.rs:140-188`

- [ ] **Step 1: Add MCTS path between mutation and fallback**

In `generate()`, after Path 4 (try_mutation) and before Path 5 (template_fallback):
```rust
// Path 5: MCTS search (systematic exploration)
if self.archive.cell_count() >= 5 {
    if let Some(result) = self.try_mcts_search() {
        return result;
    }
}
// Path 6: Template fallback (renamed from Path 5)
```

- [ ] **Step 2: Implement try_mcts_search()**

```rust
fn try_mcts_search(&mut self) -> Option<GenerateResult> {
    let best = self.archive.best_by_quality()?;
    let mut searcher = MctsSearcher::new(50, 100);
    let topology = searcher.search(best.graph.clone())?;
    let topo_id = self.cache_topology(topology.clone());
    Some(GenerateResult {
        topology, source: TopologySource::MctsSearch,
        confidence: best.quality * 0.85,
    })
}
```

- [ ] **Step 3: Add TopologySource::MctsSearch variant**

- [ ] **Step 4: Run all tests**

Run: `cargo test`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/topology/engine.rs sage-core/src/topology/mcts.rs
git commit -m "feat(mcts): wire MCTS as 6th path in DynamicTopologyEngine.generate()"
```

---

## Chunk 6: Axe 3 — Learned Routing

### Task 15: Rust-Native Shadow Trace Writer

**Files:**
- Modify: `sage-core/src/routing/router.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn test_shadow_trace_appends_to_buffer() {
    let mut router = AdaptiveRouter::new();
    router.route("simple hello task");
    assert!(router.shadow_trace_count() >= 1);
}
```

- [ ] **Step 2: Implement shadow trace buffer in AdaptiveRouter**

Add `shadow_traces: Vec<ShadowTrace>` field with `ShadowTrace { task_hash, structural_tier, onnx_tier, timestamp }`. Each `route()` call appends a trace. Add `flush_traces(path: &str)` for JSONL export.

- [ ] **Step 3: Run tests**

Run: `cargo test test_shadow_trace`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/routing/router.rs
git commit -m "feat(routing): add Rust-native shadow trace collection"
```

### Task 16: Threshold Retraining + 2-Tier Gate

**Files:**
- Modify: `sage-core/src/routing/router.rs`
- Modify: `sage-python/src/sage/routing/shadow.py:176-186`

- [ ] **Step 1: Write failing test for retrain_thresholds**

```rust
#[test]
fn test_retrain_thresholds_adjusts_from_feedback() {
    let mut router = AdaptiveRouter::new();
    // Simulate feedback: structural says S1, correct answer was S2
    for _ in 0..100 {
        router.record_feedback("complex task", 2, 0.8);  // actual=S2, quality=0.8
    }
    let (old_c0, old_c1) = (router.c0_threshold(), router.c1_threshold());
    router.retrain_thresholds();
    // Thresholds should have shifted
    assert!(router.c0_threshold() != old_c0 || router.c1_threshold() != old_c1);
}
```

- [ ] **Step 2: Implement retrain_thresholds()**

Simple logistic regression on feedback buffer: adjusts c0/c1 thresholds to minimize disagreement between structural prediction and ground truth (from feedback).

- [ ] **Step 3: Update shadow.py gate to 2-tier**

```python
def is_phase5_soft_ready(self) -> bool:
    """Soft gate: 500 traces, <10% divergence."""
    return self.total >= 500 and self.divergence_rate() < 0.10

def is_phase5_hard_ready(self) -> bool:
    """Hard gate: 1000 traces, <5% divergence. Safe to delete Python router."""
    return self.total >= 1000 and self.divergence_rate() < 0.05
```

- [ ] **Step 4: Run all tests**

Run: `cargo test && cd sage-python && python -m pytest tests/ -q`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/routing/router.rs sage-python/src/sage/routing/shadow.py
git commit -m "feat(routing): add threshold retraining + 2-tier shadow gate"
```

---

## Final Task: Build Wheel + Full Validation

### Task 17: Integration Validation

- [ ] **Step 1: Full Rust test suite**

Run: `cargo test --features smt`
Expected: 370+ tests pass (original 351 + ~20 new)

- [ ] **Step 2: Build and install wheel**

Run: `maturin build --features smt --release && pip install --force-reinstall target/wheels/*.whl`

- [ ] **Step 3: Full Python test suite**

Run: `cd sage-python && python -m pytest tests/ -q`
Expected: 1143+ passed

- [ ] **Step 4: Update CLAUDE.md**

Document new modules: ltl.rs, cma_me.rs, mcts.rs, verify_invariant_with_feedback, shadow traces, 2-tier gate.

- [ ] **Step 5: Update MEMORY.md**

Add new test counts, module descriptions.

- [ ] **Step 6: Final commit and push**

```bash
git add -A && git commit -m "docs: update CLAUDE.md and MEMORY.md for SOTA 5-axes upgrade"
git push origin master
```
