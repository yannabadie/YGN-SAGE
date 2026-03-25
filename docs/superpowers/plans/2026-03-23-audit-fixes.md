# Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 13 issues identified by 3 independent code audits (Critique 1/2/3), from CRITICAL bandit bugs to LOW documentation errors.

**Architecture:** Fixes are ordered by severity (CRITICAL → HIGH → MEDIUM → LOW). Each fix is self-contained and independently testable. The 2 CRITICAL fixes repair the bandit learning loop. The 2 HIGH fixes repair the topology execution (runner uses predecessors, upgrade resolves model).

**Tech Stack:** Rust (sage-core, PyO3), Python (sage-python), pytest, cargo test

---

## File Map

| Action | File | Fix |
|--------|------|-----|
| MODIFY | `sage-python/src/sage/pipeline.py` | C1: bandit choose() before execute, record() with decision_id |
| MODIFY | `sage-core/src/topology/engine.rs` | C2: store decision_id in generate(), use record_outcome() |
| MODIFY | `sage-python/src/sage/topology/runner.py` | H1: use get_predecessors() instead of all nodes |
| MODIFY | `sage-python/src/sage/topology_controller.py` | H2: resolve fallback_tier → new_model_id |
| MODIFY | `sage-python/src/sage/topology/runner.py` | H3: spawn_subagent adds real node (V3 marker) |
| MODIFY | `sage-python/src/sage/topology/runner.py` | M3: prune_node skips via executor |
| MODIFY | `sage-python/src/sage/verl/topology_env.py` | M4: reroute regenerates instead of terminal |
| MODIFY | `sage-core/src/memory/smmu.rs` | M1: add evict_oldest() GC method |
| MODIFY | `sage-python/src/sage/verl/training_memory.py` | M2: add LIMIT + prefilter to query |
| MODIFY | `sage-core/src/topology/engine.rs` | L3: fix "5-path" → "6-path" comment |
| CREATE | `sage-python/tests/test_audit_fixes.py` | Tests for all fixes |

---

### Task 1: C1 — Fix pipeline bandit.record() (CRITICAL)

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:540-580,670-680`
- Test: `sage-python/tests/test_audit_fixes.py`

**Problem:** `pipeline.py:677` calls `bandit.record("pipeline", ...)` but bandit expects a `decision_id` from `choose()`. The string "pipeline" is never in the pending dict → `UnknownDecision` error silently caught by `except: pass`.

- [ ] **Step 1: Add `bandit_decision_id` field to PipelineContext**

In pipeline.py, find the `PipelineContext` dataclass and add:
```python
bandit_decision_id: str | None = None
```

- [ ] **Step 2: Call bandit.choose() before topology execution (Stage 3)**

In `_stage_execute()`, before the TopologyRunner is created (~line 550), add:
```python
# Bandit: choose arm BEFORE execution to get decision_id
if self.bandit and hasattr(self.bandit, "choose"):
    try:
        decision = self.bandit.choose(0.1)  # 10% exploration
        ctx.bandit_decision_id = decision.decision_id
    except Exception:
        pass
```

- [ ] **Step 3: Fix Stage 5 bandit.record() to use real decision_id**

Replace line 677:
```python
# OLD: self.bandit.record("pipeline", quality, 0.0, ctx.latency_ms)
# NEW:
if ctx.bandit_decision_id:
    self.bandit.record(ctx.bandit_decision_id, quality, 0.0, ctx.latency_ms)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/ -k "pipeline" -v 2>&1 | tail -10`

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(critical): pipeline bandit uses choose()+record() with real decision_id

Pipeline Stage 5 called bandit.record('pipeline', ...) which always failed
with UnknownDecision because 'pipeline' was never a pending decision_id.
Now calls bandit.choose() before execution and passes the real decision_id."
```

---

### Task 2: C2 — Fix engine.record_outcome() bandit posteriors (CRITICAL)

**Files:**
- Modify: `sage-core/src/topology/engine.rs:155-170,518-520`

**Problem:** `engine.rs:519` calls `self.bandit.add_arm("observed", &template)` which only registers the arm. It never calls `record_outcome()` to update posteriors. The bandit is frozen.

- [ ] **Step 1: Store decision_id during generate()**

In `generate()` method, after the topology is selected (before returning), store the last decision_id:
```rust
// At the struct level, add:
last_decision_id: Option<String>,

// In generate(), after bandit.choose() or template selection:
self.last_decision_id = Some(ulid::Ulid::new().to_string());
// Also add the arm if not present:
self.bandit.add_arm(&source, &template);
// Store in pending:
if let Ok(decision) = self.bandit.choose(exploration_budget) {
    self.last_decision_id = Some(decision.decision_id.clone());
}
```

- [ ] **Step 2: In record_outcome(), call bandit.record_outcome() instead of add_arm()**

Replace line 519:
```rust
// OLD: self.bandit.add_arm("observed", &template);
// NEW:
if let Some(ref decision_id) = self.last_decision_id {
    if let Err(e) = self.bandit.record_outcome(decision_id, quality, cost, latency_ms) {
        debug!(error = %e, "bandit_record_failed_fallback_to_add_arm");
        self.bandit.add_arm("observed", &template);
    }
} else {
    self.bandit.add_arm("observed", &template);
}
```

- [ ] **Step 3: Run Rust tests**

Run: `cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib topology::engine 2>&1 | tail -5`

- [ ] **Step 4: Commit**

---

### Task 3: H1 — Runner uses get_predecessors() instead of all nodes (HIGH)

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py:69-82,84-90`

**Problem:** `_gather_completed_context()` iterates ALL completed nodes. Should use `graph.get_predecessors(node_idx)` to only pass predecessor context.

- [ ] **Step 1: Rewrite _gather_completed_context() to accept node_idx**

```python
def _gather_predecessor_context(self, node_idx: int) -> str:
    """Collect outputs from direct predecessors of node_idx only.

    Uses Rust TopologyGraph.get_predecessors() for correct DAG traversal.
    Falls back to all completed nodes if get_predecessors unavailable.
    """
    predecessor_indices = []
    try:
        predecessor_indices = self.graph.get_predecessors(node_idx)
    except (AttributeError, Exception):
        # Fallback: all completed nodes (old behavior)
        return self._gather_all_context()

    context_parts: list[str] = []
    for idx in predecessor_indices:
        output = self._node_outputs.get(idx)
        if output:
            node = self.graph.get_node(idx)
            role = getattr(node, "role", f"node-{idx}")
            context_parts.append(f"[{role}]: {output}")
    return "\n\n".join(context_parts)

def _gather_all_context(self) -> str:
    """Fallback: all completed nodes (legacy behavior)."""
    context_parts: list[str] = []
    for idx in sorted(self._node_outputs.keys()):
        output = self._node_outputs[idx]
        if output:
            node = self.graph.get_node(idx)
            role = getattr(node, "role", f"node-{idx}")
            context_parts.append(f"[{role}]: {output}")
    return "\n\n".join(context_parts)
```

- [ ] **Step 2: Update _execute_node() to pass node_idx**

Replace the call to `_gather_completed_context()` with `_gather_predecessor_context(node_idx)`.

- [ ] **Step 3: Remove obsolete comments**

Remove lines 10 and 73 that say "TopologyGraph does not expose get_edges() to Python" — this is false.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_topology_runner.py -v 2>&1 | tail -10`

- [ ] **Step 5: Commit**

---

### Task 4: H2 — upgrade_model resolves new_model_id (HIGH)

**Files:**
- Modify: `sage-python/src/sage/topology_controller.py:110-119`

**Problem:** Controller returns `upgrade_model` decision without setting `new_model_id`. The runner assumes it was set elsewhere.

- [ ] **Step 1: In evaluate_and_decide(), resolve fallback_tier to model_id**

After line 118, before returning the AdaptationDecision:
```python
# Resolve fallback_tier → actual model_id
new_model_id = None
if topology is not None and hasattr(topology, 'get_node'):
    try:
        node = topology.get_node(node_idx)
        fallback = getattr(node, 'fallback_tier', '')
        if fallback:
            # Use ModelAssigner to resolve tier → model_id
            try:
                from sage_core import ModelRegistry, CognitiveSystem
                from sage.llm.model_assigner import ModelAssigner
                tier_to_cs = {"reasoner": CognitiveSystem.S3, "fast": CognitiveSystem.S2, "budget": CognitiveSystem.S1}
                cs = tier_to_cs.get(fallback, CognitiveSystem.S2)
                registry = ModelRegistry.from_toml_file("config/cards.toml")
                candidates = registry.select_for_system(cs)
                if candidates:
                    new_model_id = candidates[0].id
            except (ImportError, Exception):
                new_model_id = fallback  # Use tier name as fallback
    except Exception:
        pass

return AdaptationDecision(
    action="upgrade_model",
    target_node=node_idx,
    reason=f"quality={quality:.2f} < {self.THETA_CRITICAL}",
    invariant_feedback=feedback,
    new_model_id=new_model_id,
)
```

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Commit**

---

### Task 5: H4 — Remove shadow router divergence (HIGH)

**Files:**
- Modify: `sage-python/src/sage/routing/shadow.py` (mark as deprecated)

- [ ] **Step 1: Add deprecation warning to shadow router**

At the top of shadow.py:
```python
import warnings
warnings.warn(
    "ShadowRouter is deprecated (49.6% divergence). Use Rust SystemRouter directly.",
    DeprecationWarning, stacklevel=2,
)
```

- [ ] **Step 2: Commit**

---

### Task 6: M1 — S-MMU garbage collection (MEDIUM)

**Files:**
- Modify: `sage-core/src/memory/smmu.rs`

- [ ] **Step 1: Add evict_oldest() method**

```rust
/// Evict the oldest N chunks from the graph (by ULID ordering).
/// Returns the number of chunks evicted.
pub fn evict_oldest(&mut self, count: usize) -> usize {
    let mut chunk_ids: Vec<String> = self.chunk_map.keys().cloned().collect();
    chunk_ids.sort(); // ULID sorts chronologically
    let to_evict = chunk_ids.into_iter().take(count).collect::<Vec<_>>();
    let mut evicted = 0;
    for id in to_evict {
        if let Some(node_idx) = self.chunk_map.remove(&id) {
            self.graph.remove_node(node_idx);
            evicted += 1;
        }
    }
    evicted
}

/// Current number of chunks in the graph.
pub fn chunk_count(&self) -> usize {
    self.chunk_map.len()
}
```

- [ ] **Step 2: Add PyO3 exports**

- [ ] **Step 3: Add Rust tests**

- [ ] **Step 4: Commit**

---

### Task 7: M2 — TrainingMemory query optimization (MEDIUM)

**Files:**
- Modify: `sage-python/src/sage/verl/training_memory.py:77-102`

- [ ] **Step 1: Add LIMIT and domain prefilter to query_similar()**

```python
def query_similar(self, query_embedding: np.ndarray, k: int = 3, domain: str = "") -> list[dict]:
    """Find top-k similar episodes. Uses domain prefilter + LIMIT for scale."""
    query = "SELECT * FROM episodes WHERE embedding IS NOT NULL"
    params = []
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    query += " ORDER BY created_at DESC LIMIT 500"  # Cap at 500 most recent
    rows = self._conn.execute(query, params).fetchall()
    # ... rest unchanged
```

- [ ] **Step 2: Commit**

---

### Task 8: M3 — prune_node implementation (MEDIUM)

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py:254,283`

- [ ] **Step 1: Implement prune via executor skip**

Replace "no special handling needed" with:
```python
elif decision.action == "prune_node":
    # Mark node as skipped in executor to prevent downstream scheduling
    try:
        self.executor.mark_skipped(decision.target_node)
    except (AttributeError, Exception):
        pass  # Executor may not support skip
    log.info("Node %d pruned by controller", decision.target_node)
```

- [ ] **Step 2: Commit**

---

### Task 9: L3 — Fix documentation "7-path" → "6-path" (LOW)

**Files:**
- Modify: `sage-core/src/topology/engine.rs:155`
- Modify: `CLAUDE.md` if it says "7-path"
- Modify: `AI-ARCHITECTURE.md` if it says "7-path"

- [ ] **Step 1: Fix engine.rs comment**

Line 155: "5-path strategy" → "6-path strategy" (S-MMU, archive, LLM synthesis, mutation, MCTS, template)

- [ ] **Step 2: Grep and fix all "7-path" references**

```bash
grep -rn "7.path\|7 paths\|7-path" . --include="*.md" --include="*.rs" --include="*.py" | grep -v ".git"
```

- [ ] **Step 3: Commit**

---

## Test Coverage

All fixes should be verified by:
```bash
# Rust
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib 2>&1 | tail -5

# Python (existing + new)
cd sage-python && python -m pytest tests/test_verl_v2.py tests/test_verl_reward.py tests/test_verl_micro_decisions.py tests/test_audit_fixes.py -v 2>&1 | tail -20
```
