# Audit V2 Fixes — Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 12 issues from 3 independent audits (Audit1/2/3), from CRITICAL reroute executor bug to LOW task truncation.

**Architecture:** Fixes grouped by file to minimize merge conflicts. P0 fixes first (blocking correctness), then P1 (training quality), then P2-P3 (production hardening). Each task is independently testable.

**Tech Stack:** Rust (sage-core, PyO3, maturin), Python (sage-python), pytest, cargo test

---

## File Map

| File | Fixes |
|------|-------|
| `sage-python/src/sage/pipeline.py` | T6 (reroute executor), B4 (bandit mutation), T2 (S-MMU), M2 (embeddings+cost) |
| `sage-python/src/sage/topology/runner.py` | T4 (upgrade applies model_id) |
| `sage-python/src/sage/verl/reward.py` | R4 (trivial topology penalty) |
| `sage-python/src/sage/verl/training_memory.py` | R6 (replay buffer) |
| `sage-python/src/sage/verl/topology_env.py` | R8 (real embeddings placeholder) |
| `sage-python/src/sage/topology/llm_caller.py` | R5 (task 500→2000) |
| `sage-core/src/memory/smmu.rs` | M1+M5 (utility eviction + auto-trigger) |
| `sage-core/src/topology/engine.rs` | T8 (enriched archive descriptor), M3 (quality eviction) |

---

### Task 1: P0 — T6 Reroute creates fresh executor (CRITICAL)

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:582-593`

- [ ] **Step 1: Fix the reroute block to create a fresh executor**

At line 588, `executor` refers to the stale executor built for the OLD topology. After reroute, ctx.topology is a new graph. Replace:

```python
# OLD (line 587-593):
runner2 = TopologyRunner(
    graph=ctx.topology, executor=executor,  # ← STALE executor
    ...
)

# NEW:
from sage_core import TopologyExecutor as _TE
executor_rerouted = _TE(ctx.topology)  # ← FRESH executor for new graph
runner2 = TopologyRunner(
    graph=ctx.topology, executor=executor_rerouted,
    llm_provider=self.llm_provider, llm_config=self.llm_config,
    provider_pool=self.provider_pool,
    controller=None,
)
```

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**: `fix(critical): reroute creates fresh TopologyExecutor for regenerated topology`

---

### Task 2: P0 — T4 _retry_with_upgrade applies new_model_id

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py:198-209`

- [ ] **Step 1: Replace the `pass` with actual model_id application**

```python
async def _retry_with_upgrade(self, node_idx: int, decision: Any, task: str) -> str:
    """Model upgrade: apply new_model_id to node, then re-execute."""
    if decision.new_model_id:
        try:
            self.graph.set_node_model_id(node_idx, decision.new_model_id)
            log.info("Node %d model upgraded to %s", node_idx, decision.new_model_id)
        except (AttributeError, Exception) as exc:
            log.warning("Could not set model_id on node %d: %s", node_idx, exc)
    return await self._execute_node(node_idx, task)
```

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**: `fix: _retry_with_upgrade applies new_model_id via graph.set_node_model_id()`

---

### Task 3: P0 — B4 Bandit decision_id updated after controller mutations

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:582-593`

- [ ] **Step 1: After reroute, get a new bandit decision_id**

In the reroute block (after creating the new topology), add:
```python
if result == "__REROUTE__" and self.engine:
    log.info("Topology reroute triggered — regenerating")
    ctx = self._stage_select_topology(ctx)
    ctx = self._stage_assign_models(ctx)
    # Get a fresh bandit decision for the new topology
    if self.bandit and hasattr(self.bandit, "choose"):
        try:
            new_decision = self.bandit.choose(0.1)
            ctx.bandit_decision_id = new_decision.decision_id
        except Exception:
            pass
    ...
```

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**: `fix: bandit decision_id refreshed after reroute mutation`

---

### Task 4: P1 — R4 Penalty for trivial topologies in reward

**Files:**
- Modify: `sage-python/src/sage/verl/reward.py:59-80`

- [ ] **Step 1: Add complexity penalty to _score_structure()**

After the existing scoring (line 78), add a penalty for trivially small topologies:

```python
def _score_structure(text: str) -> float:
    """Structural quality. Range: [0.0, 1.0]."""
    try:
        text = _strip_code_fence(text)
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "nodes" not in data:
            return 0.0
        nodes = data.get("nodes", [])
        if not isinstance(nodes, list):
            return 0.0
        score = 0.0
        if 1 <= len(nodes) <= 10:
            score += 0.3
        if data.get("edges"):
            score += 0.2
        if all(isinstance(n, dict) and "role" in n for n in nodes):
            score += 0.3
        if data.get("reasoning"):
            score += 0.2

        # Penalty for trivially small topologies (reward hacking mitigation)
        # A moderate/complex task with 1 node is suspicious
        difficulty = data.get("difficulty", "moderate")
        expected_min = {"simple": 1, "moderate": 2, "complex": 3}.get(str(difficulty).lower(), 2)
        if len(nodes) < expected_min:
            score *= 0.5  # halve the score for under-sized topologies

        return score
    except Exception:
        return 0.0
```

- [ ] **Step 2: Run existing reward tests to verify no regression**

Run: `python -m pytest tests/test_verl_reward.py -v`

- [ ] **Step 3: Commit**: `fix: penalty for trivially small topologies in reward (anti reward-hacking)`

---

### Task 5: P1 — R6 Replay buffer in TrainingMemory

**Files:**
- Modify: `sage-python/src/sage/verl/training_memory.py`

- [ ] **Step 1: Add replay_candidates column and sampling method**

Add to schema (in _init_schema):
```sql
ALTER TABLE episodes ADD COLUMN is_replay_candidate BOOLEAN DEFAULT 0;
```

Add method:
```python
def mark_replay_candidates(self, fraction: float = 0.1) -> int:
    """Mark top fraction of episodes as replay candidates (diversity-based).

    Selects episodes with highest reward variance across domains.
    """
    total = self.count()
    if total == 0:
        return 0
    limit = max(1, int(total * fraction))
    # Select diverse high-quality episodes
    self._conn.execute("""
        UPDATE episodes SET is_replay_candidate = 0
    """)
    self._conn.execute("""
        UPDATE episodes SET is_replay_candidate = 1
        WHERE id IN (
            SELECT id FROM episodes
            ORDER BY total_reward DESC, RANDOM()
            LIMIT ?
        )
    """, (limit,))
    self._conn.commit()
    return limit

def get_replay_batch(self, k: int = 50) -> list[dict]:
    """Get k replay candidates for mixing into training batches."""
    rows = self._conn.execute(
        "SELECT * FROM episodes WHERE is_replay_candidate = 1 ORDER BY RANDOM() LIMIT ?",
        (k,)
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d.pop("embedding", None)
        if d.get("per_node_results"):
            try:
                d["per_node_results"] = json.loads(d["per_node_results"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results
```

- [ ] **Step 2: Handle ALTER TABLE gracefully for existing DBs**

In _init_schema, wrap the ALTER in a try/except (column may already exist):
```python
try:
    self._conn.execute("ALTER TABLE episodes ADD COLUMN is_replay_candidate BOOLEAN DEFAULT 0")
    self._conn.commit()
except Exception:
    pass  # Column already exists
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**: `feat: replay buffer in TrainingMemory (anti-catastrophic forgetting)`

---

### Task 6: P1 — M2+R8 Real embeddings in pipeline and training env

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:700-708`
- Modify: `sage-python/src/sage/verl/topology_env.py:150-153,715-720`

- [ ] **Step 1: Pipeline — compute real embedding for engine.record_outcome()**

Replace `None` embedding (line 704) with actual computation:
```python
# Compute real task embedding for S-MMU retrieval
task_embedding = None
try:
    from sage.memory.embedder import get_embedder
    embedder = get_embedder()
    if embedder:
        task_embedding = embedder.embed(ctx.task[:500])
except Exception:
    pass  # Embedding unavailable, degrade gracefully

self.engine.record_outcome(
    topology_id,
    ctx.task[:200],
    keywords,
    task_embedding,  # real embedding instead of None
    quality,
    ctx.cost if hasattr(ctx, 'cost') else 0.0,  # real cost when available
    ctx.latency_ms,
)
```

- [ ] **Step 2: Training env — replace np.zeros(768) with lazy embedding**

In topology_env.py, create a helper that tries to compute real embeddings:
```python
def _get_embedding(self, text: str) -> np.ndarray:
    """Compute embedding via arctic-embed-m, fallback to zeros."""
    try:
        from sage.memory.embedder import get_embedder
        embedder = get_embedder()
        if embedder:
            emb = embedder.embed(text[:500])
            if emb is not None and len(emb) == 768:
                return np.array(emb, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(768, dtype=np.float32)
```

Replace the two `np.zeros(768)` calls at lines 152 and 718 with `self._get_embedding(prompt)` and `self._get_embedding(self._trace.prompt)`.

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**: `fix: real embeddings in pipeline record_outcome and training env`

---

### Task 7: P2 — T8 Enriched archive descriptor

**Files:**
- Modify: `sage-core/src/topology/engine.rs:316-354`

- [ ] **Step 1: Make BehaviorDescriptor use task features, not just tier**

In `try_archive_hit()`, incorporate task length and keyword count:
```rust
// OLD: fixed descriptor from tier only
// NEW: incorporate task-level signal
let task_len = task_description.len();
let (agent_count, max_depth) = match system {
    1 => (1u32, 1u32),
    2 => {
        // Vary based on task complexity signal
        if task_len > 500 { (4u32, 3u32) } else { (3u32, 2u32) }
    }
    3 => {
        if task_len > 1000 { (5u32, 4u32) } else { (4u32, 3u32) }
    }
    _ => (2u32, 2u32),
};
```

- [ ] **Step 2: Run Rust tests**
- [ ] **Step 3: Commit**: `fix: archive descriptor uses task length for richer behavior matching`

---

### Task 8: P2 — T2 Pipeline passes S-MMU to engine.generate()

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:230-235`

- [ ] **Step 1: Pass smmu_context from pipeline to engine**

Replace the `None` in engine.generate():
```python
# OLD: self.engine.generate(ctx.task, None, ctx.system, ctx.budget)
# NEW:
smmu = getattr(self, '_smmu', None)
self.engine.generate(ctx.task, smmu, ctx.system, ctx.budget)
```

The pipeline needs the S-MMU instance from boot.py. In the Pipeline.__init__(), add a parameter:
```python
def __init__(self, ..., smmu=None):
    ...
    self._smmu = smmu
```

And in boot.py, pass it when creating the pipeline.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**: `fix: pipeline passes S-MMU to engine.generate() for retrieval`

---

### Task 9: P2 — M1+M5 Utility-based eviction + auto-trigger in S-MMU

**Files:**
- Modify: `sage-core/src/memory/smmu.rs`

- [ ] **Step 1: Add access_count tracking to ChunkMetadata**

Add field: `pub access_count: u32` with default 0. Increment in retrieval methods.

- [ ] **Step 2: Replace FIFO eviction with utility scoring**

```rust
pub fn evict_by_utility(&mut self, count: usize) -> usize {
    // Utility = quality * recency_decay * access_count
    // Evict lowest utility chunks
    let mut scored: Vec<(String, f64)> = self.chunk_map.keys()
        .filter_map(|id| {
            let idx = *self.chunk_map.get(id)?;
            let meta = self.graph.node_weight(idx)?;
            let age_days = (ulid::Ulid::from_string(id).ok()?.timestamp_ms() as f64) / 86400000.0;
            let recency = 1.0 / (1.0 + age_days);
            let utility = recency * (meta.access_count as f64 + 1.0);
            Some((id.clone(), utility))
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let to_evict: Vec<String> = scored.into_iter().take(count).map(|(id, _)| id).collect();
    // ... eviction logic same as evict_oldest
}
```

- [ ] **Step 3: Add auto-trigger in register_chunk()**

```rust
const CAPACITY_THRESHOLD: usize = 10_000;

pub fn register_chunk(...) -> String {
    // Auto-GC if approaching capacity
    if self.chunk_map.len() >= CAPACITY_THRESHOLD {
        let evict_count = CAPACITY_THRESHOLD / 10; // evict 10%
        self.evict_by_utility(evict_count);
    }
    // ... rest of register_chunk unchanged
}
```

- [ ] **Step 4: Add Rust tests**
- [ ] **Step 5: Rebuild**: `maturin develop --features smt,onnx,cognitive,tool-executor --release`
- [ ] **Step 6: Commit**: `feat: S-MMU utility-based eviction + auto-trigger at 10K chunks`

---

### Task 10: P2 — M6 Evolution statistical validation

**Files:**
- Create: `sage-python/src/sage/evolution/evaluator.py`

- [ ] **Step 1: Create evaluator with Wilcoxon signed-rank test**

```python
"""Statistical validation for EvolutionEngine.

Implements Wilcoxon signed-rank test (N>=10 runs) to prove evolution
improves topology quality. Blocks promotion to production if p > 0.05.
"""
from scipy import stats

def validate_evolution(baseline_scores: list[float], evolved_scores: list[float]) -> dict:
    """Compare baseline vs evolved topology scores.

    Returns dict with: p_value, effect_size (Cohen's d), significant (bool),
    mean_improvement, n_runs.
    """
    assert len(baseline_scores) == len(evolved_scores), "Paired samples required"
    n = len(baseline_scores)
    if n < 10:
        return {"error": f"Need N>=10 runs, got {n}", "significant": False}

    stat, p_value = stats.wilcoxon(baseline_scores, evolved_scores, alternative="greater")

    # Cohen's d
    import numpy as np
    diff = np.array(evolved_scores) - np.array(baseline_scores)
    d = diff.mean() / (diff.std() + 1e-8)

    return {
        "p_value": float(p_value),
        "effect_size": float(d),
        "significant": p_value < 0.05,
        "mean_improvement": float(diff.mean()),
        "n_runs": n,
        "gate_passed": p_value < 0.05 and d > 0.2,
    }
```

- [ ] **Step 2: Commit**: `feat: evolution statistical validation (Wilcoxon signed-rank, Cohen's d)`

---

### Task 11: P3 — M3 Cache eviction by quality

**Files:**
- Modify: `sage-core/src/topology/engine.rs:128-141`

- [ ] **Step 1: Evict lowest-quality topologies instead of arbitrary keys**

```rust
if self.topology_cache.len() >= 500 {
    // Evict lowest-quality entries (not arbitrary)
    let mut scored: Vec<(String, f32)> = self.topology_cache.iter()
        .map(|(id, graph)| {
            // Use node_count as proxy for complexity/quality
            (id.clone(), graph.node_count() as f32)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    // Remove bottom half (lowest complexity)
    for (key, _) in scored.into_iter().take(250) {
        self.topology_cache.remove(&key);
    }
}
```

- [ ] **Step 2: Run Rust tests**
- [ ] **Step 3: Commit**: `fix: topology cache evicts lowest-quality entries, not arbitrary`

---

### Task 12: P3 — R5 Task truncation 500→2000

**Files:**
- Modify: `sage-python/src/sage/topology/llm_caller.py:352`

- [ ] **Step 1: Increase task slice**

Replace `task[:500]` with `task[:2000]`.

- [ ] **Step 2: Commit**: `fix: task context increased from 500 to 2000 chars for topology policy`

---

## Dependency Graph

```
Task 1 (T6 reroute) ─────────────────────────── P0
Task 2 (T4 upgrade applies) ─────────────────── P0
Task 3 (B4 bandit mutation) ──── depends on T1 ─ P0
Task 4 (R4 trivial penalty) ─────────────────── P1
Task 5 (R6 replay buffer) ──────────────────── P1
Task 6 (M2+R8 embeddings) ──────────────────── P1
Task 7 (T8 archive descriptor) ─────────────── P2
Task 8 (T2 S-MMU to pipeline) ──────────────── P2
Task 9 (M1+M5 utility eviction) ────────────── P2
Task 10 (M6 evolution validation) ──────────── P2
Task 11 (M3 cache quality eviction) ────────── P3
Task 12 (R5 task 500→2000) ─────────────────── P3
```

**Parallelizable:** Tasks 1+2+4+5+6+12 are independent. Task 3 depends on Task 1. Tasks 7+9+11 are Rust (build together).
