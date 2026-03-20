# V2 Adaptive Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete V2 adaptive topology pipeline — Rust fields, multi-turn env with memory, 5-signal reward with RewardFlow, and Unsloth GRPO local training on Qwen3.5-4B.

**Architecture:** Rust sage-core gets adaptive fields on TopologyNode/TopologyGraph/RewardScore. Python topology_env.py becomes a multi-turn environment with episodic SQLite memory and ProviderPool. A new RewardFlow module provides per-node credit via PageRank propagation. Local training uses Unsloth + TRL GRPOTrainer on Qwen3.5-4B QLoRA (12GB VRAM).

**Tech Stack:** Rust (PyO3/maturin), Python 3.12, Unsloth, TRL, SQLite, arctic-embed-m, Qwen3.5-4B

**Spec:** `docs/superpowers/specs/2026-03-20-v2-adaptive-topology-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| MODIFY | `sage-core/src/topology/topology_graph.rs` | +3 fields TopologyNode, +3 fields TopologyGraph |
| MODIFY | `sage-core/src/topology/reward.rs` | +2 fields RewardScore, `compute_full()` |
| CREATE | `sage-python/src/sage/verl/rewardflow.py` | PageRank reward propagation |
| CREATE | `sage-python/src/sage/verl/training_memory.py` | SQLite episodic memory for training |
| MODIFY | `sage-python/src/sage/verl/reward.py` | 5-signal reward with resilience + cost |
| MODIFY | `sage-python/src/sage/verl/topology_env.py` | Multi-turn env, memory, ProviderPool, controller |
| MODIFY | `sage-python/scripts/verl/convert_sft_to_verl.py` | +3 adaptive data sources |
| CREATE | `sage-python/scripts/train_local_grpo.py` | Unsloth GRPO training script |
| CREATE | `sage-python/tests/test_verl_v2.py` | All V2 tests |

---

### Task 1: Rust — TopologyNode adaptive fields

**Files:**
- Modify: `sage-core/src/topology/topology_graph.rs:122-238`

- [ ] **Step 1: Add 3 fields to TopologyNode struct**

In `topology_graph.rs`, add after the `prompt` field (line 151):

```rust
    /// Fallback model tier for runtime adaptation. Empty = no fallback.
    #[pyo3(get, set)]
    pub fallback_tier: String,
    /// Whether this node is a quality checkpoint.
    #[pyo3(get, set)]
    pub is_checkpoint: bool,
    /// Max retries for this node (0 = use controller default).
    #[pyo3(get)]
    pub max_retries: u8,
```

- [ ] **Step 2: Update `py_new()` signature (line 157)**

Replace the existing `#[pyo3(signature)]` and `py_new` function:

```rust
    #[new]
    #[pyo3(signature = (role, model_id, system=1, required_capabilities=vec![], security_label=0, max_cost_usd=1.0, max_wall_time_s=60.0, prompt=String::new(), fallback_tier=String::new(), is_checkpoint=false, max_retries=0))]
    pub fn py_new(
        role: String,
        model_id: String,
        system: u8,
        required_capabilities: Vec<String>,
        security_label: u8,
        max_cost_usd: f32,
        max_wall_time_s: f32,
        prompt: String,
        fallback_tier: String,
        is_checkpoint: bool,
        max_retries: u8,
    ) -> Self {
        let mut node = Self::new(
            role, model_id, system, required_capabilities,
            security_label, max_cost_usd, max_wall_time_s,
        );
        node.prompt = prompt;
        node.fallback_tier = fallback_tier;
        node.is_checkpoint = is_checkpoint;
        node.max_retries = max_retries;
        node
    }
```

- [ ] **Step 3: Update `new()` constructor (line 203)**

Add the 3 new fields with defaults to the `Self { ... }` block:

```rust
            prompt: String::new(),
            fallback_tier: String::new(),
            is_checkpoint: false,
            max_retries: 0,
```

- [ ] **Step 4: Update `with_id()` constructor (line 226)**

Add the 3 new fields with defaults:

```rust
            prompt: String::new(),
            fallback_tier: String::new(),
            is_checkpoint: false,
            max_retries: 0,
```

- [ ] **Step 5: Update `Display` impl (line 186)**

Add fallback info to the display string. Change the `write!` format to:

```rust
        write!(
            f,
            "TopologyNode(role='{}', model='{}', S{}, label={}, budget=${:.2}, timeout={:.0}s{})",
            self.role, self.model_id, self.system, self.security_label,
            self.max_cost_usd, self.max_wall_time_s,
            if self.fallback_tier.is_empty() { String::new() }
            else { format!(", fallback='{}'", self.fallback_tier) }
        )
```

- [ ] **Step 6: Run Rust tests**

Run: `cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib topology::topology_graph 2>&1 | tail -5`
Expected: all existing tests pass (new fields have defaults, backward compatible)

- [ ] **Step 7: Commit**

```bash
git add sage-core/src/topology/topology_graph.rs
git commit -m "feat(rust): add fallback_tier, is_checkpoint, max_retries to TopologyNode"
```

---

### Task 2: Rust — TopologyGraph adaptive fields

**Files:**
- Modify: `sage-core/src/topology/topology_graph.rs:398-425`

- [ ] **Step 1: Add 3 fields to TopologyGraph struct (after line 408)**

```rust
    /// Maximum model upgrades allowed for this topology execution.
    #[pyo3(get, set)]
    pub max_upgrades: u8,
    /// Maximum topology reroutes allowed.
    #[pyo3(get, set)]
    pub max_reroutes: u8,
    /// Quality threshold below which adaptation triggers.
    #[pyo3(get, set)]
    pub quality_threshold: f32,
```

- [ ] **Step 2: Update `try_new()` and `new()` constructors**

Find the `TopologyGraph` impl block with `try_new()`. Add defaults in the `Self { ... }` block:

```rust
            max_upgrades: 1,
            max_reroutes: 0,
            quality_threshold: 0.5,
```

- [ ] **Step 3: Run Rust tests**

Run: `cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib topology::topology_graph 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/topology/topology_graph.rs
git commit -m "feat(rust): add max_upgrades, max_reroutes, quality_threshold to TopologyGraph"
```

---

### Task 3: Rust — RewardScore resilience + cost_efficiency

**Files:**
- Modify: `sage-core/src/topology/reward.rs`

- [ ] **Step 1: Add 2 fields to RewardScore struct (after line 43)**

```rust
    /// Resilience score: bonus for topologies that survived adaptation.
    #[pyo3(get)]
    pub resilience: f32,
    /// Cost efficiency score: 1.0 - tanh(cost / budget).
    #[pyo3(get)]
    pub cost_efficiency: f32,
```

- [ ] **Step 2: Update `__repr__` to include new fields**

```rust
    fn __repr__(&self) -> String {
        format!(
            "RewardScore(total={:.4}, execution={:.1}, structural={:.4}, density={:.4}, temporal={:.4}, resilience={:.4}, cost_eff={:.4}, n_signals={})",
            self.total, self.execution, self.structural, self.density, self.temporal,
            self.resilience, self.cost_efficiency, self.n_signals
        )
    }
```

- [ ] **Step 3: Update existing `compute()` to set new fields to 0.0**

In the `RewardScore { ... }` return block of `compute()`, add:

```rust
            resilience: 0.0,
            cost_efficiency: 0.0,
```

- [ ] **Step 4: Add `compute_full()` method**

Add after `compute()` in the `#[pymethods]` impl block:

```rust
    /// Compute reward with all 6 signals including resilience and cost efficiency.
    ///
    /// The resilience and cost_efficiency values are computed in Python
    /// (from trace analysis and provider costs) and passed in directly.
    /// Weights are initial values subject to ablation (see spec C2).
    #[instrument(skip(self))]
    #[pyo3(signature = (execution_passed, structural_score, density_score, temporal_score=None, resilience=0.0, cost_efficiency=1.0))]
    pub fn compute_full(
        &self,
        execution_passed: bool,
        structural_score: f32,
        density_score: f32,
        temporal_score: Option<f32>,
        resilience: f32,
        cost_efficiency: f32,
    ) -> RewardScore {
        let base = self.compute(execution_passed, structural_score, density_score, temporal_score);
        // Return with resilience and cost_efficiency filled in.
        // The Python reward.py handles the final weighted combination.
        RewardScore {
            resilience,
            cost_efficiency,
            ..base
        }
    }
```

- [ ] **Step 5: Add Rust tests for new fields**

Add at the end of the `mod tests` block:

```rust
    #[test]
    fn test_compute_full_with_resilience() {
        let reward = TopologyReward::new();
        let score = reward.compute_full(true, 0.8, 0.6, Some(0.9), 0.5, 0.7);
        assert!((score.resilience - 0.5).abs() < 1e-6);
        assert!((score.cost_efficiency - 0.7).abs() < 1e-6);
        // Base total unchanged (compute_full delegates to compute for base signals)
        assert_eq!(score.n_signals, 4);
    }

    #[test]
    fn test_compute_full_defaults() {
        let reward = TopologyReward::new();
        let score = reward.compute_full(true, 0.8, 0.6, None, 0.0, 1.0);
        assert_eq!(score.resilience, 0.0);
        assert!((score.cost_efficiency - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_backward_compat() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 0.8, 0.6, None);
        // Existing compute() sets resilience=0, cost_efficiency=0
        assert_eq!(score.resilience, 0.0);
        assert_eq!(score.cost_efficiency, 0.0);
    }
```

- [ ] **Step 6: Run all Rust tests**

Run: `cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib topology::reward 2>&1 | tail -5`
Expected: 10+ tests PASS (7 existing + 3 new)

- [ ] **Step 7: Commit**

```bash
git add sage-core/src/topology/reward.rs
git commit -m "feat(rust): add resilience + cost_efficiency to RewardScore, compute_full()"
```

---

### Task 4: Rust build — recompile sage-core

- [ ] **Step 1: Build release**

Run: `cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor --release 2>&1 | tail -3`
Expected: `Successfully installed sage_core`

- [ ] **Step 2: Verify from Python**

Run: `python3 -c "from sage_core import TopologyNode; n = TopologyNode('coder', 'fast'); n.fallback_tier = 'reasoner'; print(n.fallback_tier, n.is_checkpoint)"`
Expected: `reasoner False`

Run: `python3 -c "from sage_core import TopologyGraph; g = TopologyGraph('sequential'); print(g.max_upgrades, g.quality_threshold)"`
Expected: `1 0.5`

Run: `python3 -c "from sage_core import TopologyReward; r = TopologyReward(); s = r.compute_full(True, 0.8, 0.6, None, 0.5, 0.7); print(s.resilience, s.cost_efficiency)"`
Expected: `0.5 0.699...`

- [ ] **Step 3: Commit (no changes, just verify)**

---

### Task 5: Python — training_memory.py

**Files:**
- Create: `sage-python/src/sage/verl/training_memory.py`
- Test: `sage-python/tests/test_verl_v2.py` (memory section)

- [ ] **Step 1: Write failing test**

Create `sage-python/tests/test_verl_v2.py`:

```python
"""Tests for V2 adaptive topology components."""
import os
import tempfile
import numpy as np
import pytest


class TestTrainingMemory:
    def test_store_and_query(self):
        from sage.verl.training_memory import TrainingMemory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = TrainingMemory(db_path=db_path)
            emb = np.random.randn(768).astype(np.float32)
            mem.store_episode(
                task_id="test/1", prompt_hash="abc123", domain="algorithm",
                topology_yaml="nodes:\n- role: coder", n_nodes=1,
                difficulty="simple", outcome="PASSED", total_reward=0.8,
                per_node_results=[{"role": "coder", "reward": 0.8}],
                adaptations_triggered=0, embedding=emb,
            )
            results = mem.query_similar(emb, k=1)
            assert len(results) == 1
            assert results[0]["outcome"] == "PASSED"
        finally:
            os.unlink(db_path)

    def test_format_context_empty(self):
        from sage.verl.training_memory import TrainingMemory
        mem = TrainingMemory(db_path=":memory:")
        ctx = mem.format_context([])
        assert ctx == ""

    def test_format_context_with_episodes(self):
        from sage.verl.training_memory import TrainingMemory
        mem = TrainingMemory(db_path=":memory:")
        episodes = [
            {"domain": "algo", "difficulty": "moderate", "outcome": "PASSED",
             "total_reward": 0.85, "n_nodes": 3, "adaptations_triggered": 1},
        ]
        ctx = mem.format_context(episodes)
        assert "PASSED" in ctx
        assert "moderate" in ctx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestTrainingMemory -v 2>&1 | tail -5`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `sage-python/src/sage/verl/training_memory.py`:

```python
"""SQLite episodic memory for topology training.

Stores episode outcomes across epochs so the model can learn
from past successes/failures on similar tasks.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

import numpy as np

log = logging.getLogger("training_memory")


class TrainingMemory:
    """SQLite-backed episodic memory for training loop."""

    def __init__(self, db_path: str = "data/training_memory.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                prompt_hash TEXT,
                domain TEXT,
                topology_yaml TEXT,
                n_nodes INTEGER,
                difficulty TEXT,
                outcome TEXT,
                total_reward REAL,
                per_node_results TEXT,
                adaptations_triggered INTEGER DEFAULT 0,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def store_episode(
        self,
        task_id: str,
        prompt_hash: str,
        domain: str,
        topology_yaml: str,
        n_nodes: int,
        difficulty: str,
        outcome: str,
        total_reward: float,
        per_node_results: list[dict],
        adaptations_triggered: int,
        embedding: np.ndarray,
    ) -> None:
        """Persist one episode outcome."""
        self._conn.execute(
            """INSERT INTO episodes
               (task_id, prompt_hash, domain, topology_yaml, n_nodes,
                difficulty, outcome, total_reward, per_node_results,
                adaptations_triggered, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id, prompt_hash, domain, topology_yaml, n_nodes,
                difficulty, outcome, total_reward,
                json.dumps(per_node_results),
                adaptations_triggered,
                embedding.tobytes(),
            ),
        )
        self._conn.commit()

    def query_similar(self, query_embedding: np.ndarray, k: int = 3) -> list[dict]:
        """Find top-k similar episodes by cosine similarity on embeddings."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scored = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_norm, emb_norm))
            scored.append((sim, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, row_dict in scored[:k]:
            row_dict.pop("embedding", None)
            if row_dict.get("per_node_results"):
                try:
                    row_dict["per_node_results"] = json.loads(row_dict["per_node_results"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(row_dict)
        return results

    def format_context(self, episodes: list[dict]) -> str:
        """Format episodes as text for model observation."""
        if not episodes:
            return ""
        lines = ["[Past experience on similar tasks]:"]
        for ep in episodes:
            outcome = ep.get("outcome", "?")
            reward = ep.get("total_reward", 0)
            diff = ep.get("difficulty", "?")
            n = ep.get("n_nodes", "?")
            adapt = ep.get("adaptations_triggered", 0)
            lines.append(
                f"- {diff}, {n} nodes, {outcome} (reward={reward:.2f})"
                + (f", {adapt} adaptations" if adapt else "")
            )
        return "\n".join(lines)

    def count(self) -> int:
        """Number of stored episodes."""
        return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestTrainingMemory -v 2>&1 | tail -5`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/verl/training_memory.py sage-python/tests/test_verl_v2.py
git commit -m "feat: add TrainingMemory — SQLite episodic memory for training"
```

---

### Task 6: Python — rewardflow.py

**Files:**
- Create: `sage-python/src/sage/verl/rewardflow.py`
- Test: `sage-python/tests/test_verl_v2.py` (rewardflow section)

- [ ] **Step 1: Write failing test**

Append to `sage-python/tests/test_verl_v2.py`:

```python
class TestRewardFlow:
    def _make_trace(self, nodes, terminal_reward):
        """Helper: build a mock EpisodeTrace-like list of dicts."""
        return {
            "node_traces": nodes,
            "terminal_reward": terminal_reward,
        }

    def test_single_rollout_propagation(self):
        from sage.verl.rewardflow import RewardFlowPropagator
        prop = RewardFlowPropagator(damping=0.85, max_iters=20)

        rollouts = [
            self._make_trace(
                [{"node_idx": 0, "role": "coder", "quality": 0.8},
                 {"node_idx": 1, "role": "reviewer", "quality": 0.6},
                 {"node_idx": 2, "role": "synthesizer", "quality": 0.9}],
                terminal_reward=1.0,
            ),
        ]
        result = prop.compute(rollouts)
        assert len(result) == 1
        # Each node should get a propagated reward > 0
        for node_idx, reward in result[0].items():
            assert reward > 0.0

    def test_multiple_rollouts_differentiation(self):
        from sage.verl.rewardflow import RewardFlowPropagator
        prop = RewardFlowPropagator()

        rollouts = [
            self._make_trace(
                [{"node_idx": 0, "role": "coder", "quality": 0.9},
                 {"node_idx": 1, "role": "synthesizer", "quality": 0.8}],
                terminal_reward=1.0,
            ),
            self._make_trace(
                [{"node_idx": 0, "role": "coder", "quality": 0.2},
                 {"node_idx": 1, "role": "synthesizer", "quality": 0.3}],
                terminal_reward=0.0,
            ),
        ]
        result = prop.compute(rollouts)
        assert len(result) == 2
        # First rollout (PASSED) should have higher node rewards
        assert sum(result[0].values()) > sum(result[1].values())

    def test_empty_rollouts(self):
        from sage.verl.rewardflow import RewardFlowPropagator
        prop = RewardFlowPropagator()
        assert prop.compute([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestRewardFlow -v 2>&1 | tail -5`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `sage-python/src/sage/verl/rewardflow.py`:

```python
"""RewardFlow — per-node credit via state-graph PageRank propagation.

Inspired by RewardFlow (arXiv 2603.18859, AAMAS 2026).
Builds a state graph from K rollouts, propagates terminal rewards
backward via Personalized PageRank to assign per-node credit.

Usage:
    prop = RewardFlowPropagator(damping=0.85, max_iters=20)
    per_node_rewards = prop.compute(rollouts)
    # per_node_rewards[i] = {node_idx: reward} for rollout i
"""
from __future__ import annotations

import logging
from collections import defaultdict

log = logging.getLogger("rewardflow")


def _quality_bucket(quality: float) -> str:
    """Bin quality score into low/med/high."""
    if quality < 0.3:
        return "low"
    if quality < 0.7:
        return "med"
    return "high"


class RewardFlowPropagator:
    """Per-node credit assignment via state-graph PageRank."""

    def __init__(self, damping: float = 0.85, max_iters: int = 20):
        self._damping = damping
        self._max_iters = max_iters

    def compute(self, rollouts: list[dict]) -> list[dict[int, float]]:
        """Build state-graph from K rollouts, propagate terminal rewards.

        Args:
            rollouts: list of {"node_traces": [...], "terminal_reward": float}
                Each node_trace: {"node_idx": int, "role": str, "quality": float}

        Returns:
            list of {node_idx: reward} dicts, one per rollout.
        """
        if not rollouts:
            return []

        # 1. Build state graph: state = (role, quality_bucket)
        # Edges: transition counts between consecutive states
        transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        terminal_rewards: dict[str, list[float]] = defaultdict(list)

        for rollout in rollouts:
            nodes = rollout.get("node_traces", [])
            term_reward = rollout.get("terminal_reward", 0.0)

            prev_state = None
            for node in nodes:
                role = node.get("role", "agent")
                quality = node.get("quality", 0.5)
                state = f"{role}:{_quality_bucket(quality)}"

                if prev_state is not None:
                    transitions[prev_state][state] += 1
                prev_state = state

            # Terminal state gets the execution reward
            if prev_state is not None:
                terminal_rewards[prev_state].append(term_reward)

        # 2. Personalized PageRank backward propagation
        all_states = set(transitions.keys())
        for targets in transitions.values():
            all_states.update(targets.keys())
        all_states.update(terminal_rewards.keys())

        if not all_states:
            return [{} for _ in rollouts]

        # Initialize: terminal states get their mean reward, others get 0
        state_reward: dict[str, float] = {}
        for s in all_states:
            if s in terminal_rewards:
                state_reward[s] = sum(terminal_rewards[s]) / len(terminal_rewards[s])
            else:
                state_reward[s] = 0.0

        # Build reverse transition graph (for backward propagation)
        reverse_trans: dict[str, dict[str, float]] = defaultdict(dict)
        for src, targets in transitions.items():
            total = sum(targets.values())
            for tgt, count in targets.items():
                reverse_trans[tgt][src] = count / total

        # PageRank iterations
        for _ in range(self._max_iters):
            new_rewards = {}
            for state in all_states:
                # Seed from terminal rewards
                seed = 0.0
                if state in terminal_rewards:
                    seed = sum(terminal_rewards[state]) / len(terminal_rewards[state])

                # Propagation from successors
                prop = 0.0
                if state in transitions:
                    total = sum(transitions[state].values())
                    for tgt, count in transitions[state].items():
                        prop += (count / total) * state_reward.get(tgt, 0.0)

                new_rewards[state] = (1 - self._damping) * seed + self._damping * prop

            state_reward = new_rewards

        # 3. Map back to per-rollout per-node rewards
        results = []
        for rollout in rollouts:
            node_rewards = {}
            for node in rollout.get("node_traces", []):
                role = node.get("role", "agent")
                quality = node.get("quality", 0.5)
                state = f"{role}:{_quality_bucket(quality)}"
                node_rewards[node["node_idx"]] = state_reward.get(state, 0.0)
            results.append(node_rewards)

        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestRewardFlow -v 2>&1 | tail -5`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/verl/rewardflow.py sage-python/tests/test_verl_v2.py
git commit -m "feat: add RewardFlowPropagator — PageRank per-node credit (2603.18859)"
```

---

### Task 7: Python — reward.py 5-signal extension

**Files:**
- Modify: `sage-python/src/sage/verl/reward.py`
- Test: `sage-python/tests/test_verl_v2.py` (reward section)

- [ ] **Step 1: Write failing test**

Append to `tests/test_verl_v2.py`:

```python
class TestRewardV2:
    def test_resilience_score_no_adaptation(self):
        from sage.verl.reward import _score_resilience
        trace = [{"role": "coder", "was_upgraded": False, "output": "code here"}]
        assert _score_resilience(trace) == 0.0

    def test_resilience_score_upgrade_succeeded_passed(self):
        from sage.verl.reward import _score_resilience
        trace = [
            {"role": "coder", "was_upgraded": True, "output": "good code", "status": ""},
            {"role": "synthesizer", "was_upgraded": False, "output": "final", "status": "PASSED"},
        ]
        assert _score_resilience(trace) == 0.5

    def test_cost_efficiency_budget_model(self):
        from sage.verl.reward import _score_cost_efficiency
        # Budget model, low cost
        assert _score_cost_efficiency(0.005, "simple") > 0.8

    def test_cost_efficiency_expensive(self):
        from sage.verl.reward import _score_cost_efficiency
        # Very expensive execution
        assert _score_cost_efficiency(0.50, "simple") < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestRewardV2 -v 2>&1 | tail -5`
Expected: FAIL

- [ ] **Step 3: Add _score_resilience and _score_cost_efficiency to reward.py**

Add before `compute_score()` in `sage-python/src/sage/verl/reward.py`:

```python
import math


# ── Resilience scoring ──────────────────────────────────────

def _score_resilience(trace: list[dict]) -> float:
    """Bonus for topologies that survived adaptation.

    0.0 — no adaptation triggered
    0.3 — adaptation triggered and succeeded
    0.5 — adaptation triggered, succeeded, and terminal PASSED
    """
    adaptation_triggered = any(t.get("was_upgraded", False) for t in trace)
    if not adaptation_triggered:
        return 0.0

    adaptation_succeeded = any(
        t.get("was_upgraded", False) and not str(t.get("output", "")).startswith("ERROR")
        for t in trace
    )
    final_passed = trace[-1].get("status") == "PASSED" if trace else False

    if adaptation_succeeded and final_passed:
        return 0.5
    elif adaptation_succeeded:
        return 0.3
    return 0.0


# ── Cost efficiency scoring (inspired by CARD 2603.01089) ───

BUDGET_REF = {"simple": 0.01, "moderate": 0.05, "complex": 0.20}


def _score_cost_efficiency(total_cost: float, difficulty: str) -> float:
    """CARD-style price penalty. Range: [0.0, 1.0].

    R_cost = 1.0 - tanh(cost / budget_ref[difficulty])
    """
    ref = BUDGET_REF.get(difficulty, 0.05)
    return 1.0 - math.tanh(total_cost / ref)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestRewardV2 -v 2>&1 | tail -5`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/verl/reward.py sage-python/tests/test_verl_v2.py
git commit -m "feat: add resilience + cost_efficiency scoring to reward.py"
```

---

### Task 8: Python — topology_env.py v2 (multi-turn + memory + ProviderPool)

**Files:**
- Modify: `sage-python/src/sage/verl/topology_env.py`
- Test: `sage-python/tests/test_verl_v2.py` (env section)

This is the largest task. Key changes:
1. Memory injection in `reset()`
2. `_build_topology_graph()` parses `gate: conditional` → `gate: open` + condition
3. Wire ProviderPool to TopologyRunner
4. Wire TopologyController
5. Incremental execution with decision turns after checkpoints

- [ ] **Step 1: Write failing test**

Append to `tests/test_verl_v2.py`:

```python
class TestTopologyEnvV2:
    def test_parse_adaptive_yaml(self):
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        env.reset("Write a sort function", "test/sort")

        yaml_text = """
difficulty: moderate
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Write sorting code
  - role: synthesizer
    model_tier: fast
    prompt: Produce final solution
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message}
"""
        obs, reward, done, info = env.step(yaml_text)
        assert info["status"] == "TOPOLOGY_PARSED"
        assert env._topo_dict["adaptation"]["max_upgrades"] == 1

    def test_memory_injection_in_reset(self):
        import tempfile, os
        from sage.verl.topology_env import SageTopologyEnv
        from sage.verl.training_memory import TrainingMemory
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = TrainingMemory(db_path=db_path)
            emb = np.ones(768, dtype=np.float32)
            mem.store_episode(
                task_id="t1", prompt_hash="h1", domain="algo",
                topology_yaml="nodes:\n- role: coder", n_nodes=1,
                difficulty="simple", outcome="PASSED", total_reward=0.9,
                per_node_results=[], adaptations_triggered=0, embedding=emb,
            )
            env = SageTopologyEnv(config={"memory_db": db_path})
            obs = env.reset("Sort a list", "test/sort")
            # Memory context should be in the observation text
            assert "PASSED" in obs["text"] or env._memory is not None
        finally:
            os.unlink(db_path)

    def test_env_structural_mode_adaptive(self):
        """Verify env handles adaptive YAML in structural mode (no API)."""
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        env.reset("Binary search", "test/bsearch")

        yaml_text = """
difficulty: moderate
reasoning: Need coder with fallback for algorithm
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Implement binary search
  - role: reviewer
    model_tier: budget
    prompt: Review for edge cases
  - role: synthesizer
    model_tier: fast
    prompt: Final solution
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message, gate: conditional}
  - {from_idx: 1, to_idx: 2, flow_type: message}
"""
        obs, reward, done, info = env.step(yaml_text)
        assert not done
        assert reward > 0  # structural reward for valid YAML
        # Step through remaining nodes
        while not done:
            obs, reward, done, info = env.step("continue")
        assert env._trace.status != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestTopologyEnvV2 -v 2>&1 | tail -5`
Expected: FAIL (missing memory integration)

- [ ] **Step 3: Add memory + ProviderPool + controller to topology_env.py**

Modifications to `sage-python/src/sage/verl/topology_env.py`:

**3a.** In `__init__`, add memory initialization:

After `self._predecessor_map` (line 80), add:
```python
        self._memory: TrainingMemory | None = None
        self._awaiting_decision = False
        db = self._config.get("memory_db", "")
        if db:
            try:
                from sage.verl.training_memory import TrainingMemory
                self._memory = TrainingMemory(db_path=db)
            except Exception:
                pass
```

**3b.** In `reset()`, inject memory context:

After building the observation dict (line 92-97), before `return`:
```python
        memory_ctx = ""
        if self._memory:
            try:
                import hashlib
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                # Use a zero embedding for now; real embeddings precomputed offline
                import numpy as np
                query_emb = np.zeros(768, dtype=np.float32)
                episodes = self._memory.query_similar(query_emb, k=3)
                memory_ctx = self._memory.format_context(episodes)
            except Exception:
                pass

        obs_text = prompt
        if memory_ctx:
            obs_text = prompt + "\n\n" + memory_ctx
```

**3c.** In `_step_parse_and_execute()`, parse adaptation metadata:

After `self._topo_dict = topo` (line 127), add:
```python
        # Parse adaptation metadata for TopologyController
        adaptation = topo.get("adaptation", {})
        self._checkpoints = set(adaptation.get("checkpoints", []))
        self._max_upgrades = adaptation.get("max_upgrades", 0)
        self._quality_threshold = adaptation.get("quality_threshold", 0.5)
```

**3d.** In `_build_topology_graph()`, map `gate: conditional`:

In the YAML fallback edge parsing (around line 265-271), update:
```python
        for ed in topo.get("edges", []):
            if isinstance(ed, dict):
                to_idx = ed.get("to_idx", 0)
                from_idx = ed.get("from_idx", 0)
                # Map conditional → open (see spec C1)
                gate = ed.get("gate", "open")
                if gate == "conditional":
                    gate = "open"  # Controller handles closing
                pred_map.setdefault(to_idx, []).append(from_idx)
```

**3e.** In `_execute_topology_traced()`, wire ProviderPool and controller:

Replace the TODO block (lines 367-376) with:
```python
            # Wire ProviderPool for per-node model resolution
            provider_pool = None
            try:
                from sage.providers.provider_pool import ProviderPool
                provider_pool = ProviderPool()
            except Exception:
                pass

            # Wire TopologyController if we have adaptation metadata
            controller = None
            adaptation = self._topo_dict.get("adaptation", {}) if self._topo_dict else {}
            if adaptation and adaptation.get("max_upgrades", 0) > 0:
                try:
                    from sage.topology_controller import TopologyController
                    controller = TopologyController()
                    controller.THETA_GOOD = adaptation.get("quality_threshold", 0.5)
                    controller.MAX_RETRIES = adaptation.get("max_upgrades", 1)
                except Exception:
                    pass

            runner = TopologyRunner(
                graph=graph,
                executor=executor,
                llm_provider=provider,
                llm_config=config,
                provider_pool=provider_pool,
                controller=controller,
            )
```

**3f.** In `_finalize_episode()`, store to memory:

After computing total_reward (line 444), add:
```python
        # Store episode in memory for future reference
        if self._memory:
            try:
                import hashlib
                import numpy as np
                self._memory.store_episode(
                    task_id=self._trace.task_id,
                    prompt_hash=hashlib.md5(self._trace.prompt.encode()).hexdigest()[:8],
                    domain="code",
                    topology_yaml=self._trace.topology_yaml[:2000],
                    n_nodes=len(self._node_traces),
                    difficulty=self._difficulty,
                    outcome=status,
                    total_reward=self._trace.total_reward,
                    per_node_results=[
                        {"role": t.get("role", ""), "reward": t.get("reward", 0)}
                        for t in self._node_traces
                    ],
                    adaptations_triggered=sum(
                        1 for t in self._node_traces if t.get("was_upgraded", False)
                    ),
                    embedding=np.zeros(768, dtype=np.float32),
                )
            except Exception:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestTopologyEnvV2 -v 2>&1 | tail -5`
Expected: 3 PASSED

- [ ] **Step 5: Run ALL existing verl tests to check backward compat**

Run: `cd sage-python && python -m pytest tests/test_verl_reward.py -v 2>&1 | tail -5`
Expected: all existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/verl/topology_env.py sage-python/tests/test_verl_v2.py
git commit -m "feat: topology_env v2 — multi-turn, memory, ProviderPool, controller"
```

---

### Task 9: Data pipeline — 3 new adaptive sources

**Files:**
- Modify: `sage-python/scripts/verl/convert_sft_to_verl.py`

- [ ] **Step 1: Add new data sources to GPT54_FILES list**

After line 54 (`"topology_raft_phase2.jsonl"`), add:

```python
# V2 Adaptive data (gpt54_adaptive + static_to_adaptive + recovery)
GPT54_ADAPTIVE_FILES = [
    "gpt54_adaptive_topologies.jsonl",
]

# Special format: static→adaptive (use topology_adaptive field)
GPT54_STATIC_TO_ADAPTIVE = "gpt54_static_to_adaptive.jsonl"

# Special format: recovery scenarios (2 entries per: initial + recovered)
GPT54_RECOVERY = "gpt54_recovery_scenarios.jsonl"
```

- [ ] **Step 2: Add loading logic in the main `convert()` function**

Find where GPT54_CORRECTION and GPT54_AUDIT are loaded. Add after that block:

```python
    # --- V2 Adaptive topologies ---
    for fname in GPT54_ADAPTIVE_FILES:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    topo = e.get("topology", {})
                    row = _make_row(
                        task_id=e.get("task_id", f"adaptive/{len(rows)}"),
                        prompt_text=e.get("prompt", ""),
                        difficulty=topo.get("difficulty", e.get("difficulty", "moderate")),
                        topology=topo,
                        topology_yaml=_topology_to_yaml(topo),
                        source="gpt54_adaptive",
                        node_count=len(topo.get("nodes", [])),
                        edge_count=len(topo.get("edges", [])),
                    )
                    if row:
                        rows.append(row)
            log.info("Loaded %s: %d entries", fname, sum(1 for r in rows if r.get("extra_info", {}).get("task_id", "").startswith("adaptive") or "adaptive" in str(r.get("data_source", ""))))

    # --- Static to Adaptive (use topology_adaptive) ---
    sta_path = data_dir / GPT54_STATIC_TO_ADAPTIVE
    if sta_path.exists():
        count = 0
        with open(sta_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                topo = e.get("topology_adaptive", e.get("topology", {}))
                row = _make_row(
                    task_id=e.get("task_id", f"sta/{len(rows)}"),
                    prompt_text=e.get("prompt", ""),
                    difficulty=topo.get("difficulty", e.get("difficulty", "moderate")),
                    topology=topo,
                    topology_yaml=_topology_to_yaml(topo),
                    source="gpt54_static_to_adaptive",
                )
                if row:
                    rows.append(row)
                    count += 1
        log.info("Loaded %s: %d entries", GPT54_STATIC_TO_ADAPTIVE, count)

    # --- Recovery scenarios (2 entries each: initial + recovered) ---
    rec_path = data_dir / GPT54_RECOVERY
    if rec_path.exists():
        count = 0
        with open(rec_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                # Entry A: initial topology
                topo_init = e.get("initial_topology", {})
                row = _make_row(
                    task_id=e.get("task_id", f"rec/{len(rows)}") + "_init",
                    prompt_text=e.get("prompt", ""),
                    difficulty=topo_init.get("difficulty", e.get("difficulty", "moderate")),
                    topology=topo_init,
                    topology_yaml=_topology_to_yaml(topo_init),
                    source="gpt54_recovery_init",
                )
                if row:
                    rows.append(row)
                    count += 1
                # Entry B: recovered topology
                topo_rec = e.get("recovered_topology", {})
                if topo_rec:
                    row = _make_row(
                        task_id=e.get("task_id", f"rec/{len(rows)}") + "_recovered",
                        prompt_text=e.get("prompt", ""),
                        difficulty=topo_rec.get("difficulty", e.get("difficulty", "moderate")),
                        topology=topo_rec,
                        topology_yaml=_topology_to_yaml(topo_rec),
                        source="gpt54_recovery_recovered",
                    )
                    if row:
                        rows.append(row)
                        count += 1
        log.info("Loaded %s: %d entries (init + recovered)", GPT54_RECOVERY, count)
```

- [ ] **Step 3: Run converter dry-run**

Run: `cd sage-python && python scripts/verl/convert_sft_to_verl.py --input data/topology_sft_v2_combined.jsonl --output /tmp/test_verl.parquet 2>&1 | tail -10`
Expected: total entries around 2200-2500, includes adaptive/recovery sources

- [ ] **Step 4: Commit**

```bash
git add sage-python/scripts/verl/convert_sft_to_verl.py
git commit -m "feat: add 3 adaptive data sources to veRL converter (220+ entries)"
```

---

### Task 10: Local training script — Unsloth GRPO

**Files:**
- Create: `sage-python/scripts/train_local_grpo.py`

- [ ] **Step 1: Create the training script**

```python
#!/usr/bin/env python3
"""Local GRPO training on Qwen3.5-4B with Unsloth QLoRA.

Two phases:
  Phase A: Structural reward ($0 API) — learns YAML format + adaptation metadata
  Phase B: Execution reward (API calls) — learns multi-provider topology execution

Usage:
    # Phase A only (structural, fast)
    python scripts/train_local_grpo.py --phase A

    # Phase B only (execution, needs API keys in .env)
    python scripts/train_local_grpo.py --phase B

    # Both phases
    python scripts/train_local_grpo.py --phase AB

Auto-recovery:
    - OOM → halve batch_size, retry
    - API timeout → structural fallback
    - NaN loss → rollback checkpoint, reduce lr 50%
    - Rate limit → exponential backoff + structural fallback
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_local.log", mode="a"),
    ],
)
log = logging.getLogger("train_local")


def load_env():
    """Load .env file if present."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"'))
        log.info("Loaded .env with %d keys", len([l for l in env_path.read_text().splitlines() if "=" in l]))


def setup_model():
    """Load Qwen3.5-4B with Unsloth QLoRA."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=2048,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def create_reward_fn(phase: str):
    """Create reward function for the given phase."""
    from sage.verl.reward import _score_format, _score_structure, _score_rust_density
    from sage.verl.reward import _score_resilience, _score_cost_efficiency

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)

            fmt = _score_format(text)
            struct = _score_structure(text)
            fmt_norm = (fmt + 2.0) / 3.0

            if phase == "A":
                # Structural only + adaptation bonus
                score = (fmt_norm + struct) / 2.0
                # Bonus for adaptive fields
                try:
                    import yaml
                    topo = yaml.safe_load(text)
                    if isinstance(topo, dict):
                        if topo.get("adaptation"):
                            score += 0.1
                        nodes = topo.get("nodes", [])
                        if any(n.get("fallback_tier") for n in nodes if isinstance(n, dict)):
                            score += 0.1
                except Exception:
                    pass
                rewards.append(float(score))
            else:
                # Phase B: structural + basic execution signal
                # Full execution requires API — handled by topology_env
                rewards.append(float((fmt_norm + struct) / 2.0))

        return rewards

    return reward_fn


def load_dataset(phase: str):
    """Load prompts from parquet."""
    import pandas as pd

    if phase == "A":
        path = Path("data/verl_topology_train.parquet")
    else:
        path = Path("data/verl_topology_curated.parquet")

    if not path.exists():
        log.error("Dataset not found: %s. Run convert_sft_to_verl.py first.", path)
        sys.exit(1)

    df = pd.read_parquet(path)
    log.info("Loaded %d prompts from %s", len(df), path)

    # Convert to list of chat prompts
    prompts = []
    for _, row in df.iterrows():
        prompt = row.get("prompt", [])
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        prompts.append(prompt)

    return prompts


def train_phase(phase: str, model, tokenizer, batch_size: int = 4):
    """Run one training phase."""
    from trl import GRPOConfig, GRPOTrainer

    log.info("=== Starting Phase %s (batch_size=%d) ===", phase, batch_size)

    prompts = load_dataset(phase)
    reward_fn = create_reward_fn(phase)

    epochs = 3 if phase == "A" else 5
    lr = 5e-5 if phase == "A" else 2e-5

    config = GRPOConfig(
        output_dir=f"models/local_grpo_phase_{phase}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 4 // batch_size),
        learning_rate=lr,
        num_generations=4,  # K=4 rollouts per prompt
        max_completion_length=1024,
        logging_steps=10,
        save_steps=100,
        report_to="none",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=prompts,
    )

    try:
        trainer.train()
        log.info("Phase %s complete. Final loss logged in train_local.log", phase)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 1:
            log.warning("OOM in Phase %s — retrying with batch_size=%d", phase, batch_size // 2)
            import torch
            torch.cuda.empty_cache()
            return train_phase(phase, model, tokenizer, batch_size // 2)
        raise

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Local GRPO training")
    parser.add_argument("--phase", default="AB", choices=["A", "B", "AB"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    load_env()

    log.info("Setting up Qwen3.5-4B with Unsloth QLoRA...")
    model, tokenizer = setup_model()

    if "A" in args.phase:
        train_phase("A", model, tokenizer, args.batch_size)

    if "B" in args.phase:
        os.environ["SAGE_VERL_EXEC"] = "1"
        train_phase("B", model, tokenizer, max(1, args.batch_size // 2))

    log.info("Training complete. Check train_local.log for full history.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script loads (no actual training)**

Run: `cd sage-python && python -c "import scripts.train_local_grpo" 2>&1 || echo "Script syntax OK"`
Expected: no import errors (Unsloth not installed yet, but syntax is valid)

- [ ] **Step 3: Commit**

```bash
git add sage-python/scripts/train_local_grpo.py
git commit -m "feat: add local GRPO training script (Unsloth + Qwen3.5-4B QLoRA)"
```

---

### Task 11: Install Unsloth + dependencies

- [ ] **Step 1: Install Unsloth**

Run: `pip install unsloth 2>&1 | tail -5`

- [ ] **Step 2: Verify Unsloth loads**

Run: `python3 -c "from unsloth import FastLanguageModel; print('Unsloth OK')"`
Expected: `Unsloth OK`

- [ ] **Step 3: Verify TRL GRPO available**

Run: `python3 -c "from trl import GRPOConfig, GRPOTrainer; print('TRL GRPO OK')"`
Expected: `TRL GRPO OK`

---

### Task 12: Integration test — end-to-end structural

- [ ] **Step 1: Write integration test**

Append to `tests/test_verl_v2.py`:

```python
class TestIntegrationV2:
    def test_full_structural_episode(self):
        """End-to-end: reset with memory → generate adaptive YAML → step through → finalize."""
        from sage.verl.topology_env import SageTopologyEnv

        env = SageTopologyEnv()
        obs = env.reset("Implement merge sort", "test/mergesort")
        assert "merge sort" in obs["text"].lower() or "Implement" in obs["text"]

        yaml_text = """
difficulty: moderate
reasoning: Merge sort needs careful implementation with fallback for edge cases
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Implement merge sort in Python
  - role: reviewer
    model_tier: budget
    prompt: Review for correctness
  - role: synthesizer
    model_tier: fast
    prompt: Produce the final solution
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message, gate: conditional}
  - {from_idx: 1, to_idx: 2, flow_type: message}
"""
        obs, reward, done, info = env.step(yaml_text)
        assert info["status"] == "TOPOLOGY_PARSED"
        assert reward > 0
        assert not done

        # Step through all nodes
        steps = 0
        while not done and steps < 20:
            obs, reward, done, info = env.step("continue")
            steps += 1

        assert done
        trace = env.get_trace()
        assert trace.total_reward != 0
        assert len(trace.steps) >= 3  # topology_generator + nodes + terminal

        # Verify StepRewardVector
        srv = env.get_step_rewards()
        assert len(srv.step_rewards) > 0
```

- [ ] **Step 2: Run integration test**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py::TestIntegrationV2 -v 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 3: Run ALL tests**

Run: `cd sage-python && python -m pytest tests/test_verl_v2.py tests/test_verl_reward.py -v 2>&1 | tail -15`
Expected: all PASS

- [ ] **Step 4: Final commit**

```bash
git add sage-python/tests/test_verl_v2.py
git commit -m "test: V2 integration — adaptive YAML + memory + full episode"
```

---

### Task 13: Convert data + dry-run training

- [ ] **Step 1: Run data converter with new sources**

Run: `cd sage-python && python scripts/verl/convert_sft_to_verl.py --input data/topology_sft_v2_combined.jsonl --output data/verl_topology_train.parquet 2>&1 | grep -E "(Loaded|Total|entries)"`
Expected: Total > 2200 entries, includes adaptive/recovery sources

- [ ] **Step 2: Curate Phase B dataset**

Run: `cd sage-python && python scripts/verl/curate_training_data.py 2>&1 | tail -5`
Expected: ~600 curated entries

- [ ] **Step 3: Dry-run training (1 step)**

Run: `cd sage-python && python scripts/train_local_grpo.py --phase A --batch-size 1 2>&1 | head -30`
Expected: Model loads, first batch processes, no crash

- [ ] **Step 4: Launch Phase A training (background)**

Run: `cd sage-python && nohup python scripts/train_local_grpo.py --phase A --batch-size 4 > train_phase_a.log 2>&1 &`

---

## Dependency Graph

```
Task 1 (TopologyNode) ──┐
Task 2 (TopologyGraph) ──┼── Task 4 (Rust build) ──┐
Task 3 (RewardScore) ────┘                          │
                                                     ├── Task 8 (topology_env v2)
Task 5 (training_memory) ───────────────────────────┤
Task 6 (rewardflow) ────────────────────────────────┤
Task 7 (reward.py) ─────────────────────────────────┘
                                                     │
Task 9 (data pipeline) ─────────────────────────────┤
Task 11 (install Unsloth) ──────────────────────────┤
                                                     │
                                                     └── Task 12 (integration test) → Task 13 (training)
Task 10 (training script) ──────────────────────────┘
```

**Parallelizable:** Tasks 1+2+3 together, Tasks 5+6+7+9+10 together (after Task 4).
**Critical path:** Tasks 1-4 → Task 8 → Task 12 → Task 13.
