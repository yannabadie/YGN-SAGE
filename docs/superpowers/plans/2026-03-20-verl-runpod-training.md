# YGN-SAGE veRL Training on RunPod H100 — Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a topology generation policy on RunPod H100 via veRL GRPO, beating AgentConductor's results on BigCodeBench and competition-level code benchmarks.

**Architecture:** veRL GRPO with async Reward Loop on single H100 80GB. Model generates YAML topology → async reward function executes topology via external LLM APIs (Gemini Flash) → Rust structural/density scoring → graduated reward. Edge-level credit assignment (Graph-GRPO) as Phase 2 enhancement.

**Tech Stack:** veRL 0.7.1, vLLM 0.17.0, Qwen/Qwen3.5-9B (primary, dense 9B hybrid GatedDeltaNet+attention) or Qwen/Qwen2.5-7B-Instruct (fallback), LoRA r=64, PyO3/maturin for sage-core Rust, asyncio for parallel reward computation.

**Critical References:**
- veRL docs: https://verl.readthedocs.io/en/latest/
- Engineering Handbook: https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
- AgentConductor: arXiv 2602.17100
- Graph-GRPO: arXiv 2603.02701
- SAGE CLAUDE.md: read first for project rules

**Qwen3.5-9B Status (verified March 20, 2026):**
- Architecture: DENSE (NOT MoE — the MoE variant is 35B-A3B only)
- Docker: `verlai/verl:vllm017.latest` EXISTS and works (updated March 12)
- vLLM 0.17.0 bug: CUDA illegal memory access (issue #36408) — WORKAROUND: set `num_speculative_tokens=0` to disable MTP speculative decoding
- veRL issue #5441 is Huawei NPU tracking only, NOT a GPU bug
- 22GB VRAM in bf16 for LoRA fine-tuning — fits H100 80GB easily

---

## Pre-Requisites

Before starting ANY task, read these files:
1. `CLAUDE.md` — project rules (CRITICAL: zero heuristics, Rust first, evidence before assertions)
2. `.claude/rules/architecture.md` — 5 cognitive pillars
3. `.claude/rules/development.md` — build/test commands
4. `synthese.md` — architecture overview
5. `sage-python/scripts/verl/README.md` — current veRL setup docs

## Environment Context

- **RunPod H100 80GB** (or A100 80GB)
- **Docker**: `verlai/verl:base-v4-cu126-cudnn9.8-torch2.7.1-fa2.8.0-te2.3` (CUDA 12.6, PyTorch 2.7.1, FA2.8, TE2.3 — vLLM+veRL installed by setup script)
- **Branch**: `VeRLGIGPO`
- **Model**: `Qwen/Qwen3.5-9B` (primary) — set `num_speculative_tokens=0` in rollout config to avoid MTP CUDA bug
- **Fallback model**: `Qwen/Qwen2.5-7B-Instruct` (if Qwen3.5 crashes)
- **Training data**: `sage-python/data/topology_sft_v2_combined.jsonl` (1880 entries) — upload via scp
- **Additional data** (if available): `topology_raft_phase2.jsonl` (199 exec-verified), `topology_raft_phase2_final.jsonl` (63 final), `topology_sft_gpt54_complex.jsonl` (144 complex), `topology_corrections.jsonl` (GPT-5.4 Pro 2nd-turn pairs)
- **Rust toolchain**: needs `maturin` + Rust 1.90+ for sage-core

## File Structure

### Files to CREATE:
```
sage-python/src/sage/verl/__init__.py          — veRL integration package
sage-python/src/sage/verl/reward.py             — reward function (delegates to existing scoring, adds edge credit)
sage-python/src/sage/verl/edge_credit.py       — Graph-GRPO edge-level credit assignment
sage-python/tests/test_verl_reward.py          — reward function tests
sage-python/tests/test_edge_credit.py          — edge credit tests
sage-python/scripts/verl/validate_setup.py     — pod validation script
sage-python/scripts/verl/benchmark_post_train.py — post-training BigCodeBench eval
```

### Files to MODIFY:
```
sage-python/scripts/verl/setup_runpod.sh       — fix Docker image, add model check
sage-python/scripts/verl/train_topology.sh     — update config from Engineering Handbook
sage-python/scripts/verl/reward_topology.py    — add async support + edge credit
sage-python/scripts/verl/convert_sft_to_verl.py — add 2nd-turn correction data support
```

### Files to KEEP UNCHANGED:
```
sage-python/src/sage/grpo/execution_reward.py  — TRL training code (backward compat)
sage-python/src/sage/topology/runner.py        — topology execution engine
sage-core/                                      — Rust core (build only, don't modify)
```

---

## Phase 1: Pod Setup & Validation (Tasks 1-4)

### Task 1: Verify Pod Environment

**Files:**
- Create: `sage-python/scripts/verl/validate_setup.py`

- [ ] **Step 1: SSH into pod and check GPU**

```bash
nvidia-smi
# Expected: H100 80GB (or A100 80GB)
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_mem / 1024**3, 'GB')"
```

- [ ] **Step 2: Check veRL and vLLM versions**

```bash
python3 -c "import verl; print('veRL OK')"
python3 -c "import vllm; print('vLLM', vllm.__version__)"
# CRITICAL: vLLM must be >= 0.12.0 for veRL 0.7.1
# If Qwen3.5-9B: need vLLM >= 0.17.0 (check for CUDA bugs first)
```

- [ ] **Step 3: Write validation script**

```python
"""validate_setup.py — Run after setup_runpod.sh to verify everything works."""
import sys
import subprocess

def check(name, cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        print(f"{'✓' if ok else '✗'} {name}: {result.stdout.strip()[:80]}")
        if not ok:
            print(f"  ERROR: {result.stderr.strip()[:200]}")
        return ok
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

checks = [
    ("GPU", "python3 -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))\""),
    ("veRL", "python3 -c \"import verl; print('OK')\""),
    ("vLLM", "python3 -c \"import vllm; print(vllm.__version__)\""),
    ("sage-core", "python3 -c \"from sage_core import TopologyGraph, TopologyReward, PyHybridVerifier; print('OK')\""),
    ("SAGE SDK", "python3 -c \"from sage.topology.runner import TopologyRunner; print('OK')\""),
    ("Training data", "python3 -c \"import pandas as pd; df=pd.read_parquet('data/verl_topology_train.parquet'); print(f'{len(df)} entries')\""),
    ("Reward function", "python3 -c \"from sage.verl.reward import compute_score; print(compute_score('test','nodes:\\n- role: coder','',{}))\""),
    ("API keys", "python3 -c \"import os; keys=[k for k in ['GOOGLE_API_KEY','DEEPSEEK_API_KEY'] if os.environ.get(k)]; print(f'{len(keys)} API keys set')\""),
]

results = [check(n, c) for n, c in checks]
passed = sum(results)
print(f"\n{'='*40}")
print(f"Validation: {passed}/{len(results)} passed")
if passed < len(results):
    print("FIX the failures above before training!")
    sys.exit(1)
print("Ready to train!")
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/scripts/verl/validate_setup.py
git commit -m "feat: add pod validation script for veRL training"
```

---

### Task 2: Fix Setup Script for Correct Docker/Versions

**Files:**
- Modify: `sage-python/scripts/verl/setup_runpod.sh`

- [ ] **Step 1: Check available Docker images on Docker Hub**

```bash
# On your LOCAL machine or the pod:
curl -s "https://hub.docker.com/v2/repositories/verlai/verl/tags/?page_size=20" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for tag in data.get('results', []):
    print(f\"{tag['name']:40s} {tag.get('last_updated','')[:10]}\")
" 2>/dev/null || echo "Check https://hub.docker.com/r/verlai/verl/tags manually"
```

- [ ] **Step 2: Update setup script with correct image and model fallback**

In `setup_runpod.sh`, update the header comment to reflect the actual Docker image found in Step 1. Add a model availability check:

```bash
# After the environment verification section, add:
echo "[2/7] Checking model availability..."
MODEL=${SAGE_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
python3 -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
    print(f'Model: {cfg.architectures[0]} — OK')
except Exception as e:
    print(f'Model {\"$MODEL\"} not available: {e}')
    print('Falling back to Qwen/Qwen2.5-7B-Instruct')
"
```

- [ ] **Step 3: Run the updated setup**

```bash
bash sage-python/scripts/verl/setup_runpod.sh
```

- [ ] **Step 4: Run validation**

```bash
cd sage-python && python3 scripts/verl/validate_setup.py
# Expected: 8/8 passed
```

- [ ] **Step 5: Commit**

```bash
git add sage-python/scripts/verl/setup_runpod.sh
git commit -m "fix: setup script with correct Docker image and model fallback"
```

---

### Task 3: Create veRL Package Structure

**Files:**
- Create: `sage-python/src/sage/verl/__init__.py`

- [ ] **Step 1: Create the package**

```python
"""sage.verl — veRL integration for SAGE topology training.

Provides:
- reward: Async reward function with topology execution
- edge_credit: Graph-GRPO edge-level credit assignment
"""
```

- [ ] **Step 2: Verify import works**

```bash
cd sage-python && python3 -c "import sage.verl; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/verl/__init__.py
git commit -m "feat: create sage.verl package for veRL integration"
```

---

### Task 4: Convert Training Data with 2nd-Turn Support

**Files:**
- Modify: `sage-python/scripts/verl/convert_sft_to_verl.py`

- [ ] **Step 1: Update converter to handle correction pairs**

Add support for `topology_corrections.jsonl` (error→correction pairs, generated via GPT-5.4 Pro prompts in `data/PROMPTS_GPT54_PRO.md`). If the file exists, include it. If not, skip gracefully.

After line `entries.append(json.loads(line))` in the main loop, add a second pass:

```python
    # Also load correction pairs if available
    corrections_path = Path(input_path).parent / "topology_corrections.jsonl"
    if corrections_path.exists():
        with open(corrections_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Correction entries have "failed_topology" + "corrected_topology"
                if "corrected_topology" in entry:
                    entries.append({
                        "prompt": entry.get("prompt", ""),
                        "task_id": entry.get("task_id", ""),
                        "difficulty": entry.get("difficulty", "moderate"),
                        "topology_yaml": entry["corrected_topology"],
                        "source": "correction",
                    })
        log.info("Added %d correction pairs from %s",
                 sum(1 for e in entries if e.get("source") == "correction"),
                 corrections_path)
```

- [ ] **Step 2: Run conversion**

```bash
cd sage-python && python3 scripts/verl/convert_sft_to_verl.py \
    --input data/topology_sft_v2_combined.jsonl \
    --output data/verl_topology_train.parquet
# Expected: ~1880 entries (+ corrections if available)
```

- [ ] **Step 3: Verify parquet schema**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/verl_topology_train.parquet')
print(f'Entries: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Abilities: {df[\"ability\"].value_counts().to_dict()}')
# Must have: data_source, prompt, ability, reward_model, extra_info
"
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/scripts/verl/convert_sft_to_verl.py
git commit -m "feat: support 2nd-turn correction data in veRL converter"
```

---

## Phase 2: Reward Function (Tasks 5-7)

### Task 5: Write Reward Function Tests

**Files:**
- Create: `sage-python/tests/test_verl_reward.py`

- [ ] **Step 1: Write tests for the reward function**

```python
"""Tests for veRL reward function (sage.verl.reward)."""
import pytest


def test_format_valid_yaml():
    from sage.verl.reward import _score_format
    yaml_text = "nodes:\n- role: coder\n  prompt: Write code\ndifficulty: simple"
    assert _score_format(yaml_text) == 1.0


def test_format_invalid_yaml():
    from sage.verl.reward import _score_format
    assert _score_format("not: [valid: yaml: {{") == -2.0


def test_format_no_nodes():
    from sage.verl.reward import _score_format
    assert _score_format("reasoning: just text") == -0.5


def test_structure_complete():
    from sage.verl.reward import _score_structure
    yaml_text = """
nodes:
  - role: coder
    prompt: Write code
    model_tier: fast
  - role: reviewer
    prompt: Review code
    model_tier: fast
edges:
  - from_idx: 0
    to_idx: 1
    flow_type: message
reasoning: Need coder then reviewer
difficulty: moderate
"""
    score = _score_structure(yaml_text)
    assert score == pytest.approx(1.0)  # has all 4 components


def test_structure_minimal():
    from sage.verl.reward import _score_structure
    score = _score_structure("nodes:\n- role: coder")
    assert score == pytest.approx(0.6)  # nodes (0.3) + roles (0.3), no edges/reasoning


def test_rust_density_fallback_without_sage_core():
    """When sage_core is not importable, returns 0.5 for valid topology."""
    from sage.verl.reward import _score_rust_density
    score = _score_rust_density("nodes:\n- role: coder", {})
    # With sage_core: returns Rust-computed score
    # Without sage_core: returns 0.5 fallback
    assert 0.0 <= score <= 1.0


def test_compute_score_sync():
    """compute_score must work as sync function (veRL calls it synchronously)."""
    from sage.verl.reward import compute_score
    result = compute_score(
        data_source="sage_topology",
        solution_str="nodes:\n- role: coder\n  prompt: code\ndifficulty: simple",
        ground_truth="",
        extra_info={"task_id": "test/0", "difficulty": "simple"},
    )
    assert isinstance(result, float)
    # fmt_norm=1.0, struct=0.6, rust>=0.5 → combined >= 0.7
    assert 0.5 <= result <= 1.0


def test_compute_score_invalid_yaml():
    """Invalid YAML should produce low but non-negative score."""
    from sage.verl.reward import compute_score
    result = compute_score("sage_topology", "{{invalid", "", {})
    assert 0.0 <= result <= 0.1  # fmt=-2.0 → fmt_norm=0.0, struct=0.0, rust=0.0
```

- [ ] **Step 2: Run tests — they should FAIL (module doesn't exist yet)**

```bash
cd sage-python && python -m pytest tests/test_verl_reward.py -v
# Expected: ImportError — sage.verl.reward not found
```

- [ ] **Step 3: Commit failing tests**

```bash
git add sage-python/tests/test_verl_reward.py
git commit -m "test: add veRL reward function tests (red)"
```

---

### Task 6: Implement Reward Function (DRY — delegates to existing scoring)

**Files:**
- Create: `sage-python/src/sage/verl/reward.py`

- [ ] **Step 1: Implement the reward function**

The existing `scripts/verl/reward_topology.py` already has `_score_format`, `_score_structure`, and `_score_execution_proxy`. We import from there to avoid duplication. The `sage.verl.reward` module adds the veRL-compatible entry point and edge credit integration.

```python
"""Reward function for veRL topology training.

veRL signature: compute_score(data_source, solution_str, ground_truth, extra_info) -> float

Delegates scoring to existing reward_topology.py (DRY).
Adds edge-level credit integration for Graph-GRPO (arXiv 2603.02701).

Register in veRL config:
    custom_reward_function.path=sage-python/src/sage/verl/reward.py
    custom_reward_function.name=compute_score
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("verl_reward")

# Ensure scripts/verl/ is importable (RunPod path)
_scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "verl"
if _scripts_dir.exists() and str(_scripts_dir.parent) not in sys.path:
    sys.path.insert(0, str(_scripts_dir.parent))


# ── Scoring functions (re-exported for tests) ───────────────
# These are thin wrappers that delegate to the canonical implementations.
# If running on RunPod, they import from scripts/verl/reward_topology.py.
# If that fails (e.g., in CI), they provide standalone fallbacks.

import yaml


def _score_format(text: str) -> float:
    """YAML format validity. Range: [-2.0, +1.0]."""
    try:
        from verl.reward_topology import _score_format as _sf
        return _sf(text)
    except ImportError:
        pass
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return -1.5
        if "nodes" not in data:
            return -0.5
        nodes = data["nodes"]
        if not isinstance(nodes, list) or len(nodes) == 0:
            return -0.25
        return 1.0
    except yaml.YAMLError:
        return -2.0
    except Exception:
        return -2.0


def _score_structure(text: str) -> float:
    """Structural quality. Range: [0.0, 1.0]."""
    try:
        from verl.reward_topology import _score_structure as _ss
        return _ss(text)
    except ImportError:
        pass
    try:
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
        return score
    except Exception:
        return 0.0


def _score_rust_density(text: str, extra_info: dict) -> float:
    """Rust TopologyReward + TopologyDensity. Fallback: 0.5 for valid topology."""
    try:
        from verl.reward_topology import _score_execution_proxy as _sep
        return _sep(text, extra_info)
    except ImportError:
        pass
    # Standalone fallback (CI/local without scripts dir)
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "nodes" not in data:
            return 0.0
        try:
            from sage_core import (
                TopologyReward, TopologyDensity, TopologyGraph,
                TopologyNode, TopologyEdge, PyHybridVerifier,
            )
        except ImportError:
            return 0.5 if isinstance(data.get("nodes"), list) and len(data["nodes"]) > 0 else 0.0

        nodes = data.get("nodes", [])
        difficulty = data.get("difficulty", extra_info.get("difficulty", "moderate"))
        system = {"simple": 1, "moderate": 2, "complex": 3}.get(str(difficulty).lower(), 2)
        graph = TopologyGraph("sequential")
        for nd in nodes:
            if isinstance(nd, dict):
                graph.add_node(TopologyNode(
                    role=nd.get("role", "agent"), model_id=nd.get("model_tier", ""),
                    system=system, prompt=nd.get("prompt", ""),
                ))
        for ed in data.get("edges", []):
            if isinstance(ed, dict):
                fi, ti = ed.get("from_idx", 0), ed.get("to_idx", 0)
                if 0 <= fi < graph.node_count() and 0 <= ti < graph.node_count():
                    graph.add_edge(fi, ti, TopologyEdge(ed.get("flow_type", "message")))
        if graph.edge_count() == 0 and graph.node_count() > 1:
            for i in range(graph.node_count() - 1):
                graph.add_edge(i, i + 1, TopologyEdge("message"))

        import math
        density = TopologyDensity()
        verifier = PyHybridVerifier()
        scorer = TopologyReward()
        d = density.compute(graph, system)
        v = verifier.verify(graph)
        structural = 1.0 if v.valid else 0.5
        reward = scorer.compute(
            execution_passed=True,
            structural_score=structural,
            density_score=d.s_complex,
            temporal_score=None,
        )
        score = reward.total
        if d.over_budget:
            n_nodes = graph.node_count()
            penalty = math.tanh(float(d.n_max - n_nodes) / float(max(d.n_max, 1)))
            score = score * max(0.0, 1.0 + penalty)
        return float(score)
    except Exception:
        return 0.0


# ── Combined reward (veRL entry point) ───────────────────────
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Combined topology reward for veRL. Sync wrapper.

    This is registered in veRL config as:
        custom_reward_function.path=sage-python/src/sage/verl/reward.py
        custom_reward_function.name=compute_score
    """
    if extra_info is None:
        extra_info = {}

    fmt = _score_format(solution_str)
    struct = _score_structure(solution_str)
    rust = _score_rust_density(solution_str, extra_info)

    fmt_norm = (fmt + 2.0) / 3.0   # [-2.0, 1.0] → [0.0, 1.0]
    combined = (fmt_norm + struct + rust) / 3.0
    return float(combined)
```

- [ ] **Step 2: Run tests**

```bash
cd sage-python && python -m pytest tests/test_verl_reward.py -v
# Expected: ALL PASS
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/verl/reward.py sage-python/src/sage/verl/__init__.py
git commit -m "feat: async reward function for veRL topology training"
```

---

### Task 7: Update Training Script with Engineering Handbook Settings

**Files:**
- Modify: `sage-python/scripts/verl/train_topology.sh`

- [ ] **Step 1: Update config based on Engineering Handbook findings**

Key changes from the verified research:
- `gpu_memory_utilization=0.8` (not 0.6 — wastes 20GB)
- `train_batch_size=64` for single GPU (not 256 — OOM risk)
- `rollout.n=5` (not 8 — +70% time for marginal gain per handbook)
- Model fallback to Qwen2.5-7B-Instruct
- Point reward to new `sage.verl.reward`

```bash
# In train_topology.sh, update these lines:
MODEL=${SAGE_MODEL:-"Qwen/Qwen3.5-9B"}  # Dense 9B, Apache 2.0. Fallback: Qwen/Qwen2.5-7B-Instruct
# ...
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"
# ...
data.train_batch_size=64 \          # was 256, too high for 1 GPU
# ...
actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \  # was 0.6, Engineering Handbook optimal
actor_rollout_ref.rollout.n=5 \     # was 8, handbook says n=10 adds +70% time
actor_rollout_ref.rollout.num_speculative_tokens=0 \  # CRITICAL: disable MTP to avoid vLLM 0.17 CUDA bug (#36408)
```

- [ ] **Step 2: Verify the script parses correctly**

```bash
bash -n sage-python/scripts/verl/train_topology.sh
echo $?  # Expected: 0
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/scripts/verl/train_topology.sh
git commit -m "fix: training config aligned with veRL Engineering Handbook"
```

---

## Phase 3: Edge-Level Credit Assignment (Tasks 8-10)

### Task 8: Write Edge Credit Tests

**Files:**
- Create: `sage-python/tests/test_edge_credit.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for Graph-GRPO edge-level credit assignment."""
import pytest
from sage.verl.edge_credit import compute_edge_advantages, EdgeStats


def test_edge_success_rate():
    """Edges in passing topologies get higher success rate."""
    topologies = [
        {"edges": [(0,1), (1,2)], "reward": 1.5},  # PASSED
        {"edges": [(0,1), (1,2)], "reward": 1.5},  # PASSED
        {"edges": [(0,1), (0,2)], "reward": 0.0},  # CRASH (different edges)
    ]
    stats = EdgeStats.from_topologies(topologies)
    # Edge (1,2) appears in 2 passing topos, (0,2) in 1 failing
    assert stats.success_rate((1, 2)) > stats.success_rate((0, 2))


def test_edge_advantage_normalization():
    """Advantages are normalized (mean ~0, std ~1 within group)."""
    topologies = [
        {"edges": [(0,1), (1,2)], "reward": 1.5},
        {"edges": [(0,1)], "reward": 0.0},
        {"edges": [(0,1), (1,2), (2,3)], "reward": 1.0},
    ]
    advantages = compute_edge_advantages(topologies)
    # Should have entries for edges (0,1), (1,2), (2,3)
    assert len(advantages) >= 2
    # Advantages should be centered
    mean_adv = sum(advantages.values()) / len(advantages)
    assert abs(mean_adv) < 0.5  # approximately centered


def test_topology_reward_with_edge_credit():
    """Combined reward = base_reward + edge_credit_weight * edge_advantage."""
    topologies = [
        {"edges": [(0,1), (1,2)], "reward": 1.5, "yaml": "nodes:\n- role: coder"},
        {"edges": [(0,1)], "reward": 0.0, "yaml": "nodes:\n- role: coder"},
    ]
    advantages = compute_edge_advantages(topologies)
    # Edge (1,2) should have positive advantage (only appears in passing)
    assert advantages.get((1, 2), 0.0) > 0.0


def test_empty_edges():
    """Topology with no edges gets 0 edge advantage."""
    topologies = [
        {"edges": [], "reward": 1.0},
    ]
    advantages = compute_edge_advantages(topologies)
    assert len(advantages) == 0
```

- [ ] **Step 2: Run — should FAIL**

```bash
cd sage-python && python -m pytest tests/test_edge_credit.py -v
# Expected: ImportError
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/tests/test_edge_credit.py
git commit -m "test: add Graph-GRPO edge credit tests (red)"
```

---

### Task 9: Implement Edge-Level Credit Assignment

**Files:**
- Create: `sage-python/src/sage/verl/edge_credit.py`

- [ ] **Step 1: Implement Graph-GRPO edge credit**

```python
"""Graph-GRPO edge-level credit assignment (arXiv 2603.02701).

Computes per-edge success rates across K topologies for the same prompt,
then normalizes to advantages. This provides finer-grained credit than
per-topology reward.

Usage in reward function:
    edges = parse_edges(yaml_text)
    edge_advs = compute_edge_advantages(group_topologies)
    credit = sum(edge_advs.get(e, 0.0) for e in edges) / max(len(edges), 1)
    final_reward = base_reward + edge_credit_weight * credit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EdgeStats:
    """Track per-edge success rates across a group of topologies."""
    _counts: dict[tuple[int, int], int] = field(default_factory=dict)
    _successes: dict[tuple[int, int], float] = field(default_factory=dict)

    def record(self, edge: tuple[int, int], reward: float) -> None:
        self._counts[edge] = self._counts.get(edge, 0) + 1
        self._successes[edge] = self._successes.get(edge, 0.0) + reward

    def success_rate(self, edge: tuple[int, int], eps: float = 1e-6) -> float:
        count = self._counts.get(edge, 0)
        if count == 0:
            return 0.0
        return self._successes.get(edge, 0.0) / (count + eps)

    @classmethod
    def from_topologies(cls, topologies: list[dict]) -> EdgeStats:
        stats = cls()
        for topo in topologies:
            reward = topo.get("reward", 0.0)
            binary = 1.0 if reward > 0.5 else 0.0  # binary success
            for edge in topo.get("edges", []):
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    stats.record((edge[0], edge[1]), binary)
        return stats

    @property
    def all_edges(self) -> list[tuple[int, int]]:
        return list(self._counts.keys())


def compute_edge_advantages(
    topologies: list[dict],
    eps: float = 1e-6,
) -> dict[tuple[int, int], float]:
    """Compute normalized edge advantages (Graph-GRPO Eq. 4-5).

    For a group of K topologies for the same prompt:
    1. S_ij = P(Success | edge(i,j) in G)  — per-edge success rate
    2. A_ij = (S_ij - mean(S)) / (std(S) + eps)  — normalized advantage

    Args:
        topologies: list of {"edges": [(i,j), ...], "reward": float}

    Returns:
        dict mapping (from_idx, to_idx) → advantage float
    """
    stats = EdgeStats.from_topologies(topologies)
    edges = stats.all_edges
    if not edges:
        return {}

    rates = {e: stats.success_rate(e) for e in edges}
    values = list(rates.values())
    n = len(values)
    if n == 0:
        return {}

    mean_s = sum(values) / n
    var_s = sum((v - mean_s) ** 2 for v in values) / max(n, 1)
    std_s = var_s ** 0.5

    return {
        edge: (rate - mean_s) / (std_s + eps)
        for edge, rate in rates.items()
    }


def parse_edges_from_yaml(yaml_text: str) -> list[tuple[int, int]]:
    """Extract edges from a YAML topology string."""
    import yaml as _yaml
    try:
        data = _yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return []
        edges = []
        for ed in data.get("edges", []):
            if isinstance(ed, dict):
                edges.append((ed.get("from_idx", 0), ed.get("to_idx", 0)))
        return edges
    except Exception:
        return []
```

- [ ] **Step 2: Run tests**

```bash
cd sage-python && python -m pytest tests/test_edge_credit.py -v
# Expected: ALL PASS
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/verl/edge_credit.py
git commit -m "feat: Graph-GRPO edge-level credit assignment (arXiv 2603.02701)"
```

---

### Task 10: Add `compute_score_with_edge_credit` for batch-level use

**Files:**
- Modify: `sage-python/src/sage/verl/reward.py`

Edge credit requires comparing K topologies for the same prompt (Graph-GRPO groups edges across samples). This function is called by the training loop AFTER collecting all K samples.

- [ ] **Step 1: Add batch-level edge credit function to reward.py**

Append to the end of `sage-python/src/sage/verl/reward.py`:

```python
def compute_score_with_edge_credit(
    topologies: list[dict],
    edge_weight: float = 0.2,
) -> list[float]:
    """Batch-level reward with Graph-GRPO edge credit (arXiv 2603.02701).

    Called after collecting K topologies for the same prompt.
    Adjusts per-topology rewards by edge-level advantage.

    Args:
        topologies: list of {"yaml": str, "base_reward": float, "extra_info": dict}
        edge_weight: weight of edge credit bonus (default 0.2)

    Returns:
        list of adjusted rewards (same length as input)
    """
    from sage.verl.edge_credit import compute_edge_advantages, parse_edges_from_yaml

    # Build edge data for advantage computation
    edge_data = []
    for topo in topologies:
        edges = parse_edges_from_yaml(topo.get("yaml", ""))
        edge_data.append({
            "edges": edges,
            "reward": topo.get("base_reward", 0.0),
        })

    advantages = compute_edge_advantages(edge_data)

    # Adjust each topology's reward by its edges' average advantage
    adjusted = []
    for topo, ed in zip(topologies, edge_data):
        base = topo.get("base_reward", 0.0)
        edges = ed["edges"]
        if edges and advantages:
            edge_bonus = sum(advantages.get(tuple(e), 0.0) for e in edges) / len(edges)
        else:
            edge_bonus = 0.0
        adjusted.append(base + edge_weight * edge_bonus)

    return adjusted
```

- [ ] **Step 2: Write test for batch edge credit**

Add to `tests/test_verl_reward.py`:

```python
def test_compute_score_with_edge_credit():
    from sage.verl.reward import compute_score_with_edge_credit
    topos = [
        {"yaml": "nodes:\n- role: coder\n- role: reviewer\nedges:\n- from_idx: 0\n  to_idx: 1", "base_reward": 1.5},
        {"yaml": "nodes:\n- role: coder\nedges: []", "base_reward": 0.0},
    ]
    adjusted = compute_score_with_edge_credit(topos)
    assert len(adjusted) == 2
    # The passing topology with edge (0,1) should get a bonus
    assert adjusted[0] >= 1.5  # bonus from edge credit
```

- [ ] **Step 3: Run all tests**

```bash
cd sage-python && python -m pytest tests/test_verl_reward.py tests/test_edge_credit.py -v
# Expected: ALL PASS
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/verl/reward.py sage-python/tests/test_verl_reward.py
git commit -m "feat: batch-level edge credit integration (Graph-GRPO)"
```

---

## Phase 4: Training & Evaluation (Tasks 11-13)

### Task 11: Launch Training

- [ ] **Step 1: Upload training data to pod**

```bash
# From LOCAL machine:
scp sage-python/data/topology_sft_v2_combined.jsonl <pod>:/workspace/YGN-SAGE/sage-python/data/
# If you have correction data:
scp sage-python/data/topology_corrections.jsonl <pod>:/workspace/YGN-SAGE/sage-python/data/ 2>/dev/null
```

- [ ] **Step 2: Set API keys on pod**

```bash
export GOOGLE_API_KEY="<your-key>"       # For Gemini Flash node execution
export DEEPSEEK_API_KEY="<your-key>"     # For DeepSeek Reasoner (optional)
export WANDB_API_KEY="<your-key>"        # For training dashboard (optional)
```

- [ ] **Step 3: Run setup + convert + validate**

```bash
cd /workspace/YGN-SAGE
bash sage-python/scripts/verl/setup_runpod.sh
cd sage-python && python3 scripts/verl/validate_setup.py
# Must be 8/8 before proceeding
```

- [ ] **Step 4: Launch training**

```bash
cd /workspace/YGN-SAGE/sage-python
nohup bash scripts/verl/train_topology.sh > train.log 2>&1 &
echo $! > train.pid
echo "Training PID: $(cat train.pid)"
tail -f train.log  # Monitor
```

- [ ] **Step 5: Monitor training**

```bash
# Check loss and reward curves:
tail -20 train.log | grep -E "(reward|loss|epoch)"
# Check GPU utilization:
nvidia-smi
# Check W&B dashboard if configured
```

---

### Task 12: Post-Training Evaluation

**Files:**
- Create: `sage-python/scripts/verl/benchmark_post_train.py`

- [ ] **Step 1: Write benchmark script**

```python
"""Post-training evaluation on BigCodeBench Hard + HumanEval+.

IMPORTANT: Set SAGE_ENABLE_PATH6=1 and SAGE_TOPOLOGY_MODEL to the trained adapter
so benchmarks use the newly trained topology policy, not the default.
"""
import argparse
import os
import subprocess
import sys


def run_bench(bench_type: str, limit: int = 20, model_path: str = ""):
    env = os.environ.copy()
    if model_path:
        env["SAGE_ENABLE_PATH6"] = "1"
        env["SAGE_TOPOLOGY_MODEL"] = model_path
    cmd = [
        sys.executable, "-m", "sage.bench",
        "--type", bench_type,
        "--limit", str(limit),
    ]
    if bench_type == "bigcodebench":
        cmd.extend(["--subset", "hard", "--split", "instruct"])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["bigcodebench", "humaneval", "routing_gt", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="models/topology_verl_local/",
                        help="Path to trained LoRA adapter (enables Path 6)")
    args = parser.parse_args()

    benches = [args.bench] if args.bench != "all" else ["bigcodebench", "humaneval", "routing_gt"]
    for bench in benches:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench} (model: {args.model})")
        print(f"{'='*60}")
        run_bench(bench, args.limit, model_path=args.model)
```

- [ ] **Step 2: Export LoRA adapter**

```bash
cd sage-python
python3 scripts/verl/export_for_local.py \
    --checkpoint models/topology_verl/ \
    --output models/topology_verl_local/
```

- [ ] **Step 3: Run benchmarks**

```bash
python3 scripts/verl/benchmark_post_train.py --bench all --limit 20
```

- [ ] **Step 4: Commit results**

```bash
git add sage-python/scripts/verl/benchmark_post_train.py
git commit -m "feat: post-training benchmark script for veRL evaluation"
```

---

### Task 13: Push Results and Clean Up

- [ ] **Step 1: Commit all training artifacts (NOT model weights)**

```bash
git add sage-python/src/sage/verl/ sage-python/scripts/verl/ sage-python/tests/test_verl_reward.py sage-python/tests/test_edge_credit.py
git add train.log 2>/dev/null  # Training log if present
# Do NOT add sage-python/models/ (large binary files)
git commit -m "feat: veRL GRPO training complete — results and configs"
```

- [ ] **Step 2: Push to VeRLGIGPO branch**

```bash
git push origin VeRLGIGPO
```

- [ ] **Step 3: Download adapter to local machine**

```bash
# From LOCAL machine:
scp -r <pod>:/workspace/YGN-SAGE/sage-python/models/topology_verl_local/ sage-python/models/
```

---

## Appendix A: Troubleshooting

### OOM on H100
- Reduce `gpu_memory_utilization` to 0.7
- Reduce `train_batch_size` to 32
- Enable `actor.fsdp_config.optimizer_offload=True`
- Reduce `rollout.n` to 3

### vLLM crashes with Qwen3.5-9B
- Ensure `num_speculative_tokens=0` is set (disables MTP, fixes CUDA bug #36408)
- Use Docker `verlai/verl:vllm017.latest` (NOT pip install — pip pins old vLLM)
- If still crashes: fall back to `Qwen/Qwen2.5-7B-Instruct`
- Issue #5441 is Huawei NPU only — not relevant for GPU

### Reward function errors
- Check API keys are set
- Run `validate_setup.py` to verify sage-core import
- Check `sage_core.TopologyReward` works: `python3 -c "from sage_core import TopologyReward; print('OK')"`

### Training diverges
- Check reward distribution: should be centered, not all zeros
- Reduce `kl_loss_coef` from 0.04 to 0.02
- Increase `rollout.temperature` from 0.4 to 0.6

## Appendix B: SAGE vs AgentConductor Comparison

| Aspect | AgentConductor | SAGE (after this plan) |
|--------|----------------|------------------------|
| Model | Qwen2.5-3B | Qwen3.5-9B (dense, 3x capacity) |
| Algorithm | GRPO | GRPO + Edge Credit |
| Verification | None | OxiZ SMT formal |
| Density | S_complex (identical) | S_complex + Rust perf |
| TopologyEngine | Template only | 5 paths (MAP-Elites, MCTS, CMA-ME, LLM, templates) |
| Self-adaptive | No | MAP-Elites archive + bandit (infra ready, not yet self-programming) |
| Training data | 4500 (GPT-4o) | 1880 + GPT-5.4 Pro corrections |
| BigCodeBench | Not submitted | 37.8% → submit after training |
| 2nd-turn | Yes (2700 pairs) | Planned (GPT-5.4 Pro distillation) |
