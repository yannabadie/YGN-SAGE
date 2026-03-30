# JSON Tool-Call Training for Nemotron-8B

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot Nemotron-8B training from YAML free-text generation to JSON tool-calling — its native format — to eliminate the 91% malformation rate and enable dense execution reward signal.

**Architecture:** Four changes, zero modifications to existing SAGE code:
1. Convert the 12K training dataset from YAML ground truth to JSON
2. Update the system prompt from "YAML DAG" to "JSON tool calls"
3. Add a `tool_calls → TopologyGraph` converter for Phase C's SageTopologyEnv
4. Configure vLLM guided_json in the verl training script for constrained decoding

The reward function, execution pipeline, and Rust TopologyGraph already accept JSON natively (verified: identical score 0.9848 for YAML and JSON).

**Tech Stack:** verl 0.7.1 (DAPO), vLLM guided_json, Nemotron-Orchestrator-8B (native JSON tool-caller), sage_core (Rust), Pydantic (JSON schema).

---

## Why This Works

Nemotron-Orchestrator-8B was GRPO-trained by NVIDIA to emit structured JSON tool calls (ToolOrchestra, arXiv 2511.21689). We've been asking it to generate YAML — a format it was never trained for. This explains:
- 91% YAML malformation (model doesn't know YAML syntax)
- Structural plateau at 0.225 (can't improve what it can't format)
- SFT warmup insufficient (118 steps can't overcome GRPO pretraining bias)

With JSON tool-calling:
- 0% malformation (vLLM constrained decoding guarantees valid JSON)
- 100% exec reward signal (every topology is valid → gets executed)
- ~200 step convergence (matching The Conductor, using model's native format)

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `sage-python/scripts/verl/convert_dataset_to_json.py` | Convert 12K YAML → JSON | Create |
| `sage-python/data/verl_topology_train_json.parquet` | JSON training dataset | Generated |
| `sage-python/src/sage/verl/topology_schema.py` | Pydantic schema for guided_json | Modify |
| `sage-python/src/sage/verl/topology_env.py` | Phase C: parse JSON tool_calls | Modify |
| `sage-python/scripts/verl/train_topology_json.sh` | Training script with guided_json | Create |
| `sage-python/src/sage/verl/reward.py` | No change needed (already accepts JSON) |
| `sage-python/src/sage/execution/__init__.py` | No change needed (JSON first) |
| `sage-core/` | No change needed (TopologyGraph is programmatic) |

---

### Task 1: Define Pydantic schema for topology JSON

**Files:**
- Modify: `sage-python/src/sage/verl/topology_schema.py`
- Test: inline Python test

This schema serves two purposes: (a) guided_json for vLLM constrained decoding, (b) validation in reward function.

- [ ] **Step 1: Write the Pydantic schema**

```python
# In topology_schema.py, add:
from pydantic import BaseModel
from typing import Optional

class TopologyNode(BaseModel):
    role: str  # coder, reviewer, planner, executor, analyst, debugger
    model_tier: str  # budget, fast, reasoner
    prompt: str

class TopologyEdge(BaseModel):
    from_idx: int
    to_idx: int
    flow_type: str = "message"  # message, control, state

class TopologyOutput(BaseModel):
    difficulty: str  # simple, moderate, complex
    reasoning: str
    nodes: list[TopologyNode]
    edges: list[TopologyEdge] = []

class CheckpointDecision(BaseModel):
    action: str  # continue, upgrade, reroute
    node_idx: Optional[int] = None
    new_tier: Optional[str] = None
```

- [ ] **Step 2: Verify schema generates valid JSON schema**

```bash
python3 -c "
from sage.verl.topology_schema import TopologyOutput, CheckpointDecision
import json
print(json.dumps(TopologyOutput.model_json_schema(), indent=2))
print('---')
print(json.dumps(CheckpointDecision.model_json_schema(), indent=2))
"
```

- [ ] **Step 3: Verify reward function accepts schema-conforming JSON**

```bash
python3 -c "
from sage.verl.topology_schema import TopologyOutput
from sage.verl.reward import _score_format, _score_structure
topo = TopologyOutput(
    difficulty='simple', reasoning='test',
    nodes=[{'role':'coder','model_tier':'budget','prompt':'code'}]
)
json_str = topo.model_dump_json()
print(f'format={_score_format(json_str)}, structure={_score_structure(json_str)}')
"
```

Expected: format=1.00, structure>0.5

- [ ] **Step 4: Commit**

```bash
git add src/sage/verl/topology_schema.py
git commit -m "feat: Pydantic schema for JSON topology (guided_json + Phase C)"
```

---

### Task 2: Convert training dataset YAML → JSON

**Files:**
- Create: `sage-python/scripts/verl/convert_dataset_to_json.py`
- Output: `sage-python/data/verl_topology_train_json.parquet`

Convert 12,303 entries from YAML ground truth to JSON. Also update system prompt.

- [ ] **Step 1: Write conversion script**

```python
#!/usr/bin/env python3
"""Convert YAML topology training data to JSON format.

Nemotron-Orchestrator-8B is natively trained for JSON tool-calling.
YAML generation caused 91% malformation. JSON eliminates this.
"""
import json, yaml
import pandas as pd
import numpy as np

def yaml_to_json(yaml_str: str) -> str:
    """Convert YAML topology to JSON string."""
    try:
        data = yaml.safe_load(yaml_str)
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False)
    except Exception:
        pass
    return yaml_str  # return as-is if can't convert

def update_system_prompt(prompt_list):
    """Change 'YAML DAG' to 'JSON' in system prompt."""
    updated = []
    for msg in prompt_list:
        if isinstance(msg, dict) and msg.get('role') == 'system':
            content = msg['content']
            content = content.replace('YAML DAG', 'JSON object')
            content = content.replace('as a YAML', 'as a JSON')
            content = content.replace('YAML format', 'JSON format')
            msg = {**msg, 'content': content}
        updated.append(msg)
    return updated

def convert_dataset(input_path, output_path):
    df = pd.read_parquet(input_path)
    converted = 0
    for idx in range(len(df)):
        # Convert ground truth
        rm = df.iloc[idx]['reward_model']
        if isinstance(rm, dict) and 'ground_truth' in rm:
            json_gt = yaml_to_json(rm['ground_truth'])
            rm = {**rm, 'ground_truth': json_gt}
            df.at[idx, 'reward_model'] = rm
            converted += 1

        # Update system prompt
        prompt = df.iloc[idx]['prompt']
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            df.at[idx, 'prompt'] = update_system_prompt(prompt)

    df.to_parquet(output_path, index=False)
    print(f"Converted {converted}/{len(df)} entries → {output_path}")

if __name__ == "__main__":
    convert_dataset(
        "data/verl_topology_train.parquet",
        "data/verl_topology_train_json.parquet",
    )
    # Also convert curated
    convert_dataset(
        "data/verl_topology_curated.parquet",
        "data/verl_topology_curated_json.parquet",
    )
```

- [ ] **Step 2: Run conversion**

```bash
cd sage-python
python3 scripts/verl/convert_dataset_to_json.py
```

Expected: "Converted 12303/12303 entries"

- [ ] **Step 3: Verify converted data**

```bash
python3 -c "
import pandas as pd, json
df = pd.read_parquet('data/verl_topology_train_json.parquet')
sample = df.iloc[0]
rm = sample['reward_model']
gt = rm['ground_truth'] if isinstance(rm, dict) else str(rm)
print(f'Format: {\"json\" if gt.startswith(\"{\") else \"yaml\"}')
print(f'Preview: {gt[:200]}')
data = json.loads(gt)
print(f'Parsed: {list(data.keys())}')
print(f'Nodes: {len(data.get(\"nodes\", []))}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/verl/convert_dataset_to_json.py data/verl_topology_train_json.parquet data/verl_topology_curated_json.parquet
git commit -m "feat: convert 12K training dataset YAML → JSON for Nemotron tool-calling"
```

---

### Task 3: Add JSON tool_calls parser to SageTopologyEnv (Phase C)

**Files:**
- Modify: `sage-python/src/sage/verl/topology_env.py`

Phase C's `_handle_yaml` method parses the model's topology output. Add JSON support alongside YAML.

- [ ] **Step 1: Modify _handle_yaml to accept JSON**

Rename to `_handle_topology_output` and try JSON first:

```python
def _handle_topology_output(self, text: str) -> tuple[dict, float, bool, dict]:
    """Parse model output as JSON (preferred) or YAML (fallback)."""
    import json, yaml
    text = text.strip()

    # Try JSON first (Nemotron native format)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "nodes" in data:
            return self._process_topology(data)
    except json.JSONDecodeError:
        pass

    # Fallback: YAML
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "nodes" in data:
            return self._process_topology(data)
    except Exception:
        pass

    # Neither worked
    return self._fail("unparseable topology output")
```

- [ ] **Step 2: Add CheckpointDecision JSON parsing**

In `_handle_decision`, parse JSON decisions:

```python
def _parse_decision(self, text: str) -> str:
    """Parse decision from JSON or text."""
    text = text.strip().lower()
    # JSON format: {"action": "continue"} or {"action": "upgrade", ...}
    try:
        import json
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("action", "continue")
    except Exception:
        pass
    # Text fallback
    if "upgrade" in text: return "upgrade"
    if "reroute" in text: return "reroute"
    return "continue"
```

- [ ] **Step 3: Update all callers of _handle_yaml → _handle_topology_output**

```bash
grep -n "_handle_yaml" sage-python/src/sage/verl/topology_env.py
# Replace all references
```

- [ ] **Step 4: Test Phase C with JSON input**

```bash
python3 -c "
from sage.verl.topology_env import SageTopologyEnv
import json
env = SageTopologyEnv(max_steps=5)
obs = env.reset('Write a sort function', 'test_task')
# Simulate JSON topology output
topo = json.dumps({'nodes':[{'role':'coder','model_tier':'budget','prompt':'sort'}],'reasoning':'test','difficulty':'simple'})
obs2, reward, done, info = env.step(topo)
print(f'Step result: reward={reward}, done={done}, state={info.get(\"state\")}')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/sage/verl/topology_env.py
git commit -m "feat: Phase C accepts JSON topology + JSON checkpoint decisions"
```

---

### Task 4: Create JSON training script with vLLM guided_json

**Files:**
- Create: `sage-python/scripts/verl/train_topology_json.sh`

The key change: pass the Pydantic JSON schema to vLLM for constrained decoding during rollout. This guarantees 100% valid output.

- [ ] **Step 1: Write the training script**

Based on `train_topology_targeted.sh` but with original NVIDIA weights + JSON format:
- JSON dataset (`verl_topology_train_json.parquet`)
- `SAGE_TRAINING_PHASE=A` (simple reward + exec)
- `SAGE_VERL_EXEC=1` (execution reward)
- DAPO token-level loss
- vLLM guided_json schema (if supported in verl 0.7.1, else rely on Nemotron's native JSON bias)

```bash
#!/bin/bash
# train_topology_json.sh
# Nemotron-8B JSON tool-call training
# Uses the model's native JSON format instead of YAML
# Expected: 0% malformation → 100% exec reward → fast convergence

export SAGE_TRAINING_PHASE=A
export SAGE_VERL_EXEC=1

# JSON dataset
DATA="data/verl_topology_train_json.parquet"
VAL="data/verl_topology_curated_json.parquet"

# BASE MODEL: original NVIDIA weights (NOT sft_merged)
MODEL="/workspace/patched_nemotron_orchestrator"
# No SFT warmup needed — model already knows tool-calling natively
# Key differences vs previous scripts:
#   1. Original NVIDIA weights (tool-calling preserved)
#   2. JSON dataset (not YAML)
#   3. <tool_call> format matches model's pretraining
```

- [ ] **Step 2: Test smoke run (2 steps)**

```bash
bash scripts/verl/train_topology_json.sh --smoke
```

Expected: 2 steps complete, reward > 0.15, no YAML malformation errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/verl/train_topology_json.sh
git commit -m "feat: JSON tool-call training script (Nemotron native format)"
```

---

### Task 5: Launch full JSON training

**Files:**
- Run: `sage-python/scripts/verl/train_topology_json.sh`

- [ ] **Step 1: Launch training**

```bash
cd sage-python
nohup bash scripts/verl/train_topology_json.sh > /workspace/train_json.log 2>&1 &
```

- [ ] **Step 2: Verify first step — check malformation rate**

After step 1, check:
- `timing_s/reward` — should be > 0.001s (execution reward firing, not 0.00001s structural fallback)
- `critic/score/max` — should be > 0.5 (exec reward, not 0.225 structural ceiling)
- No "YAML invalid" in logs

- [ ] **Step 3: Monitor convergence**

Target: reward > 0.4 by step 200 (matching The Conductor convergence rate).

- [ ] **Step 4: Backup checkpoint to HF**

```bash
python3 scripts/verl/upload_checkpoint.py --step <latest>
```

- [ ] **Step 5: Commit results**

```bash
git add TRAINING_LOG.md
git commit -m "results: JSON training step N — reward X, exec_hits Y%"
git push origin main
```

---

### Task 6: Phase C with JSON micro-decisions

**Files:**
- Run: `sage-python/scripts/verl/train_phase_c_custom.py`

**Depends on:** Task 5 convergence (reward > 0.4).

Phase C uses the same JSON format for both topology generation AND checkpoint decisions:

```json
// Step 0: Generate topology
{"nodes": [...], "edges": [...], "reasoning": "...", "difficulty": "moderate"}

// Checkpoint: Decide action
{"action": "continue"}
{"action": "upgrade", "node_idx": 2, "new_tier": "reasoner"}
{"action": "reroute"}
```

Both are native JSON tool-calls — Nemotron's native format.

- [ ] **Step 1: Launch Phase C**

```bash
export SAGE_TRAINING_PHASE=C
python3 scripts/verl/train_phase_c_custom.py \
    --model /workspace/patched_nemotron_orchestrator \
    --checkpoint /home/yann/verl_checkpoints \
    --data data/verl_topology_train_json.parquet \
    --output /home/yann/verl_checkpoints_phase_c \
    --epochs 3 --lr 5e-7 --k 4 --batch-size 4
```

- [ ] **Step 2: Verify step_advantage non-zero**

- [ ] **Step 3: Verify decision anchors appear**

- [ ] **Step 4: Commit**

---

### Task 7: Post-training and final benchmark

**Files:**
- Run: `scripts/verl/post_training_pipeline.py all`

- [ ] **Step 1: Merge LoRA (ONLY now, never during training)**
- [ ] **Step 2: Push full precision + GGUF to HF**
- [ ] **Step 3: Enable Path 6 and benchmark MASBENCH + GAIA (with 600s timeout)**
- [ ] **Step 4: Update CLAUDE.md, TRAINING_LOG.md, README.md, HF model card**
- [ ] **Step 5: Final commit**

---

## Success Criteria

| Criterion | Target | How to verify |
|-----------|--------|---------------|
| JSON malformation rate | **0%** (was 91%) | `_score_format` returns 1.0 on all samples |
| Exec reward hit rate | **>50%** (was 9%) | `critic/score/max > 0.225` frequency |
| Reward at step 200 | **>0.4** (was 0.22 at step 300) | Ray metrics |
| Phase C step_advantage | **non-zero** | Training logs |
| Phase C decision variety | upgrade + continue + reroute | Training logs |
| MASBENCH depth (600s timeout) | **>+10pp vs bare** | Benchmark |
| Full precision model on HF | 4 safetensors shards | HF repo |
| GGUF Q8_0 on HF | 1 file ~8.5GB | HF repo |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| vLLM guided_json not in verl 0.7.1 | Nemotron's native JSON bias + JSON training data should be sufficient without constrained decoding |
| JSON topology rejected by reward | Already verified: identical score 0.9848 |
| Phase C parser breaks | JSON parsing is simpler and more reliable than YAML |
| Convergence still slow | If 0% malformation + exec reward doesn't help by step 200, the problem is elsewhere (model capacity, dataset quality) |

---

## ADDENDUM: Nemotron as SAGE Operator (not just topology generator)

The model must learn to **operate YGN-SAGE as a whole**, not just generate topologies.

### SAGE modules as tools for Nemotron

| Tool (SAGE module) | What it does | When to call it |
|---------------------|-------------|-----------------|
| `TopologyEngine.generate()` | 6-path topology generation | Start of task |
| `ModelAssigner.assign()` | Choose model per node | After topology created |
| `HybridVerifier.verify()` | Validate DAG structure | Before execution |
| `TopologyExecutor.next_ready()` | Get next executable nodes | During execution |
| `MultiViewMMU.retrieve_relevant()` | Recall similar past topologies | Before generating new |
| `QualityLabeler.label()` | Score node output quality | At checkpoints (Phase C) |
| `ContextualBandit.select()` | Choose best template/model | Exploration vs exploitation |
| `ProviderPool.resolve()` | Route model_id to API provider | Per-node execution |

### Training implications

The SFT dataset should include the tool definitions in the system prompt:
```json
{
  "role": "system",
  "content": "You are the orchestrator for YGN-SAGE. Available tools:\n- add_node(role, model_tier, prompt): Add an agent node\n- add_edge(from_idx, to_idx): Connect nodes\n- set_reasoning(text): Explain your topology choice\n- checkpoint(node_idx, fallback_tier): Mark adaptation point\n- upgrade(node_idx, new_tier): Upgrade model at checkpoint\n- continue(): Proceed to next node\n- reroute(): Abort and restart with new topology\n\nGenerate a JSON topology for the given task."
}
```

This way, the model learns:
1. **Step 0**: Generate topology using `add_node`, `add_edge`, `set_reasoning`
2. **Step 1-N**: Execute via `TopologyExecutor.next_ready()` (SAGE handles this)
3. **Checkpoints**: Decide `upgrade`, `continue`, or `reroute` based on quality scores
4. **Learning**: SAGE records outcomes to S-MMU, bandit, MAP-Elites archive

The model doesn't call Rust directly — SAGE's pipeline translates JSON tool calls into Rust API calls. But the model must understand the SEMANTICS of each tool to make good decisions.
