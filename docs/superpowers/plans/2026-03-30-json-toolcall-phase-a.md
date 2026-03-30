# Phase A: JSON Tool-Call SFT Warmup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the local training pipeline from YAML to `<tool_call>` JSON format and run SFT warmup with Qwen3-4B-Instruct.

**Architecture:** Define 7 SAGE tool schemas in a shared module. Convert 1880 SFT examples to `<tool_call>` format with tool definitions baked in system prompt. Use Qwen3-4B-Instruct's native chat template (no custom override). SFT warmup then N1 eval.

**Tech Stack:** Python 3.12, TRL 0.29.1 SFTTrainer, PEFT LoRA, bitsandbytes NF4, Qwen/Qwen3-4B-Instruct

---

### File Structure

```
sage-python/
  scripts/
    sage_tool_schemas.py           # NEW: 7 SAGE tool definitions + system prompt
    convert_sft_to_toolcall.py     # NEW: convert SFT data to <tool_call> format
    train_local_qwen3_4b.py        # MODIFY: use Instruct model, native chat template
    eval_reward_holdout.py          # MODIFY: use Instruct model, native chat template
    create_holdout.py               # MODIFY: generate tool-call holdout
  data/
    topology_sft_v2_toolcall.jsonl  # NEW: 1880 entries in <tool_call> format
  experiments/
    holdout_50_toolcall.json        # NEW: holdout in tool-call format
    configs/
      toolcall_sft_full.json        # NEW: SFT config for full 1880 examples
```

---

### Task 1: SAGE Tool Schemas Module

**Files:**
- Create: `sage-python/scripts/sage_tool_schemas.py`

- [ ] **Step 1: Write the tool schemas module**

```python
#!/usr/bin/env python3
"""7 SAGE tool schemas for tool-call training.

Defines the JSON function schemas that go into the <tools> block
of the system prompt. Matches the Rust+Python SAGE pipeline.
"""
import json

SAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_topology",
            "description": "Design a multi-agent DAG topology to solve a coding task. Choose nodes (role + model_tier + prompt), edges (flow_type), and difficulty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {"type": "string", "enum": ["simple", "moderate", "complex"]},
                    "reasoning": {"type": "string", "description": "Why this topology is optimal for the task"},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "description": "Agent role (coder, reviewer, planner, synthesizer, etc.)"},
                                "model_tier": {"type": "string", "enum": ["budget", "fast", "balanced", "reasoner", "codex"]},
                                "prompt": {"type": "string", "description": "System prompt for this agent node"},
                            },
                            "required": ["role", "model_tier", "prompt"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from_idx": {"type": "integer"},
                                "to_idx": {"type": "integer"},
                                "flow_type": {"type": "string", "enum": ["message", "control", "state"]},
                            },
                            "required": ["from_idx", "to_idx", "flow_type"],
                        },
                    },
                },
                "required": ["difficulty", "reasoning", "nodes", "edges"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_task",
            "description": "Classify task complexity as S1 (simple), S2 (moderate), or S3 (complex) using kNN routing (92% accuracy).",
            "parameters": {
                "type": "object",
                "properties": {
                    "system": {"type": "string", "enum": ["S1", "S2", "S3"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"},
                },
                "required": ["system", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_models",
            "description": "Map model_tier to real model from cards.toml (affinity 0.4 + domain 0.4 + cost 0.2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_idx": {"type": "integer"},
                                "model_id": {"type": "string"},
                                "provider": {"type": "string"},
                            },
                            "required": ["node_idx", "model_id", "provider"],
                        },
                    },
                },
                "required": ["assignments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_topology",
            "description": "Run HybridVerifier (Rust Z3/OxiZ + LTL temporal checks) on the topology.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checks": {"type": "array", "items": {"type": "string", "enum": ["reachability", "acyclicity", "role_coverage", "budget_constraint"]}},
                },
                "required": ["checks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_topology",
            "description": "Runtime adaptation: upgrade model_tier, reroute to different node, or continue execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["continue", "upgrade", "reroute"]},
                    "node_idx": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run code in 3-layer sandbox (tree-sitter → Wasm WASI → subprocess).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string", "enum": ["python", "javascript", "rust"]},
                    "timeout_sec": {"type": "integer", "default": 30},
                },
                "required": ["code", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_memory",
            "description": "S-MMU operations: write to STM, read from episodic/semantic, evict stale entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["write", "read", "evict", "consolidate"]},
                    "tier": {"type": "string", "enum": ["stm", "episodic", "semantic", "causal"]},
                    "content": {"type": "string"},
                },
                "required": ["operation", "tier"],
            },
        },
    },
]

TOOLS_JSON = json.dumps(SAGE_TOOLS, indent=2)

TOOLCALL_SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "You have access to 7 tools that control the SAGE pipeline: topology creation, "
    "task routing, model assignment, verification, adaptation, code execution, and memory management.\n\n"
    f"<tools>\n{TOOLS_JSON}\n</tools>\n\n"
    "For each task, call the appropriate tool(s) using <tool_call> JSON format. "
    "Always start by calling create_topology to design the agent DAG."
)


def wrap_toolcall(topology_dict: dict) -> str:
    """Wrap a topology dict as a <tool_call> string."""
    call = {"name": "create_topology", "arguments": topology_dict}
    return f"<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>"
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `cd sage-python && python -c "from scripts.sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT, wrap_toolcall, SAGE_TOOLS; print(f'{len(SAGE_TOOLS)} tools, prompt={len(TOOLCALL_SYSTEM_PROMPT)} chars')"`

Expected: `7 tools, prompt=~3500 chars`

- [ ] **Step 3: Commit**

```bash
git add scripts/sage_tool_schemas.py
git commit -m "feat: 7 SAGE tool schemas for tool-call training"
```

---

### Task 2: Convert SFT Data to Tool-Call Format

**Files:**
- Create: `sage-python/scripts/convert_sft_to_toolcall.py`
- Output: `sage-python/data/topology_sft_v2_toolcall.jsonl`

- [ ] **Step 1: Write the conversion script**

```python
#!/usr/bin/env python3
"""Convert SFT topology data to <tool_call> JSON format.

Wraps each topology in <tool_call>{"name": "create_topology", "arguments": ...}</tool_call>
and bakes the 7 SAGE tool definitions into the system prompt.

Usage:
    python scripts/convert_sft_to_toolcall.py
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT, wrap_toolcall


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to tool-call format")
    parser.add_argument("--input", default="data/topology_sft_v2_combined.jsonl")
    parser.add_argument("--output", default="data/topology_sft_v2_toolcall.jsonl")
    args = parser.parse_args()

    count = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            topology = entry.get("topology")
            if not topology or not isinstance(topology, dict):
                continue

            toolcall_text = wrap_toolcall(topology)

            new_entry = {
                "task_id": entry.get("task_id", ""),
                "prompt": entry["prompt"],
                "topology": topology,
                "topology_toolcall": toolcall_text,
                "system_prompt": TOOLCALL_SYSTEM_PROMPT,
                "node_count": entry.get("node_count", len(topology.get("nodes", []))),
                "edge_count": entry.get("edge_count", len(topology.get("edges", []))),
                "difficulty": entry.get("difficulty", "simple"),
                "model": entry.get("model", "converted"),
            }
            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} entries to tool-call format")
    print(f"Output: {args.output}")

    # Verify first entry
    with open(args.output, encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert "<tool_call>" in first["topology_toolcall"]
    assert "create_topology" in first["topology_toolcall"]
    print("Verification OK: <tool_call> format valid")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run conversion**

Run: `cd sage-python && python scripts/convert_sft_to_toolcall.py`

Expected: `Converted 1880 entries to tool-call format` + `Verification OK`

- [ ] **Step 3: Commit**

```bash
git add -f scripts/convert_sft_to_toolcall.py data/topology_sft_v2_toolcall.jsonl
git commit -m "feat: 1880 SFT entries converted to <tool_call> JSON format"
```

---

### Task 3: Update Training Script for Tool-Call + Instruct

**Files:**
- Modify: `sage-python/scripts/train_local_qwen3_4b.py`

Three changes: (a) new system prompt, (b) support `topology_toolcall` field, (c) default model → Instruct, (d) remove custom chat template.

- [ ] **Step 1: Update SYSTEM_PROMPT and load_sft_dataset**

In `train_local_qwen3_4b.py`, replace the SYSTEM_PROMPT and load_sft_dataset:

Replace the current SYSTEM_PROMPT (lines 47-54) with:

```python
# System prompt loaded from tool schemas (includes 7 SAGE tool definitions)
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT as SYSTEM_PROMPT
```

Replace `load_sft_dataset` to support tool-call format:

```python
def load_sft_dataset(path: str, max_samples: int = 0):
    """Load SFT data from JSONL → messages format for TRL SFTTrainer.

    Supports: topology_toolcall (preferred) > topology_json > topology_yaml.
    """
    from datasets import Dataset

    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry.get("prompt", "")
            topology_text = (entry.get("topology_toolcall")
                           or entry.get("topology_json")
                           or entry.get("topology_yaml", ""))
            if not prompt or not topology_text:
                continue
            # Use entry's system_prompt if available (has tool definitions)
            sys_prompt = entry.get("system_prompt", SYSTEM_PROMPT)
            messages.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": topology_text},
            ])

    if max_samples > 0:
        messages = messages[:max_samples]

    log.info("Loaded %d SFT examples from %s", len(messages), path)
    return Dataset.from_dict({"messages": messages})
```

- [ ] **Step 2: Change default model to Instruct and remove custom chat template**

Change the argparse default:

```python
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct",
```

In `load_model()`, remove the entire chat template block (lines ~143-171 starting with `# Set chat template`). Qwen3-4B-Instruct has its own native template with tool-call support. Replace with:

```python
    # Qwen3-4B-Instruct has native tool-call chat template — do NOT override
    if tokenizer.chat_template is None:
        log.warning("No chat template found, model may not support tool-call format")
```

- [ ] **Step 3: Verify the script loads**

Run: `cd sage-python && HF_HUB_OFFLINE=1 python -c "from scripts.train_local_qwen3_4b import SYSTEM_PROMPT; print(len(SYSTEM_PROMPT), 'chars'); print('<tools>' in SYSTEM_PROMPT)"`

Expected: `~3500 chars` and `True`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_local_qwen3_4b.py
git commit -m "feat: training script uses tool-call format + Qwen3-4B-Instruct"
```

---

### Task 4: Update N1 Evaluator for Tool-Call + Instruct

**Files:**
- Modify: `sage-python/scripts/eval_reward_holdout.py`

- [ ] **Step 1: Update SYSTEM_PROMPT and remove custom chat template**

Replace SYSTEM_PROMPT import at top:

```python
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT as SYSTEM_PROMPT
```

Change default model:

```python
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
```

Remove the entire custom chat template block in `main()` (the `tokenizer.chat_template = (...)` block). Replace with:

```python
    # Qwen3-4B-Instruct has native tool-call chat template — do NOT override
```

- [ ] **Step 2: Commit**

```bash
git add scripts/eval_reward_holdout.py
git commit -m "feat: N1 eval uses tool-call system prompt + Qwen3-4B-Instruct"
```

---

### Task 5: Create Tool-Call Holdout Set

**Files:**
- Create: `sage-python/experiments/holdout_50_toolcall.json`

- [ ] **Step 1: Generate holdout from tool-call data**

```bash
cd sage-python
python scripts/create_holdout.py
```

But first update `create_holdout.py` to read from tool-call data:

Replace in `create_holdout.py`:
- `SFT_DATA = "data/topology_sft_v2_combined.jsonl"` → `SFT_DATA = "data/topology_sft_v2_toolcall.jsonl"`
- `OUTPUT = "experiments/holdout_50.json"` → `OUTPUT = "experiments/holdout_50_toolcall.json"`
- In the entry dict, add: `"reference_toolcall": entry.get("topology_toolcall", ""),`
- Keep `"reference_yaml"` as `entry.get("topology_yaml", entry.get("topology_toolcall", ""))`

Run: `python scripts/create_holdout.py`

Expected: `Created experiments/holdout_50_toolcall.json: 50 prompts`

Update `eval_reward_holdout.py` default: `HOLDOUT_PATH = "experiments/holdout_50_toolcall.json"`

- [ ] **Step 2: Commit**

```bash
git add scripts/create_holdout.py experiments/holdout_50_toolcall.json scripts/eval_reward_holdout.py
git commit -m "feat: tool-call holdout set (50 prompts with SAGE tool definitions)"
```

---

### Task 6: Create Config + Download Model + Run SFT

**Files:**
- Create: `sage-python/experiments/configs/toolcall_sft_full.json`

- [ ] **Step 1: Create the SFT config**

```json
{
  "model": "Qwen/Qwen3-4B-Instruct",
  "sft_data": "data/topology_sft_v2_toolcall.jsonl",
  "sft_epochs": 2,
  "sft_lr": 2e-5,
  "sft_max_samples": 0,
  "output": "models/toolcall_qwen3_4b",
  "lora_rank": 32
}
```

- [ ] **Step 2: Download Qwen3-4B-Instruct (first time only)**

```bash
cd sage-python
# Temporarily disable offline to download
HF_HUB_OFFLINE=0 python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct'); print('Downloaded')"
```

Check disk: `df -h /c` — need ~7.6 GB free for the model.

- [ ] **Step 3: Run SFT warmup**

```bash
nvidia-smi -lgc 3105
HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --config experiments/configs/toolcall_sft_full.json --sft-only
```

Expected: ~30-60 min, loss should drop from ~3.0 to <1.5.
Monitor: `cat models/toolcall_qwen3_4b/sft_metrics.jsonl`

- [ ] **Step 4: Commit**

```bash
git add experiments/configs/toolcall_sft_full.json
git commit -m "feat: tool-call SFT config + warmup complete"
```

---

### Task 7: N1 Eval on Tool-Call SFT Checkpoint

**Files:** None new — runs existing eval.

- [ ] **Step 1: Run N1 evaluation**

```bash
nvidia-smi -lgc 3105
HF_HUB_OFFLINE=1 python -u scripts/eval_reward_holdout.py \
  --adapter models/toolcall_qwen3_4b/sft_checkpoint \
  --output experiments/n1_toolcall_sft.json
```

Expected: ~10 min (batched), results in JSON.

- [ ] **Step 2: Compare with YAML baseline**

```bash
python -c "
import json
yaml_bl = {'avg': 0.391, 'max': 0.987, 'above_03': 0.26, 'clipped': 0.42}
tc = json.load(open('experiments/n1_toolcall_sft.json'))
print('=== YAML baseline vs Tool-Call SFT ===')
for k in ['n1_reward_avg', 'n1_reward_max', 'n1_above_03', 'n1_clipped_ratio']:
    short = k.replace('n1_','')
    bl = yaml_bl.get(short.replace('reward_','').replace('_ratio',''), '?')
    print(f'  {short}: {bl} → {tc[k]:.4f}')
"
```

- [ ] **Step 3: Record in journal and commit**

```bash
python scripts/autoresearch_loop.py --eval-only \
  --adapter models/toolcall_qwen3_4b/sft_checkpoint \
  --hypothesis "Tool-call JSON SFT on Qwen3-4B-Instruct (1880 examples, 7 SAGE tools)" \
  --phase "A-toolcall-sft"

git add experiments/
git commit -m "metrics: Phase A tool-call SFT — N1 eval on Qwen3-4B-Instruct"
git push origin local
```

---

### Summary: Phase A Exit Criteria

| Criteria | How to verify |
|----------|---------------|
| 7 tool schemas defined | `python -c "from scripts.sage_tool_schemas import SAGE_TOOLS; print(len(SAGE_TOOLS))"` → 7 |
| 1880 tool-call entries | `wc -l data/topology_sft_v2_toolcall.jsonl` → 1880 |
| Qwen3-4B-Instruct loaded | Model downloads + SFT starts without error |
| SFT loss < 1.5 | Check `models/toolcall_qwen3_4b/sft_metrics.jsonl` |
| N1 avg > 0.391 (YAML baseline) | Check `experiments/n1_toolcall_sft.json` |
| N1 clipped < 0.42 (YAML baseline) | Check `experiments/n1_toolcall_sft.json` |
| Tool-call format in output | Model generates `<tool_call>` JSON, not YAML |
