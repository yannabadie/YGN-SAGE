#!/usr/bin/env python3
"""Upload tool-call SFT model + training data to HuggingFace.

Creates repo yannabadie/sage-topology-policy-local with:
- LoRA adapter (132 MB)
- Training data (topology_sft_v2_toolcall.jsonl)
- Adaptation data (adapt_decisions_toolcall.jsonl)
- Expert topologies (expert_topologies.jsonl)
- Tool schemas (sage_tool_schemas.py)
- Training metrics (sft_metrics.jsonl)
- Model card with full documentation

Usage:
    python scripts/upload_to_hf.py
"""
import os
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)

from huggingface_hub import HfApi, create_repo

REPO_ID = "yannabadie/sage-topology-policy-local"
ADAPTER_DIR = "models/toolcall_qwen3_4b/sft_checkpoint"

MODEL_CARD = """---
library_name: peft
license: mit
base_model: Qwen/Qwen3-4B
tags:
  - tool-calling
  - topology
  - multi-agent
  - sage
  - lora
  - ygn-sage
language:
  - en
pipeline_tag: text-generation
---

# SAGE Topology Policy — Local (Qwen3-4B + LoRA)

**Trained locally on RTX 3500 Ada 12GB** for the [YGN-SAGE](https://github.com/yannabadie/YGN-SAGE) multi-agent orchestration framework.

## What it does

Generates multi-agent DAG topologies in `<tool_call>` JSON format. Given a coding task, the model decides:
- How many agent nodes (1-7)
- What role per node (coder, reviewer, planner, synthesizer...)
- What model tier (budget, fast, balanced, reasoner, codex)
- How to connect them (message, control, state edges)
- Where to place adaptation checkpoints

## Format

The model outputs `<tool_call>` JSON (Qwen3 native format):

```xml
<tool_call>
{"name": "create_topology", "arguments": {
  "difficulty": "moderate",
  "reasoning": "Multi-step code task needs coder + reviewer",
  "nodes": [
    {"role": "coder", "model_tier": "codex", "prompt": "..."},
    {"role": "reviewer", "model_tier": "fast", "prompt": "..."}
  ],
  "edges": [{"from_idx": 0, "to_idx": 1, "flow_type": "message"}]
}}
</tool_call>
```

## 2 SAGE Tools

1. **create_topology** — Design multi-agent DAG (Phase A/B)
2. **adapt_topology** — Runtime adaptation decisions: continue/upgrade/reroute (Phase C)

## Results

| Metric | YAML Baseline | Tool-Call SFT |
|--------|--------------|---------------|
| N1 avg reward | 0.391 | **0.865 (+121%)** |
| N1 max reward | 0.987 | **1.024** |
| P(reward > 0.3) | 26% | **90%** |
| Simple | 0.567 | **0.780** |
| Moderate | 0.441 | **0.949** |
| Complex | 0.148 | **0.837 (+567%)** |
| SFT loss | 0.92 | **0.225 (4.1x better)** |

## Training Details

- **Base model**: Qwen/Qwen3-4B (4-bit NF4 quantization)
- **Method**: LoRA (rank 32, alpha 64) via TRL SFTTrainer
- **Data**: 1880 tool-call topologies from GPT-5.4 distillation
- **Format**: `<tool_call>` JSON with 2 SAGE tool definitions in system prompt
- **Hardware**: RTX 3500 Ada 12GB, Windows, ~65 min training
- **Epochs**: 2, lr=2e-5, cosine schedule

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
    quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(model, "yannabadie/sage-topology-policy-local")

messages = [
    {"role": "system", "content": "<system prompt with tool definitions>"},
    {"role": "user", "content": "Write a function that sorts a list using merge sort."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:]))
```

## Training Data

Included in this repo:
- `topology_sft_v2_toolcall.jsonl` — 1880 SFT examples
- `adapt_decisions_toolcall.jsonl` — 5139 adaptation decisions for Phase C
- `expert_topologies.jsonl` — 8 Claude Opus 4.6 distilled examples
- `sage_tool_schemas.py` — Tool definitions + system prompt

## Part of YGN-SAGE

[GitHub](https://github.com/yannabadie/YGN-SAGE) | [PyPI](https://pypi.org/project/ygn-sage/) | MIT License
"""


def main():
    api = HfApi()

    # Create repo
    print(f"Creating repo {REPO_ID}...")
    create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)

    # Upload model card
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )

    # Upload adapter files (only the final checkpoint, not sub-checkpoints)
    adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    ]
    for fname in adapter_files:
        fpath = os.path.join(ADAPTER_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"Uploading {fname} ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=REPO_ID,
            )

    # Upload training data
    data_files = [
        ("data/topology_sft_v2_toolcall.jsonl", "training_data/topology_sft_v2_toolcall.jsonl"),
        ("data/adapt_decisions_toolcall.jsonl", "training_data/adapt_decisions_toolcall.jsonl"),
        ("data/expert_topologies.jsonl", "training_data/expert_topologies.jsonl"),
    ]
    for local, remote in data_files:
        if os.path.exists(local):
            size_mb = os.path.getsize(local) / 1024 / 1024
            print(f"Uploading {local} ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=REPO_ID,
            )

    # Upload tool schemas
    print("Uploading sage_tool_schemas.py...")
    api.upload_file(
        path_or_fileobj="scripts/sage_tool_schemas.py",
        path_in_repo="sage_tool_schemas.py",
        repo_id=REPO_ID,
    )

    # Upload metrics
    metrics_path = "models/toolcall_qwen3_4b/sft_metrics.jsonl"
    if os.path.exists(metrics_path):
        print("Uploading sft_metrics.jsonl...")
        api.upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo="sft_metrics.jsonl",
            repo_id=REPO_ID,
        )

    # Upload N1 eval results
    n1_path = "experiments/n1_toolcall_sft.json"
    if os.path.exists(n1_path):
        print("Uploading n1_eval_results.json...")
        api.upload_file(
            path_or_fileobj=n1_path,
            path_in_repo="n1_eval_results.json",
            repo_id=REPO_ID,
        )

    print(f"\nDone! https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
