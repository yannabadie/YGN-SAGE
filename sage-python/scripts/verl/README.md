# veRL + GiGPO Training for YGN-SAGE Topology

## Overview

Train a topology generation model (Qwen3-8B/9B) via GRPO/GiGPO on RunPod H100.
The model learns to generate optimal multi-agent YAML topologies for coding tasks.

## Architecture

```
RunPod H100 (80GB)                    Local Ada 3500 (12GB)
┌─────────────────────┐               ┌─────────────────────┐
│ veRL + vLLM          │               │ Quantized inference  │
│ Qwen3-8B bf16        │  ──export──►  │ Qwen3-8B-GPTQ 4-bit │
│ LoRA r=32            │               │ ~5GB VRAM            │
│ GiGPO/GRPO training  │               │ SAGE pipeline Path 6 │
└─────────────────────┘               └─────────────────────┘
```

## Quick Start (RunPod)

```bash
# 1. Clone repo
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE
git checkout VeRLGIGPO

# 2. Setup environment (installs veRL, vLLM, sage-core, converts data)
bash sage-python/scripts/verl/setup_runpod.sh

# 3. Copy training data (from local machine)
# scp data/topology_sft_v2_combined.jsonl runpod:/workspace/YGN-SAGE/sage-python/data/

# 4. Train
bash sage-python/scripts/verl/train_topology.sh

# 5. Export for local inference
python scripts/verl/export_for_local.py
```

## Files

| File | Purpose |
|------|---------|
| `setup_runpod.sh` | One-shot environment setup |
| `train_topology.sh` | Training launch script (veRL + GRPO/GiGPO) |
| `convert_sft_to_verl.py` | JSONL → veRL parquet conversion |
| `reward_topology.py` | Combined reward function (format + structure + Rust density) |
| `export_for_local.py` | Export LoRA adapter for local 12GB inference |

## Training Data

Source: `data/topology_sft_v2_combined.jsonl` (1880 entries)
- 60.6% BigCodeBench
- 20.0% GSM8K
- 10.6% RAFT (execution-verified)
- 8.8% CodeContests

## Reward Function

Combined reward = (format + structure + execution_proxy) / 3

- **format**: YAML validity [-2.0, +1.0]
- **structure**: nodes, edges, roles, reasoning [0.0, 1.0]
- **execution_proxy**: Rust TopologyReward (structural + density + verification) [0.0, 1.0]

Note: Full TopologyRunner execution (with LLM agent calls) is used for EVALUATION,
not during training. Training uses the Rust structural proxy for speed.

## Model

- **Training**: Qwen3-8B (or Qwen3.5-9B when available) in bf16 with LoRA r=32
- **Local inference**: GPTQ/AWQ 4-bit quantized version (~5GB VRAM)
- **Fallback**: Phi-4-mini-instruct 3.8B (current, works on 12GB)

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 1e-6 | AgentConductor |
| LoRA rank | 32 | veRL default for 8B |
| Batch size | 16 | veRL default |
| K (generations/prompt) | 8 | AgentConductor |
| Temperature | 0.4 | Phase 1 validated |
| KL coef (beta) | 0.04 | Phase 2 validated |
| Max response length | 512 | Topology fits in ~300 tokens |
| Epochs | 3 | Verl Engineering Handbook |
