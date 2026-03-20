# veRL GRPO Training for YGN-SAGE Topology

## Overview

Train Qwen3.5-9B via GRPO on RunPod H100 to generate optimal multi-agent YAML topologies.
Export LoRA adapter for local 12GB inference via AWQ 4-bit.

## Architecture

```
RunPod H100 (80GB)                          Local Ada 3500 (12GB)
┌──────────────────────────┐                ┌──────────────────────────┐
│ Docker: verlai/verl:      │                │ cyankiwi/Qwen3.5-9B-     │
│   vllm017.latest          │                │   AWQ-4bit (~5GB)        │
│ Qwen/Qwen3.5-9B bf16     │  ── export ──► │ LoRA adapter merged      │
│ LoRA r=64, GRPO           │                │ SAGE pipeline Path 6     │
│ vLLM rollout, 8 samples   │                │ TopologyEngine inference │
└──────────────────────────┘                └──────────────────────────┘
```

## Quick Start (RunPod)

```bash
# 1. Create RunPod pod
#    - GPU: H100 80GB (or A100 80GB)
#    - Docker image: verlai/verl:vllm017.latest
#    - Disk: 100GB+

# 2. SSH into pod, clone repo
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE && git checkout VeRLGIGPO

# 3. Upload training data (from local machine)
scp sage-python/data/topology_sft_v2_combined.jsonl <pod>:/workspace/YGN-SAGE/sage-python/data/

# 4. Setup environment (~5 min)
bash sage-python/scripts/verl/setup_runpod.sh

# 5. Train (~2-4h on H100)
cd sage-python && bash scripts/verl/train_topology.sh

# 6. Export for local inference
python scripts/verl/export_for_local.py --checkpoint models/topology_verl/
# Then scp the adapter back to local machine
```

## Why GRPO, not GiGPO?

GiGPO (Group-in-Group Policy Optimization) uses step-level advantage grouping
designed for **multi-turn** agentic environments (ALFWorld, WebShop). Our topology
generation is **single-turn** (1 prompt → 1 YAML). For single-turn, GiGPO
degenerates to GRPO — the step-level advantage collapses. So we use standard GRPO.

Ref: GiGPO paper (arXiv 2505.10978), Section 3.2

## Files

| File | Purpose |
|------|---------|
| `setup_runpod.sh` | One-shot environment setup on RunPod |
| `train_topology.sh` | Training launch (veRL GRPO, Qwen3.5-9B) |
| `convert_sft_to_verl.py` | SAGE JSONL → veRL parquet conversion |
| `reward_topology.py` | Combined reward (format + structure + Rust density) |
| `export_for_local.py` | Export LoRA for local 12GB inference |

## Training Config

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | `Qwen/Qwen3.5-9B` | 9B, Apache 2.0, hybrid attention |
| LoRA rank | 64 | veRL default for 9B |
| LoRA alpha | 32 | alpha = rank/2 convention |
| Learning rate | 1e-6 | AgentConductor (2602.17100) |
| K (samples/prompt) | 8 | AgentConductor |
| Temperature | 0.4 | Phase 1 validated |
| KL coef | 0.04 | Phase 2 validated |
| Max response | 512 tokens | Topologies fit ~300 tokens |
| Batch size | 256 | veRL default, scaled for 1 GPU |
| Epochs | 15 | Verl Engineering Handbook |
| Ref model | CPU offload | `ref.fsdp_config.param_offload=True` |

## VRAM Budget (H100 80GB)

```
Model bf16:       ~18 GB
LoRA r=64:        ~0.2 GB
Ref model:        CPU (offloaded)
vLLM KV cache:    ~48 GB (gpu_memory_utilization=0.6)
Optimizer:        ~0.8 GB (LoRA params only)
Activations:      ~4 GB (gradient checkpointing)
────────────────────────
Total:            ~71 GB ✓ (fits 80GB)
```

## Reward Function

```
compute_score(data_source, solution_str, ground_truth, extra_info) -> float

= (format_norm + structure_norm + execution_proxy_norm) / 3

format:          YAML validity [-2.0, +1.0] → normalized [0, 1]
structure:       nodes, edges, roles, reasoning [0, 1]
execution_proxy: Rust TopologyReward (structural + density) [0, 1]
                 Per-difficulty penalty: tanh((N_max - |V|) / N_max)
```

## Local Inference (12GB)

After exporting the LoRA adapter from H100:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Option A: AWQ 4-bit base + LoRA (recommended, ~5GB)
base = AutoModelForCausalLM.from_pretrained(
    "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "models/topology_verl_local/")

# Option B: GGUF via llama.cpp/Ollama (simpler, no Python)
# Download: unsloth/Qwen3.5-9B-GGUF Q4_K_M (5.68 GB)
```

## Data

Source: `data/topology_sft_v2_combined.jsonl` (1880 entries)
- 60.6% BigCodeBench (code tasks)
- 20.0% GSM8K (math tasks, downsampled)
- 10.6% RAFT (execution-verified from Phase 2)
- 8.8% CodeContests (competitive programming)

Converted to veRL parquet via `convert_sft_to_verl.py`.
