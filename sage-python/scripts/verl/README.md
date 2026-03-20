# veRL GRPO Training for YGN-SAGE Topology

## Overview

Train Qwen3.5-9B via GRPO on RunPod H100 to generate optimal multi-agent YAML topologies.
Mixed reward: structural (format/density) + execution (real multi-provider TopologyRunner).
Edge-level credit assignment via Graph-GRPO (arXiv 2603.02701).

**See `RUNPOD_PLAN.md` at repo root for the step-by-step execution guide.**

## Architecture

```
RunPod H100 (80GB)                          Local Ada 3500 (12GB)
┌──────────────────────────┐                ┌──────────────────────────┐
│ Docker: verlai/verl:      │                │ Qwen3.5-9B AWQ-4bit     │
│   vllm017.latest          │                │   (~5GB VRAM)           │
│ Qwen/Qwen3.5-9B bf16     │  ── export ──► │ LoRA adapter merged      │
│ LoRA r=64, GRPO           │                │ SAGE pipeline Path 6     │
│ K=4 samples, 10 epochs    │                │ TopologyEngine inference │
│ Multi-provider execution  │                │ 8 providers available    │
└──────────────────────────┘                └──────────────────────────┘
```

## Training Strategy (Mixed Reward)

| Phase | Epochs | Reward | Cost |
|-------|--------|--------|------|
| **A: Structural** | 5 | YAML format + structure + Rust density | $0 API |
| **B: Execution** | 5 | Real topology execution via ProviderPool (multi-provider) | ~$60-80 API |

Phase B uses `TopologyRunner` with `ProviderPool.resolve()` — each node is executed
by its assigned provider (DeepSeek Chat, Gemini Flash, Grok, etc.). The model learns
that assigning the right model_tier to the right node MATTERS.

Set `SAGE_VERL_EXEC=1` to activate execution mode (done automatically by train_topology.sh).

## Why GiGPO (not GRPO standard)

Our topology execution IS multi-step: the model generates YAML at step 0, then
the environment executes each node sequentially (steps 1..N). Each step has its
own reward and anchor state. GiGPO compares actions at identical anchor states
across trajectories — this provides temporal credit assignment that flat GRPO
cannot: "when a reviewer sees good coder output (same anchor), does it produce
better feedback?"

Ref: GiGPO (arXiv 2505.10978), verl-agent (langfengQ/verl-agent)

## Files

| File | Purpose |
|------|---------|
| `setup_runpod.sh` | 8-step environment setup (vLLM check, veRL install, sage-core build) |
| `train_topology.sh` | Mixed training: Phase A structural + Phase B execution |
| `convert_sft_to_verl.py` | Auto-loads 10 data sources → veRL parquet (1965 entries) |
| `curate_training_data.py` | Curate 499 diverse prompts (GSM8K capped, GPT-5.4 Pro prioritized) |
| `validate_setup.py` | 8-point pod validation |
| `benchmark_post_train.py` | Post-training BigCodeBench eval with trained model |
| `export_for_local.py` | Export LoRA adapter for local 12GB inference |
| *(deleted)* | Legacy TRL reward removed — single reward in `sage.verl.reward` |

## Training Config

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | `Qwen/Qwen3.5-9B` | Dense 9B, Apache 2.0, hybrid GatedDeltaNet |
| Docker | `verlai/verl:vllm017.latest` | CUDA 12.9.1, vLLM 0.17, PyTorch 2.10 |
| LoRA rank | 64 | veRL Engineering Handbook |
| K (samples/prompt) | 4 | Cost-optimized (was 8) |
| Temperature | 0.4 | Phase 1 validated |
| KL coef | 0.04 | Phase 2 validated |
| Batch size | 64 | Single GPU H100 |
| gpu_memory_utilization | 0.7 | Conservative (0.8 risks OOM) |
| num_speculative_tokens | 0 | vLLM 0.17 CUDA bug workaround (#36408) |
| Agent provider | DeepSeek Chat V3.2 | $0.28/$0.42, no rate limits, no CoT waste |
| Fallback provider | Gemini 3 Flash | Rate-limited but fast per call |

## Reward Function

```
sage.verl.reward.compute_score(data_source, solution_str, ground_truth, extra_info)

SAGE_VERL_EXEC=0 (Phase A):
  = (format_norm + structure + rust_density) / 3
  Format:    YAML validity [-2.0, +1.0] → normalized [0, 1]
  Structure: nodes, edges, roles, reasoning [0, 1]
  Density:   Rust TopologyReward + per-difficulty penalty [0, 1]

SAGE_VERL_EXEC=1 (Phase B):
  = 0.3 × structural + 0.7 × execution_score
  Execution: TopologyRunner + ProviderPool (multi-provider) + sandbox testing
  Graduated: PASSED=1.5, WRONG_ANSWER=1.0, RUNTIME_ERROR=0.7, RUNS_OK=0.3
  + Graph-GRPO edge-level credit (sage.verl.edge_credit)
```

## Data (1965 entries, all in repo)

| Source | Entries | Type |
|--------|---------|------|
| SFT v2 combined | 1532 | BigCodeBench, GSM8K, CodeContests |
| RAFT Phase 2 | 199 | Execution-verified |
| GPT-5.4 Pro complex | 144 | 5-7 node topologies |
| GPT-5.4 Pro CF/GCJ | 20 | Real Codeforces/Google Code Jam |
| GPT-5.4 Pro corrections | 20 | Error→correction pairs (2nd-turn) |
| GPT-5.4 Pro deep reasoning | 20 | Chain-of-thought topologies |
| GPT-5.4 Pro simple | 20 | Calibrated 2-3 node |
| GPT-5.4 Pro audit | 10 | Improved from existing |

Curated subset: 499 entries in `verl_topology_curated.parquet` (GSM8K capped at 50).

## Providers (8 active)

| Provider | Models | Pricing |
|----------|--------|---------|
| Google | gemini-3.1-pro, flash-lite, 3-flash | $0.25-$12/M |
| OpenAI | gpt-5.4, 5.4-mini, 5.4-nano | $0.20-$15/M |
| DeepSeek | deepseek-chat (primary), reasoner | $0.28/$0.42 |
| xAI | grok-4-1-fast (2M context) | $0.20/$0.50 |
| MiniMax | minimax-m2.7, M2.5 | $0.30/$1.20 |
| Kimi | kimi-k2.5 (Agent Swarm) | $0.60/$2.50 |
| OpenRouter | qwen3.5-plus | $0.26/$1.56 |
| Codex | gpt-5.3-codex (CLI) | N/A |
