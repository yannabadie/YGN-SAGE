# veRL GRPO Training for YGN-SAGE Topology

## Overview

Train Qwen3.5-9B via GRPO on RunPod H100 to generate optimal multi-agent YAML topologies.
Mixed reward: structural (format/density) + execution (real multi-provider TopologyRunner).

**See `RUNPOD_PLAN.md` at repo root for the step-by-step execution guide.**

## Architecture

```
RunPod H100 (80GB)                          Local RTX 3500 (12GB)
┌──────────────────────────┐                ┌──────────────────────────┐
│ Docker: verlai/verl:      │                │ Qwen3.5-9B Q5_K_M GGUF  │
│   vllm011.latest          │                │   (~6.6GB VRAM)          │
│ Qwen/Qwen3.5-9B bf16     │  ── export ──► │ LoRA adapter merged      │
│ LoRA r=64, GRPO           │                │ Ollama / llama.cpp       │
│ K=4 samples, 8 epochs     │                │ SAGE pipeline Path 6     │
│ Multi-provider execution  │                │ 8 providers available    │
└──────────────────────────┘                └──────────────────────────┘
```

## Algorithm: GiGPO (Group-in-Group Policy Optimization)

Topologies are inherently multi-step: the model generates YAML, then the
`SageTopologyEnv` executes each node sequentially (coder→reviewer→synthesizer).
Each node has a (role, difficulty, context) tuple that forms an **anchor state**.
Across K=4 trajectories for the same prompt, GiGPO groups actions at identical
anchor states to compute step-level advantages: "when a reviewer sees good coder
output, does it produce better feedback?" This credit assignment is impossible
with flat GRPO.

GiGPO config (validated against `ppo_trainer.yaml`):
- `algorithm.adv_estimator=gigpo` — AdvantageEstimator enum
- `algorithm.enable_similarity=True` — similarity-based anchor grouping
- `algorithm.similarity_thresh=0.85` — threshold (lower than ALFWorld's 0.95
  because topology roles are more varied than household rooms)

Ref: GiGPO (arXiv 2505.10978), verl-agent (langfengQ/verl-agent)

## Training Strategy (Mixed Reward)

| Phase | Epochs | Algorithm | Reward | Cost |
|-------|--------|-----------|--------|------|
| **A: Structural** | 5 | GiGPO | YAML format + structure + Rust density + per-node credit | $0 API |
| **B: Execution** | 3 | GiGPO | Real topology execution via ProviderPool + step-level credit | ~$50-80 API |

Phase B uses `TopologyRunner` with `ProviderPool.resolve()` — each node is executed
by its assigned provider (DeepSeek Chat, Gemini Flash, Grok, etc.). The model learns
that assigning the right model_tier to the right node MATTERS.

Set `SAGE_VERL_EXEC=1` to activate execution mode (done automatically by train_topology.sh).

## Critical: Qwen3.5 Thinking Mode

Qwen3.5 has a "thinking mode" that inserts `<think>` blocks before the assistant response.
This is controlled by the Jinja chat_template, NOT by a `/no_think` prompt prefix (which
is a Qwen3 feature, not Qwen3.5).

**Without patching, vLLM will generate thinking tokens that consume the entire response
budget, resulting in reward=0 for every rollout.**

Fix: `patch_tokenizer.py` removes the thinking conditional from the chat_template and
saves a patched tokenizer. The training script points to this patched tokenizer directory.

## Files

| File | Purpose |
|------|---------|
| `setup_runpod.sh` | 9-step environment setup (vLLM, veRL, flash-linear-attention, tokenizer patch) |
| `train_topology.sh` | GiGPO training: Phase A structural + Phase B execution (with config validation) |
| `patch_tokenizer.py` | **Critical**: patches Qwen3.5 tokenizer to remove thinking mode |
| `convert_sft_to_verl.py` | Auto-loads 11 data sources → veRL parquet |
| `curate_training_data.py` | Curate diverse prompts (GSM8K capped, GPT-5.4 Pro prioritized) |
| `validate_setup.py` | 10-point pod validation (includes tokenizer + flash-linear-attention checks) |
| `benchmark_post_train.py` | Post-training BigCodeBench eval with trained model |
| `export_for_local.py` | Export LoRA adapter → GGUF for local inference |

## Setup Sequence

```bash
# On RunPod H100 pod (Docker: verlai/verl:vllm011.latest)
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE && git checkout VeRLGIGPO

# Step 1: Full setup (installs deps, patches tokenizer, converts data)
bash sage-python/scripts/verl/setup_runpod.sh

# Step 2: Validate
cd sage-python && python3 scripts/verl/validate_setup.py

# Step 3: Train
bash scripts/verl/train_topology.sh
```

## Training Config (validated against verl-agent schema)

| Parameter | Value | Validated |
|-----------|-------|-----------|
| Model | `Qwen/Qwen3.5-9B` (patched tokenizer) | ✓ |
| Algorithm | `algorithm.adv_estimator=gigpo` | ✓ (AdvantageEstimator enum) |
| GiGPO similarity | `enable_similarity=True`, `thresh=0.85` | ✓ (ppo_trainer.yaml) |
| LoRA rank | 64 | ✓ (veRL standard) |
| K (samples/prompt) | 4 | ✓ (`actor_rollout_ref.rollout.n=4`) |
| Temperature | 0.4 | ✓ |
| KL coef | 0.04, `low_var_kl` | ✓ (ppo_trainer.yaml L59-62) |
| Batch size | 64 (Phase A), 32 (Phase B) | ✓ |
| gpu_memory_utilization | 0.7 | ✓ |

### Parameters that were REMOVED (fabricated by Claude Code)

| Parameter | Status | Fix |
|-----------|--------|-----|
| `algorithm.gigpo.step_advantage_w` | ✗ Does not exist | Removed |
| `algorithm.gigpo.mode` | ✗ Does not exist | Removed |
| `actor_rollout_ref.rollout.multi_turn=True` | ✗ Bool, needs dict | Fixed: `{"format":"chatml"}` |
| `actor_rollout_ref.rollout.num_speculative_tokens` | ✗ vLLM param | Removed |

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
```

## Data (2200+ entries)

| Source | Entries | Type |
|--------|---------|------|
| SFT v2 combined | 1532 | BigCodeBench, GSM8K, CodeContests |
| RAFT Phase 2 | 199 | Execution-verified |
| GPT-5.4 Pro complex | 144 | 5-7 node topologies |
| GPT-5.4 Pro (6 files) | 90 | CF/GCJ, deep reasoning, simple, corrections, audit |
| V2 Adaptive | 220 | Adaptive topologies, static→adaptive, recovery |

Curated subset: ~500 entries in `verl_topology_curated.parquet` (GSM8K capped at 50).

## Integration Risks (validated on pod)

- **Env registration**: `env_register.py` monkey-patches `env_manager.make_envs`.
  The train script runs a Step 0 validation that checks the actual verl-agent source.
  If the monkey-patch fails, the script will error before wasting GPU time.

- **Graph-GRPO edge credit** (`edge_credit.py`): Per-edge advantage normalization.
  Implemented, wired into `compute_score_with_edge_credit()`, not yet called by trainer.

- **RewardFlow** (`rewardflow.py`): PageRank-based per-node credit propagation.
  Implemented, compatible with StepRewardVector, not yet wired.
