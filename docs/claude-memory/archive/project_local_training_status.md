---
name: V2 Training Status (April 3, 2026)
description: Qwen3-4B local training — V2 SFT done, GRPO broken (environment_factory), Phase C still best model, MASBENCH regressed 40%→20%
type: project
---

## Branch `local` (worktree C:/Code/worktrees/local)

**Model**: Qwen/Qwen3-4B + LoRA r=32
**Format**: `<tool_call>` JSON, 2 tools (create_topology + adapt_topology)
**HuggingFace**: yannabadie/sage-topology-policy-local

## Training History (chronological)

1. **Phase A SFT** (local RTX 3500 Ada): avg 0.865 on structural reward
2. **Phase C SFT** (local): avg 0.922 (+6.6%). **BEST MODEL** as of 2026-04-03.
3. **V2 SFT** (RunPod H200 SXM 141GB, 2026-04-01): 8633 entries, 3 epochs, lr=1e-5. Loss 0.341→0.056.
4. **V2 GRPO** (RunPod H200, 2026-04-01): 500 steps, K=4, reward=1.463. **BROKEN** — environment_factory destroyed `<tool_call>` format → model generates `<think>`.
5. **MASBENCH** (2026-04-01): V2 SFT = 2/10 (20%) depth. Phase C = 4/10 (40%). **V2 regressed -20pp.**

## Checkpoints on HuggingFace

| Path | Usable? | Notes |
|------|---------|-------|
| `phase_c/` | **YES — BEST** | 40% MASBENCH depth |
| `v2_sft_checkpoint/` | YES but regressed | 20% MASBENCH depth |
| `v2_sft_merged/` | YES but regressed | 8 GB merged model |
| `v2_grpo_checkpoint/` | **NO** | Format broken (environment_factory) |

## V2 Data

- `sage-python/data/v2_final.jsonl`: 8633 entries (7063 single + 1570 multi-turn)
- **Data imbalance**: 60% adapt_topology, 22% create_topology — likely cause of regression
- Scripts: `enrich_v2_data.py`, `generate_v2_expert.py`

## Critical Bugs Fixed During V2 Sprint

1. **Path 6 silently dead** (P1, FIXED commit `6375ddc`): TopologyEdge wrong constructor signature
2. **Tokenizer crash** (FIXED commit `258e383`): V2 adapter had broken extra_special_tokens
3. **gpt-5.4-pro removed** (FIXED): model doesn't exist in OpenAI API

## Open Problems

- **P2**: GRPO GRPO checkpoint unusable (environment_factory destroyed format)
- **P3**: V2 SFT regressed vs Phase C (data imbalance hypothesis)
- **P4**: Inter-node context truncated to 500 chars (runner.py ~L240, topology_env.py L605)
- **P5**: No bidirectional agent communication (single DAG pass)

**Why:** Phase C is still the best model. V2.1 GRPO must use plain reward_funcs (not environment_factory) and start from Phase C checkpoint.

**How to apply:** For any inference/benchmark, use Phase C adapter. For V2.1 training, start from Phase C, rebalance data (50%+ create_topology), use TRL GRPOTrainer with reward_funcs only.
