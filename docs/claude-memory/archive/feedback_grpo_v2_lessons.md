---
name: V2 GRPO Training Lessons Learned
description: Critical lessons from failed V2 GRPO training on RunPod H200 (April 1, 2026) — environment_factory destroys custom format, use plain reward_funcs
type: feedback
---

## NEVER use TRL `environment_factory` for models with custom output formats

TRL's `environment_factory` takes over the generation format — it injects tool schemas, manages tool-calling loops, and overwrites what SFT taught the model. The model learns TRL's internal format, not ours.

**Why:** The Qwen3-4B model was SFT-trained to output `<tool_call>` JSON. After GRPO with `environment_factory`, it generated `<think>` instead. Training reward was 1.46 but model was useless at inference — catastrophic format mismatch.

**How to apply:** For topology/orchestration GRPO, use plain `reward_funcs` that parse the model's free-form output. No `environment_factory`, no `tools`, no `rollout_func`.

Both AgentConductor (arXiv 2602.17100) and The Conductor (Sakana AI, ICLR 2026) validated this pattern:
- SFT to teach format
- Plain GRPO with reward functions that parse structured output
- beta=0.0 (no KL penalty)
- 200 iterations, 4-64 rollouts per prompt

## Custom GRPO training loops are fragile

**Why:** Our `train_v2_multiturn.py` had: manual gradient computation (loss=0.0000), manual log_prob tracking (GPU crashes), sequential rollouts (55h for 1 epoch). Three attempts, three failures.

**How to apply:** Use TRL GRPOTrainer. It handles rollouts, gradients, advantages, checkpointing. Only override via reward_funcs.

## Existing train_local_qwen3_4b.py was already correct

The script we started with (using TRL GRPOTrainer + reward_funcs from reward.py) was the right approach. The missing piece was execution reward (SAGE_VERL_EXEC=1), not multi-turn or custom loops.
