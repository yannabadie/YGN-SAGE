---
name: Training Tracks — Nemotron (master) vs Qwen3-4B (local)
description: Two parallel training tracks exist. Nemotron-8B on master (RunPod 2xH100, stalled at step 39). Qwen3-4B on local branch (Phase C = best model, 40% MASBENCH).
type: project
---

## Track 1: Nemotron-Orchestrator-8B (branch `master`, RunPod 2xH100 NVL)

- **Status (2026-03-28)**: STALLED — Phase A V5 at step 39/1152, reward ~0.06
- **Stack**: veRL 0.7.1 + FSDP, Nemotron-Orchestrator-8B (Qwen3 arch)
- **HuggingFace**: yannabadie/sage-topology-policy-v2 (SFT merged 16 GB + step 150 LoRA)
- **Blocked by**: training convergence (reward never exceeded 0.06)
- **E2E command**: `bash scripts/verl/train_nemotron_e2e.sh`

## Track 2: Qwen3-4B (branch `local`, local RTX 3500 Ada + RunPod H200)

- **Status (2026-04-03)**: Phase C SFT = BEST MODEL (0.922 structural, 40% MASBENCH depth)
- **Stack**: TRL + PEFT + 4-bit NF4, LoRA r=32
- **HuggingFace**: yannabadie/sage-topology-policy-local (Phase C + V2 SFT/GRPO)
- **V2 SFT**: done but regressed to 20% MASBENCH (data imbalance)
- **V2 GRPO**: done but BROKEN (environment_factory destroyed format)
- **Next**: V2.1 GRPO with plain reward_funcs from Phase C checkpoint

**Why:** Two tracks exist because Nemotron-8B on RunPod never converged. Local Qwen3-4B training proved faster to iterate. Phase C on Qwen3-4B is currently the best model.

**How to apply:** Use Qwen3-4B Phase C for all inference/benchmarks. Nemotron track is dormant — revive only if pod budget available and convergence issues solved.
