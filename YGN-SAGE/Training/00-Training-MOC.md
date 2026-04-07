---
title: Training MOC
type: moc
tags:
  - training
  - moc
updated: 2026-04-07
---

# Training Pipeline

Deux tracks paralleles pour entrainer la policy de generation de topologie (Path 6).

## Track 1 — Local (RTX 3500 Ada 12 GB)

**Modele** : [[Qwen3-4B-Local|Qwen3-4B]]
- Phase A SFT : loss 1.59 → 0.225, N1 = 0.865
- Phase C SFT Adaptive : N1 = **0.922** (+6.6%)
- Stack : bitsandbytes NF4 + PEFT LoRA + TRL
- HuggingFace : `yannabadie/sage-topology-policy-local`

## Track 2 — Pod (RunPod 2xH100 NVL)

**Modele** : [[Nemotron-8B-Pod|Nemotron-Orchestrator-8B]]
- SFT warmup : loss 2.87 → 1.30
- Phase A GRPO : step 1050, reward 0.225 (structural ceiling)
- DAPO targeted : **stalled** (V2 SFT regressed, GRPO broken — voir memory)
- Stack : verl 0.7.1 + vLLM + Ray + FSDP
- HuggingFace : `yannabadie/sage-topology-policy-v2`

## Phases

| Phase | Type | Description |
|-------|------|-------------|
| A | GRPO warm-up | Single-turn, apprend le format YAML |
| B | GRPO extended | Extension phase A |
| C | GiGPO multi-step | 4-state machine, checkpoints, micro-decisions |

## Commandes

```bash
# Smoke test (CPU, <2min)
bash scripts/verl/train_nemotron_e2e.sh --smoke

# Full (RunPod H100, ~30h)
bash scripts/verl/train_nemotron_e2e.sh
```

## Donnees

- Dataset : `yannabadie/sage-training-data` (HuggingFace)
- Format : toolcall JSON (genere par `generate_toolcall_dataset.py`)

> [!warning] Path 6 opt-in
> La policy entrainee est opt-in (`SAGE_ENABLE_PATH6=1`).
> Lazy-loaded au premier usage, fallback sur templates si output invalide.
> Pas dans le pipeline par defaut.
