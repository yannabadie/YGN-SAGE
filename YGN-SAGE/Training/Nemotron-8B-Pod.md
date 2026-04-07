---
title: "Training — Nemotron-8B — RunPod"
type: training
model: "Nemotron-Orchestrator-8B"
phase: A (GRPO)
hardware: runpod-H100
date: 2026-03-20
tags:
  - training
  - nemotron
  - runpod
---

# Training — Nemotron-Orchestrator-8B — RunPod 2xH100 NVL

## Configuration

- **Modele base** : Nemotron-Orchestrator-8B
- **Hardware** : 2x H100 NVL (RunPod)
- **Framework** : verl 0.7.1 + vLLM + Ray + FSDP
- **Duree** : ~30h pour full run

## Resultats

| Phase | Metrique | Debut | Fin | Notes |
|-------|----------|-------|-----|-------|
| SFT warmup | Loss | 2.87 | 1.30 | Pre-training format |
| A GRPO | Reward | — | 0.225 | Step 1050, plafond structural |
| DAPO | — | — | — | En cours |

## Artefacts

- HuggingFace : `yannabadie/sage-topology-policy-v2`
- Script : `scripts/verl/train_nemotron_e2e.sh`

## Observations

Le reward plafonne a 0.225 en Phase A — c'est un plafond structural
du format de reward, pas du modele. DAPO est cense casser ce plafond.

> [!warning] Non verifiable localement
> Le training full necessite RunPod H100. Le smoke test (`--smoke`)
> valide la plomberie mais pas la convergence.

## Questions ouvertes

- DAPO va-t-il casser le plafond 0.225 ?
- Quelle est la performance reelle de la policy Nemotron vs Qwen3 sur des taches ?
- Le cout RunPod est-il justifie par le gain de performance ?
