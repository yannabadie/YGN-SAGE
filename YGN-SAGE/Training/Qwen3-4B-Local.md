---
title: "Training — Qwen3-4B — Local"
type: training
model: "Qwen3-4B"
phase: A+C
hardware: local-RTX3500
date: 2026-03-15
tags:
  - training
  - qwen
  - local
---

# Training — Qwen3-4B — Local (RTX 3500 Ada 12 GB)

## Configuration

- **Modele base** : Qwen3-4B
- **Hardware** : RTX 3500 Ada 12 GB (local)
- **Quantization** : bitsandbytes NF4
- **LoRA** : PEFT LoRA
- **Framework** : TRL (SFTTrainer)

## Resultats

| Phase | Metrique | Debut | Fin | Notes |
|-------|----------|-------|-----|-------|
| A SFT | Loss | 1.59 | 0.225 | Tool-call JSON format |
| A SFT | N1 | — | 0.865 | Format compliance |
| C SFT Adaptive | N1 | 0.865 | **0.922** | +6.6%, checkpoints + adapt_topology |

## Artefacts

- HuggingFace : `yannabadie/sage-topology-policy-local`
- Script : `scripts/train_local_qwen3_4b.py`
- Donnees : `data/topology_sft_v2_combined.jsonl`

## Observations

Phase C ajoute les decisions `adapt_topology` (continue/upgrade/reroute)
en plus du format de base. Le gain de +6.6% sur N1 montre que le modele
apprend les micro-decisions, pas juste le format.
