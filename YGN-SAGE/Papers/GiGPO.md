---
title: "GiGPO — Multi-Step RL Training"
type: paper
arxiv: "2505.10978"
venue: NeurIPS 2025
year: 2025
status: integre
tags:
  - paper
  - training
  - rl
created: 2026-04-07
---

# GiGPO — Generalized Group Policy Optimization

**arXiv** : [2505.10978](https://arxiv.org/abs/2505.10978)
**Venue** : NeurIPS 2025

## Resume

Extension multi-step de GRPO pour le training RL d'agents.
Permet le training sur des trajectoires multi-tour avec per-node rewards
et anchor states (etats de reference).

## Claims cles

1. Multi-step training avec per-node rewards surpasse single-turn GRPO
2. Anchor states stabilisent le training multi-tour
3. Decomposition StepRewardVector pour credit assignment temporel

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Multi-step env | SageTopologyEnv (4-state machine) | sage-python/src/sage/verl/topology_env.py | integre |
| StepRewardVector | Decomposition reward par step | topology_env.py | integre |
| Training pipeline | Nemotron E2E script | scripts/verl/train_nemotron_e2e.sh | integre |
| Phase C | GiGPO multi-step avec checkpoints | Phase C training | integre |

## 4-State Machine (SageTopologyEnv)

```
AWAITING_YAML → EXECUTING → AWAITING_DECISION → TERMINAL
```

1. **AWAITING_YAML** : le modele genere la topologie (format YAML)
2. **EXECUTING** : la topologie est executee, scores per-node collectes
3. **AWAITING_DECISION** : le modele decide (continue/upgrade/reroute)
4. **TERMINAL** : fin de l'episode, reward final

## Phases de training

| Phase | Type | Description |
|-------|------|-------------|
| A/B | GRPO warm-up | Single-turn, apprend le format YAML |
| **C** | **GiGPO multi-step** | 4-state machine, checkpoints, micro-decisions |

## Notes personnelles

GiGPO est ce qui differencie la Phase C de la Phase A.
Phase A apprend le format (YAML structurel).
Phase C apprend les **decisions** (continue, upgrade, reroute) — le vrai objectif.
Combine avec [[Graph-GRPO]] pour le credit assignment per-edge.
