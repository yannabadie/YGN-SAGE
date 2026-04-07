---
title: "PILOT — Contextual Bandit for Agent Routing"
type: paper
arxiv: "2508.21141"
venue: EMNLP 2025
year: 2025
status: integre
tags:
  - paper
  - strategy
  - bandit
created: 2026-04-07
---

# PILOT

**arXiv** : [2508.21141](https://arxiv.org/abs/2508.21141)
**Venue** : EMNLP 2025

## Resume

Utilise un bandit contextuel pour selectionner dynamiquement le meilleur
template/modele pour chaque tache, avec Thompson sampling.

## Claims cles

1. Thompson sampling per-arm converge plus vite que UCB sur les agents
2. Posterieurs Beta/Gamma sont adequats pour des recompenses binaires/continues
3. Selection front Pareto (cout vs qualite) pour multi-objectif

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Thompson sampling | ContextualBandit | sage-core/src/routing/ | integre |
| Beta/Gamma posterieurs | Per-arm sampling | bandit implementation | integre |
| Pareto selection | Multi-objectif | ModelAssigner | integre |

## Notes personnelles

Le bandit est un composant discret mais crucial : il module l'exploration
dans le TopologyEngine et ecrase les assignments des modeles sous-performants.
Persiste en SQLite, apprend cross-session.
