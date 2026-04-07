---
title: "AgentConductor — RL Topology Evolution"
type: paper
arxiv: "2602.17100"
venue: arXiv 2026
year: 2026
status: integre
tags:
  - paper
  - topology
  - verification
created: 2026-04-07
---

# AgentConductor

**arXiv** : [2602.17100](https://arxiv.org/abs/2602.17100)
**Venue** : arXiv 2026

## Resume

Evolution de topologies par RL avec une metrique de densite formelle (S_complex).
Atteint 97.5% HumanEval avec un modele 3B — preuve que la topologie peut compenser
la taille du modele.

## Claims cles

1. RL-based topology evolution surpasse les topologies fixes
2. TopologyDensity S_complex (Theorem 1) mesure la complexite cout des topologies
3. 97.5% HumanEval avec un modele 3B (via topologie optimisee)
4. N_max bounds pour limiter la taille des topologies

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| S_complex metric | TopologyDensity | sage-core/src/topology/topology_metrics.rs | integre |
| N_max bounds | Limites de taille | topology_metrics.rs | integre |
| Density scoring | Reward multi-signal | sage-python/src/sage/verl/reward.py | integre |

## Ce qui n'a PAS ete retenu

- L'algorithme RL specifique d'AgentConductor (SAGE utilise GRPO/GiGPO via verl)
- Le framework complet (SAGE a sa propre architecture)

## Metriques rapportees dans le paper

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| HumanEval | **97.5%** (3B) | 89.6% (multi-provider) | -7.9pp |

## Notes personnelles

Competitor direct de SAGE sur la these "topology > model".
AgentConductor atteint 97.5% HumanEval avec un 3B — c'est impressionnant
et montre le potentiel de l'approche. SAGE cite ce paper comme benchmark
de reference (The Conductor a 40.0% sur BigCodeBench Hard).

Le S_complex de Theorem 1 est une contribution formelle propre — donne
une metrique mathematique pour la complexite d'une topologie, pas juste
une heuristique.
