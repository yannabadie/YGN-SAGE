---
title: "Graph-GRPO — Edge-Level Credit Assignment"
type: paper
arxiv: "2603.02701"
venue: arXiv 2026
year: 2026
status: integre
tags:
  - paper
  - evolution
  - training
created: 2026-04-07
---

# Graph-GRPO — Edge-Level Credit Assignment

**arXiv** : [2603.02701](https://arxiv.org/abs/2603.02701)
**Venue** : arXiv 2026

## Resume

Credit assignment au niveau des edges (connexions entre agents) plutot qu'au niveau
de la topologie entiere. Permet d'attribuer le succes/echec aux connexions specifiques.

## Claims cles

1. Le credit au niveau edge est plus informatif que le credit au niveau topologie
2. Normalisation des avantages per-edge (Eq. 4-5) stabilise le training
3. Per-edge success rate : S_ij = P(Success | edge(i,j) in G)

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Edge-level credit | compute_edge_advantages() | sage-python/src/sage/verl/edge_credit.py | integre |
| EdgeStats tracking | Per-edge success rates across K topologies | edge_credit.py | integre |
| Advantage normalization | A_ij = (S_ij - mean(S)) / (std(S) + eps) | edge_credit.py | integre |
| Reward integration | Batch-level reward with edge credit | sage-python/src/sage/verl/reward.py:573-579 | integre |

## Ce qui n'a PAS ete retenu

- Rien de rejete — implementation fidele des equations 4-5 du paper

## Notes personnelles

Composant discret mais crucial du pipeline de training : permet au modele
d'apprendre quelles connexions entre agents sont benefiques vs nuisibles,
pas juste si la topologie globale marche. Utilise dans le reward multi-signal
de la Phase C (GiGPO).
