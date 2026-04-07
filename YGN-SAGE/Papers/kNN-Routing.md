---
title: "kNN Routing"
type: paper
arxiv: "2505.12601"
venue: arXiv 2025
year: 2025
status: integre
tags:
  - paper
  - routing
  - strategy
created: 2026-04-07
---

# kNN Routing

**arXiv** : [2505.12601](https://arxiv.org/abs/2505.12601)
**Venue** : arXiv 2025

## Resume

Routage cognitif par k-nearest neighbors sur embeddings de taches.
Approche simple mais tres efficace pour classifier la complexite cognitive.

## Claims cles

1. kNN sur embeddings surpasse les heuristiques de complexite
2. Embeddings denses (arctic-embed-m) capturent la complexite cognitive
3. Methode scalable avec peu de donnees d'entrainement

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| kNN routing | Routeur principal | sage-core/src/routing/ | integre |
| arctic-embed-m | Embeddings 768-dim | kNN implementation | integre |
| 92% GT | 46/50 sur ground truth | benchmark routing_gt | verifie |

## Ce qui n'a PAS ete retenu

- Rien — implementation fidele au paper

## Metriques

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| GT accuracy | reference | 92% (46/50) | — |
| vs ComplexityRouter | — | 34% (dead code) | +58pp |
| vs SystemRouter | — | 88% | +4pp |

## Notes personnelles

60 exemplaires etiquetes suffisent pour 92%. C'est la decision architecturale
la plus impactante du projet — simple, efficace, et validee empiriquement.
Le ComplexityRouter heuristic a 34% est officiellement dead code.
