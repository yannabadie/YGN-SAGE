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

> **⚠ Non-autoritative / archive** — Obsidian-vault snapshot, historic reference only. See `docs/CLAIMS.yaml` for the authoritative status. Routing accuracy claims are `evidence_pending`.

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| kNN routing | Routeur principal | sage-core/src/routing/ | integre (`routing.knn_92pct` `evidence_pending`) |
| arctic-embed-m | Embeddings 768-dim | kNN implementation | integre |
| Historic 93.3% GT (56/60) — `evidence_pending` | non-autoritative | benchmark routing_gt | see `docs/CLAIMS.yaml` |

## Ce qui n'a PAS ete retenu

- Rien — implementation fidele au paper

## Metriques (historic / non-autoritative)

| Benchmark | Score paper | Score SAGE (historic, `evidence_pending`) | Delta |
|-----------|-----------|-----------|-------|
| GT accuracy | reference | historic 93.3% (56/60) — non-autoritative | — |
| vs ComplexityRouter | — | historic 45% (27/60) — non-autoritative | +48pp |
| vs SystemRouter | — | historic 88% — non-autoritative | +5pp |

## Notes personnelles

60 exemplaires etiquetes suffisent pour la performance historique 93.3% (non-autoritative; `routing.knn_92pct` `evidence_pending` in `docs/CLAIMS.yaml`). C'est la decision architecturale la plus impactante du projet — simple, efficace, et validee empiriquement.
Le ComplexityRouter heuristic est Priority-3 emergency fallback only, NOT dead code (AUDIT2 2026-04-24 corrected); historic ~34% non-autoritative, see `docs/CLAIMS.yaml`.
