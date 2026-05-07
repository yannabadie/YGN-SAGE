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

> **⚠ Non-autoritative / archive snapshot** — Obsidian-vault, historic reference only. Routing accuracy claims were `evidence_pending` in this archive snapshot; current authoritative status: `docs/CLAIMS.yaml`.

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| kNN routing | Routeur principal | sage-core/src/routing/ | archive snapshot, was `evidence_pending`; current status: `routing.knn_92pct` `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml` |
| arctic-embed-m | Embeddings 768-dim | kNN implementation | integre |
| Historic 93.3% GT (56/60) — non-autoritative archive snapshot, was `evidence_pending`; current status: `docs/CLAIMS.yaml` | non-autoritative | benchmark routing_gt | see `docs/CLAIMS.yaml` |

## Ce qui n'a PAS ete retenu

- Rien — implementation fidele au paper

## Metriques (historic / non-autoritative)

| Benchmark | Score paper | Score SAGE (archive snapshot, was `evidence_pending`; current status: `docs/CLAIMS.yaml`) | Delta |
|-----------|-----------|-----------|-------|
| GT accuracy | reference | historic 93.3% (56/60) — non-autoritative | — |
| vs ComplexityRouter | — | historic 45% (27/60) — non-autoritative | +48pp |
| vs SystemRouter | — | historic 88% — non-autoritative | +5pp |

## Notes personnelles

60 exemplaires etiquetes suffisent pour la performance historique 93.3% (archive snapshot non-autoritative; `routing.knn_92pct` was `evidence_pending` in this archive snapshot — current status: `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml`). C'est la decision architecturale la plus impactante du projet — simple, efficace, et validee empiriquement.
Le ComplexityRouter heuristic est Priority-3 emergency fallback only, NOT dead code (AUDIT2 2026-04-24 corrected); historic ~34% non-autoritative, see `docs/CLAIMS.yaml`.
