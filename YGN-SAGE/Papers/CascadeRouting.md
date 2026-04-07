---
title: "Cascade Routing — ETH-SRI"
type: paper
arxiv: "2410.10347"
venue: ICML 2025
year: 2025
status: integre
tags:
  - paper
  - routing
  - strategy
created: 2026-04-07
---

# Cascade Routing — ETH-SRI

**arXiv** : [2410.10347](https://arxiv.org/abs/2410.10347)
**Venue** : ICML 2025
**Origine** : ETH Zurich SRI Lab

## Resume

Demontre que les **estimateurs de qualite** sont le bottleneck du routing,
pas les algorithmes de routing eux-memes. Propose un routing en cascade
4 etapes avec exclusion de modeles pour garantir la diversite.

## Claims cles

1. Quality estimators are the bottleneck, not routing algorithms
2. FrugalGPT cascade avec exclusion de modeles garantit la diversite
3. Le routing en cascade surpasse le routing upfront sur les taches de code difficiles

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| FrugalGPT cascade | Skip excluded models | sage-core/src/routing/model_assigner.rs:242 | integre |
| Quality estimator focus | DistilBERT QualityEstimator (ONNX) | sage-python/src/sage/ | integre |
| 4-stage cascade | Pipeline routing complet | architecture globale | integre |

## Ce qui n'a PAS ete retenu

- Rien de rejete — le paper a valide l'architecture de routing existante de SAGE

## Metriques

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| Quality estimation | baseline | +34.4pp Pearson correlation | improvement |

## Notes personnelles

Ce paper a ete un tournant : il a justifie l'investissement dans le DistilBERT
QualityEstimator (ONNX, 0.9 MB) plutot que dans des algorithmes de routing plus complexes.
Decision documentee dans `.claude/rules/research-decisions.md:12-13`.
