---
title: "Pilier 5 — Strategy"
type: architecture
pillar: 5
tags:
  - architecture
  - strategy
  - routing
updated: 2026-04-07
---

# Pilier 5 — Strategy (Cognitive Routing)

## S1/S2/S3 — Dual-Process (Kahneman)

| Systeme | Type | Exemple |
|---------|------|---------|
| S1 | Simple, reflexe | "What is 2+2?" |
| S2 | Modere, raisonnement | "Write a sorting algorithm" |
| S3 | Complexe, multi-step | "Build a full REST API with auth" |

## Routeurs

### kNN Router (Primaire)
- **Precision** : **93.3%** GT (56/60)
- **Embeddings** : arctic-embed-m 768-dim, Rust ONNX
- **Ref** : [[kNN-Routing|arXiv 2505.12601]]
- **Exemplaires** : 60 taches etiquetees

### SystemRouter (Rust)
- **Precision** : 88% GT
- **Usage** : Fallback si kNN indisponible
- **Implementation** : Rust natif, pas d'embeddings

### ComplexityRouter (DEAD CODE)
- **Precision** : 34% GT
- **Status** : Ne pas utiliser. Heuristic pur, largement surpasse par kNN.

### ShadowRouter (DEPRECATED)
- **Divergence** : 49.6% avec SystemRouter
- **Status** : `@deprecated`, desactive par defaut (`SAGE_ENABLE_SHADOW`)
- **Usage originel** : comparaison dual-path Rust/Python

## Contextual Bandit (Rust)

- Thompson sampling per-arm avec posterieurs Beta/Gamma
- Selection front Pareto (cout vs qualite)
- Ref: [[PILOT]] (EMNLP 2025, arXiv 2508.21141)

## ModelAssigner (Rust)

- Score : `0.4 * affinity + 0.4 * domain + 0.2 * (1 - cost)`
- Provider hints : +0.15 bonus
- Override bandit : ecrase si quality < 0.4
- Source : `cards.toml` (20 modeles, 7 providers)

## Cascade Routing 4-etape

Valide par [[CascadeRouting|ETH-SRI ICML 2025]] et [[Routing-Survey|survey 2603.04445]].
