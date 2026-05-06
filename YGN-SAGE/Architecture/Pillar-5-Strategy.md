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

> **⚠ Non-autoritative / archive (cycle-13 K Phase 0.6c, 2026-05-06)**
> This Obsidian-vault snapshot is **historical reference only — non-autoritative**.
> Historic figures (kNN 93.3%/92% — non-autoritative) and (SystemRouter 88% — non-autoritative) and (ComplexityRouter 34% — non-autoritative) are tracked in `docs/CLAIMS.yaml` as `evidence_pending`.
> The `DEAD CODE` framing for ComplexityRouter is corrected (AUDIT2 2026-04-24): Priority-3 emergency fallback only, NOT dead code.
> Do not cite numbers from this file as evidence of capability — see `docs/CLAIMS.yaml`.

## S1/S2/S3 — Dual-Process (Kahneman)

| Systeme | Type | Exemple |
|---------|------|---------|
| S1 | Simple, reflexe | "What is 2+2?" |
| S2 | Modere, raisonnement | "Write a sorting algorithm" |
| S3 | Complexe, multi-step | "Build a full REST API with auth" |

## Routeurs

### kNN Router (Primaire)
- **Precision historique** : 93.3% GT (56/60) — archive, non-autoritative; current status: `routing.knn_92pct` `evidence_pending` in `docs/CLAIMS.yaml`.
- **Embeddings** : arctic-embed-m 768-dim, Rust ONNX
- **Ref** : [[kNN-Routing|arXiv 2505.12601]]
- **Exemplaires** : 60 taches etiquetees

### SystemRouter (Rust)
- **Precision historique** : 88% GT — archive, non-autoritative; current status: `routing.system_router_88pct` `evidence_pending` in `docs/CLAIMS.yaml`.
- **Usage** : Fallback si kNN indisponible
- **Implementation** : Rust natif, pas d'embeddings

### ComplexityRouter (Priority-3 emergency fallback)
- **Precision historique** : 34% GT — archive, non-autoritative; historical figure also `evidence_pending`.
- **Status** : Priority-3 emergency fallback only, NOT dead code (AUDIT2 2026-04-24 corrected the prior "DEAD CODE" framing). Heuristic pur, surpasse par kNN — see `docs/CLAIMS.yaml`.

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
