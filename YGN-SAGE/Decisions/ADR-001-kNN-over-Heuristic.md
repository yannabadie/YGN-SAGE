---
title: "ADR-001: kNN over Heuristic Router"
type: adr
status: accepte
date: 2026-02-15
tags:
  - adr
  - routing
---

# ADR-001: kNN plutot que Heuristic pour le Routing

## Contexte

Deux approches de routing S1/S2/S3 etaient en competition :
- ComplexityRouter : heuristique pure (longueur, mots-cles, structure)
- kNN Router : k-nearest neighbors sur embeddings arctic-embed-m

## Decision

kNN Router est le routeur principal. ComplexityRouter est dead code.

> **⚠ Non-autoritative / archive snapshot** — Obsidian-vault decision record. Routing accuracy claims (`routing.knn_92pct`, `routing.system_router_88pct`) were `evidence_pending` in this archive snapshot; current authoritative status: `docs/CLAIMS.yaml` (`delivered` floors since 2026-05-07).

## Alternatives considerees (historic / non-autoritative archive snapshot)

| Option | Accuracy GT (historic) | Pour | Contre |
|--------|------------|------|--------|
| **kNN Router** | historic 93.3% (56/60) — non-autoritative archive snapshot; `routing.knn_92pct` was `evidence_pending` in this snapshot, current status: `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml` | Simple, empiriquement superieur | Necessite embeddings (latence boot) |
| ComplexityRouter | historic 45% (27/60) — non-autoritative archive snapshot, was `evidence_pending`; current status: `retired` in `docs/CLAIMS.yaml` | Zero-cost, pas d'embeddings | Priority-3 emergency fallback only, NOT dead code (AUDIT2 2026-04-24 corrected) |
| SystemRouter (Rust) | historic 88% — non-autoritative archive snapshot; `routing.system_router_88pct` was `evidence_pending` in this snapshot, current status: `delivered` floor ≥52/60 in `docs/CLAIMS.yaml` | Natif Rust, rapide | 5pp inferieur au kNN historic |

## Consequences

- Positives : +48pp de precision par rapport a l'heuristic (historic, non-autoritative; voir `docs/CLAIMS.yaml`)
- Negatives : Dependance aux embeddings arctic-embed-m au boot
- Risques residuels : 60 exemplaires seulement — fragile si distribution des taches change

## Evidence (historic / non-autoritative)

- Benchmark routing_gt : historic 56/60 = 93.3% (avril 7 2026) — non-autoritative archive snapshot; `routing.knn_92pct` was `evidence_pending` in this snapshot, current status: `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml`
- Paper : [[kNN-Routing|arXiv 2505.12601]]
- ComplexityRouter is Priority-3 emergency fallback only (NOT dead code; AUDIT2 corrected); kept in the code as the live fallback path
