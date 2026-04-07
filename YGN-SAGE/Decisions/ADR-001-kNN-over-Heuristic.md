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

## Alternatives considerees

| Option | Accuracy GT | Pour | Contre |
|--------|------------|------|--------|
| **kNN Router** | **93.3%** (56/60) | Simple, empiriquement superieur | Necessite embeddings (latence boot) |
| ComplexityRouter | 45% (27/60) | Zero-cost, pas d'embeddings | Precision catastrophique |
| SystemRouter (Rust) | 88% | Natif Rust, rapide | 5pp inferieur au kNN |

## Consequences

- Positives : +48pp de precision par rapport a l'heuristic
- Negatives : Dependance aux embeddings arctic-embed-m au boot
- Risques residuels : 60 exemplaires seulement — fragile si distribution des taches change

## Evidence

- Benchmark routing_gt : 56/60 = 93.3% (avril 7 2026)
- Paper : [[kNN-Routing|arXiv 2505.12601]]
- ComplexityRouter garde dans le code mais jamais appele en prod
