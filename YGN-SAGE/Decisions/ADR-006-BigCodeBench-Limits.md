---
title: "ADR-006: BigCodeBench Hard Limits — Topology Not the Lever"
type: adr
status: accepte
date: 2026-04-07
tags:
  - adr
  - benchmarks
  - topology
  - adversarial
---

# ADR-006: BigCodeBench Hard — La topologie n'est pas le bon levier

## Contexte

BigCodeBench Hard (148 taches, ICLR '25) est utilise comme benchmark principal.
L'analyse des traces (run 37.2%, avril 7 2026) revele :
- **omega moyen = 1.32** (parallelisme tres bas)
- **100% S2 routing** (toutes les taches sont "moderate")
- **AVR topology 91%** du temps, pass rate 45%
- **Sequential (fallback) : 55%** sur 11 taches (echantillon faible)

## Decision

BigCodeBench Hard **n'est pas le terrain de preuve** pour la these "topology > model".
C'est un benchmark utile pour le diagnostic pipeline mais pas pour prouver le delta framework.

## Evidence (AdaptOrch 2602.16873)

| omega taches | Cluster | Gain topologie | BigCodeBench Hard |
|---|---|---|---|
| omega >= 3 | Wide-Shallow | **+12.6pp** | NON (omega=1.3) |
| omega ~2 | Diamond | +11.4pp | NON |
| omega < 2 | Chain | +3.8pp | **OUI** — gain marginal |
| omega = 1 | Atomique | **-2.1pp** | Possible — topologie nuit |

La these Var_topology/Var_model >= 20 **tient sur omega >= 3** (SWE-bench, MASBENCH breadth).
Elle **ne tient pas sur omega ~1** (BigCodeBench Hard, GPQA).

## Alternatives pour prouver le delta

| Benchmark | omega moyen | Gain predit | Status |
|---|---|---|---|
| **MASBENCH breadth** | eleve | **+22pp mesure** | Terrain de preuve #1 |
| **SWE-bench** | ~3.4 | +23% (AdaptOrch) | Phase D.3 |
| BigCodeBench Hard | 1.3 | +3.8pp max | Diagnostic seulement |

## Consequences

- **BigCodeBench** reste utile pour : baseline, diagnostic, comparaison The Conductor
- **L'ablation** doit aussi tourner sur MASBENCH (pas seulement BigCodeBench)
- **Le gain BigCodeBench** vient du repair (AVR) et model selection, pas de la topologie
- **Ne pas survendre** un 40%+ BigCodeBench comme preuve de topology > model

## Risques

L'ablation BigCodeBench full vs baseline pourrait montrer un delta **nul ou negatif**
pour la topologie. C'est **honnete** et **attendu** (AdaptOrch le predit).
L'ablation MASBENCH breadth devrait montrer le vrai delta.
