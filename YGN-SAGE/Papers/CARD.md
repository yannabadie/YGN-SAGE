---
title: "CARD — Conditional Topology"
type: paper
arxiv: "2603.01089"
venue: ICLR 2026
year: 2026
status: partiel
tags:
  - paper
  - topology
  - training
created: 2026-04-07
---

# CARD — Conditional Agent Routing & Design

**arXiv** : [2603.01089](https://arxiv.org/abs/2603.01089)
**Venue** : ICLR 2026

## Resume

Generation conditionnelle de topologies basee sur les proprietes de la tache.
Introduit une penalite de prix dans la loss function pour l'efficience cout.

## Claims cles

1. Conditioning de la generation de topologie sur les capabilities requises
2. Price penalty dans la loss pour optimiser le cout
3. Generation conditionnelle surpasse la generation inconditionnelle

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Price penalty | R_cost_efficiency (CARD-style) | sage-python/src/sage/verl/reward.py:403-409 | integre |
| Capability conditioning | — | — | **non implemente** |

## Ce qui n'a PAS ete retenu

- La generation conditionnelle complete (capability-aware) est documentee mais pas implementee en production
- SAGE utilise le weight 0.10 pour la penalite cout (vs 0.01 dans CARD — plus agressif sur le cout)

## Notes personnelles

> [!warning] Integration partielle
> Seule la penalite de prix est reprise. La generation conditionnelle
> (le coeur du paper) n'est pas implementee. C'est un candidat pour
> une future iteration du training pipeline.
