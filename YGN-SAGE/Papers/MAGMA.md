---
title: "MAGMA — Memory Consolidation for Agents"
type: paper
arxiv: "2601.03236"
venue: arXiv 2026
year: 2026
status: partiel
tags:
  - paper
  - memory
  - consolidation
created: 2026-04-07
---

# MAGMA

**arXiv** : [2601.03236](https://arxiv.org/abs/2601.03236)
**Venue** : arXiv 2026

## Resume

Pipeline de consolidation memoire : transforme les memoires episodiques
en connaissances semantiques structurees. Equivalent cognitif du sommeil.

## Claims cles

1. +45.5% raisonnement avec consolidation vs sans
2. La consolidation periodique (tous les N steps) est optimale
3. La transition episodique → semantique preserve les relations causales

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Consolidation periodique | Tous les 10 steps | sage-python/src/sage/memory/ | **partiel** |
| Episodique → semantique | Pipeline consolidation | memory/consolidation | **partiel** |
| Causal wiring | agent_loop traces | agent_loop.py | integre |

## Ce qui n'a PAS ete retenu

- Le pipeline complet MAGMA n'est pas replique : seule la consolidation basique est implementee

> [!warning] Implementation incomplete
> Le design est documente, le code existe en partie, mais la consolidation
> complete (episodique → semantique → causal) n'est pas validee en production.
> C'est le gap le plus important du pilier Memory.

## Metriques

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| Raisonnement +consolidation | +45.5% | non mesure | inconnu |
