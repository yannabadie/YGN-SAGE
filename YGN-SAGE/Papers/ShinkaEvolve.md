---
title: "ShinkaEvolve — LLM-as-Mutator"
type: paper
arxiv: "2601.04170"
venue: ICLR 2026
year: 2026
status: integre
tags:
  - paper
  - evolution
  - drift
created: 2026-04-07
---

# ShinkaEvolve

**arXiv** : [2601.04170](https://arxiv.org/abs/2601.04170)
**Venue** : ICLR 2026

## Resume

Utilise des LLMs comme operateurs de mutation pour l'evolution de programmes/agents.
Introduit l'Agent Stability Index (ASI) pour detecter la derive de performance
sur 12 dimensions. Surpasse OpenEvolve de 60%.

## Claims cles

1. LLM-as-mutator surpasse les mutations aleatoires de 60%
2. Agent Stability Index (ASI) sur 12 dimensions detecte la derive
3. Bandit-based LLM ensemble selection pour les operateurs de mutation
4. La consolidation episodique atenue la derive

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| LLM-as-mutator | AdaptiveMutator | sage-python/src/sage/evolution/llm_mutator.py:3,105 | integre |
| Bandit-based ensemble | Thompson sampling sur operateurs | llm_mutator.py | integre |
| ASI 12D drift | ExtendedDriftMonitor | sage-python/src/sage/monitoring/extended_drift.py:1-116 | integre |
| ASI weights | Constantes drift | sage-python/src/sage/constants.py:122 | integre |

## Signaux ASI (12 dimensions)

1. Response consistency
2. Tool usage patterns
3. Reasoning pathway stability
4. Inter-agent agreement rates
5-12. (semantique, comportemental, thematique — 9 signaux additionnels)

## Ce qui n'a PAS ete retenu

- OpenEvolve comparison directe (contexte different)

## Metriques rapportees dans le paper

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| vs OpenEvolve | +60% | N/A | non mesure |

## Notes personnelles

Double contribution : le LLM-as-mutator rend l'evolution intelligente (pas juste
random), et l'ASI 12D donne une alerte precoce quand le systeme derive.
Les deux sont essentiels pour l'evolution online (`should_evolve()`).
