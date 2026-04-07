---
title: "Routing Survey — 6 Paradigms"
type: paper
arxiv: "2603.04445"
venue: arXiv 2026
year: 2026
status: reference
tags:
  - paper
  - routing
  - survey
created: 2026-04-07
---

# Routing Survey — 6 Paradigms

**arXiv** : [2603.04445](https://arxiv.org/abs/2603.04445)
**Venue** : arXiv 2026

## Resume

Survey comprehensive identifiant 6 paradigmes de routing pour les systemes multi-agent.
Valide l'architecture AdaptiveRouter de SAGE comme pattern SOTA.

## Claims cles

1. 6 paradigmes de routing identifies dans la litterature
2. Le routing multi-stage en cascade avec composants appris est SOTA
3. Taxonomie utile pour positionner les approches

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Architecture validation | AdaptiveRouter confirme SOTA | docs seulement | reference |
| 4-stage cascade | Pipeline routing | README.md:157 | cite |

## Ce qui n'a PAS ete retenu

- Pas d'implementation directe — c'est un survey, pas un algorithme

> [!info] Paper de reference uniquement
> Ce paper ne fournit pas d'algorithme a implementer.
> Il valide que l'architecture de routing de SAGE (cascade multi-stage
> avec kNN + bandit + ModelAssigner) correspond au pattern SOTA identifie
> dans la litterature. C'est une validation externe, pas une source d'inspiration.

## Notes personnelles

Utile pour le positionnement academique et le README.
Confirme que l'approche SAGE n'est pas un hack mais une architecture validee.
