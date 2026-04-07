---
title: "OpenSAGE — Open-Source Sage Agent"
type: paper
arxiv: "2602.16891"
venue: ICML 2026
year: 2026
status: integre
tags:
  - paper
  - topology
  - architecture
created: 2026-04-07
---

# OpenSAGE — Open-Source Sage Agent

**arXiv** : [2602.16891](https://arxiv.org/abs/2602.16891)
**Venue** : ICML 2026
**Origine** : UC Berkeley

## Resume

Framework multi-agent open-source qui apprend autonomement quelle topologie utiliser.
YGN-SAGE etend OpenSAGE avec verification formelle, evolution de topologies, et un core Rust.

## Claims cles

1. Les agents multi-agent auto-organises surpassent les pipelines fixes
2. Self-programming : les agents creent dynamiquement de nouveaux sous-agents
3. L'architecture doit etre apprise, pas prescrite

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Self-programming | `agent_mgmt.py` | sage-python/src/sage/tools/ | implemente |
| Auto-topologie | DynamicTopologyEngine | sage-core/src/topology/ | integre |
| Agent composition | AgentTool.from_agent() | sage-python/src/sage/tools/ | integre |

## Ce qui n'a PAS ete retenu

- Le protocole de communication interne d'OpenSAGE (remplace par 3-flow edges de [[MASFactory]])
- L'architecture specifique de routing (remplacee par kNN + bandit)

## Metriques rapportees dans le paper

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| SWE-Bench | 59% | non soumis | — |

## Notes personnelles

Paper fondateur du projet. YGN-SAGE diverge significativement de l'implementation originale
mais conserve la these centrale : les architectures multi-agent doivent etre apprises.
