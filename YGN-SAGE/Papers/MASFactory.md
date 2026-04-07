---
title: "MASFactory — 3-Flow Edge Model"
type: paper
arxiv: "2603.06007"
venue: arXiv 2026
year: 2026
status: integre
tags:
  - paper
  - topology
created: 2026-04-07
---

# MASFactory — Multi-Agent System Factory

**arXiv** : [2603.06007](https://arxiv.org/abs/2603.06007)
**Venue** : arXiv 2026

## Resume

Framework de generation de topologies multi-agent avec un modele d'edges
a 3 flux : Control, Message, State. Vibe Graphing pour conversion LLM→graph.

## Claims cles

1. 3-flow edge model (Control + Message + State) capture la semantique des interactions
2. 84.76% HumanEval avec 97% de reduction de code via approche graph-centric
3. Vibe Graphing : pipeline LLM 3 etapes (Role Assignment → Structure Design → Validation)

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| 3-flow edges | TopologyGraph IR | sage-core/src/topology/topology_graph.rs | integre |
| Vibe Graphing 3-stage | TopologySynthesizer (Path 3) | DynamicTopologyEngine | integre |
| Visualisation 3-flow | Edge stroke colors | ui/static/js/topology.js:4-5,40,192 | integre |

## Ce qui n'a PAS ete retenu

- Le framework MASFactory complet (SAGE a sa propre architecture)
- Seul le modele d'edges et le pipeline de synthese sont repris

## Metriques rapportees dans le paper

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| HumanEval | 84.76% | 89.6% | +4.84pp |
| Code reduction | 97% | N/A | — |

## Notes personnelles

Le modele 3-flow est un des emprunts les plus elegants du projet.
Distinguer Control (orchestration), Message (data), et State (shared state)
donne une semantique riche aux connexions entre agents sans complexite excessive.
