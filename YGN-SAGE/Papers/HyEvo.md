---
title: "HyEvo — Hybrid LLM+Code Nodes"
type: paper
arxiv: "2603.19639"
venue: arXiv 2026
year: 2026
status: integre
tags:
  - paper
  - topology
  - code-nodes
created: 2026-04-07
---

# HyEvo — Hybrid Evolution

**arXiv** : [2603.19639](https://arxiv.org/abs/2603.19639)
**Venue** : arXiv 2026

## Resume

Noeuds hybrides dans les topologies multi-agent : certains noeuds executent
du code deterministe au lieu d'appeler un LLM. Reduction de cout massive.

## Claims cles

1. Code nodes : 13-19x reduction de cout sur MBPP
2. Execution sandbox deterministe plus fiable que LLM pour les operations connues
3. L'evolution peut decouvrir quels noeuds beneficient de code vs LLM

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| node_type="code" | TopologySchema | sage-python/src/sage/verl/topology_schema.py | integre |
| Sandbox execution | TopologyRunner dispatch | sage-python/src/sage/topology/ | integre |
| Cost reduction | Code nodes dans templates | templates avec code nodes | integre |

## Notes personnelles

Approche elegante et pragmatique. Les code nodes sont un des differenciateurs
de SAGE par rapport aux frameworks purement LLM.
Le sandbox 3-couches (tree-sitter → Wasm → subprocess) est la fondation de cette feature.
