---
title: "Pilier 1 — Topology"
type: architecture
pillar: 1
tags:
  - architecture
  - topology
updated: 2026-04-07
---

# Pilier 1 — Topology

> "Which agents work together matters more than which model you use."
> — [[AdaptOrch]] : Var_topology / Var_model >= 20

## Composants Rust (sage-core)

- **DynamicTopologyEngine** : generation 6-path avec exploration bandit
- **MAP-Elites** : archive QD 4D (agent_count, max_depth, cost, model_diversity) avec insertion Pareto
- **CMA-ME** : adaptation matrice covariance pour parametres topology continus, sigma decay, warm_start
- **MCTS** : UCB1 tree search dans l'espace de mutation
- **TopologyGraph** (petgraph) : DAG avec 3 types d'edges (Control, Message, State) — [[MASFactory]]
- **HyEvo hybrid nodes** : `node_type="code"` pour execution sandbox deterministe (-13-19x cout sur MBPP) — [[HyEvo]]

## Composants Python (sage-python)

- **TopologySchema** : contrat partage entre training et runtime — nodes, edges, node_type, model_tier, provider_hint
- **TopologyRunner** : execution DAG, dispatch code nodes, streaming per-node
- **TopologyController** : decisions post-noeud (continue/upgrade/prune/reroute/spawn/gate)

## 11 Templates

| Template | Usage | Notes |
|----------|-------|-------|
| sequential | Taches lineaires | Baseline |
| parallel | Taches independantes | |
| AVR | Aggregation-Voting-Refinement | |
| selfmoa | Self-Mixture-of-Agents | |
| hierarchical | Decomposition arborescente | |
| hub | Hub central + workers | |
| debate | Multi-agent debate | |
| brainstorming | Generation divergente | |
| robust | Majority voting | gamma eleve |
| horizon_pipeline | Pipelines profonds | delta eleve |
| parallel_fanout | Fan-out massif | omega eleve |

## Macro-topologie pre-selection

`select_macro_topology()` utilise 3 features structurelles du TaskDAG :
- **omega** (parallelism) : nombre de branches parallelisables
- **delta** (depth) : profondeur max de la chaine
- **gamma** (coupling) : degre de couplage entre branches

## Path 6 — Learned Policy

- **Local** : Qwen3-4B, SFT N1=0.922, LoRA NF4
- **Pod** : Nemotron-Orchestrator-8B, GRPO step 1050
- **Activation** : `SAGE_ENABLE_PATH6=1`
- **Fallback** : templates si output invalide

> [!warning] Path 6 opt-in
> Pas dans le pipeline par defaut. Necessite activation explicite.

## Papers cles

- [[OpenSAGE]] — Architecture inspiration
- [[AdaptOrch]] — Topology > model variance
- [[MASFactory]] — 3-flow edge model
- [[AgentConductor]] — Density metric S_complex
- [[HyEvo]] — Hybrid LLM+code nodes
- [[CARD]] — Conditional topology
- [[ShinkaEvolve]] — LLM-as-mutator
