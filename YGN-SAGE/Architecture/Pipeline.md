---
title: Pipeline CLASSIFY-LEARN
type: architecture
tags:
  - architecture
  - pipeline
updated: 2026-04-07
---

# Pipeline — 6 Etapes

```
CLASSIFY → DECOMPOSE → TOPOLOGY → ASSIGN → EXECUTE → LEARN
```

## Stage 1 — CLASSIFY

**Composant** : kNN Router (Rust, arctic-embed-m 768-dim)
**Precision** : 93.3% GT (56/60) — [[kNN-Routing|arXiv 2505.12601]]
**Sortie** : S1 (simple) / S2 (moderate) / S3 (complexe)

Le ComplexityRouter heuristic est **dead code** (34% GT). Ne pas utiliser.

## Stage 2 — DECOMPOSE

**Composant** : TaskPlanner
**Sortie** : TaskDAG — graphe de dependances avec features (complexity, branching, domain)

## Stage 3 — SELECT TOPOLOGY

**Composant** : DynamicTopologyEngine (Rust)
**6 chemins + fallback templates** :

1. **S-MMU retrieval** — cherche une topologie passee similaire (similarity > 0.7)
2. **MAP-Elites archive** — meilleur elite du QD archive
3. **LLM synthesis** — genere via prompt structure
4. **Mutation** — mute un elite existant (7 operateurs)
5. **MCTS** — Monte Carlo tree search dans l'espace de mutation
6. **Path 6: Learned policy** — Qwen3-4B ou Nemotron-8B (opt-in, `SAGE_ENABLE_PATH6=1`)
7. **Templates fallback** — 11 templates pre-cables

**Pre-filtre** : `select_macro_topology()` utilise les features structurelles du DAG :
- omega (parallelism) eleve → `parallel_fanout`
- delta (depth) eleve → `horizon_pipeline`
- gamma (coupling) eleve → `robust` (majority voting)

Un bandit contextuel (Thompson sampling) module exploration vs exploitation.

## Stage 4 — ASSIGN MODELS

**Composant** : ModelAssigner (Rust) + `ModelRegistry.select_for_system()` en bypass
**Score** : `0.4 * affinity + 0.4 * domain + 0.2 * (1 - cost)` (meme scoring en topologie et en bypass)
**Bypass** : `select_for_system(S1/S2/S3)` → premier candidat avec provider disponible
**Provider exclusion** : `ModelAssigner.exclude_providers(dead)` depuis le health check boot
**FrugalGPT cascade** : valide provider avant upgrade modele, `json_schema` seulement pour OpenAI
**Source** : `sage-core/config/cards.toml` — 19 modeles, 7 providers

## Stage 5 — EXECUTE (Tool-Calling Loop)

**Single entry point** : `system.run()` → pipeline → tool-calling loop (max 30 turns)
**Agent tools** : 13 outils (execute_bash, create_python_tool, create_bash_tool, 8 memoire, 2 knowledge)
**Tool-calling** : generate(tools) → LLM retourne tool_calls → execute → re-generate (boucle)
**TopologyRunner** : noeuds en ordre DAG, contexte adaptatif (mode multi-agent)
**Deduplication** : Jaccard similarity gate (S2-MAD)
**ExecutionTrace** : dataclass structuree par run (tokens, cost, latency par noeud)

Apres chaque noeud :
- **QualityEstimator** — OxiZ (code) ou DistilBERT ONNX (texte)
- **TopologyController** decide : `continue` / `upgrade_model` / `prune_node` / `reroute_topology` / `spawn_subagent` / `open_gate`
- **Escalation Conductor** : bypass → repair (reasoner tier) → topology fallback
- **Code nodes** (HyEvo) : execution sandbox
- **HITL callback** : pause optionnelle pour approbation humaine
- **Health check** au boot : circuit breaker pour providers morts

## Stage 6 — LEARN

**6 systemes de feedback** :
1. **Bandit** → met a jour quelle combo template/modele marche (SQLite, retour en Stage 4)
2. **MAP-Elites** → stocke la topologie si nouveau elite dans sa niche
3. **Episodic memory** → trace complete (SQLite, cross-session)
4. **EvolutionMemory** (CORAL Phase 1) → mutations/skills persistantes en SQLite WAL
4. **Consolidation** → transforme memoire episodique en semantique (tous les 10 steps)
5. **Online evolution** → `should_evolve()` (Rust) trigger mutation quand assez d'outcomes

> [!warning] Consolidation incomplete
> Le pipeline episodique → semantique → causal (MAGMA) est documente mais pas entierement implemente en production.
