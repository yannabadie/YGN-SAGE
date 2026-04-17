---
title: Pipeline CLASSIFY-LEARN
type: architecture
tags:
  - architecture
  - pipeline
  - f7
updated: 2026-04-17
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
**FrugalGPT cascade** : valide provider avant upgrade modele, `json_schema` seulement pour OpenAI, **forwarde `task_system` depuis Apr 17** (voir [[ADR-007-F7-Routing]])
**Source** : `sage-core/config/cards.toml` — 19 modeles, 7 providers

### F7 — Role-aware tier promotion (Apr 17)

`pipeline._stage_assign_models` forwarde `task_system=ctx.system` au Rust
`assigner.assign_models(...)`. Le scoring lit alors `effective_system(role,
node, task_system, task_domain)` au lieu de la `node.system` brute :

```
producer role + S3 task + math/formal domain  → S3 floor (full reasoner)
producer role + S3 task + autres domaines     → S2 floor (mid-tier reasoner)
producer role + S2 task                       → S1 floor (no-op)
sink role (synthesizer/aggregator/mixer/      → keep node.system
  judge/verifier/solver/formatter/output)
```

- **Sink classification** : liste maintenue dans `SINK_ROLES` const, audit
  `grep -B 1 SINK_NODE_PROMPT sage-core/src/topology/templates.rs` par
  `test_sink_drift_templates_match_classifier`.
- **Domain split** : `is_high_rigour_domain(domain)` substring-match sur
  `math` ou `formal`. Cards exposent `math` et `formal` columns.
- **FrugalGPT cascade** : `topology_controller._resolve_upgrade_model` et
  `pipeline.py` Stage 4 cascade forwardent aussi `task_system` à
  `assign_single_node`. TypeError fallback pour bindings antérieurs.

**Régression évitée par audit** : sans la classification sink, F7 promouvait
`formal_solver`'s `solver` node (model_id="", S1, Rust math evaluator)
de S1 → S3 sur math task → remplaçait Rust math gratuit par LLM call à
0.10 USD. Caught par `test_formal_solver_sink_protected_on_math_s3`.

## Stage 5 — EXECUTE (Tool-Calling Loop)

**Single entry point** : `system.run()` → pipeline.run() (Apr 9-10 unified)
**Per-node max_steps** : F1 scale par task tier (S1=5, S2=10, S3=20)
**Agent tools** : **14** outils (execute_bash, create_python_tool, create_bash_tool, 8 memoire, 2 knowledge, +`sage_recurse` Apr 17 Sprint 4)
**Tool-calling** : generate(tools) → LLM retourne tool_calls → execute → re-generate (boucle)
**Tool choice** : depuis `da839dc` (Apr 17), `tool_choice="required"` sur coder/actor steps 1-2 pour forcer F6 mandate empiriquement ignoré
**TopologyRunner** : noeuds en ordre DAG via `agent_loop` factory (Apr 9-10 Phase 2 — vrais agents par node)
**Deduplication** : Jaccard similarity gate (S2-MAD)
**ExecutionTrace** : dataclass structuree par run (tokens, cost, latency par noeud)

### Validation level — domain-symetrique au F7 (Apr 17)

```
sink role (verifier/formatter/aggregator/critic) → validation = 0
S3 task + math/formal                            → validation = 3 (PRM Z3)
S3 task + autres                                 → validation = 2 (AVR)
S2 task                                          → validation = 2
S1 task                                          → validation = 1
```

Voir [[ADR-008-PRM-Gate-Domain]]. Pré-fix, S3 + code allait chercher Z3
PRM avec `<think>` blocks → 17 CEGAR failures × 6 RESET_AGENT × 6
SWITCH_MODEL → 0/3 patches sur smoke v2.

Apres chaque noeud :
- **QualityEstimator** — OxiZ (code) ou DistilBERT ONNX (texte)
- **TopologyController** decide : `continue` / `upgrade_model` / `prune_node` / `reroute_topology` / `spawn_subagent` / `open_gate`
- **Escalation Conductor** : bypass → repair (reasoner tier) → topology fallback
- **Code nodes** (HyEvo) : execution sandbox
- **HITL callback** : pause optionnelle pour approbation humaine
- **Health check** au boot : circuit breaker pour providers morts
- **Orphan-tool guard** dans message truncation (Apr 17 `591d3c4`) : strict providers (MiniMax) rejetaient orphan `tool_call_id`

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
