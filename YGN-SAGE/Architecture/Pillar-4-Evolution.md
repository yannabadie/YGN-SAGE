---
title: "Pilier 4 — Evolution"
type: architecture
pillar: 4
tags:
  - architecture
  - evolution
updated: 2026-04-19
---

# Pilier 4 — Evolution

## MAP-Elites + CMA-ME (Rust)

- QD search dans l'espace des topologies
- Descripteurs comportementaux configurables (4D : agent_count, max_depth, cost, model_diversity)
- Insertion Pareto dans l'archive
- CMA-ME : adaptation matrice covariance, sigma decay, warm_start

## Online Evolution

`should_evolve()` (Rust) : trigger mutation quand >5 outcomes accumules.
Les topologies s'ameliorent **entre** les taches, pas seulement pendant le training.

**Status (2026-04-19):** `[wired + empirically validated]`
- H1 commit `2cd840e` — wired `should_evolve()` + `evolve()` dans `_stage_learn`
- H4 commit `dc51976` — fix critique : ajouté `cache_topology()` après `generate()`. Sans ce fix, `record_outcome` no-op silencieusement sur le MAP-Elites archive et `should_evolve` ne déclenche jamais (caught via empirical validation — advisor warning prouvé)
- Constants : `EVOLUTION_MIN_OUTCOMES=5`, `EVOLUTION_COOLDOWN_OUTCOMES=3`, `EVOLUTION_ONLINE_POP_SIZE=5`, `EVOLUTION_ONLINE_GENERATIONS=2`
- Régression tests : `TestPipelineEvolutionWiring` (mock-level) + `TestRealEngineEvolutionLoop` (empirical end-to-end avec real Rust engine — pin la contrainte cache_topology, asserte archive grandit 0 → 7 cells sur 20 outcomes, evolve fires post-cooldown)
- Clôture la claim "SA-3 online evolution complete" de architecture.md, qui était *fausse* jusqu'au 19 avril
- Validation empirique sur un vrai smoke SWE-Lite = Phase 1.4 du plan `docs/superpowers/plans/2026-04-20-rust-first-plan.md`

## 7 Operateurs de Mutation

1. **add_node** — ajouter un noeud agent
2. **remove_node** — supprimer un noeud
3. **swap_model** — changer le modele d'un noeud
4. **rewire_edge** — recabler une connexion
5. **split_node** — diviser un noeud en deux
6. **merge_nodes** — fusionner deux noeuds
7. **mutate_prompt** — muter le prompt systeme

## LLM-as-Mutator

Mutations intelligentes via GPT avec AdaptiveMutator (Thompson sampling sur les operateurs).
Ref: [[ShinkaEvolve]] (ICLR 2026, arXiv 2601.04170)

## Validation Statistique

- Test de Wilcoxon signed-rank pour la significativite de l'evolution
- Cohen's d pour la taille d'effet

## Drift Monitor (3 signaux)

3 signaux ponderes pour detecter la degradation de performance :
- **Latency trend** (40%) : ratio seconde/premiere moitie de fenetre
- **Error rate** (40%) : proportion d'events avec erreur
- **Cost trend** (20%) : ratio de couts seconde/premiere moitie
- Actions : CONTINUE (<0.4) / SWITCH_MODEL (0.4-0.7) / RESET_AGENT (>0.7)
- Inspiration : Agent Stability Index de [[ShinkaEvolve]]

## Papers cles

- [[ShinkaEvolve]] — LLM-as-mutator, Agent Stability Index
- [[OpenSAGE]] — Self-programming agents
- [[AgentConductor]] — RL topology evolution
