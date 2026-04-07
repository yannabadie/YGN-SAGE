---
title: "Pilier 4 — Evolution"
type: architecture
pillar: 4
tags:
  - architecture
  - evolution
updated: 2026-04-07
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

> [!warning] Opt-in
> L'evolution online n'est pas activee par defaut dans `system.run()`.
> Necessite `_auto_evolve=True`.

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
