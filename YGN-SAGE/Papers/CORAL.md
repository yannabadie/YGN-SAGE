---
title: "CORAL — Autonomous Multi-Agent Evolution"
type: paper
arxiv: "2604.01658"
venue: arXiv 2026
year: 2026
status: a-integrer
tags:
  - paper
  - evolution
  - memory
  - multi-agent
created: 2026-04-07
---

# CORAL — Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery

**arXiv** : [2604.01658](https://arxiv.org/abs/2604.01658)
**Code** : [github.com/Human-Agent-Society/CORAL](https://github.com/Human-Agent-Society/CORAL)
**Venue** : arXiv, avril 2026

## Resume

Framework ou des agents LLM long-running explorent de maniere autonome via
memoire partagee persistante, execution asynchrone, et interventions heartbeat.
Remplace les heuristiques d'evolution fixes par l'autonomie des agents.
**3-10x taux d'amelioration** vs evolution fixe (AlphaEvolve, FunSearch, EvoX).

## Claims cles

1. Memoire partagee persistante (attempts/notes/skills) > evolution sans memoire
2. Heartbeat 3-niveaux (reflect/consolidate/redirect) previent la stagnation
3. Autonomie agent dans les decisions d'exploration > operateurs fixes
4. 4 agents co-evoluent : 36% code cross-agent, 87% reference knowledge
5. SOTA sur 8/11 taches (Circle Packing, EPLB, Kernel Engineering 1363→1103)

## Resultats

| Benchmark | Baseline | CORAL 1-agent | CORAL 4-agents | Delta |
|-----------|----------|---------------|----------------|-------|
| Kernel Engineering (cycles, ↓) | 1363 | 1350 | **1103** | -19.1% |
| Polyominoes (↑) | — | 80.2 | **84.2** | +5.0% |
| MMD-16-2 improvement rate | 33.3% (EvoX) | — | **83.3%** | +50pp |

## Architecture

```
Agent 1 (worktree)   Agent 2 (worktree)   Agent 3 (worktree)
    |                    |                    |
    +-------- .coral/public/ (symlinks) ------+
              ├── attempts/   (JSON, keyed by commit hash)
              ├── notes/      (markdown + YAML frontmatter)
              └── skills/     (reusable patterns + preconditions)
```

- **Pas de communication directe** entre agents — coordination par memoire partagee
- **Heartbeat** : per-iteration reflect, consolidation toutes les 10 evals, redirect apres 5 stagnations
- **Isolation** : git worktrees, grader inaccessible aux agents

## Ce qui est applicable a SAGE

| Innovation CORAL | Feature SAGE cible | Fichiers | Priorite | Status |
|---|---|---|---|---|
| **Memoire evolution persistante** | EvolutionMemory (attempts + skills) | `evolution/memory.py` | **P0** | **DONE (avril 8)** |
| **Knowledge transfer cross-agent** | Skills queryable dans LLM prompts | `evolution/llm_mutator.py` | **P0** | **DONE (wire au boot)** |
| **Heartbeat stagnation redirect** | DriftMonitor → trigger evolve | `monitoring/drift_evolution.py` | **P1** | A faire |
| **Autonomie mutation** | LLM propose strategie avant mutation | `evolution/mutation_strategy.py` | **P2** | A faire |
| **Worktree parallelization** | Evolution parallele N agents | `evolution/parallel_evolution.py` | **P1** | A faire |

## Ce qui n'est PAS applicable

- **CLI (17 commandes)** : SAGE a son propre CLI (`python -m sage.bench`)
- **Claude Code comme agent** : SAGE utilise ses propres providers, pas un agent terminal
- **LiteLLM gateway** : SAGE a ProviderPool avec circuit breaker

## Problemes SAGE resolus

1. **Evolution opt-in** : avec memoire persistante, l'evolution apprend et devient fiable → peut passer en defaut
2. **MASBENCH parallel -6pp** : les skills capturent "topologie nuit aux taches paralleles" → bypass automatique
3. **Robustness 0%** : heartbeat redirect + recovery active au lieu de reset passif
4. **DAPO/GRPO stalled** : le pattern "redirect apres 5 stagnations" est exactement ce qui manque au training

## Chiffre cle

> **3-10x taux d'amelioration** vs evolution fixe.
> La cle n'est PAS l'algorithme d'evolution — c'est la **memoire persistante** qui permet
> aux mutations futures de s'appuyer sur les succes et echecs passes.

## Notes personnelles

CORAL est au pilier Evolution ce que Write-Gate est au pilier Memory :
un paper qui transforme un composant theoriquement correct mais pratiquement inerte
en un systeme qui apprend vraiment.

L'integration Phase 1 (EvolutionMemory) est un prerequis pour rendre l'evolution
activable par defaut. Sans ca, l'evolution reste un gadget opt-in.
