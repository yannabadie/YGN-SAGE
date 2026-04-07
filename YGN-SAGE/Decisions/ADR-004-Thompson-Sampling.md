---
title: "ADR-004: Thompson Sampling over UCB"
type: adr
status: accepte
date: 2026-02-20
tags:
  - adr
  - strategy
  - bandit
---

# ADR-004: Thompson Sampling plutot que UCB pour le Bandit

## Contexte

Le ContextualBandit doit selectionner le meilleur couple (template, modele)
pour chaque tache. Deux familles d'algorithmes : UCB (deterministe) vs Thompson
sampling (stochastique).

## Decision

Thompson sampling avec posterieurs Beta (qualite) et Gamma (cout/latence).
Selection Pareto multi-dimensionnelle (qualite + cout + latence simultanes).

## Alternatives considerees

| Option | Pour | Contre |
|--------|------|--------|
| **Thompson sampling** | Multi-dimensionnel natif (Beta/Gamma), exploration naturelle, posterieurs conjugues | Plus complexe a implementer |
| UCB (Upper Confidence Bound) | Simple, bornes theoriques connues | Unidimensionnel, exploration moins fluide sur multi-objectif |
| Epsilon-greedy | Trivial | Exploration non-adaptative, pas de convergence garantie |

## Pourquoi Thompson

1. **Multi-dimensionnel** : qualite (Beta[0,1]) + cout (Gamma[0,inf]) + latence (Gamma[0,inf])
   necessitent des distributions differentes — Thompson le gere naturellement
2. **Posterieurs conjugues** : Beta-Bernoulli pour qualite, Gamma pour couts positifs
3. **Exploration adaptative** : l'incertitude guide naturellement l'exploration
4. **Front Pareto** : la selection multi-objectif s'integre bien avec le sampling stochastique
5. **Temporal decay** : un facteur de decay assure que les donnees recentes dominent

## Implementation

- `sage-core/src/routing/bandit.rs:1-143`
- Beta distribution (qualite) : Box-Muller Gaussian approximation, mean = alpha/(alpha+beta)
- Gamma distribution (cout/latence) : Box-Muller approximation, mean = shape/rate
- Pareto front selection a la ligne 323-331

## Evidence

- Paper : [[PILOT]] (EMNLP 2025) valide le bandit contextuel avec contraintes de budget
- Architecture docs : "Hard constraints → structural scoring → contextual bandit"
- Tests : `sage-core/tests/` bandit tests passent
