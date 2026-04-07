---
title: "AdaptOrch — Adaptive Orchestration"
type: paper
arxiv: "2602.16873"
venue: arXiv 2026
year: 2026
status: integre
tags:
  - paper
  - topology
  - these-centrale
created: 2026-04-07
---

# AdaptOrch — Adaptive Orchestration

**arXiv** : [2602.16873](https://arxiv.org/abs/2602.16873)
**Venue** : arXiv 2026

## Resume

Demontre empiriquement que la **variance de la topologie** a un impact 20x superieur
a la **variance du modele** sur les taches difficiles.

## Claims cles

1. Var_topology / Var_model >= 20 sur les taches complexes (coding: ratio >= 48.7)
2. Le choix de l'architecture multi-agent est plus important que le choix du LLM
3. Les topologies doivent etre adaptatives, pas fixes
4. **Topologie nuit sur taches atomiques** (|V|=1) : -2.1pp sur GPQA

## Seuils numeriques (Table 5 + Algorithm 1)

| DAG Cluster | Topologie optimale | Gain vs single | Seuil |
|---|---|---|---|
| Chain (sequential) | tau_S | +3.8pp | omega < 2 |
| **Wide-Shallow** | tau_P | **+12.6pp** | **omega >= 3** |
| Deep-Narrow | tau_H | +9.2pp | delta >= 5 |
| Diamond | tau_X | +11.4pp | omega >= 2, gamma < 0.4 |

Thresholds Algorithm 1 : theta_omega=0.5, theta_gamma=0.6, theta_delta=5

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| Topology > model | These centrale du projet | Tout | integre |
| Var ratio >= 20 | Justification du pipeline 6-path | architecture.md | cite |
| omega/delta/gamma seuils | select_macro_topology() | pipeline.py | integre |

## Implication pour BigCodeBench Hard

> [!warning] BigCodeBench Hard a omega ~1.3
> Les taches BigCodeBench Hard sont majoritairement atomiques (omega moyen = 1.32).
> AdaptOrch predit un gain de seulement +3.8pp (Chain cluster).
> Le vrai gain SAGE sur BigCodeBench vient du **repair (AVR)** et du **model selection**,
> pas de la topologie multi-agent. La these topology > model tient sur
> **MASBENCH breadth** (omega eleve) et **SWE-bench** (omega ~3.4), pas sur BigCodeBench.

## Ce qui n'a PAS ete retenu

- Rien de rejete — c'est le paper fondateur de la these du projet

## Metriques rapportees dans le paper

| Benchmark | Score paper | Score SAGE | Delta |
|-----------|-----------|-----------|-------|
| Var ratio taches difficiles | >= 20 | non mesure directement | — |

## Notes personnelles

**C'est LE paper qui justifie l'existence du projet.** Si ce result ne tient pas,
la valeur ajoutee de SAGE par rapport a un simple appel LLM est discutable.
Les resultats MASBENCH (+27pp non-pondere) supportent la these,
mais la regression sur l'axe parallel (-6pp) montre que ce n'est pas universel.
