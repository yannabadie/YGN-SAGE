---
title: "MASBENCH Parallel -6pp"
type: issue
severity: P2
status: ouvert
tags:
  - issue
  - benchmark
  - regression
created: 2026-04-07
---

# MASBENCH Parallel Axis Regression

## Probleme

Sur l'axe parallel de MASBENCH : bare 36.0% vs SAGE 30.0% = **-6pp regression**.
La topologie multi-agent **nuit** aux taches naturellement paralleles.

## Hypotheses

1. **Overhead sans valeur ajoutee** : le pipeline topology ajoute de la latence et de la complexite
   sans benefice quand les taches sont deja independantes
2. **Deduplication trop aggressive** : le Jaccard gate (S2-MAD) supprime peut-etre des outputs utiles
3. **Mauvais template** : le macro_topology selector choisit peut-etre un template inadapte

## Solutions possibles

1. **Bypass topology** : detecter les taches paralleles (omega eleve) et les executer bare
2. **Template specifique** : creer un template "transparent passthrough" qui ne fait que distribuer
3. **Desactiver deduplication** sur les taches paralleles

## Impact

Remet en question l'universalite de la these "topology helps".
La these tient pour breadth (+22pp) mais pas pour parallel.
C'est coherent avec [[AdaptOrch]] qui montre le gain surtout sur les taches **difficiles**.

## Donnees

Source : `masbench_official_results.json`
