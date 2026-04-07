---
title: "ADR-005: ShadowRouter Deprecated"
type: adr
status: accepte
date: 2026-03-15
tags:
  - adr
  - routing
  - deprecation
---

# ADR-005: Depreciation du ShadowRouter

## Contexte

Le ShadowRouter etait un systeme dual-path qui comparait en parallele :
- Le **SystemRouter Rust** (routeur principal)
- Le **AdaptiveRouter Python** (routeur candidat)

Les divergences etaient loguees en JSONL pour valider une eventuelle promotion
du routeur Python. Un gate Phase 5 exigeait <5% de divergence sur 1000+ traces
pour promouvoir le routeur Python.

## Decision

ShadowRouter deprecie. Desactive par defaut. Le Rust SystemRouter reste primaire.

## Donnees

- **1090 traces collectees**
- **49.6% de divergence** entre Rust et Python
- Gate requis : <5% divergence
- **Resultat : FAIL massif** (49.6% >> 5%)

## Consequences

- Positives : simplifie le code, pas de double-execution couteux
- Negatives : le routeur Python n'est pas ameliore (pas de feedback loop)
- Le code est conserve (pas supprime) pour de futures collectes de traces
- Activation : `SAGE_ENABLE_SHADOW=1` (opt-in)

## Lecon

49.6% de divergence signifie que les deux routeurs prennent des decisions
fondamentalement differentes sur la moitie des taches. Ce n'est pas un probleme
de calibration — c'est une difference structurelle. Le Python n'a pas ete promu
car il n'a pas prouve sa superiorite.

## Fichier

`sage-python/src/sage/routing/shadow.py:8-18,82,96,138`
Doc : `AI-ARCHITECTURE.md:696-738`
