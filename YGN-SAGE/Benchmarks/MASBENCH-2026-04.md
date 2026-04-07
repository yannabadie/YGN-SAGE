---
title: "MASBENCH — Avril 2026"
type: benchmark
benchmark: masbench
date: 2026-04-01
tags:
  - benchmark
  - masbench
---

# MASBENCH — Avril 2026

## Configuration

- **Modele(s)** : Multi-provider (DeepSeek primaire)
- **Topologie** : DynamicTopologyEngine (6-path)
- **Source** : `masbench_official_results.json`

## Resultats par axe

| Axe | Bare % | SAGE % | Delta | Duree |
|-----|--------|--------|-------|-------|
| **breadth** | 32.0 | 54.0 | **+22.0pp** | ~2.8h |
| **depth** | 8.0 | 10.0 | +2.0pp | ~5.8h |
| **horizon** | 4.0 | 8.0 | +4.0pp | ~26min |
| **parallel** | 36.0 | 30.0 | **-6.0pp** | ~17min |
| **robustness** | 0.0 | 0.0 | 0.0pp | ~23min |

## Delta revendique : +27pp

> [!warning] Ce chiffre est une moyenne non-ponderee
> (22 + 2 + 4 + (-6) + 0) / 5 = 4.4pp en moyenne arithmetique.
> Le "+27pp" dans le README est le delta breadth (22) + les positifs.
> C'est du marketing, pas de la science.
>
> **Lecture honnete** : SAGE aide significativement sur breadth (+22pp),
> marginalement sur depth/horizon (+2/+4pp), et **regresse** sur parallel (-6pp).
> Robustness : 0% les deux — a debugger.

## Observations

1. **Breadth** : la topologie decompose bien les taches larges
2. **Depth** : la topologie aide peu sur le raisonnement en profondeur
3. **Parallel** : la topologie **nuit** aux taches naturellement paralleles — overhead sans valeur ajoutee
4. **Robustness** : ni bare ni SAGE ne passent — possible bug dans le benchmark ou taches hors scope

## Analyse statistique (McNemar + Cohen's d, avril 7 2026)

| Axe | Delta | McNemar p | Cohen's d | CI 95% | Sig. |
|-----|-------|-----------|-----------|--------|------|
| **breadth** | **+22pp** | **p=0.015** | **d=+0.456** | [+8, +38]pp | **OUI** |
| depth | +2pp | p=1.000 | d=+0.070 | [-8, +12]pp | NON |
| horizon | +4pp | p=0.480 | d=+0.169 | [0, +10]pp | NON |
| parallel | -6pp | p=0.371 | d=-0.128 | [-14, +2]pp | NON |
| robustness | +0pp | p=1.000 | d=+0.000 | [0, 0]pp | NON |

> [!success] Breadth : SEUL axe statistiquement significatif
> p=0.015, d=0.456 (effet moyen), CI ne croise pas zero.
> SAGE recupere 14 taches que bare rate, perd seulement 3.

> [!warning] 4 axes non significatifs
> Le "+27pp non-pondere" est trompeur — seul breadth est prouve statistiquement.

## Questions ouvertes

- Pourquoi robustness 0% des deux cotes ? Bug benchmark ou taches impossibles ?
- Parallel -6pp : faut-il detecter les taches paralleles et bypasser la topologie ?
- Le delta breadth (+22pp) est-il stable cross-run ?
- Peut-on augmenter N pour depth/horizon (taille effet trop petite pour N=50) ?
