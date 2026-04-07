---
title: "Write-Gate — Memory Write Gate"
type: paper
arxiv: "2603.15994"
venue: arXiv 2026
year: 2026
status: integre
tags:
  - paper
  - memory
created: 2026-04-07
---

# Write-Gate — Composite Memory Write Gate

**arXiv** : [2603.15994](https://arxiv.org/abs/2603.15994)
**Venue** : arXiv 2026

## Resume

Gate de salience composite pour filtrer les ecritures memoire.
5 signaux ponderes determinent si une information merite d'etre stockee.
Sans le gate : 13% precision. Avec : 100%.

## Claims cles

1. 5 signaux composites pour la salience memoire
2. **100% precision avec gate vs 13% sans** — resultat spectaculaire
3. Ponderation calibree par ablation

## Ce qui est utilise dans SAGE

| Claim | Feature SAGE | Fichier | Statut |
|-------|-------------|---------|--------|
| 5-signal write gate | RustCompositeWriteGate | sage-core/src/memory/write_gate.rs:1-408 | integre |
| Weights from paper | w_confidence=0.25, w_novelty=0.30, w_reliability=0.20, w_recency=0.10, w_relevance=0.15 | write_gate.rs:48 | integre |
| Python mirror | CompositeWriteGate | sage-python/src/sage/memory/write_gate.py:3 | integre |
| Constants | Weights in constants | sage-python/src/sage/constants.py:104 | integre |

## Poids des signaux

| Signal | Poids | Role |
|--------|-------|------|
| Confidence | 25% | Certitude de l'information |
| Novelty | **30%** | Information nouvelle vs deja connue |
| Reliability | 20% | Fiabilite de la source |
| Recency | 10% | Fraicheur temporelle |
| Relevance | 15% | Pertinence pour la tache |

> Commentaire code : "All weights from arXiv 2603.15994, subject to ablation"

## Metriques

| Metrique | Sans gate | Avec gate | Delta |
|----------|----------|----------|-------|
| Precision | 13% | **100%** | +87pp |

## Notes personnelles

Le resultat le plus impressionnant de tous les papers integres.
Implementation double (Rust + Python) prouve l'importance accordee.
Les poids sont repris du paper et marques "subject to ablation" —
honnetete sur le fait qu'ils pourraient etre re-calibres pour SAGE.
