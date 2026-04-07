---
title: "Pilier 3 — Memory"
type: architecture
pillar: 3
tags:
  - architecture
  - memory
updated: 2026-04-07
---

# Pilier 3 — Memory

Architecture 4-tier inspiree CoALA (cognitive architecture).

## Tier 0 — Working Memory (Rust, Arrow)
- Zero-copy columnar STM
- Operations SIMD
- ULID chunk IDs
- **Status** : `[confirmed]`

## Tier 1 — Episodic Memory (SQLite)
- Log d'evenements cross-session avec requetes temporelles
- **Status** : `[confirmed]`

## Tier 2 — Semantic Memory
- Graphe entity-relation avec embeddings (arctic-embed-m 768-dim)
- **Status** : `[confirmed]` pour le stockage, **partiel** pour la consolidation

## Tier 3 — ExoCortex (Google File Search RAG)
- 500+ papers de recherche
- Pipeline de decouverte arXiv (sage-discover)
- **Status** : `[confirmed]` pour le RAG, **casse** pour le pipeline discover

## S-MMU (Selective Memory Management Unit, Rust)

4 vues du graphe :
- **Temporelle** : quand
- **Semantique** : quoi (embeddings)
- **Causale** : pourquoi (chaines tool_call → result → decision)
- **Entites** : qui/quoi (noeuds nommes)

Fonctionnalites :
- Eviction utility-based (recency x access_count)
- Auto-GC a 10K chunks
- Write gate composite (5 signaux) :
  - Salience 25%
  - Novelty 30%
  - Reliability 20%
  - Recency 10%
  - Relevance 15%
  - Ref: [[Write-Gate|arXiv 2603.15994]] — 100% precision vs 13% sans gate

## Causal Wiring

`agent_loop` trace les chaines : tool_call → result → decision.
Le graphe causal est injecte dans les prompts LLM pour du raisonnement informe.
Ref: [[AMA-Bench]] (2602.22769) — la memoire echoue sans causalite.

## Consolidation

Tous les 10 steps, les memoires episodiques sont transformees en relations semantiques.
Equivalent cognitif du sommeil.
Ref: [[MAGMA]] — +45.5% raisonnement avec consolidation.

> [!warning] Consolidation incomplete
> Le design est documente et le code existe, mais la consolidation complete
> (episodique → semantique → causal avec MAGMA) n'est pas validee en production.
> Le pipeline fonctionne en test, pas prouve a l'echelle.
