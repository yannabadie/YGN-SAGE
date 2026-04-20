---
title: "ADR-011: Singleton AgentLoop vs Factory Asymmetry"
type: adr
status: draft
date: 2026-04-20
tags:
  - adr
  - architecture
  - placeholder
---

# ADR-011: Singleton AgentLoop vs Factory Asymmetry

## Status

**DRAFT — placeholder**. Cet ADR sera finalisé après les commits Phase
1.1 à 1.3 du plan `docs/superpowers/plans/2026-04-20-rust-first-plan.md`.

Numéro réservé pour éviter qu'un ADR parallèle le prenne.

## Contexte prévu (à compléter)

Le pipeline YGN-SAGE a deux chemins d'exécution LLM :
- **Multi-node topology** (factory path) : `topology/runner.py` invoque
  `agent_loop_factory.create_node_agent_loop()` pour chaque node.
  Fresh instance par node, config. dérivée de `node.role` et `ctx.system`.
- **Single-agent bypass** (singleton path) : `pipeline.py:941+` réutilise
  un `self._agent_loop` créé à `boot.py:272` avec config par défaut.

Les H5 et H6 fixes (2026-04-19) ont révélé que la branche bypass
n'appliquait PAS les configurations que la factory applique :
- `write_gate` / `gate_current_task` / `gate_source_tier` (H5)
- `_on_drift` callback (H6)

Phases 1.1–1.3 audit les trois derniers champs suspects :
- `max_steps` (singleton figé à 20, factory scale 5/10/20)
- `stall_cap` (singleton défault 0, factory = max_steps - 1)
- `tools` (singleton = all, factory = role-filtered) — probablement
  false-positive

## Décision (à formuler après 1.3)

*Draft :* Codifier la règle : **toute state set par la factory dans
`create_node_agent_loop` DOIT être set par la branche bypass dans
`pipeline.py:941+`**. Source de vérité : le body de cet ADR, tenu à
jour au fur et à mesure.

## Conséquences (à compléter)

- Tout changement futur dans `create_node_agent_loop` DOIT être mirré
  dans `pipeline.py` branche bypass.
- Les regression tests (pattern H5/H6) DOIVENT couvrir les deux paths.

## Références

- ADR-010 (méthodologie bypass-audit)
- Plan : `docs/superpowers/plans/2026-04-20-rust-first-plan.md` §Phase 1
- Commits H5 / H6 : `27a9a4c`, `aa348e1`
- `docs/audits/bypass-patterns.md` §"Two-path check"
