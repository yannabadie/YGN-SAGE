---
title: "ADR-011: Singleton AgentLoop vs Factory Asymmetry"
type: adr
status: accepted
date: 2026-04-20
tags:
  - adr
  - architecture
  - bypass-audit
  - pipeline
---

# ADR-011: Singleton AgentLoop vs Factory Asymmetry

## Status

**ACCEPTED** — 2026-04-20 (session 1 of Rust-First plan, four commits
below).

## Context

Le pipeline YGN-SAGE expose deux chemins d'exécution LLM côte-à-côte :

1. **Factory path (multi-node topology).** `topology/runner.py` appelle
   `agent_loop_factory.create_node_agent_loop()` pour chaque node
   avant exécution. Chaque nœud reçoit une instance fraîche
   d'`AgentLoop` configurée par `node.role` et `ctx.system` :
   `tools` filtrés, `max_steps` scale 5/10/20, `stall_after_tool_steps`
   = `max_steps - 1` (ou 0 pour S1), `validation_level` selon le rôle
   et le domaine, `write_gate`/`_on_drift` câblés.

2. **Singleton path (single-agent bypass).** `pipeline.py:~941` réutilise
   `self._agent_loop`, un singleton construit à `boot.py:272`. Avant
   cet ADR, le bypass héritait des *defaults de boot* sans
   re-configurer les attributs que la factory aurait définis pour un
   node équivalent. Déclenché quand `ctx.topology is None` ou
   `node_count() <= 1` (inclut S1 non-math, SAGE_ABLATION_NO_TOPOLOGY=1,
   topologies dégradées en single-node par le budget check).

Les soirées des 2026-04-19 (G + H1-H6) et 2026-04-20 (H7-H10) ont
catalogué **10 bypasses** de ce pattern, tous cluster autour du même
anti-pattern : une feature wired ailleurs mais jamais re-câblée sur
le bypass path. Liste vérifiée de champs re-configurés par la factory
et leur statut sur le bypass :

| Attribut | Factory | Bypass (avant fix) | Commit-fix |
|---|---|---|---|
| `_skip_routing` | ✅ | ✅ | (déjà câblé) |
| `_current_topology` | ✅ | ✅ | (déjà câblé) |
| `validation_level` | ✅ (role + system + domain) | ✅ (system-only) | (déjà câblé) |
| `write_gate` + `gate_current_task` + `gate_source_tier` | ✅ (G-series) | ❌ | H5 `27a9a4c` |
| `_on_drift` | ✅ (D6 + runner) | ❌ | H6 `aa348e1` |
| `max_steps` | ✅ (5/10/20) | ❌ (figé à 20) | H7 `b7ced9d` |
| `stall_after_tool_steps` | ✅ (`max_steps - 1`) | ❌ (0) | H8 `0b5a272` |
| `tools` filter | ✅ (par role) | N/A (pas de role concept) | false positive (plan 1.3, `e5e3811`) |

Et deux bypasses non-AgentLoop mais de même famille (pipeline écrit un
identifiant incompatible avec le cache / une branche de code ne cache
pas) :

| Surface | Avant-fix | Commit-fix |
|---|---|---|
| `ctx.topology_id` = descriptor-keyed au lieu d'ULID | cache miss → archive stagnant | H9 `c65659b` |
| `cache_topology` absent sur template branch | même chose sur le chemin prod-dominant | H10 `c65659b` |

## Décision

**Toute state que la factory applique dans `create_node_agent_loop` DOIT
être appliquée par la branche bypass dans `pipeline.py:941+`.**

Règles opérationnelles :

1. **Source de vérité pour la liste de champs** : le corps de cet ADR
   (tableau ci-dessus). À mettre à jour immédiatement à chaque
   addition / suppression d'un champ dans la factory.
2. **Regression test à chaque commit factory** : tout changement dans
   `create_node_agent_loop` DOIT être accompagné d'un test pinning le
   même comportement sur la bypass branche (pattern
   `_SpyAgentLoop` — voir `test_pipeline.py::test_pipeline_single_agent_*`).
3. **Test empirical avec objets Rust réels** pour toute feature qui
   mute de l'état (archive, bandit posteriors, memory). Les mocks ne
   suffisent pas — bypass-patterns.md §4. Skip avec
   `@pytest.mark.skipif(not _HAS_SAGE_CORE)`.
4. **Audit après chaque wiring commit.** Le pattern chaîne :
   H1 → H4, G-series → H5, H5 → H6, H1+H4 → H9+H10. Chaque
   fix de bypass révèle potentiellement un bypass downstream.
5. **Commit message discipline.** Les phrases « wires X into Y »,
   « X now called from Y », « enables online / per-request Z », 
   « closes the SA-N architecture claim » doivent déclencher une
   exigence de test-d'intégration empirical avant merge
   (bypass-patterns.md §"Red-flag commit patterns").

## Conséquences

**Positives.**
- La SA-3 architecture claim ("online evolution wired and firing")
  est passée de *believing-claim* (H1 seul) à *verified-claim*
  (H1+H4+H9+H10 chaîne empiriquement validée).
- Les trois attributs suspects de la singleton AgentLoop
  (`max_steps`, `stall_cap`, `tools`) sont soit corrigés soit
  explicitement documentés comme faux-positifs.
- Un workflow reproductible pour future sessions :
  `docs/audits/bypass-patterns.md` + `docs/superpowers/plans/…`.

**Négatives.**
- Toute addition future à `create_node_agent_loop` a maintenant une
  obligation de mirror + test sur le bypass path, sinon silent
  regression. Discipline maintenue via tests ci-dessus.
- Le TopologyController Python reste la plus grosse violation Critical
  Directive #1 encore en vie. Phase 2 de `rust-first-plan` (ADR-012)
  l'adresse.

**Neutres.**
- Le codepath bypass reste un codepath distinct avec ses propres
  bugs potentiels — pas de consolidation forcée. Justification : il
  sert la « single-agent fast path » performance use-case (S1 skip
  topology; single-node templates). L'existence de deux paths est OK
  tant qu'ils sont explicitement synchronisés.

## Alternatives considérées

1. **Unifier factory + bypass en UN SEUL path** (toutes les runs
   passent par `create_node_agent_loop` même pour un seul node).
   Rejetée — overhead d'instanciation pour le cas S1 courant,
   complique le caching mémoire intra-tâche.
2. **Forcer la bypass branche à invoquer la factory** pour un node
   synthétique. Moins invasif mais perdrait le bénéfice du cache
   d'AgentLoop singleton (mémoire chaude, embedder Faiss, etc.).
3. **Status quo (laisser les asymétries se résoudre ad-hoc)**. Rejetée
   — 10 bypasses en 2 jours, pas tenable sans règle formelle.

## Références

- ADR-010 (méthodologie bypass-audit)
- Plan : `docs/superpowers/plans/2026-04-20-rust-first-plan.md` §Phase 1
- `docs/audits/bypass-patterns.md` §"Two-path check" + §4 empirical validation
- `docs/benchmarks/2026-04-20-archive-growth-smoke.md` (preuve empirique
  du bypass chaîne H1 → H4 → H9+H10)
- Commits : `27a9a4c` (H5), `aa348e1` (H6), `b7ced9d` (H7), `0b5a272` (H8),
  `e5e3811` (1.3 false positive), `c65659b` (H9+H10)
