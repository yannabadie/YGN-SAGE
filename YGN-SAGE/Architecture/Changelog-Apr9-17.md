---
title: "Changelog April 9 → 17, 2026"
type: changelog
date: 2026-04-17
tags:
  - changelog
  - architecture
  - sprint
updated: 2026-04-17
---

# Changelog — April 9 → 17, 2026

50+ commits depuis le 8 avril en deux blocs thématiques majeurs (architecture + sprints) puis intensification monodomaine (routing F7 + ExoCortex).

## Vue d'ensemble

| Bloc | Date | Commits | Impact |
|---|---|---|---|
| Unified entry point Phases 1-3 | Apr 9-10 | 12 | -944 LOC, single execution path |
| Training pipeline deletion | Apr 15 | 1 (b2f59ee) | -4.3 GB, training move to dedicated branch |
| Autonomous Sprints 1-6 | Apr 17 AM | 7 | +27 tests, decision gate ready |
| F7 routing sequence | Apr 17 PM | 7 | role-aware tier promotion (Rust+Python) |
| ExoCortex bugs + perf | Apr 17 PM | 5 | 3 bugs fixed, manifest grew 3 → 39 |
| Agent loop fixes | Apr 17 PM | 3 | orphan-tool, PRM domain gate, tool_choice |

## Bloc 1 — Unified Entry Point (April 9-10)

**Phases 1-3 mergées**. Avant : `system.run()` et le legacy `_run_legacy` co-existaient comme 2 chemins distincts. Après : `system.run() → pipeline.run()` est le seul chemin ; les nœuds de topology sont des vrais agents instanciés via `create_node_agent_loop()` factory.

- `786838a` — agent_loop factory pour per-node topology instances (H6, H8)
- `e8401dd` — TopologyRunner dispatch LLM nodes vers agent_loop (H6, H7, H8)
- `f265dba` — pipeline crée le agent_loop factory pour multi-agent topology
- `f97ddc4` — refactor : suppression `_run_legacy` + `SAGE_AGENT_LOOP_LEGACY` + tool loop fallback (-550 LOC)
- `f212f93` — merge Phase 3 (-944 LOC net)

**Impact** : code path unique, plus de drift entre legacy et pipeline. 9 hazards adressés (H1..H9).

## Bloc 2 — Training Pipeline Removal (April 15)

`b2f59ee` — Suppression de `verl/`, `scripts/`, `data/`, `models/` (~4.3 GB). Le code training (Z3 quality labeling, SFT generation, GRPO) vit désormais sur une branche dédiée. Les checkpoints HF sont préservés (`yannabadie/sage-topology-policy-local`, `yannabadie/sage-topology-policy-v2`). `SAGE_ENABLE_PATH6=1` charge un checkpoint local à inference time.

## Bloc 3 — Sprints autonomes 1-6 (April 17 AM)

Session multi-sprint. Voir `project_april17_autonomous_sprints` (memory).

- `bb70502` Sprint 2 — SWE-bench 3 gaps : `system_hint=3` forwarding, prompt fusion, multi-turn mandate
- `93f911d` Sprint 3 — ToolForge E2E wiring : `execute_tool_call` ouvre un CreationTicket et retry transparent
- `13463fb` Sprint 4 — `sage_recurse` tool : The Conductor-style self-invocation (depth tracked via contextvars)
- `3d898a3` Sprint 5 — SWE-bench Pro support + 4-config ablation runner (full / no_recurse / no_toolforge / bare)
- `2878265` Sprint 6 — decision gate framework + `decide_next_phase.py`

**+200 tests intégrés** (27 nouveaux + carryover). Cherry-picks CORAL : `984d7e6` (3 P1 fixes), `47784c7`+`d73cfa2` (TopologyController), `30ee004` (remove S2+sequential bypass), `00097d8` (-1794 LOC ComplexityRouter+ShadowRouter), `ff41e53` (kNN exact-match → 100% GT sur 60).

## Bloc 4 — F7 Routing Sequence (April 17 PM)

Voir [[ADR-007-F7-Routing|ADR-007]].

- `029701c` — F7 base : Rust ModelAssigner reçoit `task_system` et applique `effective_system(role, node, task)` avec promotion `max(node, task-1)` pour producer roles
- `2839d95` — F7 floor domain-aware : math/formal S3 → S3 (full reasoner), autres S3 → S2
- `4efa37d` — Audit `is_sink_role` : ajout de `mixer/judge/verifier/solver` (caught regression où F7 promouvait formal_solver's solver de S1 → S3, remplaçant Rust math par LLM call)
- `4c1b52a` — F7 FrugalGPT wiring : Python passe `task_system` à `assign_single_node` (cascade upgrade respecte le tier)
- `6d198db` — Apply advisor+Codex review : `ingested = max` pas sum, sink-drift Rust test, S1 strict-mandate template guard
- `ae9d10b` — Test fix : `test_oxiz_pipeline._MockAssigner` hérite de la signature F7
- `091812d` — F8 collateral : Gemini 3.x temperature=1.0 forced (degenerate au-dessous)

## Bloc 5 — ExoCortex Bugs + Perf (April 17 PM)

3 bugs réels qui faisaient que `report.ingested=0` malgré des "211 papers discovered/curated" mensongers.

- `9b8d91c` — pipeline.py if/else → both backends (Qdrant + ExoCortex)
- `2c994e0` — upload polling timeout (90s default) + manifest `store_name` persistence
- `60793b7` — bump timeout 90s → 300s (v1 hit 31/31 timeouts)
- `2780bbb` — bump timeout 300s → 600s (v2 still hit 5+ timeouts)
- `d40dc4e` — perf : reuse cached `genai.Client` au lieu de re-créer à chaque upload (~63s économisés sur 211 uploads)

**Manifest passé de 3 (verify run) → 39 papers** (rattrapage v3 actif au moment du commit).

## Bloc 6 — Agent Loop Fixes (April 17 PM)

3 bugs latents révélés par les smokes SWE-bench v1/v2/v3 successifs.

- `591d3c4` — orphan-tool guard dans message truncation : MiniMax-m2.7 rejetait les requêtes (400 tool id not found) parce que la troncature `messages[:2] + messages[-(MAX-2):]` créait des `tool` orphelins
- `97dfb2b` — PRM domain gate symétrique au F7 floor : Z3 PRM uniquement sur math/formal (sur code/general → AVR niveau 2). Pré-fix : `<think>` block requirement → 17 CEGAR failures + 6 RESET_AGENT × 6 SWITCH_MODEL → 0/3 patches
- `da839dc` — `tool_choice="required"` sur coder/actor steps 1-2 : F6 _CODER mandate "AT LEAST 3 execute_bash" empiriquement ignoré (smoke v3 : 0 tool calls / 3 tasks). Force l'API à imposer l'usage des tools

## Métriques courantes (au 2026-04-17 18:00)

| Avant | Après | Métrique |
|---|---|---|
| 2001 | **1896** Python tests | training tests retirés Apr 15 |
| 429 | **441** Rust tests | +12 (F7 floor + sink + drift tests) |
| 0 | **39** ExoCortex papers | rattrapage v3 actif après bugs fixés |
| smoke 1/3 | smoke v3 **2/3** | F2 + F7 + PRM-gate |
| smoke v3 0 tool_calls | smoke v4 testing | tool_choice forcing en cours |

## ADRs ajoutés

- [[ADR-007-F7-Routing]] — domain-aware floor + sink classification (April 17 PM)
- [[ADR-008-PRM-Gate-Domain]] — Z3 PRM uniquement sur math/formal (April 17 PM)

## Memory files créés

- `project_april17_autonomous_sprints.md` (Sprints 1-6)
- `project_april17_session_f7_exocortex.md` (F7 + ExoCortex)
- `project_april15_training_parked.md` (training branch)

## Source de vérité

`git log --since=2026-04-09 --no-merges` pour la liste exhaustive. Ce fichier doit être régénéré (ou étendu) lors du prochain refactor majeur.
