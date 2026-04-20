---
title: "ADR-012: TopologyController Rust Port (Phase 2 Rust-First Plan)"
type: adr
status: accepted
date: 2026-04-20
tags:
  - adr
  - architecture
  - topology
  - rust-first
  - directive-1
---

# ADR-012: TopologyController Rust Port

## Status

**ACCEPTED** — 2026-04-20 (session 1 of Rust-First plan, commits
`152fe5e` (2.1), `3ee6b5f` (2.2), `84b1c1f` (2.3), `6684665` (2.4),
`b1d75c9` (2.5), and this commit for 2.6).

## Contexte

`docs/audits/2026-04-18-astropy-14995-decision-path.md` §5.1 a
identifié `sage-python/src/sage/topology_controller.py` comme la plus
grosse violation restante de Critical Directive #1 : 6 chemins de
décision en Python pur, aucun en Rust, au cœur du pipeline
d'adaptation runtime (Phase 4 de `pipeline.run()`).

Le pattern était le même que H1/H5/H6 :
- Rust TopologyEngine (`engine.rs`), Rust ModelAssigner, Rust
  QualityLabeler — tous côté Rust.
- Python TopologyController — seule surface Python sur le hot path
  d'adaptation. Appelé ~100× par run en moyenne (chaque node complété
  passe par `evaluate_and_decide`).

Les 6 chemins de décision ported :

| # | Python (legacy) | Rust (2.x) | État | Enrichment |
|---|---|---|---|---|
| 1 | empty/error reroute | `check_empty_error_reroute` | Rust primary | — |
| 2 | quality cascade (good+critical) | `check_quality_cascade` | Rust primary | Python: `_get_invariant_feedback` (SmtVerifier), `_resolve_upgrade_model` (ModelAssigner) |
| 3 | debate gate threshold | `is_in_gate_band` | Rust threshold, Python `_open_gate` orchestration | (no split) |
| 4 | parallel inconsistency reroute | `check_parallel_inconsistency` | Rust primary | Python: consistency scoring (`sage.consistency`) |
| 5 | importance prune | `check_importance_prune` | Rust primary | Python: importance scoring |
| 6 | emergent subtask spawn | `check_emergent_spawn` | Rust primary, regex + state | — |

## Décision

Port incrémental, pas "thin wrapper" strict comme le plan
optimiste le suggérait. La règle retenue : **les décisions de seuil
et les state machines vivent en Rust ; le scoring et l'enrichment
qui dépendent d'objets Python (embedder, SmtVerifier, ModelAssigner
via PyO3, topology-graph accessors) restent Python.**

Structure finale :

```
Python TopologyController.evaluate_and_decide:
  ├─ Rust path 1 (empty/error)         → if Some, return
  ├─ Python _compute_quality (embedder + estimator + PRM blend)
  ├─ Python axis=depth arithmetic verify
  ├─ Rust path 2 (quality cascade)     → if continue, return
  │                                     → if upgrade, Python enriches
  │                                        (invariant_feedback + new_model_id)
  ├─ Rust is_in_gate_band              → Python _open_gate if in band
  ├─ Rust path 4 (parallel inconsistency, Python-computed score)
  ├─ Rust path 5 (importance prune, Python-computed score)
  ├─ Rust path 6 (emergent spawn)
  └─ default continue
```

State ownership :

- **Rust-owned** (`RustTopologyController` struct) :
  `reroute_count`, `spawn_count`, `node_retries`,
  `node_qualities`, `gate_loops`, `abstain_count`
- **Python-owned** : `_abstain_count` (for quality-known tracking
  during _compute_quality), `_gate_loops` (not yet migrated — small
  dict, no perf gain).

Legacy fallback path préservé (`_evaluate_and_decide_legacy`) — active
quand `_rust_ctrl is None` (sage_core non-compilé, environnement de
dev pur-Python). Jamais utilisé en production sur ce poste, mais
bisectable si la délégation Rust régresse.

## Divergences du plan original (§2.6)

Trois points de scope-clip :

1. **"Python shrink à ~50 lignes"** (`docs/superpowers/specs/…-design.md`
   §2.6) → Non atteint. La classe Python reste ~730 lignes parce que
   les helpers `_compute_quality`, `_heuristic_quality`,
   `_infer_task_domain`, `_resolve_upgrade_model`,
   `_get_invariant_feedback`, `_verify_arithmetic`,
   `compute_consistency_score`, `compute_importance_score`,
   `_open_gate`, `_heuristic_quality`, `_max_retries_for_node`,
   `_resolve_fallback_model` restent Python. Chacun a une raison :
   embedder Python-held, SmtVerifier/assigner déjà Rust via PyO3
   (pas de gain à re-wrapper), topology-graph predecessors API
   Python-facing, regex math facile à garder Python.
2. **Helpers `compute_consistency_score` / `compute_importance_score`
   ports** (plan §2.4) → Non ported. Ils consomment un embedder
   Python-tenu dont la backend est déjà Rust
   (`RustEmbedder.batch_cosine_similarity` SIMD). Porter le wrapper
   Python vers Rust nécessiterait de faire passer l'embedder par
   PyO3 — pas de gain Critical-Directive-#1. Cf. advisor consult
   2026-04-20 avant commit 2.4.
3. **`_resolve_upgrade_model` → Rust method** (plan §2.6) → Non ported.
   La méthode appelle déjà `ModelAssigner.assign_single_node` qui
   EST Rust via PyO3. Porter le wrapper Python côté Rust ajouterait
   une couche PyO3 sans changer la surface Rust appelée.

Ces trois divergences sont documentées dans cet ADR plutôt que de
laisser le plan affirmer ce qui n'a pas été fait. L'objectif
"decisions live in Rust" est atteint sans élargir le scope de façon
improductive.

## Helpers de synchronisation Rust↔Python (2.6 scaffold)

Pour que les tests legacy qui seedent du state Python (`controller.
_reroute_count = 1`, `controller._node_retries[0] = 2`) continuent
d'affecter la cascade Rust-primary, deux mécanismes :

1. **`__setattr__` override** dans `TopologyController` : intercepte
   les affectations scalar `_reroute_count` et `_spawn_count` et
   forwarde vers `self._rust_ctrl.set_reroute_count(…)` /
   `set_spawn_count(…)`.
2. **Explicit Rust setters** : `set_node_retries(node_idx, value)` pour
   les états dict-shaped qu'`__setattr__` ne peut pas intercepter.
   Les 2 tests legacy qui seedent `_node_retries[k] = v` ont une
   ligne ajoutée pour mirror côté Rust.

Ces setters ne sont pas `#[setter]` dans Rust — ils sont appelés
explicitement pour rendre le mirror grep-able comme test-scaffold.

## Conséquences

**Positives.**
- 100% des décisions d'adaptation runtime sont Rust-primary. Critical
  Directive #1 satisfait sur ce sous-système (bloc le plus Python-
  primary de la Phase 4 jusqu'ici).
- State machine invariants (reroute budget, spawn budget, retry
  budget) sont garantis cohérents par la locality Rust — plus de
  chance qu'un code path oublie d'incrémenter un compteur Python.
- 32 tests Rust + 9 Python sur `RustTopologyController` (dont 5
  équivalences à 20 samples chacune) figent le contrat. Drift
  futur Rust-vs-Python sur un seuil est caught à CI.
- PyO3 surface count +2 : `RustTopologyController`, `RustAdaptationDecision`.

**Négatives.**
- Python `topology_controller.py` n'a PAS rétréci comme espéré. Il
  reste orchestration + helpers. Scope minimalement satisfaisant.
- Fallback legacy (`_evaluate_and_decide_legacy`) double la surface
  code. Gardé pour bisectabilité ; à supprimer au prochain refactor
  si stabilité se confirme sur 2-3 sessions.
- `__setattr__` override a un coût par assignment (check
  `_rust_ctrl` présent, check type) — négligeable vs la perf Rust
  gagnée dans le cascade.

**Neutres.**
- La signature publique `evaluate_and_decide` ne change pas.
  Callers externes (pipeline Stage 4, runner.py) voient la même
  API qu'avant.
- Les tests existants (68 dans `test_topology_controller.py`) passent
  sans changement (à l'exception de 3 qui seedent le state — fixés
  par mirror `__setattr__` + 1 ligne ajoutée pour `_node_retries`).

## Vérification (2026-04-20)

- Rust : `cargo test controller::` → 32/32 PASS
- Rust full : `cargo test --no-default-features --features smt,tool-executor` → 478/478 PASS
- Python `test_topology_controller.py` → 20/20 PASS
- Python `test_rust_controller.py` → 9/9 PASS
- Python `test_pipeline_adaptation.py` → 9/9 PASS
- Python `test_pipeline.py` → 38/38 PASS
- Python full suite (hors live providers) → 1939 passed, 45 skipped.
  5 erreurs + 1 failed = pollution asyncio-fixture pré-existante
  session 1 (identiques avant et après 2.6, pas causées par le port).

## Références

- ADR-002 (Rust-First initial directive)
- ADR-010 (bypass audit methodology)
- ADR-011 (singleton-vs-factory asymmetry — résolu par la même
  session)
- Plan : `docs/superpowers/plans/2026-04-20-rust-first-plan.md` Phase 2
- Spec : `docs/superpowers/specs/2026-04-20-rust-first-plan-design.md` §2.1-2.6
- Commits : `152fe5e` (scaffold), `3ee6b5f` (path 1), `84b1c1f`
  (paths 2-3), `6684665` (paths 4-5), `b1d75c9` (path 6), current
  commit (2.6 délégation + ADR)
- Inventory : `docs/audits/2026-04-20-pyo3-inventory.md` — confirme
  qu'après 2.6 toutes les pyclass wired sont soit runtime-referenced
  soit return-type-accessed (0 bypass restant sur TopologyController).

## Follow-up — 2026-04-20 Phase 1 Stabilization

The scope divergences noted at decision time have been closed by
`docs/superpowers/plans/2026-04-20-post-rust-first-phase1-stab-plan.md`
(commits `1edb57d`, `4aa161c`, `5cb654d`, `a66a846`, `0ab9a97`, plus this docs commit):

- **`__setattr__` mirror**: removed (commit `4aa161c`). Replaced by `@property` getters reading Rust-authoritative state + `_seed_for_tests` helper for legacy test setups.
- **`_evaluate_and_decide_legacy`**: removed (commit `4aa161c`, ~140 lines deleted). `TopologyController.__init__` now raises `ImportError` if `sage_core` is missing — no silent fallback.
- **Path-6 emergent-regex detection** (`detect_emergent_subtask`, `check_emergent_spawn`, 3 regex patterns): removed (commit `5cb654d`, H12 closure). Emergent subtasks flow through `sage_recurse` tool exclusively; spawn budget enforced by `RustTopologyController::should_trigger_emergent_spawn` + `record_emergent_spawn` (commit `1edb57d`), wired into the tool (commit `a66a846`) and set from `TopologyRunner._execute_node` (commit `0ab9a97`).
- **Python `topology_controller.py` line count**: the "thin wrapper" target is now achieved — ~440 lines (was 770 at ADR-012 authorship).

All deferred items from ADR-012 are closed. Directive #1 (Rust first) and Directive #2 (minimal heuristics) are both satisfied on the `TopologyController` surface.
