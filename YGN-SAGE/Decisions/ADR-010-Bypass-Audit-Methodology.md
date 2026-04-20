---
title: "ADR-010: Bypass-Audit Methodology"
type: adr
status: accepte
date: 2026-04-19
tags:
  - adr
  - methodology
  - architecture
  - rust-first
---

# ADR-010: Bypass-Audit Methodology

## Contexte

La soirée du 19 avril 2026 a clôturé **6 bypasses architecturaux** en
une seule session. Tous suivaient le même pattern : une implémentation
Rust correcte, exposée via PyO3, importée par Python — mais **aucun
call site runtime**. L'architecture documentait les features comme
"wired" mais les call sites restaient vides.

| ID | Commit | Surface |
|---|---|---|
| G-series | `c905d06` | `CompositeWriteGate` dans `phases/act.py` (3 writes mémoire) |
| Factory guard | `0b00abd` | Regression test factory → write_gate |
| H1 | `2cd840e` | `engine.should_evolve()` / `evolve()` dans LEARN |
| H4 | `dc51976` | `cache_topology()` après generate (bypass silent caught via empirical validation) |
| H5 | `27a9a4c` | `write_gate` sur singleton AgentLoop (bypass single-agent path) |
| H6 | `aa348e1` | `_on_drift` sur singleton AgentLoop (bypass single-agent path) |

Le pattern compound : un fix wiring peut en masquer un autre en aval (H1
→ H4 : H1 correct, mais `record_outcome` silent no-op sans
`cache_topology`). Les unit tests passent, l'architecture.md continue
d'affirmer "complete", mais la feature fait *rien*.

## Pourquoi ça dure des mois

Trois raisons convergentes :

1. **Le `if X is not None` guard défaut off = indistinguable de "works".**
   Le call site lit `if self.gate is not None: self.gate.evaluate(...)`.
   Quand le wiring casse, `self.gate is None`, la branche est sautée
   silencieusement. Rien ne rouge.

2. **Les tests unitaires testent la classe Rust en isolation**, pas son
   invocation dans le runtime réel. Tester `gate.evaluate(sentinel) →
   blocked` ne prouve pas que `pipeline.run()` appelle `evaluate()`.

3. **Les benchs ne mesurent pas les invocations**, juste les résultats.
   Pass rate reste stable si une feature non-critique dort.

## Décision

Adopter systématiquement la méthodologie documentée dans
`docs/audits/bypass-patterns.md` avant de déclarer toute feature
architecturale "wired". La checklist :

1. **Inventaire PyO3.** Lister chaque `#[pyclass]` de sage-core, grep
   chaque nom dans `sage-python/src/sage/` en excluant tests +
   imports. Zero runtime ref = bypass ou dead code — trier.

2. **Two-path check.** Pour toute state set par la factory
   (`agent_loop_factory.py`), vérifier que le single-agent bypass
   branch (`pipeline.py:941+`) la set aussi. Asymétrie = bypass.

3. **Empirical validation.** Au moins un test d'intégration gardé par
   `@pytest.mark.skipif(not _HAS_SAGE_CORE)` qui :
   - utilise l'objet Rust réel (pas mock)
   - exécute la chaîne d'appel complète
   - asserte que l'état a MUTÉ comme attendu
   Les mocks prouvent le call-site, pas l'effet.

4. **Re-audit après chaque wiring commit.** H1 a introduit H4 en
   cascade. Assumer que chaque fix wiring a un bypass caché une couche
   en-dessous.

## Red-flag commit-message phrases

Phrases dans un commit message qui DOIVENT déclencher un empirical
validation check avant merge :
- "wires X into Y"
- "X now called from Y"
- "enables online / live / per-request Z"
- "closes the SA-N architecture claim"
- "gate / controller / callback now fires"

Si le commit ship seulement des mocks unit tests contre la classe cible
— **ne pas merge** sans au moins un integration test de la chaîne
complète.

## False-positives catalogués

Ne pas chasser ces fantômes (documentés dans `bypass-patterns.md` §"Not
a bypass (common false positives)") :

- **RustQualityEstimator** (lexical, 5-signal) — délibérément supprimé
  per architecture.md ("5-signal heuristic REMOVED, r=0.34"). Le code
  Rust est stale, pas bypass.
- **HardwareProfile** — utility `detect()`, trivialement optionnel.
- **RustRagCache**, **RustSmmu** — grep-name mismatch ; Python utilise
  `sage_core.WorkingMemory` etc. qui délèguent à ces classes sous
  d'autres noms.
- **RustEntityGraph** — surface dupliquée. Python `CausalMemory` est
  l'impl wirée. Refactor scope, pas bypass scope.

## Conséquences

- Chaque commit de type "wires X" DOIT inclure un integration test
  empirical (pattern H4-style).
- Chaque ADR/audit qui affirme "X is wired" DOIT citer au moins un
  integration test path name.
- Les sessions futures commencent par lire
  `docs/audits/bypass-patterns.md` (cf MEMORY.md § ⭐).
- Les reviews advisor + codex DOIVENT être appelés à poser l'empirical
  validation question : "what observable state mutation proves this
  works on a real run?"

## Références

- Méthodologie : `docs/audits/bypass-patterns.md`
- Commits : `c905d06`, `0b00abd`, `2cd840e`, `dc51976`, `27a9a4c`, `aa348e1`
- Audit qui a démarré la série : `docs/audits/2026-04-18-astropy-14995-decision-path.md`
- Advisor warning qui a sauvé H1 : "If record_outcome silently fails, you'd wire engine.evolve() perfectly and observe zero effect" (2026-04-19 conversation).
