---
title: "ADR-010: Meta-Harness divergence from stanford-iris-lab reference + Python-override extension"
type: adr
status: accepte
date: 2026-04-18
tags:
  - adr
  - meta-harness
  - search
  - evolution
---

# ADR-010 — Meta-Harness : divergence vs référence + extension Python-override

## Contexte

Audit du 18 avril 2026 suite à la question utilisateur :
« je doute que [notre Meta-Harness] soit tout a fait adapté ».

**Référence**: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness)
(push le 2026-04-16), avec l'implémentation artifact TerminalBench-2
[meta-harness-tbench2-artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact).

Paper : *Meta-Harness: End-to-End Optimization of Model Harnesses*
(arXiv 2603.28052, Stanford + MIT + KRAFTON, 2026-03-30).

## Comparaison directe

| Axe | Référence Stanford | SAGE (Apr 17 → Apr 18) |
|-----|-------------------|------------------------|
| **Format candidat** | Fichier Python (module importable) dans `agents/` | Dataclass JSON `HarnessConfig` dans `candidates/<id>/config.json` |
| **Espace de recherche** | Structurel : toute la classe agent, memory system, tool scaffolding, loop reorganization | Numérique : hyperparamètres (budget_ratio, similarity_threshold, prompts templates, debate rounds) |
| **Proposer** | Subprocess `claude_wrapper.run(prompt, model='opus', allowed_tools=[Read,Glob,Grep,Agent,Write,Edit,Bash], cwd=EVOLVE_DIR)` — "10M tokens per step" grâce au filesystem complet | `auto_propose.py` = **LLM call one-shot** (génère du JSON), OU Claude Code manuel lisant les fichiers |
| **Activation candidat** | `import text_classification.agents.<name>` puis instanciation de `AgentHarness` / `MemorySystem` | `HarnessPatcher.patch_runner()` monkey-patch via dataclass |
| **Inner loop** | `inner_loop.py` + `benchmark.py` (domain-specific eval) | MASBENCH/BCB via `MetaHarnessLoop.evaluate()` |
| **Logs** | `logs/run_name/{evolution_summary.jsonl, frontier_val.json, pending_eval.json, reports/, claude_sessions/}` | `~/.sage-meta-harness/{baseline/, candidates/<id>/, leaderboard.json}` |
| **Pareto** | `frontier_val.json` multi-axes | `leaderboard.json` flat |
| **Shell orchestration** | `scripts/run_eval.sh` (harbor runner, concurrent trials) | 100% Python |

## Gap fondamental

Notre module actuel est **un hyperparameter tuner** (narrow search space),
pas un **harness search** (structural evolution). Le papier Stanford tire
sa valeur de l'exploration de code agent entier : *Terminal-Bench 2.0
76.4% avec Opus 4.6* = des harnesses entièrement réécrits par le proposer.

Notre `HarnessConfig` actuel N'A PAS de degré de liberté pour :
- Ajouter ou retirer un stage du pipeline
- Réécrire `_gather_predecessor_context` (seulement paramétrer le format)
- Introduire un nouveau type de mémoire
- Modifier la surface de tools

Conséquence : même en bouclant 1000 itérations `auto_propose`, on
converge vers le meilleur `budget_ratio=0.72` au lieu de `0.70`. On ne
peut pas découvrir de nouveau scaffold.

## Décision — extension pragmatique (Apr 18)

Ne pas refaire le papier à l'identique (travail de plusieurs jours :
port du `claude_wrapper`, import-check, `scripts/run_eval.sh`, etc.).
Pragmatic path :

### 1. Ajouter `python_override_path: str = ""` à `HarnessConfig`

Nouveau champ optionnel. Si non-vide, le patcher importlib-load le module
Python et cherche des hooks bien connus :

- `gather_predecessor_context_override(runner, node_idx) -> str`
- `execute_llm_node_override(runner, node_idx, task, context_override) -> awaitable[str]`
- `select_macro_topology_override(omega, delta, gamma, domain) -> str | None`

Si présents, ces hooks remplacent la méthode homologue du runner pour la
durée du contexte patché. Le reste du runner (execute_node, pipeline,
memory, tools) reste intact.

Résolution de chemin :
1. Absolu → chargé direct
2. Relatif → essayé dans `~/.sage-meta-harness/candidates/<id>/`
3. Relatif → essayé dans `cwd/`

### 2. Garder le dataclass search existant

La couche hyperparamètres reste utile (tuning cheap, faible variance).
Les deux couches sont compositionnelles : un candidat peut fournir les
deux (dataclass + Python override) → le patcher applique les dataclass
d'abord, puis les hooks.

### 3. Documenter la limitation

Le `auto_propose.py` reste un LLM single-shot. Le "vrai" proposer
agentique (subprocess Claude CLI, 10M tokens/step) est une prochaine
étape non faite dans cet ADR — il faudra porter `claude_wrapper.py` de
la référence et adapter le prompt template.

## Tests

`tests/test_meta_harness.py::TestPythonOverrideModule` (5 nouveaux tests) :
- `test_no_override_path_is_noop` : champ vide → pas de load
- `test_override_path_loads_module` : importlib OK
- `test_override_path_missing_file_is_none` : fichier manquant → `None`
- `test_override_hook_replaces_gather` : bout-en-bout, patch + hook, puis unpatch restore l'original
- `test_from_json_round_trip_preserves_override_path` : sérialisation

19/19 tests meta_harness passent (était 14, +5).

## Alternatives rejetées

### A. Tout réécrire depuis zéro selon la référence Stanford

**Avantages** : parité exacte, accès aux benchmarks Terminal-Bench 2.0.
**Inconvénients** : 1-2 semaines de travail, invalide les 14 tests
existants, force à adopter `harbor` + `kira` + `Terminus2` qui ne font
pas partie de SAGE.

**Verdict** : reporté — à refaire quand on a du temps dédié.

### B. Forker claude_wrapper.py + subprocess uniquement

Adoption partielle : garde notre dataclass, remplace juste le proposer.
**Inconvénient** : le proposer continuerait à ne proposer que du JSON
dataclass (ne peut pas écrire de .py sans les hooks). Complémente ADR-010.3
mais ne le remplace pas.

### C. Rester en hyperparameter tuning "honest"

Documenter qu'on n'est PAS Meta-Harness, qu'on est un tuner.
**Inconvénient** : bloque l'amélioration structurelle — valeur cap à
+5-10pp sur benchmarks tuned vs nouveau scaffold qui peut délivrer +20pp.

**Verdict** : rejeté — la vision produit SAGE exige SOTA. Il faut
l'évolution structurelle, donc on met les hooks.

## Conséquences

- `HarnessConfig` a désormais un degré de liberté structurelle via Python
  override, tout en restant back-compat (ancien JSON sans `python_override_path`
  continue de fonctionner).
- Un proposer (Claude Code manuel ou auto_propose futur) peut écrire un
  `candidates/<id>/agent.py` avec `gather_predecessor_context_override`
  qui réorganise complètement la collecte de contexte.
- La vraie parité avec la référence demande encore (non fait ici) :
  - Port de `claude_wrapper.py` subprocess pattern
  - Pareto frontier multi-axes
  - Import-validate avant eval
  - `scripts/run_eval.sh` équivalent pour benches SAGE
- Cet ADR prépare la session suivante qui pourra faire ces ports.

## Références

- Paper: [arXiv 2603.28052](https://arxiv.org/html/2603.28052v1)
- Reference repo: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness)
- TerminalBench-2 artifact: [meta-harness-tbench2-artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact) (76.4% Opus 4.6)
- Notre impl: `sage-python/src/sage/meta_harness/`
