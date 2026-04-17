---
title: "ADR-008: PRM Domain Gate — Z3 reasoning uniquement sur math/formal"
type: adr
status: accepte
date: 2026-04-17
tags:
  - adr
  - validation
  - prm
  - z3
---

# ADR-008: PRM Z3 reasoning uniquement sur math/formal

## Contexte

`ProcessRewardModel` (sage-python/src/sage/topology/kg_rlvr.py:233) a été
conçu pour scorer les chaînes de raisonnement S3 via assertions Z3 dans
des blocs `<think>...</think>` :

```python
def calculate_r_path(content):
    steps = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    if not steps:
        return -1.0, {"error": "No <think> blocks found. System 3 reasoning required."}
```

`agent_loop_factory.py` mappait `validation_level = system_level` pour
les producer roles. Donc tout task S3 enclenchait le PRM gate, qui
pénalise (-1.0) une réponse sans `<think>` blocks.

**Problème empirique (smoke v2 SWE-bench Lite, 2026-04-17 PM)** :
- 17 CEGAR repair failures × 3 tasks × 6 RESET_AGENT × 6 SWITCH_MODEL
- 0/3 patches générés
- Tous les tasks hit le `--timeout-per-task=300s` ceiling

**Cause** : aucun model frontier (Gemini, OpenAI, MiniMax, DeepSeek)
n'émet inline `<think>` blocks par défaut. Les models qui font du
chain-of-thought (DeepSeek-Reasoner, GPT-5.4 thinking) exposent leur
reasoning via un canal séparé (`reasoning_content`), pas dans `content`.
Donc le PRM check pénalisait systematiquement → retry → CEGAR → degrade
→ thrash.

## Décision

Domain-gate symétrique au F7 routing (voir [[ADR-007-F7-Routing]]) :

```python
if any(r in role_lower for r in _NO_VALIDATION_ROLES):
    validation = 0
elif system_level >= 3:
    validation = 3 if _is_high_rigour_domain(task_domain) else 2
elif system_level >= 2:
    validation = 2
else:
    validation = 1
```

`_is_high_rigour_domain(task_domain)` est défini en miroir de la fonction
Rust `is_high_rigour_domain` dans `sage-core/src/routing/model_assigner.rs`.
Substring case-insensitive sur `math` ou `formal`.

`task_domain` plumbé via `pipeline.py:_agent_loop_factory = partial(...,
task_domain=ctx.domain or "")`.

## Pourquoi cette décision

L'architecture intent du PRM est valable pour math/formal :
- Cards exposent `math` et `formal_z3_strength` columns
- Z3 assertions sont signal réel pour proof search
- DeepSeek-R1 / Qwen2.5-Math émettent bien des `<think>` blocks

Pour code/general :
- Z3 assertions sur un Python bug-fix patch ne portent pas signal
- Les models frontier qui font du CoT le font hors `<think>`
- AVR (validation_level=2, syntax + sandbox + runtime guard) suffit

## Alternatives rejetées

- **Forcer `<think>` blocks via prompt** : on a essayé via le retry
  message dans `phases/think.py:122-133` ("SYSTEM: Use <think> tags
  with Z3 assertions"). Empiriquement ignoré : 17 CEGAR failures =
  17 retries qui n'ont pas produit `<think>` blocks.

- **Désactiver PRM globalement** : casserait le formal_solver pipeline
  (le seul endroit où Z3 assertions ont du sens dans la stack actuelle).

- **Valider sur `reasoning_content` au lieu de `content`** : nécessite
  parsing du response object par provider, pas de standard cross-API.
  Reportable mais demande couche d'abstraction.

## Tests

`tests/test_agent_loop_factory.py` :
- `test_actor_s3_math_gets_validation_3`
- `test_actor_s3_formal_gets_validation_3` (substring + case-insensitive)
- `test_actor_s3_code_gets_validation_2` (regression : pinned across
  "code", "general", "", "swe_bench")

## Évidence empirique

Smoke v3 (PRM gate landed) :
- 0 CEGAR failures
- 0 RESET_AGENT
- 0 SWITCH_MODEL
- 6 AVR feedback (legitimate S2 validation)
- 2/3 vrais patches (vs 0/3 sur v2 et v1)

## Conséquences

- **Math/formal tasks** restent S3 PRM (intentionnel, pas de perte)
- **Code/general S3 tasks** descendent à AVR S2 — perte de validation
  formelle théorique mais zéro perte empirique (PRM signal était zéro
  sur ces domaines de toute façon)
- **Symétrie avec F7** : `is_high_rigour_domain` est défini deux fois
  (Python `agent_loop_factory.py:_is_high_rigour_domain` et Rust
  `model_assigner.rs:is_high_rigour_domain`). Doit rester en sync.

## Source de vérité du domain

`ctx.domain` est inferred par Stage 0 (`_infer_domain(ctx.task)`).
Si l'inference se trompe (ex: task math classée comme `general`), on
manque le PRM. Risque acceptable : le worst case est AVR au lieu de
PRM, pas un crash.
