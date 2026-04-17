---
title: "ADR-007: F7 — Role-aware tier promotion in Rust ModelAssigner"
type: adr
status: accepte
date: 2026-04-17
tags:
  - adr
  - routing
  - rust
  - topology
---

# ADR-007: F7 — Promotion de tier role-aware

## Contexte

Pré-F7, le Rust `ModelAssigner` scorait chaque nœud uniquement sur sa
propre `node.system` (S1/S2/S3 fixé par le template) :

```rust
let system = match node.system { 1 => S1, 2 => S2, 3 => S3, _ => S1 };
let score = w_aff * affinity(card, system) + w_dom * domain_score + w_cost * (1 - cost_norm);
```

Sur un task SWE-bench (forcé à S3 via `system_hint=3`), le template
`sequential` a `planner=S1, coder=S2, synthesizer=S1`. Stage 0 routait
`task_system=S3` mais le Stage 4 ModelAssigner ne le voyait pas — il
restait sur les tiers locaux des nœuds. Résultat : un planner S1 pické
sur affinity S1 (cheap flash-tier) au lieu d'un model-tier S2 capable
de raisonner sur la décomposition d'un bug astropy.

Symptôme : 25-char fake patches en sortie (smoke 2026-04-17 matin).

## Décision

Forwarder `task_system: Option<CognitiveSystem>` jusqu'au scoring et
appliquer un **tier floor** sur les producer roles :

```rust
fn effective_system(role, node_system, task_system, task_domain) -> CognitiveSystem {
    let task = task_system?;          // None disables promotion
    if is_sink_role(role) { return node_system; }    // SINK_NODE_PROMPT nodes stay
    let floor = if task == S3 && is_high_rigour_domain(task_domain) {
        3   // math/formal S3 → S3 (full reasoner tier)
    } else {
        (task as u8).saturating_sub(1)   // S3 → S2, S2 → S1
    };
    CognitiveSystem::from(max(node_system as u8, floor))
}
```

Trois règles :

1. **Sink roles** (synthesizer, aggregator, mixer, judge, verifier, solver,
   formatter, output, sink, output_*) gardent leur tier template — ce sont
   des forwarders qui n'ont pas besoin d'un reasoner. Liste maintenue dans
   `SINK_ROLES` constant, vérifiée par `grep -B 1 SINK_NODE_PROMPT
   sage-core/src/topology/templates.rs`.

2. **Producers** (planner, coder, worker, actor, etc.) sur task S3 sont
   floor à S2. Suffit pour la majorité des SWE-bench tasks.

3. **Domain math/formal + S3** floor à S3 (full reasoner). Cards exposent
   déjà `math` et `formal` columns dans `cards.toml` ; on les utilise.

## Côté Python

`pipeline._stage_assign_models` forwarde `task_system=ctx.system` à
`assigner.assign_models(...)`. Le `topology_controller._resolve_upgrade_model`
(FrugalGPT cascade) et `pipeline.py` Stage 4 cascade le forwardent aussi
à `assign_single_node`. TypeError fallback pour binding antérieur.

## Tests

Côté Rust (sage-core/src/routing/model_assigner.rs) :
- `test_effective_system_*` (5 tests : without_task, S3_promotes, S2_no_op,
  sink_never_promoted, never_demotes)
- `test_f7_math_s3_floors_at_s3`, `test_f7_formal_s3_floors_at_s3`,
  `test_f7_code_s3_unchanged_floors_at_s2`
- `test_is_sink_role_classification`, `test_formal_solver_sink_protected_on_math_s3`
- `test_sink_drift_templates_match_classifier` (parcourt les 12 templates et
  assert que chaque SINK_NODE_PROMPT node satisfait `is_sink_role`)
- `test_no_strict_mandate_role_at_s1_in_any_template` (no coder/actor at S1)

Côté Python : `test_f7_task_system_forward.py` (6 tests batch),
`test_f7_frugalgpt_cascade.py` (4 tests cascade + TypeError fallback).

## Régression évitée par le sink audit

L'audit (item 2 de la séquence advisor) a rattrapé une régression que ma
première implémentation du floor avait introduite : le `solver` node de
`formal_solver` est `model_id=""` + `system=1` + Rust math evaluator
(prix 0.10 USD). Sans la classification sink, le domain rule pushait
`solver` de S1 → S3 sur math/formal task, **remplaçant le Rust math par
un LLM call à chaque task**. Exactement le sur-spend que F7 voulait
éviter.

## Alternatives rejetées

- **Plumbing `node.prompt == SINK_NODE_PROMPT`** (Codex's MODIFY) :
  Codex proposait de passer le booleen au `effective_system()` au lieu
  de matching sur le role string. Plus propre mais plus invasive
  (signature breaking). Préféré : test drift qui catch les divergences
  sans changer la prod signature.

- **Promotion `max(node, task)` (sans -1)** : trop agressif, S3 task
  forcerait S3 sur tous les producers ; gaspille le budget côté coder
  où S2 (gpt-5.3-codex) est meilleur que S3 (gemini-3.1-pro-preview).

- **Prompt-based force** : "you MUST use a strong reasoner" dans le
  system_prompt. Ignored par les models — voir mandate F6 `AT LEAST 3
  execute_bash` également ignoré (résolu via `tool_choice="required"`).

## Conséquences

- **Coût Stage 4** inchangé (toujours 1 lookup par node)
- **Latence** scoring Stage 4 inchangée (`is_high_rigour_domain` est O(1)
  substring match)
- **Cards.toml** doit conserver les colonnes `math` et `formal` (sinon
  domain rule ne fonctionne pas)
- **Symétrie côté validation** : voir [[ADR-008-PRM-Gate-Domain]] —
  Z3 PRM est aussi domain-gated pour la même raison
