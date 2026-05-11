# Retire `gpt-5.5-pro` from the runtime model catalogue

**Date**: 2026-05-11
**Operator**: Yann (directive); applied by autonomous canary session
**Source of truth changed**: `sage-core/config/cards.toml`
**Status**: `runtime_selectable = false` + `runtime_replacement = "gpt-5.5"`

## Decision

`gpt-5.5-pro` is removed from the set of models the runtime can newly
assign. The card is left in the catalogue with `runtime_selectable =
false` so:

- bandit / MAP-Elites posteriors that reference the historical id
  continue to load,
- cost-comparison reports that pull historical metrics still resolve
  the id,
- the runtime selector rewrites any candidate that still surfaces
  the id (e.g. from a stale prompt or hint) to `gpt-5.5`.

## Why

Yann directive 2026-05-11: *"Retire ChatGPT 5.5 Pro, aucun problème
avec 5.5 mais pas l'édition Pro, il coûte trop cher."*

Cost is the binding constraint:

- gpt-5.5-pro: **$30 / $180 per 1M tokens** (input/output)
- gpt-5.5: $5 / $30 per 1M tokens (per the card at
  `sage-core/config/cards.toml` line ~218)
- Ratio: 6× input cost, 6× output cost

The empirical SWE-bench Pro canary work on 2026-05-11 (commits
`6e0609ba → 6317d42a`) showed that the routing layer often picked
`gpt-5.5-pro` as the conceptual best-model for `domain=code system=3`
tasks, even though provider execution fell back to other providers
(google / deepseek were on the allowlist; openai was on the denylist
for the canary). The routing decision still incurred reasoning cost
inside the assigner. With B2 final close still pending and the budget
tier (`deepseek-v4-flash`) as the actual execution tier, gpt-5.5-pro
adds no observable resolution-rate signal that justifies the price
spread.

## What this changes

### Runtime behavior

- The Rust ModelAssigner filters out `gpt-5.5-pro` from the candidate
  pool at scoring time (the existing `runtime_selectable = false`
  short-circuit, same path as `deepseek-chat` / `deepseek-reasoner`
  retirement on 2026-05-10).
- The Python wrapper around the Rust assigner rewrites the id to
  `gpt-5.5` if anything upstream surfaces it (e.g. a literal model_id
  baked into a prompt).
- The card's `runtime_replacement_settings = {}` because there's no
  special config required to fall back to gpt-5.5 (same provider
  endpoint, same `supports_tools` / `supports_json_mode` / vision
  capabilities).

### What does NOT change

- The card remains in the catalogue. Costs in historical reports
  resolve normally.
- The bandit / MAP-Elites posteriors that include `gpt-5.5-pro` keep
  loading. The selector simply will not propose it.
- No CLAIMS.yaml entry was tied to `gpt-5.5-pro` specifically; the
  routing claims (`routing.knn_92pct`, `routing.system_router_88pct`)
  are model-agnostic.
- No test or benchmark was conditioned on `gpt-5.5-pro` specifically;
  tests pinning a model id pin `gpt-5.4` or a generic "any pro tier"
  shape.

## What may need follow-up

- `sage-python/config/cards.toml` is supposed to be a SYMLINK to
  `sage-core/config/cards.toml` per `.claude/rules/environment.md`,
  but on 2026-05-11 they are two separate files (the python copy
  predates the May 10 model-catalog refresh). That divergence is a
  pre-existing bug, NOT introduced by this retirement. A follow-up
  block should re-establish the symlink (or formally diverge the two
  files with explicit policy).
- `sage-python/pipeline_v2/assign_models.py:64` carries a comment
  mentioning `gpt-5.5-pro` as an example. Comment only; no code path
  references the id. Leaving as-is — it documents historical context.
- Various docs under `docs/codex-memory/` and `docs/claude-memory/`
  mention `gpt-5.5-pro` as a past-tense reference. Not load-bearing;
  no rewrite needed.

## Reproducibility / reversal

To re-enable: flip `runtime_selectable = false` to `true` in
`sage-core/config/cards.toml` for the `gpt-5.5-pro` entry, drop
`runtime_replacement` and `runtime_replacement_settings`, optionally
remove this evidence file.
