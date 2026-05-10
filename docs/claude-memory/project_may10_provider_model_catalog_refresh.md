---
name: project_may10_provider_model_catalog_refresh
description: Provider/model catalog refresh, live provider smoke, and blocked canary status as of 2026-05-10
type: project
originSessionId: codex-2026-05-10
---
# Provider/Model Catalog Refresh - 2026-05-10

This supersedes the 2026-05-09 provider-status memory where it conflicts.

## Current repository base

- Branch: `main`
- Recent pushed base before this ticket: `76f5a54be4293d8845b08ed0d95b72e0d43fcb27`
- Human status note: `docs/status/2026-05-10-current-state.md`

## Model catalog truth

- `sage-core/config/cards.toml` is the source of truth for model IDs, costs,
  context windows, runtime selectability, runtime replacements, and runtime
  settings.
- `sage-python/src/sage/providers/connector.py` is connection settings only.
- Current catalog: 24 model cards, 7 API providers.

## Corrected current models

- DeepSeek active IDs: `deepseek-v4-flash`, `deepseek-v4-pro`.
- DeepSeek legacy aliases: `deepseek-chat`, `deepseek-reasoner`; they are
  runtime-non-selectable and rewrite to `deepseek-v4-flash`.
- DeepSeek runtime settings: flash has thinking disabled, pro has thinking
  enabled.
- MiniMax active spelling: `MiniMax-M2.7`, `MiniMax-M2.5`,
  `MiniMax-M2.5-highspeed`; text context 204800.
- MiniMax compatibility alias: `minimax-m2.7` is kept as a
  runtime-non-selectable card and rewrites to `MiniMax-M2.7`.
- Kimi active card: `kimi-k2.6`.
- OpenRouter Qwen card: `qwen/qwen3.5-plus-02-15`.
- Runtime `codex` tier uses `gpt-5.4`; `gpt-5.3-codex` exists in live OpenAI
  discovery but is not the runtime chat model card.

## Evidence

- Live model discovery: `docs/benchmarks/2026-05-10-live-model-discovery.json`.
- Live provider preflight:
  `docs/benchmarks/2026-05-10-provider-preflight-post-model-catalog.json`.
- Preflight result: 10/10 OK for small responses across Google, DeepSeek,
  xAI, Kimi, MiniMax, OpenRouter, and OpenAI (`gpt-5.4`, `gpt-5.5-pro`).
- Scope label: every row has `evidence_scope = liveness_only`.
- Warnings: `grok-code-fast-1` retirement on `2026-05-15T19:00:00Z`;
  MiniMax produced non-empty content but not exactly `ok`.

This is provider/config reachability evidence, not benchmark-quality evidence.

## SWE-bench Pro / canary status

- Official local grading remains blocked:
  `docs/benchmarks/2026-05-10-grader-preflight-76f5a54b.json`.
- Decision: `NO_GO_GRADER_REPO_DIRTY`.
- Blockers: low local disk for SWE-bench Docker and dirty
  `external/SWE-bench_Pro-os`.
- N=1 canary preflight:
  `docs/benchmarks/2026-05-10-canary-n1-preflight-76f5a54b/`.
- Result: timeout at 120s, 0 patch, final provider/model
  `deepseek` / `deepseek-v4-flash`, learning evidence gate `NO_GO`,
  grading/CI gates blocked, canary decision `BLOCKED`.

Do not cite the N=1 canary as performance evidence.

## cgpro VERIFY

- cgpro VERIFY returned on conversation
  `6a0087d7-2a1c-838b-aff9-8dff56a633e4` from
  `.tmp/cgpro_model_catalog_provider_settings_verify_20260510.md`.
- Response summary:
  `docs/codex-memory/cgpro-model-catalog-provider-settings-verify-2026-05-10.md`
  (local raw copy also kept under `.tmp/`).
- Applied follow-ups: DeepSeek final-kwargs test, no DeepSeek
  `reasoning_effort` leak, MiniMax lowercase alias, xAI retirement
  metadata/warning, liveness-only preflight labeling.

The final commit was still pending when this memory note was updated.
