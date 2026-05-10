# YGN-SAGE Current State - 2026-05-10

This note is a documentation and memory sync point, not a new release claim.
Current source code, tests, claims audit, and benchmark artifacts remain the
authoritative evidence.

## Repository Snapshot

- Branch: `main`
- Snapshot base: `76f5a54be4293d8845b08ed0d95b72e0d43fcb27`
- Status snapshot: `docs/status/current.json`, generated
  `2026-05-10T14:29:02+00:00`
- Test surface in that snapshot: `3476` Python tests, `571` Rust tests,
  `100` sage-discover tests
- The snapshot was generated from a dirty working tree while the
  provider/model-catalog ticket was in progress.

## Provider And Model Catalog

`sage-core/config/cards.toml` is the source of truth for model IDs, costs,
context windows, runtime selectability, and runtime replacements. Provider
connection settings live in `sage-python/src/sage/providers/connector.py`.

Current catalog:

- `24` model cards
- `7` API providers: Google, OpenAI, DeepSeek, xAI, Kimi/Moonshot, MiniMax,
  OpenRouter
- DeepSeek active models: `deepseek-v4-flash` and `deepseek-v4-pro`
- DeepSeek legacy aliases: `deepseek-chat` and `deepseek-reasoner` are
  runtime-non-selectable and rewrite to `deepseek-v4-flash`
- MiniMax active spelling: `MiniMax-M2.7`, `MiniMax-M2.5`,
  `MiniMax-M2.5-highspeed`; text context set to `204800`
- MiniMax legacy alias: `minimax-m2.7` is retained as a
  runtime-non-selectable compatibility alias for one migration window
- Kimi active model: `kimi-k2.6`
- OpenRouter active Qwen model: `qwen/qwen3.5-plus-02-15`

Evidence:

- Live model discovery artifact:
  `docs/benchmarks/2026-05-10-live-model-discovery.json`
- Live provider preflight artifact:
  `docs/benchmarks/2026-05-10-provider-preflight-post-model-catalog.json`
- Direct provider smoke summary: `10/10` OK across Google, DeepSeek,
  xAI, Kimi, MiniMax, OpenRouter, and OpenAI (`gpt-5.4`,
  `gpt-5.5-pro`)
- Every preflight row is labeled `evidence_scope: "liveness_only"`.
  Current warnings: `grok-code-fast-1` retirement on `2026-05-15T19:00:00Z`;
  MiniMax response was non-empty but not exactly `ok`, so it is not
  instruction-following evidence.

This smoke proves that the configured provider/model IDs can produce small
responses with the current `.env`. It does not prove benchmark quality,
latency stability, official grading, or release-candidate status.

## Canary And Grader Status

SWE-bench Pro official grading remains blocked locally.

- Grader preflight artifact:
  `docs/benchmarks/2026-05-10-grader-preflight-76f5a54b.json`
- Decision: `NO_GO_GRADER_REPO_DIRTY`
- Blockers: host disk below the SWE-bench local Docker floor and dirty
  `external/SWE-bench_Pro-os`
- Docker itself is reachable and `hello-world` passed in that preflight

The real N=1 canary preflight at
`docs/benchmarks/2026-05-10-canary-n1-preflight-76f5a54b/` is a blocked
runtime artifact, not benchmark evidence:

- `tasks_run`: `1`
- `patches_extracted`: `0`
- timeout: `120s`
- final model/provider: `deepseek-v4-flash` / `deepseek`
- provider policy gate: pass
- learning evidence gate: `NO_GO`
- grading gate: `BLOCKED`
- CI gate: `BLOCKED`
- canary decision: `BLOCKED`

Do not cite this as SWE-bench Pro performance. It is useful evidence for
trace shape, timeout behavior, provider-policy enforcement, and blocked
grader status.

## Local Gates Run During The Provider Ticket

The following local gates were green before this documentation sync:

- `cargo fmt --check`
- `cd sage-core && cargo test --features smt routing::model_assigner --lib`
- Targeted Python provider/routing tests:
  `tests/test_provider_preflight.py`,
  `tests/test_provider_pool_wiring.py`,
  `tests/test_provider_pool.py`,
  `tests/providers/test_pydantic_ai_integration.py`,
  `tests/test_llm_providers.py`,
  `tests/test_provider_policy.py`
- cgpro follow-up slice:
  `tests/test_provider_preflight.py`,
  `tests/test_provider_pool.py`,
  `tests/providers/test_pydantic_ai_integration.py`
- `cd sage-python && ruff check ...` on touched Python files/tests
- `cd sage-python && python -m mypy src/sage/ --ignore-missing-imports`
- `python -m sage.ops.sage_core_version --strict`
- `python -m sage.ops.claims_audit --strict`
- `python scripts/regenerate_claims_index.py --check`
- `python scripts/sync_doc_counters.py --check`
- `python scripts/narrative_guard_phase22.py`
- `python scripts/status_snapshot.py`

The full Python suite was not re-run after this provider/model-catalog ticket.

## cgpro Review

cgpro VERIFY ran on conversation `6a0087d7-2a1c-838b-aff9-8dff56a633e4`.
The response summary is archived at
`docs/codex-memory/cgpro-model-catalog-provider-settings-verify-2026-05-10.md`
(local raw copy also kept under `.tmp/`).

Post-review changes applied:

- DeepSeek final OpenAI-compatible kwargs are tested for
  `extra_body.thinking.type = disabled`.
- DeepSeek no longer maps `reasoning_effort` into generic OpenAI reasoning
  settings.
- Lowercase `minimax-m2.7` remains as a runtime-non-selectable alias.
- `grok-code-fast-1` now carries retirement metadata and the preflight warning.
- Provider preflight rows are explicitly scoped to liveness only.

The final commit remains pending at the time of this note.
