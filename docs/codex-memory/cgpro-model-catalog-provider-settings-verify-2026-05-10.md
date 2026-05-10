# cgpro VERIFY - Provider Model Catalog - 2026-05-10

Conversation: `6a0087d7-2a1c-838b-aff9-8dff56a633e4`

Verdict: design mostly sound, with required pre-merge conditions.

Key findings:

- DeepSeek alias mapping is correct:
  - `deepseek-chat` -> `deepseek-v4-flash` with thinking disabled.
  - `deepseek-reasoner` -> `deepseek-v4-flash` with thinking enabled.
  - `deepseek-v4-pro` remains a separate active model, not an alias target.
- Runtime integrity is preserved if live discovery updates `cards.toml` through
  human-reviewed source changes and runtime never mutates candidates directly
  from discovery.
- DeepSeek thinking must be explicitly serialized through
  `extra_body.thinking.type`; provider defaults must not be trusted.
- Highest-risk integration detail: verify final OpenAI-compatible kwargs, not
  only `direct.model_request(..., model_settings=...)`.
- Keep lowercase `minimax-m2.7` as a non-selectable alias for one migration
  window.
- `grok-code-fast-1` is valid on 2026-05-10 but has an imminent retirement
  date on `2026-05-15T19:00:00Z`; catalog/preflight should warn.
- Provider preflight evidence must be labeled as liveness only, not quality,
  benchmark, tool, JSON-mode, long-context, or instruction-following evidence.

Actions taken after review:

- Added final OpenAI-compatible kwargs capture test for DeepSeek V4 flash:
  `tests/providers/test_pydantic_ai_integration.py`.
- Stopped mapping DeepSeek `reasoning_effort` into generic
  `openai_reasoning_effort`; DeepSeek uses `extra_body.thinking`.
- Added `minimax-m2.7` runtime-non-selectable compatibility card with
  replacement `MiniMax-M2.7`.
- Added `runtime_retire_after = "2026-05-15T19:00:00Z"` for
  `grok-code-fast-1`.
- Added provider-preflight `evidence_scope = "liveness_only"` and warnings for
  imminent xAI retirement and non-exact smoke output.
- Regenerated live provider preflight artifact after those changes.

Local verification after actions:

- `cargo fmt --check`: PASS
- `cd sage-core && cargo test --features smt routing::model_assigner --lib`:
  38 passed
- `cd sage-python && python -m pytest tests/test_provider_preflight.py tests/test_provider_pool.py tests/providers/test_pydantic_ai_integration.py tests/test_llm_providers.py tests/test_provider_policy.py -q`:
  86 passed
- `cd sage-python && ruff check ...`: PASS
- `cd sage-python && python -m mypy src/sage/ --ignore-missing-imports`: PASS
- `maturin develop --features smt,onnx`: blocked by known
  `--include-debuginfo cannot be used with --strip`
- Fallback `maturin build --release --features smt,onnx --out target/wheels`
  then `pip install target/wheels/sage_core-*.whl --force-reinstall --no-deps`:
  PASS
- PyO3 smoke confirmed `runtime_retire_after` and `minimax-m2.7` alias fields
  are visible from installed `sage_core`.
