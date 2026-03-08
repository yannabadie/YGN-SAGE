# CI Workflows

GitHub Actions continuous integration pipeline for YGN-SAGE.

## Workflows

### `ci.yml` -- Main CI Pipeline

Runs on every push and pull request. Executes three parallel test suites:

**1. Rust Core (`sage-core/`)**
- `cargo test --workspace` -- 36 tests (no-default-features)
- `cargo clippy` -- Lint check
- ONNX embedder tests are feature-gated and run separately (`cargo test --features onnx`)

**2. Python SDK (`sage-python/`)**
- Installs with `pip install -e ".[all,dev]"`
- `python -m pytest tests/ -v` -- 695 tests
- `ruff check src/` -- Lint
- Tests run without API keys (all LLM calls are mocked)

**3. Python Discovery (`sage-discover/`)**
- Installs with `pip install -e .`
- `python -m pytest tests/ -v` -- 52 tests

## Notes

- The CI pipeline was added during audit fix A5 to ensure Rust tests are always validated.
- Cargo.toml uses edition 2021 (downgraded from 2024 for CI compatibility).
- ONNX tests require the model file (`all-MiniLM-L6-v2`); they are skipped if the model is not present.
- Python tests use mocked providers and do not require `GOOGLE_API_KEY` or Codex CLI authentication.
