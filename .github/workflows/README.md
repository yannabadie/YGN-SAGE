# CI Workflows

GitHub Actions continuous integration pipeline for YGN-SAGE.

## Workflows

### `ci.yml` -- Main CI Pipeline

Runs on every push and pull request. Executes three parallel test suites:

**1. Rust Core (`sage-core/`)**
- `cargo fmt --check` -- Format check
- `cargo clippy --no-default-features` -- Lint check
- `cargo test --no-default-features` -- 7 tests
- `cargo check --features onnx` -- Verify ONNX feature compiles
- ONNX embedder tests are feature-gated (require model download + `ORT_DYLIB_PATH`)

**2. Python SDK (`sage-python/`)**
- Installs with `pip install -e ".[all,dev]"`
- `python -m pytest tests/ -v` -- 691 tests
- `ruff check src/` -- Lint
- Tests run without API keys (all LLM calls are mocked)

**3. Python Discovery (`sage-discover/`)**
- Installs both `sage-python` and `sage-discover`
- `python -m pytest tests/ -v` -- 52 tests

## Notes

- The CI pipeline was added during audit fix A5 to ensure Rust tests are always validated.
- Cargo.toml uses edition 2021 (downgraded from 2024 for CI compatibility).
- ONNX tests require the model file (`all-MiniLM-L6-v2`); they are skipped if the model is not present.
- Python tests use mocked providers and do not require `GOOGLE_API_KEY` or Codex CLI authentication.
