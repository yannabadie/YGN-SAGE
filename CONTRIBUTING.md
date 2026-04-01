# Contributing to YGN-SAGE

## Development Setup

```bash
# Clone
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE

# Rust core (requires Rust 1.90+)
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor

# Python SDK (requires Python 3.12+)
cd sage-python && pip install -e ".[all,dev]"
```

## Running Tests

```bash
# Rust (403+ tests)
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib

# Python (1900+ tests)
cd sage-python && python -m pytest tests/ -v
```

## Code Standards

- **Rust first**: performance-critical code in `sage-core/`, Python for orchestration
- **No `except Exception:`**: use specific exception types (ImportError, RuntimeError, etc.)
- **Research-backed**: cite arXiv when introducing new algorithms or thresholds
- **Evidence before assertions**: run tests + benchmarks before claiming completion

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass (Rust + Python)
4. Update documentation if adding features
5. Reference related issues in the PR description

## Architecture

See [AI-ARCHITECTURE.md](AI-ARCHITECTURE.md) for the full system design.

## License

MIT License. By contributing, you agree that your contributions will be licensed under MIT.
