# YGN-SAGE - CLAUDE.md

## Project Overview
YGN-SAGE (Yann's Generative Neural Self-Adaptive Generation Engine) is a next-generation
Agent Development Kit built on 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Architecture
- `sage-core/` - Rust orchestrator (PyO3 bindings to Python)
- `sage-python/` - Python SDK for building agents
- `sage-discover/` - Flagship Research & Discovery agent
- `docs/plans/` - Architecture and implementation plans
- `Researches/` - Research papers (OpenSage, AlphaEvolve, PSRO, etc.)

## Development Commands

### Rust Core
```bash
cargo build                    # Build Rust core
cargo test                     # Run Rust tests
cargo clippy                   # Lint Rust code
```

### Python SDK
```bash
cd sage-python
pip install -e ".[dev]"        # Install in dev mode
pytest                         # Run tests
ruff check src/                # Lint
mypy src/                      # Type check
```

### Full Build
```bash
cargo build --release
cd sage-python && pip install -e ".[all,dev]"
```

## Tech Stack
- Rust 1.90+ (orchestrator, via PyO3)
- Python 3.13+ (SDK, agents)
- Neo4j (graph memory, episodic + semantic)
- Qdrant (vector memory)
- Docker (tool sandboxing)

## Key Design Principles
- AI-centered: agents create their own topology, tools, and memory
- Self-evolving: evolutionary pipeline improves all components
- Game-theoretic: PSRO-based strategy for multi-agent orchestration
- Multi-provider: supports Claude, GPT, Gemini, and local LLMs
