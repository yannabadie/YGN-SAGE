# sage-python

Python SDK for YGN-SAGE (Self-Adaptive Generation Engine) -- an Agent Development Kit built on five cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Installation

```bash
cd sage-python
pip install -e ".[all,dev]"    # All providers + dev tools
pip install -e ".[google]"     # Google Gemini only
pip install -e ".[z3]"         # Z3 formal verification
```

Requires Python 3.12+.

## Quick Start

```python
from sage.boot import boot_agent_system

system = boot_agent_system()           # Auto-detects Codex CLI or Gemini
result = await system.run("Solve X")   # S1/S2/S3 routing + full agent loop
```

## Testing

```bash
python -m pytest tests/ -v             # Unit tests (846 passed, 1 skipped)
ruff check src/                        # Lint
mypy src/                              # Type check
python -m sage.bench --type routing    # Routing benchmark (no API key needed)
python -m sage.bench --type humaneval  # HumanEval 164 (needs LLM provider)
```

## Package Structure

| Subpackage | Description |
|------------|-------------|
| `sage/` | Core runtime: boot sequence, agent loop, resilience |
| `sage/agents/` | Composition patterns: sequential, parallel, loop, handoff |
| `sage/contracts/` | Contract IR, DAG verification, Z3 SMT, CEGAR repair |
| `sage/memory/` | 4-tier memory: working (Arrow), episodic (SQLite), semantic (graph), ExoCortex (RAG) |
| `sage/llm/` | LLM providers: Google Gemini, OpenAI Codex CLI, model router |
| `sage/providers/` | Provider discovery, capability matrix, OpenAI-compat adapter |
| `sage/strategy/` | ComplexityRouter (S1/S2/S3 routing), CGRS self-braking |
| `sage/topology/` | MAP-Elites topology search, KG-RLVR process reward model |
| `sage/evolution/` | Evolutionary engine, LLM-driven mutation |
| `sage/tools/` | Tool registry, dynamic tool creation (Rust ToolExecutor first, Python fallback), memory tools, ExoCortex tools |
| `sage/events/` | EventBus: in-proc event system for observability |
| `sage/guardrails/` | 3-layer guardrails: input, runtime, output |
| `sage/bench/` | Benchmarks: HumanEval, routing accuracy |
| `sage/sandbox/` | Sandbox manager (host execution disabled by default) |
| `sage/routing/` | DynamicRouter: capability-constrained model selection |

## Environment Variables

```bash
export GOOGLE_API_KEY="..."              # Required for Gemini models
export SAGE_MODEL_FAST="gemini-2.5-flash"  # Override any tier model ID
export SAGE_DASHBOARD_TOKEN="..."        # Dashboard auth (optional)
```

## Dependencies

Core: `httpx`, `pydantic`, `rich`, `anyio`, `aiosqlite`, `numpy`.
Optional: `google-genai` (Gemini), `openai` (Codex), `pyarrow` (Arrow memory), `z3-solver` (formal verification), `fastapi`/`uvicorn` (dashboard), `sentence-transformers` (Tier 2 embeddings), `onnxruntime` (Tier 1 RustEmbedder DLL), `sage_core` with `tool-executor` feature (Rust ToolExecutor: tree-sitter validation + Wasm WASI sandbox + subprocess isolation).
