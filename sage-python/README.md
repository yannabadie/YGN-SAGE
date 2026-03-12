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
python -m pytest tests/ -v             # Unit tests (1216 passed, 115 skipped)
ruff check src/                        # Lint
mypy src/                              # Type check
python -m sage.bench --type routing    # Routing benchmark (no API key needed)
python -m sage.bench --type humaneval  # HumanEval 164 (needs LLM provider)
python -m sage.bench --type evalplus --dataset humaneval   # EvalPlus HumanEval+ (80x harder)
python -m sage.bench --type evalplus --dataset mbpp         # EvalPlus MBPP+ (35x harder)
python -m sage.bench --type ablation --limit 20             # Ablation study (6 configs)
python -m sage.bench.eval_protocol --suite humaneval -v     # Official evaluation protocol
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
| `sage/strategy/` | AdaptiveRouter (5-stage learned routing: structural → kNN embeddings → BERT ONNX → entropy → cascade), KnnRouter (arXiv 2505.12601, 92% accuracy on 50 GT tasks), ComplexityRouter (heuristic fallback), CGRS self-braking, training data export |
| `sage/topology/` | MAP-Elites + CMA-ME + MCTS topology search, LLM synthesis, KG-RLVR process reward model |
| `sage/evolution/` | Evolutionary engine, LLM-driven mutation |
| `sage/tools/` | Tool registry, dynamic tool creation (Rust ToolExecutor first, Python fallback), memory tools, ExoCortex tools |
| `sage/events/` | EventBus: in-proc event system for observability |
| `sage/guardrails/` | 3-layer guardrails: input, runtime, output |
| `sage/bench/` | EvalPlus HumanEval+/MBPP+, routing accuracy, routing quality, ablation, evaluation protocol with error logging |
| `sage/sandbox/` | Sandbox manager (host execution disabled by default) |
| `sage/routing/` | DynamicRouter (capability-constrained), ShadowRouter (dual Rust/Python traces) |

## Environment Variables

```bash
export GOOGLE_API_KEY="..."              # Required for Gemini models
export SAGE_MODEL_FAST="gemini-2.5-flash"  # Override any tier model ID
export SAGE_DASHBOARD_TOKEN="..."        # Dashboard auth (optional)
```

## Dependencies

Core: `httpx`, `pydantic`, `rich`, `anyio`, `aiosqlite`, `numpy`.
Optional: `google-genai` (Gemini), `openai` (Codex), `pyarrow` (Arrow memory), `z3-solver` (formal verification), `fastapi`/`uvicorn` (dashboard), `sentence-transformers` (Tier 2 embeddings), `onnxruntime` (Tier 1 RustEmbedder DLL), `sage_core` with `tool-executor` feature (Rust ToolExecutor: tree-sitter validation + Wasm WASI sandbox + subprocess isolation).
