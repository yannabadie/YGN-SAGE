# YGN-SAGE - CLAUDE.md

## Project Overview
YGN-SAGE (Yann's Generative Neural Self-Adaptive Generation Engine) is a next-generation
Agent Development Kit built on 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Architecture
- `sage-core/` - Rust orchestrator (PyO3 bindings to Python)
- `sage-python/` - Python SDK for building agents
- `sage-discover/` - Flagship Research & Discovery agent
- `ui/` - Control Dashboard (FastAPI + Tailwind + WebSocket)
- `docs/plans/` - Architecture and implementation plans
- `Researches/` - Research papers (OpenSage, AlphaEvolve, PSRO, etc.)

### Key Python Modules (sage-python/src/sage/)
- `boot.py` - Boot sequence, wires all pillars into `AgentSystem`
- `agent_loop.py` - Structured perceive→think→act→learn runtime
- `agent_pool.py` - Dynamic sub-agent pool (create/run/ensemble)
- `llm/router.py` - Model Router with 6 tiers (fast/mutator/reasoner/codex/budget/fallback)
- `llm/google.py` - Google Gemini provider (primary LLM)
- `llm/codex.py` - OpenAI Codex CLI provider (fallback: Google)
- `strategy/metacognition.py` - SOFAI System 1/3 routing + CGRS self-braking
- `topology/evo_topology.py` - MAP-Elites evolutionary topology search
- `evolution/llm_mutator.py` - LLM-driven code mutation with structured JSON
- `memory/memory_agent.py` - Autonomous entity extraction agent

### Dashboard (ui/)
- `ui/app.py` - FastAPI backend: REST API + WebSocket event streaming
- `ui/static/index.html` - Single-file production dashboard (Tailwind + vanilla JS)
- Endpoints: `GET /`, `GET /api/state`, `POST /api/task`, `POST /api/stop`, `POST /api/reset`, `WS /ws`

## Development Commands

### Python SDK
```bash
cd sage-python
pip install -e ".[all,dev]"    # Install in dev mode with all providers
pytest                          # Run tests (59 passing)
ruff check src/                 # Lint
mypy src/                       # Type check
```

### Dashboard
```bash
# From repo root:
python ui/app.py                # Start dashboard on http://localhost:8000
```

### Rust Core
```bash
cargo build                    # Build Rust core
cargo test                     # Run Rust tests
cargo clippy                   # Lint Rust code
```

### Full Build
```bash
cargo build --release
cd sage-python && pip install -e ".[all,dev]"
```

## LLM Configuration

### Active Models (March 2026, verified working)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | `gpt-5.3-codex` | Codex CLI | Default, SOTA coding |
| codex_max | `gpt-5.2` | Codex CLI | Most powerful reasoning, xhigh effort |
| reasoner | `gemini-3.1-pro-preview` | Google API | Complex evaluation |
| mutator | `gemini-3-flash-preview` | Google API | Code mutation |
| fast | `gemini-3.1-flash-lite-preview` | Google API | Low-latency |
| budget | `gemini-2.5-flash-lite` | Google API | Cheapest |
| fallback | `gemini-2.5-flash` | Google API | If 3.x unavailable |

### Required Environment Variables
```bash
export GOOGLE_API_KEY="..."     # Required for Gemini models
# Codex CLI uses ChatGPT Pro account (codex login)
```

### Structured Output
- Codex: `--output-schema file.json` (requires `additionalProperties: false`)
- Gemini: `response_schema=PydanticModel.model_json_schema()`

## Tech Stack
- Rust 1.90+ (orchestrator, via PyO3)
- Python 3.12+ (SDK, agents)
- OpenAI Codex CLI + gpt-5.3-codex (primary LLM)
- Google Gemini 3.x (secondary LLM, fallback)
- FastAPI + WebSocket (dashboard)
- Z3 Solver (formal verification)

## Key Design Principles
- AI-centered: agents create their own topology, tools, and memory
- Self-evolving: evolutionary pipeline improves all components
- Game-theoretic: PSRO-based strategy for multi-agent orchestration
- Multi-provider: Codex CLI (primary), Gemini (fallback), Mock (testing)
- Metacognitive: SOFAI S1/S2/S3 tripartite cognitive routing with CGRS self-braking

## NotebookLM Integration

Package `notebooklm-py` (v0.3.2) is installed. CLI at `sage-python/.venv/Scripts/notebooklm.exe`.

### Available Notebooks (research knowledge base)
| ID prefix | Title | Use for |
|-----------|-------|---------|
| `34d65dbb` | YGN-SAGE: Core Research & MARL | Game theory, PSRO, VAD-CFR, SHOR-PSRO, MARL |
| `ba22b122` | YGN-SAGE: Technical Implementation | Architecture, metacognition, agent loop, memory |
| `dcf45958` | Discover AI: Frontiers of Agentic Reasoning | SOTA research, dual-process, reflection, reasoning |
| `097c4c5c` | MetaScaffold_Core | Meta-reasoning, cognitive architectures, scaffolding |
| `7ab1d708` | YGN-ExoCortex | Memory architecture, S-MMU, Arrow, graph memory |

### CLI Protocol
```bash
# IMPORTANT: Always set PYTHONIOENCODING=utf-8 on Windows to avoid encoding errors
PYTHONIOENCODING=utf-8 sage-python/.venv/Scripts/notebooklm.exe use <ID_PREFIX>
PYTHONIOENCODING=utf-8 sage-python/.venv/Scripts/notebooklm.exe ask "<question>"

# Other useful commands:
notebooklm list                    # List all notebooks
notebooklm status                  # Show current notebook context
notebooklm source list             # List sources in current notebook
notebooklm history                 # Get conversation history
```

### Best Practices
- Present the project context and problem in detail when asking questions
- Query multiple notebooks in parallel for cross-referenced insights
- Conversations persist (use `notebooklm history` to review)
- Each notebook has different source material — pick the right one for your question
