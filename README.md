# YGN-SAGE

**YGN-SAGE** (Yann's Generative Neural Self-Adaptive Generation Engine) is an Agent Development Kit built on **5 cognitive pillars**: Topology, Tools, Memory, Evolution, Strategy.

It combines a high-performance Rust execution core (`sage-core`) with a Python orchestration layer (`sage-python`), a tripartite cognitive routing system (S1/S2/S3), and a real-time control dashboard.

## Core Architecture

```
                        +------------------------------+
                        |   Control Dashboard (:8000)   |
                        |  FastAPI + WebSocket + HTML    |
                        +-------------+----------------+
                                      | POST /api/task
                        +-------------v----------------+
                        |       boot.py (AgentSystem)   |
                        |  Metacognition -> ModelRouter  |
                        +-------------+----------------+
                                      |
              +-----------------------v-----------------------+
              |              Agent Runtime Loop               |
              |       perceive -> think -> act -> learn       |
              +---+--------+----------+--------+------+------+
                  |        |          |        |      |
           +-----v--+ +---v---+ +----v---+ +--v----+ +--v------+
           | LLM    | | Tools | | Memory | |Evolve | |Strategy |
           |Provider| |Sandbox| | Agent  | |Engine | |PSRO/CFR |
           +--------+ +-------+ +--------+ +-------+ +---------+
```

### 5 Cognitive Pillars

| Pillar | Module | Description |
|--------|--------|-------------|
| **Topology** | `evo_topology.py` | MAP-Elites evolutionary search on agent DAG topologies |
| **Tools** | `tools/registry.py` | Dynamic tool creation, registration, and Docker-sandboxed execution |
| **Memory** | `memory_agent.py` | Entity extraction, Rust-backed Arrow working memory |
| **Evolution** | `llm_mutator.py` | LLM-driven code mutation + eBPF sub-ms evaluation |
| **Strategy** | `metacognition.py` | Stanovich S1/S2/S3 tripartite routing + CGRS self-braking |

### Cognitive Routing (S1/S2/S3)

YGN-SAGE implements a **tripartite cognitive model** inspired by Stanovich (2011):

| System | Role | LLM Tier | Validation | Latency |
|--------|------|----------|------------|---------|
| **S1** (Autonomous) | Fast heuristic responses | `fast` (Gemini Flash Lite) | None | <1s |
| **S2** (Algorithmic) | Step-by-step reasoning | `mutator`/`reasoner` (Gemini Flash/Pro) | Empirical (CoT enforcement) | 2-5s |
| **S3** (Reflective) | Formal verified reasoning | `codex`/`reasoner` (GPT-5.3/Gemini Pro) | Z3 PRM formal proofs | 5-30s |

The **MetacognitiveController** assesses task complexity and uncertainty, then routes to the appropriate system. S2 offloads ~80% of moderate tasks from expensive S3, reducing cost by ~20x.

See [ADR-001: System 2 Cognitive Routing](docs/ADR-cognitive-routing-system2.md) for the full decision record.

### LLM Providers

YGN-SAGE uses a tiered model router with automatic fallback:

**OpenAI (via [Codex CLI](https://github.com/openai/codex)):**

| Tier | Model | Usage |
|------|-------|-------|
| `codex` | `gpt-5.3-codex` | SOTA agentic coding (default) |
| `codex_max` | `gpt-5.2` | Most powerful general reasoning |

**Google Gemini (via `google-genai` SDK):**

| Tier | Model | Usage |
|------|-------|-------|
| `fast` | `gemini-3.1-flash-lite-preview` | High-volume, low-latency (S1) |
| `mutator` | `gemini-3-flash-preview` | Code mutation, S2 routing |
| `reasoner` | `gemini-3.1-pro-preview` | Complex reasoning, S2/S3 routing |
| `budget` | `gemini-2.5-flash-lite` | Bulk cheap transforms |
| `fallback` | `gemini-2.5-flash` | If 3.x unavailable |

All providers support **structured JSON output** (via `--output-schema` for Codex, `response_schema` for Gemini).

## Quickstart

### Prerequisites

- Python 3.12+
- A Google AI API key ([get one here](https://aistudio.google.com/apikey))
- [Codex CLI](https://github.com/openai/codex) with a ChatGPT Pro account (optional, for `codex` tier)
- Rust 1.90+ (optional, for `sage-core` native bindings)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE

# 2. Create a virtual environment
python -m venv sage-python/.venv
source sage-python/.venv/bin/activate  # Windows: sage-python\.venv\Scripts\activate

# 3. Install Python SDK with all providers
cd sage-python && pip install -e ".[all,dev]" && cd ..

# 4. Set your API key
export GOOGLE_API_KEY="your_google_api_key"
# Windows: set GOOGLE_API_KEY=your_google_api_key

# 5. (Optional) Build Rust core for native performance
pip install maturin
cd sage-core && maturin develop && cd ..
```

### Running the Dashboard

```bash
# From the repo root:
python ui/app.py
```

This starts the **FastAPI backend AND the frontend** on **http://localhost:8000**. A single command launches everything:

- **Backend**: FastAPI server with REST API + WebSocket event streaming
- **Frontend**: Single-file HTML dashboard served at `/` via `StaticFiles`
- **Agent Runtime**: Boots on-demand when you submit a task via `POST /api/task`

**Dashboard features:**
- **Task Input**: Type a prompt and run the agent
- **Response Pane**: Real-time agent response with markdown rendering, S1/S2/S3 color coding
- **Phase Indicator**: Animated perceive -> think -> act -> learn cycle
- **Live Event Stream**: Color-coded log with phase-based filtering
- **Telemetry**: AIO ratio, step count, LLM cost, memory events, inference latency
- **Evolution Grid**: MAP-Elites heatmap of topology search
- **Cognitive Routing**: S1/S2/S3 bars with active system indicator
- **Z3 Verification**: Pass/fail counter for formal proofs
- **Sub-Agent Pool**: Live tracking of dynamically spawned agents
- **LLM Providers**: Status of all configured model tiers

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the dashboard HTML |
| `/api/state` | GET | Current dashboard state (polled by frontend) |
| `/api/task` | POST | Submit a task `{"task": "..."}` |
| `/api/stop` | POST | Cancel the running agent |
| `/api/reset` | POST | Reset all counters |
| `/api/providers` | GET | List available LLM providers |
| `/ws` | WS | Real-time event stream (JSONL) |

### Running the Agent via Python

```python
import asyncio
from sage.boot import boot_agent_system

async def main():
    # With real LLM (requires GOOGLE_API_KEY)
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast")
    result = await system.run("Explain the trade-offs of B-tree vs LSM-tree indexes.")
    print(result)

asyncio.run(main())
```

```python
# For testing (no API key needed)
system = boot_agent_system(use_mock_llm=True)
result = await system.run("Test task")
```

### Running Tests

```bash
cd sage-python
python -m pytest tests/ -v

# Current status: 162 tests, all passing
```

## Project Structure

```
YGN-SAGE/
|-- sage-core/              # Rust core (PyO3 bindings)
|   +-- src/
|       |-- memory/         # Arrow-backed working memory (SIMD/AVX-512)
|       |-- sandbox/        # eBPF + Wasm sandboxing (solana_rbpf)
|       +-- z3/             # Z3 formal verification bindings
|-- sage-python/            # Python SDK
|   +-- src/sage/
|       |-- llm/            # LLM providers (Google, Codex, Mock)
|       |   |-- base.py     # LLMConfig, LLMResponse, Message types
|       |   |-- google.py   # Google Gemini provider (from google import genai)
|       |   |-- codex.py    # OpenAI Codex CLI provider (+ Google fallback)
|       |   |-- router.py   # ModelRouter with 7 tiers
|       |   +-- mock.py     # Mock provider for testing
|       |-- tools/          # Tool registry + meta-tools (Python, Bash)
|       |-- memory/         # Working memory + Memory Agent + episodic
|       |-- topology/       # KG-RLVR, Z3 validator, evo topology
|       |-- evolution/      # LLM mutator, eBPF evaluator, fitness cascade
|       |-- strategy/       # Metacognitive controller + PSRO/CFR solvers
|       |-- sandbox/        # Docker sandbox manager
|       |-- agent.py        # Core Agent class
|       |-- agent_loop.py   # Structured perceive->think->act->learn runtime
|       |-- agent_pool.py   # Dynamic sub-agent pool
|       +-- boot.py         # Boot sequence (wires all pillars)
|-- sage-discover/          # Flagship research agent + MCP Gateway
|-- ui/                     # Control Dashboard
|   |-- app.py              # FastAPI backend (REST + WebSocket)
|   +-- static/index.html   # Single-file production dashboard
|-- docs/                   # Architecture Decision Records + plans
|-- conductor/              # Strategic track planning
|-- memory-bank/            # AI agent context persistence
|-- Researches/             # Research papers + experimental code
|-- research_journal/       # Hypothesis logs (79 experiments)
+-- debug/                  # Diagnostic scripts
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key for Gemini models |

Codex CLI authenticates via `codex login` (ChatGPT Pro account).

### Model Selection

Override the default LLM tier when booting:

```python
# Use the cheapest model for high-volume tasks
system = boot_agent_system(llm_tier="budget")

# Use the reasoning model for complex analysis
system = boot_agent_system(llm_tier="reasoner")

# Use Codex for SOTA coding
system = boot_agent_system(llm_tier="codex")
```

Or edit `sage-python/src/sage/llm/router.py` to change model mappings.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Rust 1.90+ (PyO3) | High-performance core, memory, sandboxing |
| SDK | Python 3.12+ | Agent logic, LLM providers, tools |
| Primary LLM | OpenAI Codex CLI (gpt-5.3) | SOTA agentic coding |
| Secondary LLM | Google Gemini 3.x (`google-genai`) | Multi-tier routing, fallback |
| Dashboard | FastAPI + WebSocket | Real-time control interface |
| Verification | Z3 Solver 4.16 | Formal reasoning validation (S3) |
| Sandbox | eBPF (solana_rbpf) + Docker | Sub-ms code evaluation |
| Serialization | Apache Arrow (PyArrow) | Zero-copy memory compaction |

## Status (March 2026)

- **162/162 tests passing** (Python SDK)
- **Dashboard**: Production-ready with real-time telemetry and response pane
- **Cognitive Routing**: Tripartite S1/S2/S3 with validation levels (none/empirical/formal)
- **LLM Integration**: Google Gemini fully wired, Codex CLI optional
- **Agent Loop**: Full perceive->think->act->learn cycle with CGRS self-braking
- **Z3 Verification**: Formal safety gate on reasoning steps (S3)
- **Memory**: Rust-backed Arrow working memory + heuristic entity extraction
- **Evolution**: MAP-Elites topology search + LLM-driven mutation + eBPF evaluation
- **Strategy**: PSRO meta-solver with VAD-CFR and SHOR-PSRO variants

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development instructions for Claude Code |
| [GEMINI.md](GEMINI.md) | Gemini CLI memory and strategic directives |
| [ADR-001: S2 Routing](docs/ADR-cognitive-routing-system2.md) | System 2 cognitive routing decision record |
| [Codebase Audit](docs/plans/2026-03-05-codebase-audit.md) | Full connectivity audit (March 2026) |
| [OpenSAGE-surpass Plan](docs/plans/2026-03-05-opensage-surpass-implementation.md) | 5-pillar implementation plan |
| [Architecture Design](docs/plans/2026-03-02-ygn-sage-architecture-design.md) | Original architecture spec |
| [memory-bank/](memory-bank/) | AI agent context persistence files |
| [conductor/](conductor/) | Strategic track planning and roadmaps |
| [Researches/](Researches/) | Curated research papers (AlphaEvolve, PSRO, etc.) |

## License

Proprietary. All rights reserved. (c) 2026 Yann Abadie.
