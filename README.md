# YGN-SAGE

**YGN-SAGE** (Yann's Generative Neural Self-Adaptive Generation Engine) is an Agent Development Kit built on **5 cognitive pillars**: Topology, Tools, Memory, Evolution, Strategy.

It combines a Rust execution core (`sage-core`) with a Python orchestration layer (`sage-python`), a tripartite cognitive routing system (S1/S2/S3), and a real-time control dashboard.

> **Status**: Research prototype under active development. Core agent loop and all 5 pillars are functional with 245 Python tests + 38 Rust tests passing (283 total). CI via GitHub Actions. Not yet battle-tested in production environments.

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
| **Tools** | `tools/registry.py` | Dynamic tool creation, registration, and sandboxed execution |
| **Memory** | `memory_agent.py` | Entity extraction + MEM1 rolling internal state + 7 AgeMem tools + ExoCortex |
| **Evolution** | `llm_mutator.py` | LLM-driven code mutation with DGM context + eBPF sub-ms evaluation + SnapBPF |
| **Strategy** | `metacognition.py` | Stanovich S1/S2/S3 tripartite routing + CGRS self-braking |

### Cognitive Routing (S1/S2/S3)

YGN-SAGE implements a **tripartite cognitive model** inspired by Stanovich (2011):

| System | Role | LLM Tier | Validation | Latency |
|--------|------|----------|------------|---------|
| **S1** (Autonomous) | Fast heuristic responses | `fast` (Gemini Flash Lite) | None | <1s |
| **S2** (Algorithmic) | Step-by-step reasoning | `mutator`/`reasoner` (Gemini Flash/Pro) | Empirical (AVR sandbox loop) | 2-5s |
| **S3** (Reflective) | Formal verified reasoning | `codex`/`reasoner` (GPT-5.3/Gemini Pro) | Z3 PRM formal proofs | 5-30s |

The **MetacognitiveController** assesses task complexity and uncertainty, then routes to the appropriate system. S2's **Act-Verify-Refine (AVR)** loop executes code in a sandbox, checks results, and retries with structured error feedback before escalating to S3.

**S3 Z3 Alignment**: The S3 prompt teaches the LLM the exact Z3 DSL assertion syntax expected by the Process Reward Model (`assert bounds`, `assert loop`, `assert arithmetic`, `assert invariant`). The retry and escalation prompts reinforce this syntax, ensuring LLM output is parseable by the Z3 validator.

See [ADR-001: System 2 Cognitive Routing](docs/ADR-cognitive-routing-system2.md) for the full decision record.

### Memory System

YGN-SAGE uses a **three-tier memory architecture**:

**Working Memory (STM)** -- Rust-backed Arrow buffer for the current session:
- Per-step **MEM1 internal state** (`<IS_t>`): the MemoryCompressor generates a rolling summary every step, merging previous state with new observations
- Pressure-triggered **bulk compression**: when event count exceeds threshold, LLM summarizes old events and prunes the buffer

**Episodic Memory (LTM)** -- In-memory store with keyword search:
- CRUD operations (store, update, delete, search, list)
- Automatic storage of significant agent outputs during the LEARN phase

**ExoCortex (Persistent RAG)** -- Google GenAI File Search API:
- Persistent managed stores with automatic chunking and embedding
- Replaces the fragile NotebookLM CLI bridge with native SDK integration
- **Passive grounding**: automatically injected into every `_think()` call via `file_search_store_names`
- **Active grounding**: `search_exocortex` agent tool for targeted research queries
- Results cached by FIFO+TTL Rust cache (`sage_core.RagCache`) with Python fallback

**Knowledge Pipeline** (`sage-discover`) -- Auto-refreshing research paper ingestion:
- **Discovery**: Scans arXiv, Semantic Scholar, HuggingFace across 5 domains (MARL, cognitive architectures, formal verification, evolutionary computation, memory systems)
- **Curation**: Two-stage filtering (heuristic + LLM scoring with Gemini Flash)
- **Ingestion**: PDF download + ExoCortex upload with `custom_metadata` + local manifest dedup
- **Migration**: One-time NotebookLM Q&A export bootstrap
- **CLI**: `python -m discover.pipeline --mode nightly|on-demand|migrate`
- **Agent tool**: `refresh_knowledge` triggers on-demand pipeline from within the agent loop

**7 AgeMem Memory Tools** -- exposed to the LLM as callable tools:

| Tool | Memory Tier | Description |
|------|-------------|-------------|
| `retrieve_context` | STM | Get recent working memory events |
| `summarize_context` | STM | Get the current MEM1 internal state |
| `filter_context` | STM | Search working memory by keyword |
| `search_memory` | LTM | Search episodic memory |
| `store_memory` | LTM | Store a new episodic entry |
| `update_memory` | LTM | Update an existing entry by key |
| `delete_memory` | LTM | Delete an entry by key |
| `search_exocortex` | ExoCortex | Search research papers in ExoCortex store |
| `refresh_knowledge` | ExoCortex | Trigger on-demand knowledge pipeline (discover + curate + ingest) |

### Evolution System

The evolution engine uses **SAMPO game-theoretic solver** (DGM) to choose among 5 strategic actions per mutation step:

| Action | Directive |
|--------|-----------|
| 0 | Optimize execution performance and reduce latency |
| 1 | Improve correctness and fix edge cases |
| 2 | Expand search space -- explore novel algorithmic approaches |
| 3 | Tighten constraints -- make code more robust and safe |
| 4 | Simplify and reduce complexity while maintaining functionality |

The chosen action's semantic directive is injected into the LLM mutation prompt ("DGM Directive" section), so mutations are **strategically guided** rather than blind. Context includes action, description, parent score, and generation number.

**SnapBPF** provides sub-ms CoW memory snapshots for mutation rollback (Rust, DashMap-backed).

### LLM Providers

YGN-SAGE uses a tiered model router with automatic fallback:

**OpenAI (via [Codex CLI](https://github.com/openai/codex)):**

| Tier | Model | Usage |
|------|-------|-------|
| `codex` | `gpt-5.3-codex` | Agentic coding (default) |
| `codex_max` | `gpt-5.2` | Complex general reasoning |

**Google Gemini (via `google-genai` SDK):**

| Tier | Model | Usage |
|------|-------|-------|
| `fast` | `gemini-3.1-flash-lite-preview` | High-volume, low-latency (S1) |
| `mutator` | `gemini-3-flash-preview` | Code mutation, S2 routing |
| `reasoner` | `gemini-3.1-pro-preview` | Complex reasoning, S2/S3 routing |
| `budget` | `gemini-2.5-flash-lite` | Bulk cheap transforms |
| `fallback` | `gemini-2.5-flash` | If 3.x unavailable |

Model IDs are configurable via `config/models.toml` or environment variables (`SAGE_MODEL_FAST`, `SAGE_MODEL_REASONER`, etc.). See [Configuration](#configuration).

All providers support **structured JSON output** (via `--output-schema` for Codex, `response_schema` for Gemini).

`GoogleProvider.generate()` also accepts `file_search_store_names` for ExoCortex grounding.

### Sandbox Execution

Code execution uses `SandboxManager` with three backends (in priority order):

1. **Wasm** (via `sage-core` + `wasmtime`/`solana_rbpf`) -- sub-ms, highest isolation
2. **Docker** (`use_docker=True`) -- container-level isolation, requires Docker Desktop
3. **Local subprocess** (`use_docker=False`, current default) -- no isolation, for development only

> **Note**: The default is local subprocess execution. For untrusted code, enable Docker (`SandboxManager(use_docker=True)`) or use the Wasm path.

## Quickstart

### Prerequisites

- Python 3.12+
- A Google AI API key ([get one here](https://aistudio.google.com/apikey))
- [Codex CLI](https://github.com/openai/codex) with a ChatGPT Pro account (optional, for `codex` tier)
- Rust 1.90+ (optional, for `sage-core` native performance: SnapBPF, RagCache, Arrow memory)

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

# 6. (Optional) Install sage-discover for Knowledge Pipeline
cd sage-discover && pip install -e . && cd ..

# 7. Verify installation
cd sage-python && python -m pytest tests/ -v
# Expected: 200 passed, 1 skipped
```

### Running the Dashboard

```bash
# From the repo root:
python ui/app.py
```

This starts the **FastAPI backend AND the frontend** on **http://localhost:8000**:

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

### Running the Knowledge Pipeline

```bash
# Nightly: discover + curate + ingest papers from yesterday
python -m discover.pipeline --mode nightly

# On-demand: search for a specific topic
python -m discover.pipeline --mode on-demand --query "attention mechanism pruning"

# Migrate: bootstrap ExoCortex from NotebookLM exports
# First: save NotebookLM Q&A as markdowns in ~/.sage/migration/
python -m discover.pipeline --mode migrate
```

### Running Tests

```bash
# Python tests (sage-python)
cd sage-python
python -m pytest tests/ -v
# Expected: 200 passed, 1 skipped

# Discovery pipeline tests (sage-discover)
cd sage-discover
python -m pytest tests/ -v
# Expected: 45 passed

# Rust tests (requires Rust 1.90+)
cd sage-core
cargo test
# Expected: 38 passed
```

## Project Structure

```
YGN-SAGE/
|-- sage-core/              # Rust core (PyO3 bindings)
|   +-- src/
|       |-- memory/         # Arrow-backed working memory + RagCache (FIFO+TTL)
|       |-- sandbox/        # eBPF executor (solana_rbpf) + SnapBPF (CoW snapshots)
|       +-- z3/             # Z3 formal verification bindings
|-- sage-python/            # Python SDK
|   +-- src/sage/
|       |-- llm/            # LLM providers (Google, Codex, Mock)
|       |   |-- base.py     # LLMConfig, LLMResponse, Message types
|       |   |-- google.py   # Google Gemini provider + FileSearch grounding
|       |   |-- codex.py    # OpenAI Codex CLI provider (+ Google fallback)
|       |   |-- router.py   # ModelRouter with 7 tiers
|       |   +-- mock.py     # Mock provider for testing
|       |-- tools/          # Tool registry + meta-tools (Python, Bash) + memory tools
|       |-- memory/         # Working memory + episodic + compressor + ExoCortex + RAG cache
|       |-- topology/       # KG-RLVR (Z3 PRM), Z3 validator, evo topology
|       |-- evolution/      # LLM mutator (DGM context), eBPF evaluator, fitness cascade
|       |-- strategy/       # Metacognitive controller + PSRO/CFR solvers
|       |-- sandbox/        # Sandbox manager (Wasm > Docker > local subprocess)
|       |-- agent.py        # Core Agent class
|       |-- agent_loop.py   # Structured perceive->think->act->learn runtime
|       |-- agent_pool.py   # Dynamic sub-agent pool
|       +-- boot.py         # Boot sequence (wires all pillars + ExoCortex)
|-- sage-discover/          # Flagship research agent + Knowledge Pipeline
|   +-- src/discover/
|       |-- discovery.py    # arXiv + Semantic Scholar + HuggingFace scanning
|       |-- curator.py      # Heuristic filter + LLM relevance scoring
|       |-- ingestion.py    # ExoCortex upload + manifest tracking
|       |-- migration.py    # NotebookLM bootstrap to ExoCortex
|       |-- pipeline.py     # Orchestrator (nightly/on-demand/migrate)
|       +-- __main__.py     # CLI entry point
|-- ui/                     # Control Dashboard
|   |-- app.py              # FastAPI backend (REST + WebSocket)
|   +-- static/index.html   # Single-file dashboard (Tailwind + vanilla JS)
|-- docs/                   # Architecture Decision Records + plans
|-- Researches/             # Research papers + experimental code
+-- debug/                  # Diagnostic scripts
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key for Gemini models |
| `SAGE_EXOCORTEX_STORE` | No | Google GenAI File Search store resource name (e.g. `file_search_stores/...`). Enables passive + active ExoCortex grounding. |
| `SAGE_MODEL_<TIER>` | No | Override model ID per tier (e.g. `SAGE_MODEL_FAST=gemini-2.5-flash`) |

Codex CLI authenticates via `codex login` (ChatGPT Pro account).

### Model Configuration

Model IDs are resolved in order: **env var > `config/models.toml` > hardcoded defaults**.

```bash
# Override a single tier via env var
export SAGE_MODEL_FAST="gemini-2.5-flash"

# Or edit the TOML config (searched in: ./config/, ~/.sage/)
cat sage-python/config/models.toml
```

Override the default LLM tier when booting:

```python
# Use the cheapest model for high-volume tasks
system = boot_agent_system(llm_tier="budget")

# Use the reasoning model for complex analysis
system = boot_agent_system(llm_tier="reasoner")

# Use Codex for coding tasks
system = boot_agent_system(llm_tier="codex")
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Rust 1.90+ (PyO3) | High-performance core: memory, sandboxing, caching |
| SDK | Python 3.12+ | Agent logic, LLM providers, tools |
| Primary LLM | OpenAI Codex CLI (gpt-5.3) | Agentic coding |
| Secondary LLM | Google Gemini 3.x (`google-genai`) | Multi-tier routing, fallback, File Search |
| Dashboard | FastAPI + WebSocket | Real-time control interface |
| Verification | Z3 Solver 4.16 | Formal reasoning validation (S3) |
| Sandbox | Wasm (`wasmtime`) + eBPF (`solana_rbpf`) + Docker | Multi-tier code isolation |
| Serialization | Apache Arrow (PyArrow) | Zero-copy memory compaction |
| Caching | DashMap (Rust) | Lock-free FIFO+TTL RAG cache, CoW snapshots |

## Status (March 2026)

- **283 tests passing** (200 sage-python + 45 sage-discover + 38 sage-core Rust)
- **CI/CD**: GitHub Actions with 3 jobs (Rust fmt+clippy+test, Python sage, Python discover)
- **Dashboard**: Functional with real-time telemetry, response pane, and event streaming
- **Cognitive Routing**: Tripartite S1/S2/S3 with AVR sandbox loop (S2) and Z3 formal proofs (S3)
- **Agent Loop**: Full perceive->think->act->learn cycle with CGRS self-braking, async metacognition, independent S2/S3 retry budgets, structured `AgentEvent` observability
- **LLM Integration**: Google Gemini fully wired (incl. File Search grounding), Codex CLI optional
- **Model Config**: Externalized to `config/models.toml` with env var overrides (`SAGE_MODEL_<TIER>`)
- **Memory**: MEM1 per-step internal state + 9 agent tools (7 AgeMem + 2 ExoCortex) + Arrow compaction
- **Evolution**: MAP-Elites topology search + LLM-driven mutation with DGM context + eBPF evaluation + SnapBPF
- **Strategy**: PSRO meta-solver with VAD-CFR and SHOR-PSRO variants
- **Knowledge Pipeline**: Auto-refreshing research ingestion (arXiv/S2/HF -> curate -> ExoCortex)
- **RAG Cache**: Rust FIFO+TTL cache with Python fallback

### Known Limitations

- **Sandbox default is local subprocess** -- enable Docker or Wasm for isolation
- **Episodic memory is in-memory only** -- no cross-session persistence yet

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development instructions for Claude Code |
| [GEMINI.md](GEMINI.md) | Gemini CLI memory and strategic directives |
| [ADR-001: S2 Routing](docs/ADR-cognitive-routing-system2.md) | System 2 cognitive routing decision record |
| [Phase 3 Design](docs/plans/2026-03-05-exocortex-dgm-z3-design.md) | ExoCortex, DGM, Z3 design decisions |
| [Phase 3 Implementation](docs/plans/2026-03-05-phase3-implementation.md) | TDD implementation plan (5 tasks) |
| [Knowledge Pipeline Design](docs/plans/2026-03-05-knowledge-pipeline-design.md) | ExoCortex auto-refreshing pipeline design |
| [Knowledge Pipeline Plan](docs/plans/2026-03-05-knowledge-pipeline.md) | Implementation plan (6 tasks, TDD) |
| [Hardening Design](docs/plans/2026-03-06-hardening-maturation-design.md) | Expert review-driven hardening design |
| [Hardening Plan](docs/plans/2026-03-06-hardening-maturation.md) | 12-task TDD implementation plan |
| [Codebase Audit](docs/plans/2026-03-05-codebase-audit.md) | Full connectivity audit (March 2026) |
| [OpenSAGE-surpass Plan](docs/plans/2026-03-05-opensage-surpass-implementation.md) | 5-pillar implementation plan |
| [Architecture Design](docs/plans/2026-03-02-ygn-sage-architecture-design.md) | Original architecture spec |
| [Researches/](Researches/) | Curated research papers (AlphaEvolve, PSRO, etc.) |

## License

Proprietary. All rights reserved. (c) 2026 Yann Abadie.
