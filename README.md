# YGN-SAGE

**YGN-SAGE** (Yann's Generative Neural Self-Adaptive Generation Engine) is an Agent Development Kit built on **5 cognitive pillars**: Topology, Tools, Memory, Evolution, Strategy.

It combines a high-performance Rust execution core (`sage-core`) with a Python orchestration layer (`sage-python`) and a real-time control dashboard.

## Core Architecture

```
                        ┌──────────────────────────────┐
                        │   Control Dashboard (:8000)   │
                        │  FastAPI + WebSocket + React   │
                        └──────────┬───────────────────┘
                                   │ POST /api/task
                        ┌──────────▼───────────────────┐
                        │       boot.py (AgentSystem)   │
                        │  Metacognition → ModelRouter   │
                        └──────────┬───────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │            Agent Runtime Loop            │
              │     perceive → think → act → learn      │
              └───┬────────┬──────────┬────────┬───────┘
                  │        │          │        │
           ┌─────▼──┐ ┌───▼───┐ ┌────▼───┐ ┌──▼──────┐
           │ LLM    │ │ Tools │ │ Memory │ │Evolution│
           │Provider│ │Registry│ │ Agent  │ │Topology │
           └────────┘ └───────┘ └────────┘ └─────────┘
```

### 5 Cognitive Pillars

| Pillar | Module | Description |
|--------|--------|-------------|
| **Topology** | `evo_topology.py` | MAP-Elites evolutionary search on agent DAG topologies |
| **Tools** | `tools/registry.py` | Dynamic tool creation, registration, and sandboxed execution |
| **Memory** | `memory_agent.py` | Entity extraction, working memory, Neo4j/Qdrant persistence |
| **Evolution** | `llm_mutator.py` | LLM-driven code mutation with structured JSON output |
| **Strategy** | `metacognition.py` | SOFAI System 1/3 routing + CGRS self-braking |

### LLM Providers

YGN-SAGE uses a tiered model router with automatic fallback. Two provider backends:

**OpenAI (via [Codex CLI](https://github.com/openai/codex) `codex exec`):**

| Tier | Model | Effort | Usage |
|------|-------|--------|-------|
| `codex` | `gpt-5.3-codex` | configurable | SOTA agentic coding (default) |
| `codex_max` | `gpt-5.2` | xhigh | Most powerful general reasoning |

**Google Gemini (via API key):**

| Tier | Model | Usage |
|------|-------|-------|
| `fast` | `gemini-3.1-flash-lite-preview` | High-volume, low-latency |
| `mutator` | `gemini-3-flash-preview` | Code mutation, SEARCH/REPLACE |
| `reasoner` | `gemini-3.1-pro-preview` | Complex reasoning, evaluation |
| `budget` | `gemini-2.5-flash-lite` | Bulk cheap transforms |
| `fallback` | `gemini-2.5-flash` | If 3.x unavailable |

All providers support **structured JSON output** (via `--output-schema` for Codex, `response_schema` for Gemini).

The **Metacognitive Controller** automatically routes tasks:
- **System 1** (fast): Simple tasks → `fast` tier
- **System 3** (formal): Complex/uncertain tasks → `reasoner` tier with Z3 verification

## Quickstart

### Prerequisites

- Python 3.12+
- [Codex CLI](https://github.com/openai/codex) with a ChatGPT Pro account (primary LLM)
- A Google AI API key ([get one here](https://aistudio.google.com/apikey)) (fallback LLM)
- Rust (optional, for `sage-core` native bindings)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-org/ygn-sage.git
cd ygn-sage

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install Python SDK with all providers
pip install -e "sage-python[all,dev]"

# 4. Set your API key
export GOOGLE_API_KEY="your_google_api_key"
# Windows:  set GOOGLE_API_KEY=your_google_api_key

# 5. (Optional) Build Rust core for native performance
pip install maturin
cd sage-core && maturin develop && cd ..
```

### Running the Agent via Dashboard

The dashboard is the primary interface for controlling YGN-SAGE agents.

```bash
# From the repo root:
python ui/app.py
```

Open **http://localhost:8000** in your browser. You'll see:

- **Task Input** (left sidebar): Type a task and click "Run Agent"
- **Phase Indicator**: Visualizes the perceive → think → act → learn cycle
- **Live Event Stream**: Real-time log of agent actions with colored phase indicators
- **Telemetry Cards**: AIO ratio, step count, LLM cost, memory events
- **Evolution Grid**: MAP-Elites heatmap of topology search
- **Metacognitive Display**: System 1/3 routing decisions and Z3 verification

**What happens when you submit a task:**
1. Dashboard calls `POST /api/task` with your prompt
2. Backend boots `AgentSystem` via `boot.py` (wires all 5 pillars)
3. `MetacognitiveController` assesses task complexity and routes to the right LLM tier
4. `AgentLoop` executes the perceive→think→act→learn cycle
5. Events are streamed to `agent_stream.jsonl`, which the WebSocket pushes to the dashboard in real-time
6. The agent stops when it produces a final answer (no tool calls) or hits `max_steps`

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

# Current status: 59 tests, all passing
```

## Project Structure

```
ygn-sage/
├── sage-core/              # Rust core (PyO3 bindings)
│   └── src/
│       ├── memory/         # Arrow-backed working memory
│       ├── sandbox/        # eBPF + Wasm sandboxing
│       └── z3/             # Z3 formal verification bindings
├── sage-python/            # Python SDK
│   └── src/sage/
│       ├── llm/            # LLM providers (Google, Codex, Mock)
│       │   ├── base.py     # LLMConfig, LLMResponse, Message types
│       │   ├── google.py   # Google Gemini provider
│       │   ├── codex.py    # OpenAI Codex CLI provider (+ Google fallback)
│       │   ├── router.py   # ModelRouter with 6 tiers
│       │   └── mock.py     # Mock provider for testing
│       ├── tools/          # Tool registry + built-in tools
│       ├── memory/         # Working memory + Memory Agent
│       ├── topology/       # KG-RLVR, Z3 validator, evo topology
│       ├── evolution/      # LLM mutator, fitness evaluation
│       ├── strategy/       # Metacognitive controller
│       ├── agent.py        # Core Agent class
│       ├── agent_loop.py   # Structured perceive→think→act→learn runtime
│       ├── agent_pool.py   # Dynamic sub-agent pool
│       └── boot.py         # Boot sequence (wires all pillars)
├── ui/                     # Control Dashboard
│   ├── app.py              # FastAPI backend (REST + WebSocket)
│   └── static/
│       └── index.html      # Single-file production dashboard
├── sage-discover/          # Reference agents + MCP Gateway
├── docs/plans/             # Architecture docs + agent event stream
└── research_journal/       # Research hypotheses log
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key for Gemini models |
| `OPENAI_API_KEY` | No | For Codex CLI (optional, falls back to Gemini) |
| `NEO4J_URI` | No | Neo4j connection for graph memory (default: bolt://localhost:7687) |
| `QDRANT_HOST` | No | Qdrant vector DB host (default: localhost) |

### Model Selection

Override the default LLM tier when booting:

```python
# Use the cheapest model for high-volume tasks
system = boot_agent_system(llm_tier="budget")

# Use the reasoning model for complex analysis
system = boot_agent_system(llm_tier="reasoner")
```

Or edit `sage-python/src/sage/llm/router.py` to change model mappings.

## Status

- **59/59 tests passing** (Python SDK)
- **Dashboard**: Production-ready with real-time telemetry
- **LLM Integration**: Google Gemini fully wired, Codex CLI optional
- **Agent Loop**: Full perceive→think→act→learn cycle operational
- **Z3 Verification**: Formal safety gate on reasoning steps
- **Memory**: Working memory + heuristic entity extraction (Neo4j persistence ready)
- **Evolution**: MAP-Elites topology search + LLM-driven code mutation
