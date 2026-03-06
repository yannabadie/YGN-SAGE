<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Agent Development Kit with Cognitive Routing, Formal Guardrails, and Real-Time Dashboard</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-307%20passed-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-proprietary-red?style=flat-square" alt="License">
</p>

---

YGN-SAGE is an Agent Development Kit that routes tasks through three cognitive systems (S1/S2/S3), validates outputs with Z3 formal proofs, and shows everything in a real-time dashboard. It combines a Rust execution core with a Python SDK.

## What Makes It Different

| Feature | Google ADK | OpenAI Agents | LangGraph | **YGN-SAGE** |
|---------|-----------|---------------|-----------|--------------|
| Routing | Static | LLM picks | Graph edges | **S1/S2/S3 cognitive + LLM assessment** |
| Guardrails | Instructions | Heuristic | Human-in-loop | **Z3 formal proofs, composable** |
| Memory | 1 tier | 1 tier | 1 tier | **4 tiers (STM + Episodic + Semantic + ExoCortex)** |
| Sandbox | None | None | None | **Wasm + eBPF + Docker** |
| Dashboard | Cloud console | Cloud traces | Cloud (paid) | **Built-in, real-time, free** |
| Benchmarks | External | None | External | **HumanEval + Routing built-in** |
| Composition | Seq/Par/Loop | Handoffs | StateGraph | **Seq/Par/Loop + Handoffs + Cognitive routing** |

## Quick Start

```bash
# Clone and install
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE/sage-python
pip install -e ".[all,dev]"

# Set your API key
export GOOGLE_API_KEY="your_key"

# Run the dashboard
cd .. && python ui/app.py
# Open http://localhost:8000
```

## How It Works

```
User Task
    |
    v
MetacognitiveController --- assesses complexity + uncertainty via LLM
    |
    +---> S1 (Simple)   --- Gemini Flash Lite, no validation, <1s
    +---> S2 (Code)     --- Gemini Flash/Pro, sandbox AVR loop, <5s
    +---> S3 (Formal)   --- Codex/Reasoner, Z3 formal proofs, <30s
    |
    v
AgentLoop: perceive -> think -> act -> learn
    |
    +---> Guardrails (input/runtime/output) --- Z3 bounds, cost limits
    +---> Memory (4 tiers) --- STM, SQLite episodic, semantic graph, ExoCortex RAG
    +---> Tools (dynamic) --- Python, Bash, search, sub-agents
    +---> Evolution --- DGM + SAMPO + MAP-Elites topology search
    |
    v
EventBus ---> Dashboard (WebSocket, real-time)
```

## Run Benchmarks

```bash
# Routing accuracy (no API key needed, instant)
python -m sage.bench --type routing

# HumanEval pass@1 (requires GOOGLE_API_KEY)
python -m sage.bench --type humaneval --limit 20
```

## Run Tests

```bash
cd sage-python && python -m pytest tests/ -v    # 307 passed
cd sage-core && cargo test --workspace          # 38 passed
cd sage-discover && python -m pytest tests/ -v  # 45 passed
```

## Project Structure

```
YGN-SAGE/
|-- sage-core/           # Rust core (eBPF, Z3, Arrow memory, RagCache)
|-- sage-python/         # Python SDK
|   +-- src/sage/
|       |-- agents/      # Sequential, Parallel, Loop, Handoff composition
|       |-- bench/       # HumanEval + Routing benchmarks
|       |-- events/      # EventBus (central nervous system)
|       |-- guardrails/  # Cost, Schema, Z3 formal guardrails
|       |-- llm/         # Google Gemini + Codex CLI providers
|       |-- memory/      # 4-tier: STM, Episodic (SQLite), Semantic, ExoCortex
|       |-- strategy/    # S1/S2/S3 metacognitive routing + CGRS self-braking
|       |-- tools/       # Dynamic tool creation + 9 memory tools
|       |-- topology/    # MAP-Elites + Z3 PRM validator
|       |-- evolution/   # LLM mutator + DGM/SAMPO solver
|       |-- agent_loop.py
|       +-- boot.py
|-- sage-discover/       # Knowledge pipeline (arXiv/S2/HF -> ExoCortex)
|-- ui/                  # Dashboard (FastAPI + single-file HTML)
+-- docs/                # ADR, designs, plans
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key for Gemini models |
| `SAGE_MODEL_<TIER>` | No | Override model per tier (e.g. `SAGE_MODEL_FAST=gemini-2.5-flash`) |

ExoCortex is auto-configured with the default research store. No setup needed.

Model IDs are resolved: **env var > `config/models.toml` > hardcoded defaults**.

## Use as a Library

```python
import asyncio
from sage.boot import boot_agent_system

async def main():
    system = boot_agent_system(llm_tier="fast")
    result = await system.run("Write a function to check if a number is prime.")
    print(result)

asyncio.run(main())
```

### Compose Agents

```python
from sage.agents import SequentialAgent, ParallelAgent, Handoff

# Chain agents
pipeline = SequentialAgent(name="pipeline", agents=[analyzer, coder, reviewer])

# Run in parallel
team = ParallelAgent(name="team", agents=[researcher, implementer])

# Handoff to specialist
handoff = Handoff(target=debugger, description="For debugging tasks")
```

### Add Guardrails

```python
from sage.guardrails import GuardrailPipeline, CostGuardrail, SchemaGuardrail

pipeline = GuardrailPipeline([
    CostGuardrail(max_usd=0.50),
    SchemaGuardrail(required_fields=["answer"]),
])
```

## Status (March 2026)

- **307 tests passing** (Python) + 38 Rust = 345 total
- **CI/CD**: GitHub Actions (3 jobs)
- **Dashboard**: Fully wired, real-time via WebSocket
- **Cognitive Routing**: S1/S2/S3 with AVR sandbox loop and Z3 proofs
- **Memory**: 4 tiers, SQLite persistence, semantic entity graph, ExoCortex (500+ sources)
- **Guardrails**: Wired at 3 points (input/runtime/output)
- **Benchmarks**: HumanEval (164 problems) + Routing Accuracy (30/30)
- **Composition**: Sequential, Parallel, Loop, Handoff patterns
- **Evolution**: DGM + SAMPO + MAP-Elites + SnapBPF

## License

Proprietary. All rights reserved. (c) 2026 Yann Abadie.
