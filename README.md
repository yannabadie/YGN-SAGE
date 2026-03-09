<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Agent Development Kit with Cognitive Routing, Guardrails, and Real-Time Dashboard</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-yellow?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/tests-895%20passed-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

YGN-SAGE is a research prototype Agent Development Kit that combines **cognitive routing (S1/S2/S3)**, **multi-provider model selection** (7 providers), **composable guardrails**, and a **real-time dashboard**. It uses a Rust data plane with a Python SDK.

## Features

- **S1/S2/S3 cognitive routing** — word-boundary regex heuristic routes tasks to appropriate model tiers
- **Multi-provider** — 7 providers auto-discovered at boot (Google, OpenAI, xAI, DeepSeek, MiniMax, Kimi, Codex CLI)
- **Composable guardrails** — cost limits, output validation, schema validation, Z3 bounds checking at input/runtime/output
- **4-tier memory** — working memory (Rust Arrow), episodic (SQLite), semantic (entity graph), ExoCortex (Google File Search)
- **Sandbox** — Wasm (wasmtime) execution sandbox (experimental)
- **Dashboard** — built-in FastAPI + WebSocket real-time event viewer with task queue
- **Benchmarks** — HumanEval (164 problems) + routing self-consistency test built-in

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component status and known limitations.

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
ComplexityRouter --- heuristic complexity + uncertainty assessment
    |
    +---> S1 (Simple)   --- fast model, no validation
    +---> S2 (Code)     --- mid-tier model, sandbox AVR loop
    +---> S3 (Formal)   --- reasoner model, Z3 bounds checking
    |
    v
AgentLoop: perceive -> think -> act -> learn
    |
    +---> Guardrails (input/runtime/output) --- Z3 bounds, cost limits
    +---> Memory (4 tiers) --- STM, SQLite episodic, semantic graph, ExoCortex RAG
    +---> Tools (dynamic) --- Python, Bash, search, sub-agents
    +---> Evolution --- SAMPO + MAP-Elites topology search
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
cd sage-python && python -m pytest tests/ -v    # 895 passed, 1 skipped
cd sage-core && cargo test --workspace          # 7 passed (+5 ONNX feature-gated)
cd sage-discover && python -m pytest tests/ -v  # 52 passed
```

## Project Structure

```
YGN-SAGE/
|-- sage-core/           # Rust core (Z3, Arrow memory, RagCache)
|-- sage-python/         # Python SDK
|   +-- src/sage/
|       |-- agents/      # Sequential, Parallel, Loop, Handoff composition
|       |-- bench/       # HumanEval + Routing benchmarks
|       |-- events/      # EventBus (central nervous system)
|       |-- guardrails/  # Cost, Output, Schema, Z3 formal guardrails
|       |-- llm/         # Google Gemini + Codex CLI providers
|       |-- memory/      # 4-tier: STM, Episodic (SQLite), Semantic, ExoCortex
|       |-- strategy/    # S1/S2/S3 metacognitive routing + CGRS self-braking
|       |-- tools/       # Dynamic tool creation + 9 memory tools
|       |-- topology/    # MAP-Elites + Z3 PRM validator
|       |-- evolution/   # LLM mutator + SAMPO solver
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
from sage.guardrails import GuardrailPipeline, CostGuardrail, OutputGuardrail

pipeline = GuardrailPipeline([
    CostGuardrail(max_usd=0.50),
    OutputGuardrail(),  # Warns on empty, too-long, or refusal outputs
])
# For JSON mode, use SchemaGuardrail(required_fields=["answer"]) instead
```

## Status (March 2026)

> **Research prototype.** Not production-ready. See [ARCHITECTURE.md](ARCHITECTURE.md) for honest component status.

- **895 tests passed** (Python) + 7 Rust + 52 Discover
- **CI/CD**: GitHub Actions (3 parallel jobs)
- **Dashboard**: functional, real-time via WebSocket (First-Message auth pattern), task queue (up to 10)
- **Cognitive Routing**: S1/S2/S3 heuristic routing, self-consistency benchmark (30/30)
- **Memory**: 4 tiers — Tier 0 Working Memory: per-session Arrow buffer (persisted via compressor to Tier 1); Tier 1 Episodic: SQLite persistent (`~/.sage/episodic.db`); Tier 2 Semantic: SQLite persistent (`~/.sage/semantic.db`); Tier 3 ExoCortex: cloud-hosted persistent (Google File Search via KnowledgeStore protocol)
- **Embeddings**: 3-tier fallback (RustEmbedder ONNX > sentence-transformers > hash), all working on Windows
- **Guardrails**: wired at 3 points (input/runtime/output), cost + output + schema + Z3 bounds
- **Sandbox**: Wasm (wasmtime v36 LTS), host execution blocked by default
- **Benchmarks**: HumanEval 164 built-in, routing self-consistency test
- **Composition**: Sequential, Parallel, Loop, Handoff patterns
- **Evolution**: scaffolding present, not validated against baselines

## License

MIT License. (c) 2026 Yann Abadie. See [LICENSE](LICENSE).
