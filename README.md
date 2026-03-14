<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Agent Development Kit with Cognitive Routing, Guardrails, and Real-Time Dashboard</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-yellow?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/tests-1503%20passed-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

YGN-SAGE is a research prototype Agent Development Kit that combines **cognitive routing (S1/S2/S3)**, **multi-provider model selection** (7 providers), **composable guardrails**, and a **real-time dashboard**. It uses a Rust data plane with a Python SDK.

## Features

- **S1/S2/S3 cognitive routing** — adaptive 5-stage routing (structural → kNN embeddings → BERT ONNX → entropy → cascade) with Rust ContextualBandit + telemetry calibration. kNN on arctic-embed-m (arXiv 2505.12601): 52% → 92% accuracy on 50 GT tasks, Rust-native SIMD dot product
- **Multi-provider** — 7 providers auto-discovered at boot (Google, OpenAI, xAI, DeepSeek, MiniMax, Kimi, Codex CLI)
- **Composable guardrails** — cost limits, output validation, schema validation, Z3 bounds checking at input/runtime/output
- **4-tier memory** — working memory (Rust Arrow), episodic (SQLite), semantic (entity graph), ExoCortex (Google File Search)
- **Tool Security** — Rust ToolExecutor: tree-sitter AST validation (23 blocked modules + 11 blocked calls) + Wasm WASI sandbox (deny-by-default) + subprocess fallback with timeout
- **Sandbox** — Wasm (wasmtime v36 LTS) Component Model sandbox with WASI deny-by-default capabilities
- **Dashboard** — built-in FastAPI + WebSocket real-time event viewer with task queue
- **Benchmarks** — EvalPlus HumanEval+ (164), MBPP+ (378), ablation study, routing quality, official evaluation protocol with error logging

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
AdaptiveRouter --- 5-stage routing (structural → kNN → BERT → entropy → cascade)
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

## Benchmark Results

### Framework Value (Ablation Study)

YGN-SAGE adds **+15 percentage points** over a bare LLM baseline on coding tasks (N=20, paired A/B test):

| Configuration | pass@1 | Delta |
|--------------|--------|-------|
| Full framework | 100% | — |
| Bare LLM (no framework) | 85% | -15pp |

Routing contributes +5pp. Memory, AVR, and guardrails show no isolated delta on single-turn code tasks — re-run at N=100 with statistical tests pending. Non-code evaluation (reasoning, multi-turn, research) in progress.

### Code Generation (EvalPlus)

| Benchmark | Score | Model |
|-----------|-------|-------|
| HumanEval+ pass@1 | **84.1%** (138/164) | Gemini 2.5 Flash |
| MBPP+ pass@1 | **75.1%** (284/378) | Gemini 2.5 Flash |

> **Note:** These are absolute scores using a budget-tier model. Cross-model comparisons (e.g., vs GPT-4o, O1) are not meaningful — different model tiers, different cost profiles. The framework's value is the +15pp delta over the same model without it.

### Routing

| Method | Accuracy (50 GT tasks) | Notes |
|--------|----------------------|-------|
| kNN on arctic-embed-m (arXiv 2505.12601) | **92%** (46/50) | Replaces keyword heuristic |
| Keyword heuristic baseline | 52% (26/50) | Non-circular, human-labeled |

### Tests

| Suite | Result |
|-------|--------|
| Python unit tests | **1216 passed** (115 skipped) |
| Rust unit tests | **235+ passed** |

```bash
# Run benchmarks
python -m sage.bench --type evalplus --dataset humaneval   # EvalPlus HumanEval+ (requires GOOGLE_API_KEY)
python -m sage.bench --type routing                        # Routing self-consistency (instant)
python -m sage.bench --type ablation --limit 20            # Ablation study (6 configs)

# Official evaluation protocol (with full error logging)
python -m sage.bench.eval_protocol --suite humaneval --limit 20 -v   # HumanEval+ with error capture
python -m sage.bench.eval_protocol --suite mbpp --limit 20 -v       # MBPP+ with error capture
python -m sage.bench.eval_protocol --replay docs/benchmarks/errors.jsonl  # Post-mortem analysis

python tests/e2e_proof.py                                  # E2E proof (requires GOOGLE_API_KEY)
```

## Run Tests

```bash
cd sage-python && python -m pytest tests/ -v    # 1216 passed, 115 skipped
cd sage-core && cargo test --no-default-features --features smt --lib  # 235+ tests
cd sage-discover && python -m pytest tests/ -v   # 52 passed
# Integration tests: sage-python/tests/integration/ (50 tests, no mocks)
```

## Project Structure

```
YGN-SAGE/
|-- sage-core/           # Rust core (ToolExecutor, Arrow memory, Wasm sandbox)
|-- sage-python/         # Python SDK
|   +-- src/sage/
|       |-- agents/      # Sequential, Parallel, Loop, Handoff composition
|       |-- bench/       # EvalPlus HumanEval+/MBPP+, routing, ablation, evaluation protocol
|       |-- events/      # EventBus (central nervous system)
|       |-- guardrails/  # Cost, Output, Schema, Z3 formal guardrails
|       |-- llm/         # Google Gemini + Codex CLI providers
|       |-- memory/      # 4-tier: STM, Episodic (SQLite), Semantic, ExoCortex
|       |-- strategy/    # S1/S2/S3 metacognitive routing + CGRS self-braking
|       |-- tools/       # Dynamic tool creation + 9 memory tools
|       |-- topology/    # MAP-Elites + CMA-ME + MCTS + LLM synthesis + Z3 PRM
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

- **EvalPlus HumanEval+ 84.1%** pass@1 (138/164) with budget Gemini 2.5 Flash — official 80x harder tests
- **Routing 100%** self-consistency on 30 deterministic tasks (heuristic, not learned)
- **1216 tests passed** (Python) + 235+ Rust + 52 Discover + 50 integration tests (no mocks)
- **CI/CD**: GitHub Actions (5 jobs: Rust, Rust features, Python, Discover, **Windows**)
- **Dashboard**: functional, real-time via WebSocket (First-Message auth pattern), task queue (up to 10)
- **Cognitive Routing**: S1/S2/S3 adaptive 5-stage routing (structural → kNN embeddings on arctic-embed-m ONNX 768-dim → BERT ONNX → entropy → cascade). kNN routing: 92% accuracy (arXiv 2505.12601)
- **Memory**: 4 tiers wired end-to-end — Tier 0 Working Memory (Rust Arrow + S-MMU), Tier 1 Episodic (SQLite), Tier 2 Semantic (deque + lazy eviction), Tier 3 ExoCortex (Google File Search)
- **Embeddings**: RustEmbedder ONNX with auto-discovery, 3-tier fallback + hash fallback warning
- **Guardrails**: wired at 3 points (input/runtime/output), cost + output + schema + Z3 bounds
- **Sandbox**: Wasm (wasmtime v36 LTS) + WASI deny-by-default + Rust ToolExecutor (tree-sitter validator + subprocess with kill-on-drop), wired to S2 AVR path
- **Benchmarks**: EvalPlus HumanEval+ (164), MBPP+ (378), ablation study, routing self-consistency (30 tasks), E2E proof
- **Composition**: Sequential, Parallel, Loop, Handoff patterns
- **Security**: thread-safe ModelRegistry, bounded messages (MAX_MESSAGES=40), EventBus timeout, OnceLock ORT resolution

## License

MIT License. (c) 2026 Yann Abadie. See [LICENSE](LICENSE).
