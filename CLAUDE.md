# YGN-SAGE - CLAUDE.md

## Project Overview
YGN-SAGE (Yann's Generative Neural Self-Adaptive Generation Engine) is an Agent Development Kit
built on 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Architecture
- `sage-core/` - Rust orchestrator (PyO3 bindings to Python)
- `sage-python/` - Python SDK for building agents
- `sage-discover/` - Knowledge Pipeline (arXiv/S2/HF -> ExoCortex)
- `ui/` - Control Dashboard (FastAPI + WebSocket + single-file HTML)
- `docs/plans/` - Architecture designs and implementation plans
- `Researches/` - Research papers (AlphaEvolve, PSRO, etc.)
- `.github/workflows/ci.yml` - CI pipeline (Rust + Python sage + Python discover)

### Key Python Modules (sage-python/src/sage/)
- `boot.py` - Boot sequence, wires all pillars + EventBus + GuardrailPipeline into `AgentSystem`
- `agent_loop.py` - Structured perceive->think->act->learn runtime with S2 AVR loop, Z3 S3 prompts, guardrails (input/runtime/output), semantic context injection, AgentEvent schema
- `agent_pool.py` - Dynamic sub-agent pool (create/run/ensemble)
- `agents/sequential.py` - SequentialAgent: chain agents in series
- `agents/parallel.py` - ParallelAgent: run agents concurrently with aggregator
- `agents/loop_agent.py` - LoopAgent: iterate until exit condition
- `agents/handoff.py` - Handoff: transfer control to specialist agent
- `events/bus.py` - EventBus: in-proc event system (emit/subscribe/stream/query)
- `guardrails/base.py` - GuardrailResult, Guardrail, GuardrailPipeline
- `guardrails/builtin.py` - CostGuardrail, SchemaGuardrail
- `bench/runner.py` - BenchmarkRunner, BenchReport, TaskResult
- `bench/humaneval.py` - HumanEval benchmark (164 problems, pass@1)
- `bench/routing.py` - Routing accuracy benchmark (30 labeled tasks)
- `llm/router.py` - Model Router with 7 tiers, data-driven lookup from TOML + env vars
- `llm/config_loader.py` - TOML config loader + env var resolution (SAGE_MODEL_<TIER>)
- `llm/google.py` - Google Gemini provider + File Search grounding (google_search/file_search mutually exclusive)
- `llm/codex.py` - OpenAI Codex CLI provider (+ Google fallback)
- `strategy/metacognition.py` - Stanovich S1/S2/S3 tripartite routing + CGRS self-braking
- `topology/evo_topology.py` - MAP-Elites evolutionary topology search
- `topology/kg_rlvr.py` - Process Reward Model (Z3 DSL)
- `evolution/engine.py` - Evolution engine with DGM context injection (5 SAMPO actions)
- `evolution/llm_mutator.py` - LLM-driven code mutation with DGM Directive prompt section
- `memory/memory_agent.py` - Autonomous entity extraction (heuristic or LLM), wired to LEARN phase
- `memory/compressor.py` - MEM1 per-step internal state + pressure-triggered compression
- `memory/episodic.py` - SQLite-backed episodic store (cross-session persistence) with in-memory fallback
- `memory/semantic.py` - In-memory entity-relation graph built by MemoryAgent
- `memory/remote_rag.py` - ExoCortex (Google GenAI File Search API), auto-configured with DEFAULT_STORE
- `memory/causal.py` - CausalMemory: entity-relation graph with directed causal edges + temporal ordering
- `memory/write_gate.py` - WriteGate: confidence-based write gating with abstention tracking
- `tools/memory_tools.py` - 7 AgeMem tools (3 STM + 4 LTM) exposed to agent
- `tools/exocortex_tools.py` - `search_exocortex` + `refresh_knowledge` agent tools
- `contracts/task_node.py` - TaskNode IR: typed I/O schemas, capabilities, security labels, budgets
- `contracts/verification.py` - VFResult, pre_check, post_check, run_verification
- `contracts/dag.py` - TaskDAG: Kahn's topo sort, cycle detection, IO validation, ready_nodes
- `contracts/z3_verify.py` - Z3 SMT: capability coverage, budget feasibility, type compatibility
- `contracts/policy.py` - PolicyVerifier: info-flow labels, budget, fan-in/fan-out limits
- `contracts/executor.py` - DAGExecutor: topo execution with VF pre/post checks + policy gate
- `contracts/planner.py` - TaskPlanner: Plan-and-Act decomposition into verified TaskDAG
- `contracts/repair.py` - RepairLoop: counterexample-guided retry with hard fences (CEGAR)
- `contracts/cost_tracker.py` - CostTracker: cumulative per-node cost accounting with budget cap
- `routing/dynamic.py` - DynamicRouter: capability-constrained model selection with feedback

### Key Rust Modules (sage-core/src/)
- `memory/mod.rs` - Arrow-backed working memory (SIMD/AVX-512) + S-MMU paging
- `memory/rag_cache.rs` - FIFO+TTL cache for File Search results (DashMap + atomic counters)
- `sandbox/ebpf.rs` - eBPF executor (solana_rbpf) + SnapBPF (CoW memory snapshots)
- `sandbox/wasm.rs` - Wasm sandbox (wasmtime)
- `z3/` - Z3 formal verification bindings

### Dashboard (ui/)
- `ui/app.py` - FastAPI backend: EventBus WebSocket push + REST API
- `ui/static/index.html` - Single-file dark-theme dashboard (Tailwind + Chart.js)
- WebSocket `/ws` pushes all AgentEvents in real-time (replaces JSONL polling)
- Sections: Routing S1/S2/S3, Response, Memory 4-tier, Guardrails, Events, Benchmarks

## Development Commands

### Python SDK
```bash
cd sage-python
pip install -e ".[all,dev]"    # Install in dev mode with all providers
python -m pytest tests/ -v     # Run tests (596 passed, 1 skipped)
ruff check src/                 # Lint
mypy src/                       # Type check
```

### Benchmarks
```bash
cd sage-python
python -m sage.bench --type routing                    # Routing accuracy (instant, no API key)
python -m sage.bench --type humaneval --limit 20       # HumanEval smoke test
python -m sage.bench --type humaneval                  # Full HumanEval (164 problems)
```

### Dashboard
```bash
python ui/app.py                # Start dashboard on http://localhost:8000
```

### Rust Core
```bash
cd sage-core
cargo build                    # Build Rust core
cargo test --workspace         # Run Rust tests (38 passing)
cargo clippy                   # Lint Rust code
maturin develop                # Build + install Python bindings
```

### Discovery Pipeline
```bash
cd sage-discover
pip install -e .               # Install sage-discover
python -m pytest tests/ -v     # Run tests (45 passed)
python -m discover.pipeline --mode nightly -v  # Run nightly pipeline
```

## LLM Configuration

### Active Models (March 2026, verified working)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | `gpt-5.3-codex` | Codex CLI | Default, SOTA coding |
| codex_max | `gpt-5.2` | Codex CLI | Most powerful reasoning |
| reasoner | `gemini-3.1-pro-preview` | Google API | Complex evaluation |
| mutator | `gemini-3-flash-preview` | Google API | Code mutation |
| fast | `gemini-3.1-flash-lite-preview` | Google API | Low-latency |
| budget | `gemini-2.5-flash-lite` | Google API | Cheapest |
| fallback | `gemini-2.5-flash` | Google API | If 3.x unavailable |

### Required Environment Variables
```bash
export GOOGLE_API_KEY="..."                  # Required for Gemini models
export SAGE_MODEL_FAST="gemini-2.5-flash"    # Optional: override any tier model ID
# Codex CLI uses ChatGPT Pro account (codex login)
# ExoCortex auto-configured (DEFAULT_STORE hardcoded, no env var needed)
```

### Model Config Resolution
Model IDs resolved in order: env var `SAGE_MODEL_<TIER>` > `config/models.toml` > hardcoded defaults.
TOML searched in: `cwd/config/`, `sage-python/config/` (package), `~/.sage/`.

## Memory System (4 Tiers)
- **Tier 0 — Working Memory (STM)**: Rust Arrow buffer. MEM1 internal state every step. Pressure-triggered compression.
- **Tier 1 — Episodic Memory**: SQLite-backed (`~/.sage/episodic.db`), cross-session persistent. CRUD + keyword search.
- **Tier 2 — Semantic Memory**: In-memory entity-relation graph. MemoryAgent extracts entities in LEARN phase. `get_context_for(task)` injected before LLM calls.
- **Tier 3 — ExoCortex (Persistent RAG)**: Google GenAI File Search API. Auto-configured with `DEFAULT_STORE`. 500+ research sources. Passive grounding in `_think()` + active `search_exocortex` tool.
- **9 Agent Tools**: 7 AgeMem (3 STM + 4 LTM) + `search_exocortex` + `refresh_knowledge`

## Guardrails (3-layer)
- **Input** (PERCEIVE): checks task before LLM call. Blocks if severity="block".
- **Runtime** (ACT): checks code before sandbox execution. Best-effort.
- **Output** (LEARN): checks result before return. Cost + schema validation.
- Wired via `GuardrailPipeline` in boot.py. Events emitted on EventBus.

## Agent Composition
- **SequentialAgent**: chain agents in series (output feeds next input)
- **ParallelAgent**: run agents concurrently via asyncio.gather, pluggable aggregator
- **LoopAgent**: iterate until exit condition or max_iterations
- **Handoff**: transfer control to specialist with input_filter and on_handoff callback

## EventBus (Observability)
- Central in-proc event bus (`events/bus.py`). All components emit AgentEvents.
- `emit()`, `subscribe()`, `stream()` (async iterator for WebSocket), `query(phase, last_n)`
- Dashboard consumes via WebSocket `/ws`. JSONL export is optional subscriber.
- Event types: PERCEIVE, THINK, ACT, LEARN, ROUTING, GUARDRAIL_CHECK, GUARDRAIL_BLOCK, BENCH_RESULT, etc.

## Benchmarks
- **HumanEval**: 164 problems bundled as JSON. pass@1 with subprocess sandbox. CLI: `python -m sage.bench --type humaneval`
- **Routing Accuracy**: 30 labeled tasks (10 S1 + 10 S2 + 10 S3). Measures MetacognitiveController precision.
- **Metrics per task**: pass_rate, avg_latency_ms, avg_cost_usd, routing_breakdown S1/S2/S3

## Evolution System
- **DGM Context**: SAMPO solver chooses 1 of 5 strategic actions. Context injected into LLM mutation prompt.
- **Self-modification**: Actions 2/3/4 modify engine hyperparameters (mutations_per_generation, clip_epsilon, filter_threshold)
- **SnapBPF**: Rust CoW memory snapshots for mutation rollback

## Z3 Formal Verification (S3)
- S3 system prompt teaches Z3 DSL: `assert bounds/loop/arithmetic/invariant`
- S2->S3 escalation when AVR budget exhausted
- `kg_rlvr.py` parses `<think>` blocks, scores each step via regex + Z3

## ExoCortex
Auto-configured with `DEFAULT_STORE = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"`.
Resolution: explicit param > env var `SAGE_EXOCORTEX_STORE` > DEFAULT_STORE.

```python
from sage.memory.remote_rag import ExoCortex
exo = ExoCortex()  # Works automatically when GOOGLE_API_KEY is set
```

### Knowledge Pipeline (sage-discover)
```bash
python -m discover.pipeline --mode nightly           # Papers from yesterday
python -m discover.pipeline --mode on-demand --query "PSRO"  # Targeted search
python -m discover.pipeline --mode migrate           # Bootstrap from NotebookLM exports
```

## Tech Stack
- Rust 1.90+ (orchestrator, via PyO3) -- SnapBPF, RagCache, Arrow memory, eBPF
- Python 3.12+ (SDK, agents)
- OpenAI Codex CLI + gpt-5.3-codex (primary LLM)
- Google Gemini 3.x via `google-genai` SDK (secondary LLM, fallback, File Search)
- FastAPI + WebSocket + EventBus (dashboard)
- Z3 Solver 4.16 (formal verification, S3)
- aiosqlite (episodic memory persistence)
- Apache Arrow / PyArrow (zero-copy memory compaction)
- Wasm (wasmtime) + eBPF (solana_rbpf) + Docker (multi-tier sandboxing)
- DashMap (Rust) -- lock-free FIFO+TTL RAG cache + CoW snapshots
