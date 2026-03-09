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
- `boot.py` - Boot sequence, wires all pillars + EventBus + GuardrailPipeline into `AgentSystem`. Auto-discovers models via registry.refresh(), populates CapabilityMatrix, logs per-provider summary
- `agent_loop.py` - Structured perceive->think->act->learn runtime with SLF-based S2 AVR (syntax-first + stagnation detection), Z3 S3 prompts, guardrails (input/runtime/output), CRAG-gated memory injection, code-task detection, AgentEvent schema
- `agent_pool.py` - Dynamic sub-agent pool (create/run/ensemble)
- `agents/sequential.py` - SequentialAgent: chain agents in series
- `agents/parallel.py` - ParallelAgent: run agents concurrently with aggregator
- `agents/loop_agent.py` - LoopAgent: iterate until exit condition
- `agents/handoff.py` - Handoff: transfer control to specialist agent
- `events/bus.py` - EventBus: in-proc event system (emit/subscribe/stream/query)
- `guardrails/base.py` - GuardrailResult, Guardrail, GuardrailPipeline
- `guardrails/builtin.py` - CostGuardrail, OutputGuardrail (default for text), SchemaGuardrail (for JSON mode)
- `bench/runner.py` - BenchmarkRunner, BenchReport, TaskResult
- `bench/humaneval.py` - HumanEval benchmark (164 problems, pass@1)
- `bench/routing.py` - Routing accuracy benchmark (30 labeled tasks)
- `bench/routing_quality.py` - Routing quality benchmark (45 labeled tasks) for ComplexityRouter and AdaptiveRouter ground truth evaluation
- `bench/routing_downstream.py` - DownstreamEvaluator: tier precision, escalation rate, routing P50/P99 latency, quality tracking
- `llm/router.py` - Model Router with 7 tiers, data-driven lookup from TOML + env vars
- `llm/config_loader.py` - TOML config loader + env var resolution (SAGE_MODEL_<TIER>)
- `llm/google.py` - Google Gemini provider + File Search grounding (google_search/file_search mutually exclusive)
- `llm/codex.py` - OpenAI Codex CLI provider (+ Google fallback)
- `orchestrator.py` - CognitiveOrchestrator: primary routing engine with cascade fallback (FrugalGPT: 3 attempts across providers), ModelAgent class
- `providers/openai_compat.py` - OpenAI-compatible provider with per-provider quirk dispatch (DeepSeek, Kimi, Grok, MiniMax, OpenAI)
- `providers/capabilities.py` - Provider capability registry: auto-populated at boot from discovered providers, 7 known provider capability sets
- `strategy/metacognition.py` - ComplexityRouter (ex-MetacognitiveController): S1/S2/S3 tripartite routing via word-boundary regex (`\b`) heuristic + CGRS self-braking + speculative zone detection (0.35-0.55). Supports provider injection for LLM assessment (no vendor lock-in)
- `strategy/adaptive_router.py` - AdaptiveRouter: 4-stage learned routing (structural → BERT ONNX → entropy probe → cascade). Duck-type compat with ComplexityRouter. Falls back to heuristic if sage_core[onnx] unavailable
- `strategy/training.py` - Training data export (JSONL) for BERT classifier retraining
- `topology/evo_topology.py` - MAP-Elites evolutionary topology search
- `topology/kg_rlvr.py` - Process Reward Model (Z3 DSL, safe AST evaluator — no eval())
- `resilience.py` - CircuitBreaker: per-subsystem failure tracking (max_failures=3, opens with WARNING)
- `evolution/engine.py` - Evolution engine with DGM context injection (5 SAMPO actions)
- `evolution/llm_mutator.py` - LLM-driven code mutation with DGM Directive prompt section
- `memory/memory_agent.py` - Autonomous entity extraction (heuristic or LLM), wired to LEARN phase. Supports provider injection (no vendor lock-in), falls back to GoogleProvider if none injected
- `memory/compressor.py` - MEM1 per-step internal state + pressure-triggered compression + S-MMU write (compact_to_arrow_with_meta with keywords/embedding/summary)
- `memory/embedder.py` - Embedder adapter: 3-tier fallback (RustEmbedder ONNX > sentence-transformers > hash, 384-dim) for S-MMU semantic edges. `_ensure_ort_dylib_path()` auto-discovers onnxruntime DLL from pip package.
- `memory/smmu_context.py` - S-MMU context retrieval: queries multi-view graph (BFS, configurable weights), returns formatted context for THINK phase injection
- `memory/episodic.py` - SQLite-backed episodic store (cross-session persistence) with in-memory fallback
- `memory/semantic.py` - Entity-relation graph built by MemoryAgent, with SQLite persistence (`~/.sage/semantic.db`)
- `memory/remote_rag.py` - ExoCortex (Google GenAI File Search API), implements `KnowledgeStore` protocol, auto-configured with DEFAULT_STORE. Configurable model via SAGE_EXOCORTEX_MODEL env var
- `memory/relevance_gate.py` - CRAG-style keyword overlap gate: scores context vs task, threshold=0.3, blocks irrelevant memory injection
- `memory/rag_backend.py` - `KnowledgeStore` protocol: pluggable RAG backend interface (search/ingest/store_name)
- `memory/causal.py` - CausalMemory: entity-relation graph with directed causal edges + temporal ordering + SQLite persistence (optional `db_path`)
- `memory/write_gate.py` - WriteGate: confidence-based write gating with abstention tracking
- `tools/memory_tools.py` - 7 AgeMem tools (3 STM + 4 LTM) exposed to agent
- `tools/exocortex_tools.py` - `search_exocortex` + `refresh_knowledge` agent tools
- `contracts/task_node.py` - TaskNode IR: typed I/O schemas, capabilities, security labels, budgets
- `contracts/verification.py` - VFResult, pre_check, post_check, run_verification
- `contracts/dag.py` - TaskDAG: Kahn's topo sort, cycle detection, IO validation, ready_nodes
- `contracts/z3_verify.py` - Z3 SMT: provider assignment via genuine Z3 SAT (`z3.PbEq` exactly-one). capability_coverage, budget_feasibility, type_compatibility are Python-native (~2000x faster)
- `contracts/policy.py` - PolicyVerifier: info-flow labels, budget, fan-in/fan-out limits
- `contracts/executor.py` - DAGExecutor: topo execution with VF pre/post checks + policy gate
- `contracts/planner.py` - TaskPlanner: Plan-and-Act decomposition into verified TaskDAG
- `contracts/repair.py` - RepairLoop: counterexample-guided retry with hard fences (CEGAR)
- `contracts/cost_tracker.py` - CostTracker: cumulative per-node cost accounting with budget cap
- `routing/dynamic.py` - DynamicRouter: capability-constrained model selection with feedback

### Key Rust Modules (sage-core/src/)
- `memory/mod.rs` - Arrow-backed working memory (SIMD/AVX-512) + S-MMU paging (wired: write via compressor, read via THINK phase)
- `memory/rag_cache.rs` - FIFO+TTL cache for File Search results (DashMap + atomic counters)
- `sandbox/ebpf.rs` - eBPF executor (solana_rbpf) + SnapBPF (CoW memory snapshots)
- `sandbox/wasm.rs` - Wasm sandbox (wasmtime v36 LTS). `WasmSandbox` PyClass + `WasiState` (deny-by-default). `execute_precompiled()` / `execute_precompiled_wasi()` for Windows (no cranelift), `execute()` with JIT on Linux CI. Standalone `execute_wasi_component()` / `execute_bare_component()` for ToolExecutor. Behind `sandbox` feature flag.
- `sandbox/validator.rs` — tree-sitter-python AST validation: 23 blocked modules + 11 blocked calls. Error-tolerant (partial trees on broken code). Behind `tool-executor` feature flag.
- `sandbox/subprocess.rs` — Subprocess executor with tokio timeout + kill_on_drop. Writes code to temp file, feeds args via stdin. Behind `tool-executor` feature flag.
- `sandbox/tool_executor.rs` — `ToolExecutor` PyO3 class: combines validator + Wasm WASI sandbox + subprocess fallback. `validate()`, `validate_and_execute()`, `execute_raw()`, `load_precompiled_component()`, `load_component()`, `has_wasm()`, `has_wasi()`. Execution priority: Wasm WASI → bare Wasm → subprocess. Releases GIL via `py.allow_threads()`. Behind `tool-executor` feature flag; Wasm paths behind `sandbox` feature.
- `memory/embedder.rs` - RustEmbedder: ONNX Runtime embedder (all-MiniLM-L6-v2, 384-dim, L2-normalized) via `ort` crate (`load-dynamic` feature). Behind `onnx` feature flag. PyO3 class: `embed(text)`, `embed_batch(texts)`. Auto-discovers `onnxruntime.dll` from pip package or `ORT_DYLIB_PATH` env var.
- `routing/features.rs` - StructuralFeatures: zero-cost keyword/structural feature extraction for Stage 0 pre-routing. Always compiled.
- `routing/router.rs` - AdaptiveRouter PyO3 class: Stage 0 (structural) + Stage 1 (BERT ONNX classifier). Behind `onnx` feature. Dynamic input discovery, 512-token truncation, binary/multi-class support.
- `z3/` - Z3 formal verification bindings

### Dashboard (ui/)
- `ui/app.py` - FastAPI backend: EventBus WebSocket push + REST API (HTTPBearer auth via `SAGE_DASHBOARD_TOKEN`)
- `ui/static/index.html` - Single-file dark-theme dashboard (Tailwind + Chart.js)
- WebSocket `/ws` pushes all AgentEvents in real-time. Uses First-Message auth pattern: client sends `{action:"auth", token:"..."}` as first message.
- Task queue: `asyncio.Queue(maxsize=10)` replaces single-task slot. New `/api/tasks` endpoint for queue status.
- Sections: Routing S1/S2/S3, Response, Memory 4-tier, Guardrails, Events, Benchmarks

## Development Commands

### Python SDK
```bash
cd sage-python
pip install -e ".[all,dev]"    # Install in dev mode with all providers
python -m pytest tests/ -v     # Run tests (1036 passed, 91 skipped)
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
cargo test --features onnx     # Run all Rust tests (57 passing: 30 lib + 27 integration)
cargo clippy                   # Lint Rust code
maturin develop                # Build + install Python bindings
maturin develop --features onnx  # Build with ONNX embedder support (auto-discovers DLL)
maturin develop --features tool-executor  # Build with ToolExecutor (tree-sitter + subprocess)
maturin develop --features sandbox,tool-executor  # Build with ToolExecutor + Wasm WASI sandbox
```

### End-to-End Proof
```bash
python tests/e2e_proof.py      # Full-stack E2E: 25/25 tests, ~35s, real LLM, no mocks
                               # Tests: Rust core + Python components + Gemini LLM + benchmarks
                               # Report: docs/benchmarks/YYYY-MM-DD-e2e-proof.json
```

### Discovery Pipeline
```bash
cd sage-discover
pip install -e .               # Install sage-discover
python -m pytest tests/ -v     # Run tests (52 passed)
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
export SAGE_DASHBOARD_TOKEN="..."            # Optional: dashboard auth (no token = open dev mode)
# Codex CLI uses ChatGPT Pro account (codex login)
# ExoCortex auto-configured (DEFAULT_STORE hardcoded, no env var needed)
```

### Sandbox & Tool Security
The sandbox blocks host code execution by default. To allow local execution (e.g., for development):
```python
sandbox = Sandbox(allow_local=True)  # Required — default is False
```

**ToolExecutor security pipeline** (when `sage_core` compiled with `tool-executor` + `sandbox`):
1. **tree-sitter AST validation** — blocks 23 dangerous modules (os, sys, subprocess, etc.) + 11 dangerous calls (exec, eval, open, etc.)
2. **Wasm WASI sandbox** (if component loaded) — deny-by-default: NO filesystem, NO env vars, NO network, NO subprocess. Only stdout/stderr inherited
3. **Subprocess fallback** — timeout + kill-on-drop isolation

Python `create_python_tool` tries Rust ToolExecutor first, falls back to Python `sandbox_executor.py`.

**6 Phoenix Security Vectors blocked:**
- Filesystem read/write, env var access, network access, subprocess spawn, dangerous module import

### Model Config Resolution
Model IDs resolved in order: env var `SAGE_MODEL_<TIER>` > `config/models.toml` > hardcoded defaults.
TOML searched in: `cwd/config/`, `sage-python/config/` (package), `~/.sage/`.

## Memory System (4 Tiers)
- **Tier 0 — Working Memory (STM)**: Rust Arrow buffer. MEM1 internal state every step. Pressure-triggered compression. Falls back to Python mock with warning if `sage_core` not installed.
- **S-MMU (wired)**: Write path: compressor calls `compact_to_arrow_with_meta()` with keywords + embedding (via `Embedder`) + dynamic summary. `register_chunk()` uses bounded recency scan (last 128 chunks, `MAX_SEMANTIC_NEIGHBORS = 128`). Read path: `retrieve_smmu_context()` queries the multi-view S-MMU graph during THINK phase and injects top-k chunk summaries (via `get_chunk_summary()`) as a SYSTEM message. In mock mode, write runs but chunk count stays 0 so read returns "".
- **Embedder (3-tier fallback)**: RustEmbedder (ONNX via ort `load-dynamic`, native SIMD) > sentence-transformers (Python, in `[embeddings]` extra) > SHA-256 hash. Auto-detected at init. All 3 tiers work on Windows MSVC. Model: all-MiniLM-L6-v2 (384-dim). Download: `python sage-core/models/download_model.py` + `pip install onnxruntime`
- **Tier 1 — Episodic Memory**: SQLite-backed (`~/.sage/episodic.db`), cross-session persistent. CRUD + keyword search. Defaults to SQLite (was in-memory before audit fix).
- **Tier 2 — Semantic Memory**: In-memory entity-relation graph. MemoryAgent extracts entities in LEARN phase. `get_context_for(task)` injected before LLM calls.
- **Tier 3 — ExoCortex (Persistent RAG)**: Google GenAI File Search API. Auto-configured with `DEFAULT_STORE`. 500+ research sources. Active `search_exocortex` tool only (passive grounding removed — Sprint 3 evidence showed it adds latency without benefit for code tasks).
- **9 Agent Tools**: 7 AgeMem (3 STM + 4 LTM) + `search_exocortex` + `refresh_knowledge`

## Guardrails (3-layer)
- **Input** (PERCEIVE): checks task before LLM call. Blocks if severity="block".
- **Runtime** (ACT): checks code before sandbox execution. Best-effort.
- **Output** (LEARN): checks result before return. Cost + output validation (empty/too-long/refusal).
- Default pipeline uses `OutputGuardrail` for text output. `SchemaGuardrail` available for JSON mode.
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
- **Routing Accuracy**: 30 labeled tasks (10 S1 + 10 S2 + 10 S3). Measures ComplexityRouter precision.
- **Routing Quality**: 45 labeled tasks. Measures both ComplexityRouter and AdaptiveRouter against human-labeled ground truth.
- **Downstream Quality**: DownstreamEvaluator tracks tier precision, escalation rate (<20% target), routing P50/P99 latency (<50ms target).
- **Metrics per task**: pass_rate, avg_latency_ms, avg_cost_usd, routing_breakdown S1/S2/S3

## Evolution System
- **DGM Context**: SAMPO solver chooses 1 of 5 strategic actions. Context injected into LLM mutation prompt.
- **Self-modification**: Actions 2/3/4 modify engine hyperparameters (mutations_per_generation, clip_epsilon, filter_threshold)
- **SnapBPF**: Rust CoW memory snapshots for mutation rollback

## Z3 Formal Verification (S3)
- S3 system prompt teaches Z3 DSL: `assert bounds/loop/arithmetic/invariant`
- S2->S3 escalation when AVR budget exhausted
- `kg_rlvr.py` parses `<think>` blocks, scores each step via safe AST evaluator (no `eval()`) + Z3
- `z3_verify.py`: 3 of 4 checks are Python-native (capability_coverage, budget_feasibility, type_compatibility — ~2000x faster). Only `verify_provider_assignment()` uses Z3 SAT with `z3.PbEq` exactly-one constraint (was at-least-one via `z3.Or`)

## Resilience
- **CircuitBreaker** (`resilience.py`): per-subsystem failure tracking (max_failures=3)
- 6 breakers in agent_loop: semantic_memory, smmu_context, runtime_guardrails, episodic_store, entity_extraction, evolution_stats
- After 3 consecutive failures, circuit opens and skips calls with WARNING log
- `record_success()` resets the counter

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
- 7 LLM providers: Google, OpenAI, xAI (Grok), DeepSeek, MiniMax, Kimi, Codex CLI — auto-discovered at boot, cascade fallback (FrugalGPT)
- FastAPI + WebSocket + EventBus (dashboard)
- Z3 Solver 4.16 (formal verification, S3)
- aiosqlite (episodic + semantic memory persistence)
- Apache Arrow / PyArrow (zero-copy memory compaction)
- Wasm (wasmtime v36 LTS, Component Model, WASI p2 deny-by-default) + Docker (multi-tier sandboxing)
- tree-sitter 0.26 + tree-sitter-python 0.25 (AST-based code validation, `tool-executor` feature)
- process-wrap 9 (subprocess executor with tokio timeout + kill-on-drop, `tool-executor` feature)
- DashMap (Rust) -- lock-free FIFO+TTL RAG cache + CoW snapshots
- ort 2.0 (ONNX Runtime for Rust, `load-dynamic`, optional `onnx` feature) — native embeddings, works on Windows MSVC
- tokenizers 0.21 (HuggingFace tokenizer, optional `onnx` feature)
