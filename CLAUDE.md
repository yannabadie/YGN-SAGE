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
- `boot.py` - Boot sequence, wires all pillars + EventBus + GuardrailPipeline + TopologyEngine + ContextualBandit into `AgentSystem`. Auto-discovers models via registry.refresh(), populates CapabilityMatrix, logs per-provider summary. Phase 6: topology generation on every task, outcome recording feeds S-MMU + MAP-Elites + bandit learning loop
- `agent_loop.py` - Structured perceive->think->act->learn runtime with SLF-based S2 AVR (syntax-first + stagnation detection), Z3 S3 prompts, guardrails (input/runtime/output), CRAG-gated memory injection, code-task detection, AgentEvent schema
- `agent_pool.py` - Dynamic sub-agent pool (create/run/ensemble)
- `agents/sequential.py` - SequentialAgent: chain agents in series
- `agents/parallel.py` - ParallelAgent: run agents concurrently with aggregator
- `agents/loop_agent.py` - LoopAgent: iterate until exit condition
- `agents/handoff.py` - Handoff: transfer control to specialist agent
- `events/bus.py` - EventBus: in-proc event system (emit/subscribe/stream/query)
- `guardrails/base.py` - GuardrailResult, Guardrail, GuardrailPipeline
- `guardrails/builtin.py` - CostGuardrail, OutputGuardrail (default for text), SchemaGuardrail (for JSON mode)
- `quality_estimator.py` - QualityEstimator: 5-signal quality scoring (non-empty, length adequacy, code presence, error absence, AVR convergence). Replaces `len>10` heuristic
- `bench/runner.py` - BenchmarkRunner, BenchReport, TaskResult
- `bench/humaneval.py` - HumanEval benchmark (164 problems, pass@1)
- `bench/routing.py` - Routing accuracy benchmark (30 labeled tasks)
- `bench/routing_quality.py` - Routing quality benchmark (45 labeled tasks) for ComplexityRouter and AdaptiveRouter ground truth evaluation
- `bench/routing_downstream.py` - DownstreamEvaluator: tier precision, escalation rate, routing P50/P99 latency, quality tracking
- `bench/evalplus_bench.py` - EvalPlus HumanEval+/MBPP+ adapter: 80x harder tests, subprocess evaluator (Windows-compatible)
- `bench/ablation.py` - 6-config ablation framework: full, baseline, no-memory, no-avr, no-routing, no-guardrails
- `bench/eval_protocol.py` - Official evaluation protocol: real-condition benchmarks with full error logging (traceback, phase, model, routing), JSONL error logs, JSON reports, post-mortem replay
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
- `topology/kg_rlvr.py` - Process Reward Model (Z3 DSL, safe AST evaluator — no eval()). All SMT paths (verify_invariant, verify_arithmetic, prove_memory_safety, check_loop_bound, score_with_z3) use Rust OxiZ first with z3-solver fallback. verify_invariant uses Rust verify_invariant_with_feedback() for clause-level diagnostic feedback stored in _last_invariant_feedback
- `topology/llm_caller.py` - LLM topology synthesis (Path 3): role prompt → structure prompt → Rust TopologySynthesizer. Completes the 5-path strategy in DynamicTopologyEngine
- `resilience.py` - CircuitBreaker: per-subsystem failure tracking (max_failures=3, opens with WARNING)
- `evolution/engine.py` - Evolution engine with DGM context injection (5 SAMPO actions)
- `evolution/llm_mutator.py` - LLM-driven code mutation with DGM Directive prompt section
- `memory/memory_agent.py` - Autonomous entity extraction (heuristic or LLM), wired to LEARN phase. Supports provider injection (no vendor lock-in), falls back to GoogleProvider if none injected
- `memory/compressor.py` - MEM1 per-step internal state + pressure-triggered compression + S-MMU write (compact_to_arrow_with_meta with keywords/embedding/summary)
- `memory/embedder.py` - Embedder adapter: 3-tier fallback (RustEmbedder ONNX > sentence-transformers > hash, 768-dim) for S-MMU semantic edges. `_ensure_ort_dylib_path()` auto-discovers onnxruntime DLL from pip package.
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
- `routing/shadow.py` - ShadowRouter: dual Rust/Python routing with JSONL divergence traces. 2-tier Phase 5 gate: soft (500 traces, <10% divergence), hard (1000 traces, <5% divergence)

### Key Rust Modules (sage-core/src/)
- `memory/mod.rs` - Arrow-backed working memory (SIMD/AVX-512) + S-MMU paging (wired: write via compressor, read via THINK phase)
- `memory/rag_cache.rs` - FIFO+TTL cache for File Search results (DashMap + atomic counters)
- `sandbox/ebpf.rs` - eBPF executor (solana_rbpf) + SnapBPF (CoW memory snapshots)
- `sandbox/wasm.rs` - Wasm sandbox (wasmtime v36 LTS). `WasmSandbox` PyClass + `WasiState` (deny-by-default). `execute_precompiled()` / `execute_precompiled_wasi()` for Windows (no cranelift), `execute()` with JIT on Linux CI. Standalone `execute_wasi_component()` / `execute_bare_component()` for ToolExecutor. Behind `sandbox` feature flag.
- `sandbox/validator.rs` — tree-sitter-python AST validation: 23 blocked modules + 11 blocked calls. Error-tolerant (partial trees on broken code). Behind `tool-executor` feature flag.
- `sandbox/subprocess.rs` — Subprocess executor with tokio timeout + kill_on_drop. Writes code to temp file, feeds args via stdin. Behind `tool-executor` feature flag.
- `sandbox/tool_executor.rs` — `ToolExecutor` PyO3 class: combines validator + Wasm WASI sandbox + subprocess fallback. `validate()`, `validate_and_execute()`, `execute_raw()`, `load_precompiled_component()`, `load_component()`, `has_wasm()`, `has_wasi()`. Execution priority: Wasm WASI → bare Wasm → subprocess. Releases GIL via `py.allow_threads()`. Behind `tool-executor` feature flag; Wasm paths behind `sandbox` feature.
- `memory/embedder.rs` - RustEmbedder: ONNX Runtime embedder (snowflake-arctic-embed-m, 768-dim, L2-normalized) via `ort` crate (`load-dynamic` feature). Behind `onnx` feature flag. PyO3 class: `embed(text)`, `embed_batch(texts)`. Auto-discovers `onnxruntime.dll` from pip package or `ORT_DYLIB_PATH` env var. Robust token_type_ids handling (auto-detects from model inputs).
- `routing/features.rs` - StructuralFeatures: zero-cost keyword/structural feature extraction for Stage 0 pre-routing. Always compiled.
- `routing/router.rs` - AdaptiveRouter PyO3 class: Stage 0 (structural) + Stage 1 (BERT ONNX classifier). Behind `onnx` feature. Dynamic input discovery, 512-token truncation, binary/multi-class support. Rust-native shadow trace collection (JSONL), `retrain_thresholds()` logistic regression on feedback, `flush_shadow_traces(path)` PyO3 method.
- `routing/model_card.rs` - ModelCard + CognitiveSystem (S1/S2/S3) + domain_scores (HashMap<String, f32>) + safety_rating + TOML parsing with `[models.domain_scores]` sub-tables. PyO3 class with `domain_score(domain)` method.
- `routing/model_registry.rs` - ModelRegistry: TOML-loaded model catalog with system-based selection, telemetry calibration (quality + latency P95 via VecDeque ring buffer), `select_best_for_domain(domain, budget)` for domain-aware model selection. PyO3 class.
- `routing/system_router.rs` - SystemRouter: cognitive system decision engine (hard constraints → structural scoring → domain hint → bandit/budget selection). `record_outcome()` updates both bandit AND registry telemetry via decision→model mapping. PyO3 class with RoutingDecision + RoutingConstraints (includes `domain_hint`).
- `routing/bandit.rs` - ContextualBandit: per-arm Beta/Gamma posteriors, Thompson sampling, Pareto front. PyO3 class. SQLite persistence behind `cognitive` feature.
- `routing/smmu_bridge.rs` - S-MMU bridge for routing: stores routing decisions as S-MMU chunks for similarity retrieval.
- `topology/topology_graph.rs` - TopologyGraph: unified IR wrapping petgraph::DiGraph with typed nodes (roles, capabilities, budgets) and three-flow edges (Control, Message, State). PyO3 class.
- `topology/templates.rs` - 8 topology templates (Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming). PyTemplateStore PyO3 class.
- `topology/verifier.rs` - HybridVerifier: 6 structural + 4 semantic checks + LTL integration (safety→errors, liveness→warnings), all O(V+E). PyO3 class.
- `topology/smmu_bridge.rs` - TopologySmmuBridge: stores topology outcomes in S-MMU, retrieves similar topologies, injects bandit priors.
- `topology/map_elites.rs` - MapElitesArchive: quality-diversity archive with 4-dim BehaviorDescriptor (108 cells), Pareto dominance. SQLite persistence behind `cognitive` feature.
- `topology/cma_me.rs` - CmaEmitter: CMA-ME (Covariance Matrix Adaptation MAP-Elites) for directional search on continuous topology parameters. Diagonal covariance, elite-weighted mean/variance update. Integrated into DynamicTopologyEngine.evolve() (50% random / 50% CMA-sampled mutations).
- `topology/mutations.rs` - 7 mutation operators (add_node, remove_node, swap_model, rewire_edge, split_node, merge_nodes, mutate_prompt). Validation via HybridVerifier.
- `topology/llm_synthesis.rs` - TopologySynthesizer: 3-stage LLM pipeline (Role Assignment → Structure Design → Validation). Rate-limited.
- `topology/executor.rs` - TopologyExecutor: dual-mode scheduling — Static (Kahn's toposort) for acyclic, Dynamic (gate-based readiness) for cyclic topologies. PyO3 class.
- `topology/mcts.rs` - MctsSearcher: Monte Carlo Tree Search for topology space exploration. UCB1 selection, random mutation expansion, HybridVerifier-based rollout heuristic. Budget: 50 simulations or 100ms. 6th path in DynamicTopologyEngine.generate().
- `topology/engine.rs` - DynamicTopologyEngine: 6-path generate strategy (S-MMU → archive → LLM → mutation → MCTS → template fallback). Evolution loop via MAP-Elites + CMA-ME refinement.
- `topology/pyo3_wrappers.rs` - PyO3 thin wrappers: PyTopologyEngine (owns internal S-MMU), PyTopologyExecutor, PyGenerateResult (with opaque topology_id for lazy-load).
- `verification/mod.rs` - Module hub: re-exports `smt` (behind `smt` feature) and `ltl` (always compiled).
- `verification/smt.rs` - SmtVerifier + SmtVerificationResult: OxiZ pure-Rust SMT verifier (QF_LIA with `set_logic("QF_LIA")` for branch-and-bound integer solving). Memory safety, loop bounds, arithmetic, invariant verification (expression parser), provider assignment (exactly-one boolean encoding). Recursive descent parser for constraint strings ("x > 0 and x < 100"). 10 PyO3 methods: prove_memory_safety, check_loop_bound, verify_arithmetic, verify_arithmetic_expr, verify_invariant, verify_invariant_with_feedback, synthesize_invariant, verify_array_bounds, validate_mutation, verify_provider_assignment. `#[instrument]` tracing on all public methods. Behind `smt` feature flag. ALL Python callers fully wired to Rust — zero Z3-only code paths remain.
- `verification/ltl.rs` - LtlVerifier: temporal property verification on TopologyGraph via petgraph. 4 checks: check_reachability (BFS), check_safety (no HIGH→LOW paths), check_liveness (all entries reach exits), check_bounded_liveness (depth ≤ K). LtlResult PyO3 class. Wired into HybridVerifier (safety→errors, liveness→warnings). Always compiled (no feature flag).

### Dashboard (ui/)
- `ui/app.py` - FastAPI backend: EventBus WebSocket push + REST API (HTTPBearer auth via `SAGE_DASHBOARD_TOKEN`)
- `ui/static/index.html` - Single-file dark-theme dashboard (Tailwind + Chart.js)
- WebSocket `/ws` pushes all AgentEvents in real-time. Uses First-Message auth pattern: client sends `{action:"auth", token:"..."}` as first message.
- Task queue: `asyncio.Queue(maxsize=10)` replaces single-task slot. New `/api/tasks` endpoint for queue status.
- Sections: Routing S1/S2/S3, Response, Memory 4-tier, Guardrails, Events, Benchmarks

### Protocols (MCP + A2A)
- `protocols/__init__.py` - Feature detection (HAS_MCP, HAS_A2A)
- `protocols/mcp_server.py` - MCP server: ToolRegistry → MCP tools, `run_task` meta-tool, EventBus resource
- `protocols/a2a_server.py` - A2A agent: AgentLoop → AgentExecutor, AgentCard with 3 skills (general, code, research). Uses a2a-sdk >= 1.0
- `protocols/serve.py` - Unified CLI: `python -m sage.protocols.serve --mcp --a2a`

## Development Commands

### Python SDK
```bash
cd sage-python
pip install -e ".[all,dev]"    # Install in dev mode with all providers
python -m pytest tests/ -v     # Run tests (1170 passed, 102 skipped)
ruff check src/                 # Lint
mypy src/                       # Type check
```

### Benchmarks
```bash
cd sage-python
# Official benchmarks (EvalPlus — 80x more tests than HumanEval)
python -m sage.bench --type evalplus --dataset humaneval          # HumanEval+ (164 problems)
python -m sage.bench --type evalplus --dataset humaneval --limit 20  # Quick smoke test
python -m sage.bench --type evalplus --dataset mbpp               # MBPP+ (378 problems)

# Ablation study (proves each pillar's value vs bare LLM)
python -m sage.bench --type ablation --limit 20                   # 6 configs x 20 tasks

# Evidence collection benchmarks (non-circular ground truth)
python -m sage.bench --type routing_gt                              # Non-circular routing ground truth (50 human-labeled tasks)
python -m sage.bench --type memory_ablation                         # Memory tier ablation (4 configs)
python -m sage.bench --type evolution_ablation                      # Evolution search ablation (3 configs)

# Legacy benchmarks
python -m sage.bench --type routing                    # Routing accuracy (instant, no API key)
python -m sage.bench --type humaneval --limit 20       # Original HumanEval (custom tests)

# Official evaluation protocol (full error logging for debugging)
python -m sage.bench.eval_protocol --suite humaneval --limit 20 -v  # HumanEval+ with error capture
python -m sage.bench.eval_protocol --suite mbpp --limit 20 -v      # MBPP+ with error capture
python -m sage.bench.eval_protocol --replay docs/benchmarks/errors.jsonl  # Post-mortem error analysis
```

### Dashboard
```bash
python ui/app.py                # Start dashboard on http://localhost:8000
```

### Protocol Servers
```bash
pip install ygn-sage[protocols]                           # Install MCP + A2A deps
python -m sage.protocols.serve --mcp --mcp-port 8001     # MCP server
python -m sage.protocols.serve --a2a --a2a-port 8002     # A2A agent
python -m sage.protocols.serve --mcp --a2a               # Both
```

### Rust Core
```bash
cd sage-core
cargo build                    # Build Rust core
cargo test --no-default-features --lib  # Run unit tests (~200 baseline)
cargo test --no-default-features --features smt --lib  # +25 SMT tests
cargo test --no-default-features --features sandbox,cranelift --lib  # +sandbox tests (Linux)
cargo clippy --no-default-features  # Lint Rust code
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

### SSL / Corporate Proxy
This dev machine sits behind a corporate proxy that injects a self-signed certificate.
All outbound HTTPS calls (Google GenAI, pip, etc.) will fail with `CERTIFICATE_VERIFY_FAILED` unless SSL verification is bypassed.

**Protocol for any Python call to external APIs:**
```python
import httpx
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                      http_options={"api_version": "v1beta"})
# REQUIRED on this machine — corporate proxy self-signed cert
client._api_client._httpx_client = httpx.Client(verify=False, timeout=60)
```

**For CLI / env-level bypass:**
```bash
set -a && source .env && set +a          # Load API keys
export PYTHONIOENCODING=utf-8            # Fix Windows console encoding
export REQUESTS_CA_BUNDLE=""             # Disable cert bundle for requests
export NODE_TLS_REJECT_UNAUTHORIZED=0    # For Node.js tools
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
- **Embedder (3-tier fallback)**: RustEmbedder (ONNX via ort `load-dynamic`, native SIMD) > sentence-transformers (Python, in `[embeddings]` extra) > SHA-256 hash. Auto-detected at init. All 3 tiers work on Windows MSVC. Model: snowflake-arctic-embed-m (768-dim, 109M params). Download: `python sage-core/models/download_model.py` + `pip install onnxruntime`. Robust token_type_ids handling (auto-detects from model inputs).
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
- **EvalPlus HumanEval+** (official): 164 problems with 80x more tests (up to 999 plus_inputs per task). pass@1 with subprocess sandbox. Windows-compatible custom evaluator. CLI: `python -m sage.bench --type evalplus --dataset humaneval`
- **EvalPlus MBPP+** (official): 378 Python problems with 35x more tests. CLI: `python -m sage.bench --type evalplus --dataset mbpp`
- **Ablation Study**: 6-config framework value proof (full, baseline, no-memory, no-avr, no-routing, no-guardrails). Quantifies each pillar's contribution vs bare LLM. CLI: `python -m sage.bench --type ablation --limit 20`
- **HumanEval** (legacy): 164 problems bundled as JSON. pass@1 with subprocess sandbox. CLI: `python -m sage.bench --type humaneval`
- **Routing Ground Truth**: 50 human-labeled tasks (10 S1 + 20 S2 + 20 S3). Non-circular: labels by domain expertise, NOT reverse-engineered from heuristic. CLI: `python -m sage.bench --type routing_gt`
- **Memory Ablation**: 4-config measurement (none, tier0, tier01, full). Proves memory tier value. CLI: `python -m sage.bench --type memory_ablation`
- **Evolution Ablation**: 3-config measurement (none, random, full). Proves evolutionary search value. CLI: `python -m sage.bench --type evolution_ablation`
- **Routing Accuracy** (legacy): 30 labeled tasks (10 S1 + 10 S2 + 10 S3). Measures ComplexityRouter precision.
- **Routing Quality**: 45 labeled tasks. Measures both ComplexityRouter and AdaptiveRouter against human-labeled ground truth.
- **Downstream Quality**: DownstreamEvaluator tracks tier precision, escalation rate (<20% target), routing P50/P99 latency (<50ms target).
- **Metrics per task**: pass_rate, avg_latency_ms, avg_cost_usd, routing_breakdown S1/S2/S3

### Benchmark Results (March 10, 2026)
| Benchmark | Score | Notes |
|-----------|-------|-------|
| **EvalPlus HumanEval+ (164)** | **84.1%** pass@1 (138/164) | Official 80x harder tests. Base=90.9%, Plus=84.1% |
| **EvalPlus MBPP+ (378)** | **75.1%** pass@1 (284/378) | Official 35x harder tests. Base=88.9%, Plus=75.1% |
| **EvalPlus MBPP+ (20 smoke)** | **80.0%** pass@1 (16/20) | Quick validation subset |
| Ablation: full vs baseline | **+15pp** (100% vs 85%) | A/B paired, same model (20 tasks) |
| Ablation: routing contribution | **+5pp** (100% vs 95%) | Isolated delta (20 tasks) |
| Routing quality (30 GT) | 100% (30/30) | Self-consistency |

**SOTA context** (HumanEval+ pass@1): O1 ~89%, GPT-4o ~87%, Qwen2.5-Coder-32B ~87%, **YGN-SAGE 84.1%** (using budget Gemini 2.5 Flash), Claude Sonnet 3.5 ~82%

**AVR (Act-Verify-Refine) improvements** (v2, March 10):
- Edge case prompt injection for S2 code tasks (empty inputs, negatives, boolean-is-int, float precision)
- Rich traceback feedback (LLMLOOP/Review-then-fix pattern): full stderr + stdout in AVR retry messages
- Improved syntax error formatting with fenced code blocks
- 60s per-task timeout prevents AVR hangs

## Evolution System
- **DGM Context**: SAMPO solver chooses 1 of 5 strategic actions. Context injected into LLM mutation prompt.
- **Self-modification**: Actions 2/3/4 modify engine hyperparameters (mutations_per_generation, clip_epsilon, filter_threshold)
- **SnapBPF**: Rust CoW memory snapshots for mutation rollback

## SMT Formal Verification (S3)
- **Backend**: Rust OxiZ (sage_core.SmtVerifier, `smt` feature) preferred; Python z3-solver as fallback
- **Zero Z3-only paths**: ALL Python SMT callers fully wired to Rust OxiZ backend (verify_invariant, verify_arithmetic_expr, prove_memory_safety, check_loop_bound, verify_array_bounds, validate_mutation, verify_provider_assignment)
- **QF_LIA integer sort**: `solver.set_logic("QF_LIA")` enables OxiZ 0.1.3 branch-and-bound for proper integer domain reasoning (e.g. x > 0 → x >= 1)
- **Expression parser**: Recursive descent parser in Rust for constraint strings ("x > 0", "x >= -1 and x < 100", "2 + 2 * 3"). Supports variables, integer literals, comparisons (>, <, >=, <=, ==, !=), arithmetic (+, -, *, /), boolean connectives (and, or, not), parentheses
- **CEGAR invariant synthesis**: `synthesize_invariant(pre, post_candidates, max_rounds)` iteratively weakens/strengthens post-conditions over max 5 rounds. `verify_invariant_with_feedback()` returns clause-level diagnostic violations. Python `kg_rlvr.py` wires feedback into S3 escalation prompt via `_last_invariant_feedback`
- **LTL model checking**: `LtlVerifier` checks temporal properties on TopologyGraph — reachability (BFS), safety (no HIGH→LOW paths), liveness (all entries reach exits), bounded liveness (depth ≤ K). Wired into HybridVerifier
- S3 system prompt teaches Z3 DSL: `assert bounds/loop/arithmetic/invariant`
- S2->S3 escalation when AVR budget exhausted (now includes clause-level invariant feedback)
- `kg_rlvr.py` parses `<think>` blocks, scores each step via safe AST evaluator (no `eval()`) + OxiZ/Z3
- `z3_verify.py`: 3 of 4 checks are Python-native. `verify_provider_assignment()` uses Rust OxiZ SAT (exactly-one encoding) or Z3 `PbEq` fallback
- `z3_validator.py`: Z3Validator auto-delegates to Rust SmtVerifier (sub-0.1ms) with z3-solver fallback

## Resilience
- **CircuitBreaker** (`resilience.py`): per-subsystem failure tracking (max_failures=3)
- 6 breakers in agent_loop: semantic_memory, smmu_context, runtime_guardrails, episodic_store, entity_extraction, evolution_stats
- After 3 consecutive failures, circuit opens and skips calls with WARNING log
- `record_success()` resets the counter

## ExoCortex
Auto-configured with `DEFAULT_STORE = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"`.
Resolution: explicit param > env var `SAGE_EXOCORTEX_STORE` > DEFAULT_STORE.

```python
# Via SAGE SDK
from sage.memory.remote_rag import ExoCortex
exo = ExoCortex()  # Works automatically when GOOGLE_API_KEY is set
```

**Direct query protocol** (for scripts / Claude Code):
```python
import httpx, os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                      http_options={"api_version": "v1beta"})
client._api_client._httpx_client = httpx.Client(verify=False, timeout=60)

store_id = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"
tools = [types.Tool(file_search=types.FileSearch(file_search_store_names=[store_id]))]

result = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="<your query>",
    config=types.GenerateContentConfig(tools=tools, temperature=0.1),
)
print(result.text)
```
**API note:** Use `types.FileSearch(file_search_store_names=[...])` — NOT `types.FileSearchTool` (does not exist in google-genai SDK).

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
- OxiZ 0.1 (pure Rust SMT solver, S3 formal verification) + Z3 Solver 4.16 (fallback)
- aiosqlite (episodic + semantic memory persistence)
- Apache Arrow / PyArrow (zero-copy memory compaction)
- Wasm (wasmtime v36 LTS, Component Model, WASI p2 deny-by-default) + Docker (multi-tier sandboxing)
- tree-sitter 0.26 + tree-sitter-python 0.25 (AST-based code validation, `tool-executor` feature)
- process-wrap 9 (subprocess executor with tokio timeout + kill-on-drop, `tool-executor` feature)
- DashMap (Rust) -- lock-free FIFO+TTL RAG cache + CoW snapshots
- ort 2.0 (ONNX Runtime for Rust, `load-dynamic`, optional `onnx` feature) — native embeddings, works on Windows MSVC
- tokenizers 0.21 (HuggingFace tokenizer, optional `onnx` feature)
