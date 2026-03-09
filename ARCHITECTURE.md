# YGN-SAGE Architecture

> **Status: Research Prototype** — Last verified: 2026-03-09

This document describes what YGN-SAGE **actually implements**, with honest evidence levels and known limitations.

## Evidence Levels

| Level | Meaning |
|-------|---------|
| **implemented** | Code exists and runs |
| **tested-unit** | Has unit tests passing |
| **tested-integration** | Tested with real APIs end-to-end |
| **benchmarked** | Measured against an external baseline |

---

## Components

### 1. Cognitive Routing (S1/S2/S3)

**What it does:** Routes tasks to model tiers based on complexity/uncertainty assessment.

**Evidence:** tested-unit (self-consistency only)

**How it works:**
- `ComplexityRouter._assess_heuristic()` (renamed from MetacognitiveController) uses word-boundary regex matching (`\b`) on task text to compute complexity/uncertainty scores
- Thresholds: S1 (simple) ≤0.35, S3 (formal) >0.7, S2 (code) = everything else
- **Speculative zone** (0.35-0.55 complexity): detected and logged; designed for future parallel S1+S2 execution but currently routes normally
- LLM-based assessment available async only (requires `GOOGLE_API_KEY`); sync callers always get heuristic

**Known limitations:**
- The 30/30 routing benchmark is a **self-consistency test** — labels were calibrated against the heuristic, so 100% agreement proves nothing about downstream task quality
- No evidence that S1/S2/S3 routing improves outcomes vs. always using the best model
- Heuristic is word-boundary regex-based, not semantic
- Speculative parallel execution not yet implemented (zone detection only)

### 2. CognitiveOrchestrator

**What it does:** Decomposes tasks into subtasks and routes them through Sequential or Parallel topologies.

**Evidence:** tested-unit

**How it works:**
- S1/S2: single-agent execution (fast or quality tier)
- S3: optional decomposition into subtasks via tag parsing, then Sequential (dependencies) or Parallel (independent)

**Known limitations:**
- Only Sequential/Parallel — no Loop, Handoff, or hierarchical topologies in the orchestrator itself
- Subtask decomposition is simple tag parsing, not a real planner
- Binary routing: seq if dependencies detected, par otherwise

### 3. Model Registry & Providers

**What it does:** Auto-discovers models from 7 providers at boot. Routes LLM calls to appropriate provider.

**Evidence:** tested-integration (7 providers verified working)

**Providers:** Google Gemini, OpenAI, xAI, DeepSeek, MiniMax, Kimi, Codex CLI

**Known limitations:**
- **OpenAI-compatible providers silently degrade:**
  - `file_search_store_names` parameter is dropped (logged at debug level, not warning)
  - `tool` role messages are rewritten to `user` role (semantic loss)
- **Provider discovery returns empty list on missing SDK** — no crash, but silent capability loss
- **MiniMax has no models.list()** — model list is hardcoded
- **Structured output + tools are mutually exclusive on Google** — not enforced, can cause API errors

### 4. Memory (4 Tiers)

**What it does:** Hierarchical memory from fast working memory to persistent RAG.

| Tier | Backend | Evidence | Persistence |
|------|---------|----------|-------------|
| 0 — Working (STM) | Rust Arrow / Python mock | tested-unit | None (session) |
| 1 — Episodic | SQLite | tested-unit | SQLite (`~/.sage/episodic.db`) |
| 2 — Semantic | In-memory graph + SQLite | tested-unit | SQLite (`~/.sage/semantic.db`) |
| 3 — ExoCortex | Google File Search API (via KnowledgeStore protocol) | tested-integration | Vendor-managed |

**S-MMU Integration (wired March 2026):**

The S-MMU (Structured Memory Management Unit) in `sage-core` is now wired end-to-end:

- **Write path:** `MemoryCompressor.step()` compresses events, then calls `compact_to_arrow_with_meta()` with extracted keywords, embedding vector (via `Embedder`), and dynamic summary. This populates the S-MMU graph with temporal, semantic, and entity edges.
- **Read path:** During THINK phase, `agent_loop.py` calls `retrieve_smmu_context()` (from `memory/smmu_context.py`) which queries the S-MMU graph (BFS multi-path traversal with configurable weights) and injects the top-k chunk summaries (via `get_chunk_summary()`) as a SYSTEM message before the LLM call.
- **Embedder:** `memory/embedder.py` auto-selects from 3-tier fallback: RustEmbedder (ONNX, native) > sentence-transformers (Python) > hash. All 3 tiers working on Windows. Wired into `MemoryCompressor` at boot time.

**ONNX Embedder (3-tier fallback):**

The Python `Embedder` adapter (`memory/embedder.py`) now supports a 3-tier fallback chain, auto-detected at init:

1. **RustEmbedder (preferred):** Native ONNX Runtime embedder in `sage-core/src/memory/embedder.rs`, behind the `onnx` Cargo feature flag. Loads the `all-MiniLM-L6-v2` ONNX model (384-dim, L2-normalized output). Exposed to Python via PyO3 class with `embed(text) -> list[float]` and `embed_batch(texts) -> list[list[float]]`. Fastest option with native SIMD acceleration.
2. **sentence-transformers (fallback):** Python `sentence-transformers` library with `all-MiniLM-L6-v2`. Same model, same 384-dim output, but slower than native Rust.
3. **SHA-256 hash (last resort):** Deterministic hash-based fallback producing 384-dim vectors. No semantic meaning, but ensures the pipeline never crashes due to missing dependencies.

Key details:
- **Feature flag:** `onnx` in `sage-core/Cargo.toml` gates `ort` (ONNX Runtime, `load-dynamic`) and `tokenizers` (HuggingFace) dependencies
- **Dynamic linking:** Uses ort's `load-dynamic` feature — loads `onnxruntime.dll` at runtime via `ORT_DYLIB_PATH` env var or auto-discovery from pip `onnxruntime` package. Avoids static linking errors (LNK1120) on Windows MSVC.
- **Model download:** `python sage-core/models/download_model.py` fetches the ONNX model and tokenizer files
- **Build command:** `maturin develop --features onnx` + `pip install onnxruntime` for the runtime DLL
- **Tests:** `cargo test --features onnx` runs 5 Rust unit tests (requires model download + `ORT_DYLIB_PATH`)

In pure-Python mock mode (no Rust), the write path runs but S-MMU chunk count stays at 0, so the read path returns empty string. No errors in either direction.

**Known limitations:**
- **Tier 0 falls back to Python mock** when Rust `sage_core` extension is not available. ~~No warning~~ Now emits `warnings.warn()` (audit fix A3).
- ~~Tier 1 defaults to volatile in-memory~~ **Tier 1 now defaults to SQLite** at `~/.sage/episodic.db` (audit fix A4)
- ~~**Tier 2 is in-memory only**~~ **Tier 2 now persists to SQLite** at `~/.sage/semantic.db` (audit fix T8). Auto-loads at boot, auto-saves after each run
- **Tier 3** uses `KnowledgeStore` protocol (`memory/rag_backend.py`); ExoCortex implements it, but only the Google backend exists today
- **No evidence** that 4-tier memory improves outcomes vs. long-context baseline
- **S-MMU read path** returns chunk summaries (via `get_chunk_summary()`) and scores — full content retrieval requires Rust extension

### 5. Guardrails

**What it does:** Composable validation pipeline at 3 points: input (PERCEIVE), runtime (ACT), output (LEARN).

**Evidence:** tested-unit

**Built-in guardrails:** CostGuardrail (USD limits), OutputGuardrail (empty/too-long/refusal detection, default pipeline), SchemaGuardrail (required fields, for JSON mode), Z3 bounds checking

**Known limitations:**
- Z3 guardrails check arithmetic bounds on generated code, not semantic correctness
- Runtime guardrails are best-effort (don't block execution)
- No formal proof that guardrails prevent all policy violations

### 6. Contract IR + Verification (Phase 2+3)

**What it does:** Typed task DAG with formal verification, policy enforcement, and repair loops.

**Evidence:** tested-unit + tested-integration (846 tests total, 6 E2E + 14 stress + 10 ablation + 14 bugfix + 18 audit-response + 17 audit-fixes + Sprint 1-3 tests)

**Components:**
| Component | Module | Tests |
|-----------|--------|-------|
| TaskNode IR | `contracts/task_node.py` | 9 unit tests |
| Verification Functions | `contracts/verification.py` | 11 unit tests |
| TaskDAG + Scheduler | `contracts/dag.py` | 14 unit tests |
| Z3 Contract Verification | `contracts/z3_verify.py` | 17 unit tests |
| PolicyVerifier | `contracts/policy.py` | 12 unit tests |
| DAGExecutor | `contracts/executor.py` | 7 unit tests |
| TaskPlanner (Plan-and-Act) | `contracts/planner.py` | 10 unit tests |
| RepairLoop (CEGAR) | `contracts/repair.py` | 8 unit tests |
| DynamicRouter (DyTopo) | `routing/dynamic.py` | 9 unit tests |
| CausalMemory | `memory/causal.py` | 10 unit tests |
| WriteGate | `memory/write_gate.py` | 9 unit tests |
| Synthetic Failure Lab | `tests/test_failure_modes.py` | 10 tests (MAST taxonomy) |
| CostTracker | `contracts/cost_tracker.py` | 7 unit tests |
| Contract Stress Tests | `tests/test_stress_contracts.py` | 14 stress tests |
| Ablation Study | `tests/test_ablation.py` | 10 ablation tests |
| Phase 3 Integration | `tests/test_integration_phase3.py` | 6 E2E tests |

**Key capabilities:**
- **Z3 SMT proofs:** Provider assignment uses genuine Z3 SAT (`z3.PbEq` for exactly-one constraint). Capability coverage, budget feasibility, and type compatibility use Python-native checks (~2000x faster than Z3 for these trivial set/arithmetic operations) — all checked at plan time
- **Info-flow enforcement:** No HIGH→LOW data flow (lattice-based security labels)
- **CEGAR repair:** Counterexample-guided retry → escalate → abort with hard fences
- **DyTopo routing:** Capability-constrained model selection with adaptive feedback
- **Causal memory:** Directed causal edges with BFS chain traversal + ancestor queries
- **Write gating:** Confidence-based abstention ("better to forget than to store noise")
- **Cost tracking:** Cumulative per-node cost accounting with mid-loop budget halt

**Known limitations:**
- Z3 verifies structural properties, not semantic correctness of LLM outputs
- DynamicRouter uses static quality scores, not live profiling
- CausalMemory now has SQLite persistence (`save()`/`load()`, optional `db_path`). Bounded via `max_entities` + `max_context_lines`
- Planner only supports static plan specs (no LLM-driven planning yet)

### 7. Dashboard

**What it does:** Real-time event viewer via FastAPI + WebSocket.

**Evidence:** tested-unit (auth + CORS added in Phase 0)

**Security (Phase 0 fix + Sprint 1):**
- WebSocket uses First-Message auth pattern: client sends `{action:"auth", token:"..."}` as first message (was query param)
- HTTPBearer auth for REST API with `SAGE_DASHBOARD_TOKEN` env var (no token = open dev mode)
- CORS middleware configured for localhost:8000 and :3000
- `EventBus.clear()` public API replaces private field access

**Task Queue (Sprint 2):**
- Replaces single-task slot (was 409 on concurrent requests)
- `asyncio.Queue(maxsize=10)` with background worker
- New `/api/tasks` endpoint for queue status and task history

**Known limitations:**
- ~~Global mutable state~~ Encapsulated in `DashboardState` class (audit fix C13), but still single-process

### 8. Evolution Engine

**What it does:** LLM-driven code mutation with DGM (Dynamic Goal Management) context injection and SAMPO strategic solver.

**Evidence:** tested-unit (code runs, but not validated against baselines)

**How it works:**
- DGM self-modifies engine hyperparameters (mutations_per_generation, clip_epsilon, filter_threshold)
- Z3 safety gate validates mutations before evaluation
- SAMPO solver tracks rewards, batches policy updates every 5 generations
- MAP-Elites topology search explores agent configurations

**Known limitations:**
- **Not validated**: no evidence that evolution improves outcomes vs. random search or manual tuning
- **No ablation study**: unclear which components (DGM, SAMPO, MAP-Elites) contribute value
- Depends on eBPF/Wasm for sandboxed evaluation (optional features)

### 9. Rust Core (sage-core)

**What it does:** High-performance data plane: Arrow working memory, ToolExecutor (security + sandboxed execution), Wasm sandbox, RAG cache.

**Evidence:** tested-unit (63 Rust tests passing with sandbox+tool-executor features, +5 ONNX feature-gated)

| Component | Status |
|-----------|--------|
| Arrow working memory | Solid, SIMD/AVX-512 |
| S-MMU paging | Wired (write via compressor, read via THINK phase) |
| RagCache (FIFO+TTL) | Implemented (DashMap) |
| **ToolExecutor** | **Implemented** (tree-sitter validator + Wasm WASI sandbox + subprocess fallback). PyO3 class with GIL release. Behind `tool-executor` + `sandbox` feature flags |
| **tree-sitter validator** | **Implemented** (23 blocked modules + 11 blocked calls, error-tolerant). Behind `tool-executor` feature |
| **Subprocess executor** | **Implemented** (tokio timeout + kill-on-drop). Behind `tool-executor` feature |
| Wasm sandbox | Implemented (wasmtime v36 LTS, Component Model + WASI p2 deny-by-default). `cranelift` excluded on Windows MSVC. `execute_precompiled()` / `execute_precompiled_wasi()` for Windows |
| eBPF sandbox | Implemented (solana_rbpf), but optional feature. `snap_bpf.c` is a stub (printk only) |
| Z3 bindings | Implemented |

**Known limitations:**
- eBPF and Wasm are behind `sandbox` feature flag (not built by default, not tested in CI)
- eBPF compiles for BPF target requiring `core` stdlib — not available on all platforms
- `snap_bpf.c` is a stub (printk only) — real SnapBPF is Rust userspace CoW in `ebpf.rs`
- Python falls back silently to mock when Rust extension unavailable

### 9b. ToolExecutor (Rust Security Pipeline)

**What it does:** End-to-end code security pipeline for dynamic tool creation: AST validation → Wasm WASI sandbox → subprocess fallback.

**Evidence:** tested-unit (63 Rust tests + 184 security tests covering 6 Phoenix attack vectors)

**Execution priority:**
1. **Wasm WASI sandbox** (if component loaded) — deny-by-default capabilities: NO filesystem, NO env vars, NO network, NO subprocess. Only stdout/stderr inherited.
2. **Bare Wasm** (for simple components without WASI imports)
3. **Subprocess fallback** (always available) — timeout + kill-on-drop isolation

**tree-sitter validator (23 modules + 11 calls blocked):**
- Modules: `os`, `sys`, `subprocess`, `shutil`, `ctypes`, `importlib`, `socket`, `http`, `ftplib`, `smtplib`, `xmlrpc`, `multiprocessing`, `threading`, `signal`, `resource`, `code`, `codeop`, `pathlib`, `glob`, `tempfile`, `pickle`, `shelve`, `builtins`
- Calls: `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`

**WASI deny-by-default (verified with Context7 + wasmtime v36 docs):**
- `WasiCtxBuilder::new()` starts with NO capabilities
- Only `inherit_stdout()` + `inherit_stderr()` added
- No `inherit_env()`, no `preopened_dir()`, no `inherit_stdin()`

**6 Phoenix Security Vectors — all blocked:**
| Vector | Blocked by |
|--------|------------|
| Filesystem read | tree-sitter (`os`, `pathlib`, `open`) + WASI (no preopened dirs) |
| Filesystem write | tree-sitter (`os`, `shutil`, `open`) + WASI (no preopened dirs) |
| Env var access | tree-sitter (`os`) + WASI (no `inherit_env`) |
| Network access | tree-sitter (`socket`, `http`) + WASI (no network in preview2) |
| Subprocess spawn | tree-sitter (`subprocess`, `os`) + WASI (no subprocess in preview2) |
| Dangerous import | tree-sitter (23 modules + `__import__` + `importlib`) |

**Feature flags:**
- `tool-executor`: tree-sitter validator + subprocess executor + ToolExecutor PyO3 class
- `sandbox`: wasmtime + wasmtime-wasi (Wasm paths in ToolExecutor)
- `cranelift`: JIT compilation (Linux only; `load_component()` requires this)

**Python integration:** `sage.tools.meta.create_python_tool` tries Rust `ToolExecutor` first, falls back to Python `sandbox_executor.py`.

### 10. Agent Composition

**What it does:** Patterns for composing agents: Sequential, Parallel, Loop, Handoff.

**Evidence:** tested-unit

**Known limitations:**
- ParallelAgent uses string concatenation as default aggregator (not semantic merge)
- Only used standalone — not wired into CognitiveOrchestrator's decomposition path
- Loop and Handoff patterns available but not exercised in any benchmark

### 11. Benchmarks

**What it does:** Built-in HumanEval (164 problems) and routing self-consistency test.

**Evidence:**
- HumanEval: tested-integration (85% pass@1 on 20-problem subset with Gemini Flash Lite)
- Routing: self-consistency only (30/30, circular — see routing.py docstring)

**Known limitations:**
- HumanEval full 164 not yet run with published per-task traces
- No bare-model baseline (measures framework overhead)
- Routing benchmark proves nothing about downstream quality (labels calibrated to heuristic)

---

## Silent Degradation Modes

These components degrade silently instead of failing hard:

| Component | Trigger | Degradation |
|-----------|---------|-------------|
| Working Memory | `sage_core` not installed | Falls back to Python mock (**now warns**, audit fix A3) |
| Episodic Memory | Default config | ~~In-memory only~~ **Now defaults to SQLite** (audit fix A4) |
| Semantic Memory | Default config | ~~In-memory only~~ **Now persists to SQLite** (audit fix T8) |
| Provider Discovery | Missing SDK (openai, etc.) | Returns empty list, skips provider |
| OpenAI Compat | file_search parameter | Dropped (**now warns at WARNING level**, audit fix B9) |
| OpenAI Compat | tool role messages | Rewritten to user role (**now warns**, audit fix T4) |
| LLM Assessment | No GOOGLE_API_KEY | Falls back to heuristic (sync path) |
| Evolution | No sandbox feature | Skips sandboxed evaluation |
| Best-effort subsystems | 3+ consecutive failures | **Circuit breaker opens** with WARNING log (audit fix T6) |

**V2 design goal:** Replace all silent degradation with hard failures or explicit warnings.
Phase 0 fixed: dashboard auth, provider warnings. Phase 2+3 added: CapabilityMatrix.require() hard-fails.
Audit response (March 2026) fixed: sandbox blocks host by default, working memory fallback warns, episodic defaults to SQLite.

---

## Audit Response (March 2026)

Three independent audits (Opus 4.6, GPT-5.4 Pro, GPT-5.4 Codex) identified 20 confirmed findings.
16 tasks organized in 4 phases (A-D) were executed, followed by a cross-verification audit (5 audits, 78 assertions) that confirmed 15 problems requiring further fixes. Two additional sprints addressed the cross-verification findings. Current test count: **846 passed, 1 skipped**.

### Phase A — Kill Unsafe Defaults (Tasks 1-5)
| Task | Finding | Fix |
|------|---------|-----|
| A1 | Sandbox allowed host code execution | `allow_local=False` by default; callers must opt in with `allow_local=True` |
| A2 | Dashboard WebSocket unauthenticated | HTTPBearer auth via `SAGE_DASHBOARD_TOKEN`; binds localhost only |
| A3 | Working memory fallback silent | `warnings.warn()` emitted when falling back to Python mock |
| A4 | Episodic memory volatile by default | Now defaults to SQLite at `~/.sage/episodic.db` |
| A5 | CI skipped Rust tests; invalid edition | CI runs `cargo test --no-default-features`; edition fixed to 2021 |

### Phase B — Honesty Reset (Tasks 6-9)
| Task | Finding | Fix |
|------|---------|-----|
| B6 | 24+ AI confabulation markers in docs/code | Stripped all "novel", "groundbreaking", etc. from docstrings and comments |
| B7 | SAMPO/DGM terminology misleading | Honestly documented as heuristic solvers, not RL or formal optimizers |
| B8 | Dual control planes (boot.py confusion) | Consolidated to single `AgentSystem` entry point |
| B9 | OpenAI-compat silently drops semantics | Now warns at `WARNING` level on file_search drop and tool-role rewrite |

### Phase C — Evidence & Baselines (Tasks 10-13)
| Task | Finding | Fix |
|------|---------|-----|
| C10 | No baseline benchmark mode | Added `--baseline` flag: direct LLM call, no routing/memory overhead |
| C11 | Routing value unproven | Added ablation tests: routing ON vs OFF, measuring actual delta |
| C12 | Only 2 guardrails (cost + Z3) | Added `SchemaGuardrail` for required-field validation |
| C13 | Dashboard global mutable state | Encapsulated into `DashboardState` class; thread-safe access |

### Phase D — Documentation Sync (Tasks 14-16)
Updated ARCHITECTURE.md, CLAUDE.md, and MEMORY.md to reflect all changes above.

### Audit Verification Fixes (March 2026, Sprints 1-2)

Cross-verification report: `docs/audits/2026-03-07-audit-verification.md` (78 assertions, 52 confirmed, 14 partial, 8 infirmed).

**Sprint 1 — Stop the Bleeding:**
| Task | Finding | Fix |
|------|---------|-----|
| T1 | `eval()` RCE in kg_rlvr.py | Safe AST evaluator + fail-closed (not fail-open). z3_topology.py silent catch now logs WARNING |
| T2 | eBPF claimed in README but not built | Removed stale eBPF/solana_rbpf claims |
| T3 | Test badge count stale (413) | Updated to actual count (846) |
| T4 | tool→user role rewrite silent | Extracted `_convert_messages()` + WARNING log |
| T5 | snap_bpf.c claimed as SOTA | Marked as STUB — NOT FUNCTIONAL |
| T6-a | SchemaGuardrail wrong for text output | Added `OutputGuardrail` (empty/too-long/refusal detection) as default. SchemaGuardrail kept for JSON mode |
| T6-b | Routing used substring matching | Word-boundary regex (`\b`) for heuristic keyword matching |
| T6-c | Cost estimation inaccurate | Uses API `usage_metadata` (Google `prompt_token_count`/`candidates_token_count`) when available, falls back to `len(text)//4` |
| T6-d | S-MMU returned bare chunk IDs | Returns chunk summaries via `get_chunk_summary()` |
| T6-e | WebSocket auth via query param | First-Message pattern: `{action:"auth", token:"..."}` as first WS message |
| T6-f | CausalMemory lost on restart | SQLite persistence (`save()`/`load()`, optional `db_path`) |
| T6-g | sentence-transformers not in deps | Added `[embeddings]` extra in pyproject.toml |

**Sprint 2 — Observability + Z3:**
| Task | Finding | Fix |
|------|---------|-----|
| T6 | 6 silent `except: pass` in agent_loop | CircuitBreaker per subsystem (max_failures=3, opens with WARNING) |
| T7 | Z3 used for trivial set checks | 3 of 4 checks now Python-native (~2000x faster). Only `verify_provider_assignment()` uses Z3 SAT (`z3.PbEq` exactly-one constraint) |
| T8 | SemanticMemory lost on restart | SQLite persistence (`~/.sage/semantic.db`), auto-load at boot, auto-save after run |
| T9 | Benchmarks not reproducible | Truth-pack: BenchmarkManifest + per-task JSONL traces |
| T10-a | ExoCortex vendor-locked | `KnowledgeStore` protocol in `memory/rag_backend.py`. ExoCortex implements it; future backends can plug in |
| T10-b | Dashboard single-task slot (409) | Task queue (asyncio.Queue, maxsize=10) + `/api/tasks` status endpoint |
| T10-c | S-MMU register_chunk O(n²) | Bounded recency scan: only last 128 chunks scanned (`MAX_SEMANTIC_NEIGHBORS = 128`) |

**Sprint 3 — Prove or Remove (partial):**
| Task | Finding | Status |
|------|---------|--------|
| T10 | No cost-performance frontier | Pending (requires API keys) |
| T11 | Full HumanEval 164 | Smoke test 80% pass@1 (20 problems); full 164 requires dedicated run |
| T12 | Evolution engine unvalidated | Pending (ablation study) |
| T13 | wasmtime v29 EOL | **DONE**: Upgraded to v36 LTS. cranelift excluded on Windows MSVC (stack overflow); `execute_precompiled()` for Windows, `execute()` with JIT on Linux CI |

### Deferred to Phase E
- **ExoCortex vendor lock** (Google File Search API): `KnowledgeStore` protocol added (Sprint 2), but only the Google backend exists today. Needs at least one alternative backend implementation.
- **Evolution engine validation**: needs controlled experiment with ablation
- **Cost-performance frontier benchmark**: requires API keys for multi-tier comparison

---

## Research References

This project draws on ideas from:

- [AdaptOrch](https://arxiv.org/abs/2602.16873) — topology-adaptive routing
- [DyTopo](https://arxiv.org/abs/2602.06039) — round-level semantic routing
- [VeriMAP](https://arxiv.org/abs/2510.17109) — verified multi-agent planning
- [AMA-Bench](https://arxiv.org/abs/2602.22769) — causal memory evaluation
- [PCAS](https://arxiv.org/abs/2602.16708) — policy compiler for agent systems
- [MAST](https://arxiv.org/abs/2503.13657) — multi-agent failure taxonomy
- [Plan-and-Act](https://arxiv.org/abs/2503.09572) — planner/executor separation

See `docs/plans/2026-03-06-v2-evidence-first-design.md` for the V2 rebuild plan.
