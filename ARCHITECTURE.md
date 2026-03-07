# YGN-SAGE Architecture

> **Status: Research Prototype** — Last verified: 2026-03-06

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
- `MetacognitiveController._assess_heuristic()` uses keyword matching (regex on task text) to compute complexity/uncertainty scores
- Thresholds: S1 (simple) ≤0.35, S3 (formal) >0.7, S2 (code) = everything else
- LLM-based assessment available async only (requires `GOOGLE_API_KEY`); sync callers always get heuristic

**Known limitations:**
- The 30/30 routing benchmark is a **self-consistency test** — labels were calibrated against the heuristic, so 100% agreement proves nothing about downstream task quality
- No evidence that S1/S2/S3 routing improves outcomes vs. always using the best model
- Heuristic is keyword-based, not semantic

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
| 1 — Episodic | SQLite / in-memory | tested-unit | Optional (SQLite) |
| 2 — Semantic | In-memory graph | tested-unit | None |
| 3 — ExoCortex | Google File Search API | tested-integration | Vendor-managed |

**Known limitations:**
- **Tier 0 silently falls back to Python mock** when Rust `sage_core` extension is not available. No warning to caller. Mock methods return dummy values.
- **Tier 1 defaults to volatile in-memory** — SQLite persistence is opt-in, not default
- **Tier 2 is in-memory only** — no persistence, no export, lost on restart
- **Tier 3 is vendor-locked** to Google GenAI File Search API
- **No evidence** that 4-tier memory improves outcomes vs. long-context baseline

### 5. Guardrails

**What it does:** Composable validation pipeline at 3 points: input (PERCEIVE), runtime (ACT), output (LEARN).

**Evidence:** tested-unit

**Built-in guardrails:** CostGuardrail (USD limits), SchemaGuardrail (required fields), Z3 bounds checking

**Known limitations:**
- Z3 guardrails check arithmetic bounds on generated code, not semantic correctness
- Runtime guardrails are best-effort (don't block execution)
- No formal proof that guardrails prevent all policy violations

### 6. Contract IR + Verification (Phase 2+3)

**What it does:** Typed task DAG with formal verification, policy enforcement, and repair loops.

**Evidence:** tested-unit + tested-integration (599 tests total, 6 E2E + 14 stress + 10 ablation + 11 bugfix)

**Components:**
| Component | Module | Tests |
|-----------|--------|-------|
| TaskNode IR | `contracts/task_node.py` | 9 unit tests |
| Verification Functions | `contracts/verification.py` | 11 unit tests |
| TaskDAG + Scheduler | `contracts/dag.py` | 14 unit tests |
| Z3 Contract Verification | `contracts/z3_verify.py` | 11 unit tests |
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
- **Z3 SMT proofs:** Capability coverage, budget feasibility, type compatibility — checked at plan time
- **Info-flow enforcement:** No HIGH→LOW data flow (lattice-based security labels)
- **CEGAR repair:** Counterexample-guided retry → escalate → abort with hard fences
- **DyTopo routing:** Capability-constrained model selection with adaptive feedback
- **Causal memory:** Directed causal edges with BFS chain traversal + ancestor queries
- **Write gating:** Confidence-based abstention ("better to forget than to store noise")
- **Cost tracking:** Cumulative per-node cost accounting with mid-loop budget halt

**Known limitations:**
- Z3 verifies structural properties, not semantic correctness of LLM outputs
- DynamicRouter uses static quality scores, not live profiling
- CausalMemory is in-memory only (no persistence). Bounded via `max_entities` + `max_context_lines`
- Planner only supports static plan specs (no LLM-driven planning yet)

### 7. Dashboard

**What it does:** Real-time event viewer via FastAPI + WebSocket.

**Evidence:** tested-unit (auth + CORS added in Phase 0)

**Security (Phase 0 fix):**
- HTTPBearer auth with `SAGE_DASHBOARD_TOKEN` env var (no token = open dev mode)
- CORS middleware configured for localhost:8000 and :3000
- `EventBus.clear()` public API replaces private field access

**Known limitations:**
- **Global mutable state** — single-task only, race conditions under concurrent use

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

**What it does:** High-performance data plane: Arrow working memory, eBPF/Wasm sandboxes, RAG cache.

**Evidence:** tested-unit (38 Rust tests passing)

| Component | Status |
|-----------|--------|
| Arrow working memory | Solid, SIMD/AVX-512 |
| S-MMU paging | Implemented |
| RagCache (FIFO+TTL) | Implemented (DashMap) |
| eBPF sandbox | Implemented (solana_rbpf), but optional feature |
| Wasm sandbox | Implemented (wasmtime + WASI), but optional feature |
| Z3 bindings | Implemented |

**Known limitations:**
- eBPF and Wasm are behind `sandbox` feature flag (not built by default, not tested in CI)
- eBPF compiles for BPF target requiring `core` stdlib — not available on all platforms
- Python falls back silently to mock when Rust extension unavailable

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
| Working Memory | `sage_core` not installed | Falls back to Python mock (dummy values) |
| Episodic Memory | Default config | In-memory only (no persistence) |
| Provider Discovery | Missing SDK (openai, etc.) | Returns empty list, skips provider |
| OpenAI Compat | file_search parameter | Silently dropped (debug log only) |
| OpenAI Compat | tool role messages | Rewritten to user role |
| LLM Assessment | No GOOGLE_API_KEY | Falls back to heuristic (sync path) |
| Evolution | No sandbox feature | Skips sandboxed evaluation |

**V2 design goal:** Replace all silent degradation with hard failures or explicit warnings.
Phase 0 fixed: dashboard auth, provider warnings. Phase 2+3 added: CapabilityMatrix.require() hard-fails.

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
