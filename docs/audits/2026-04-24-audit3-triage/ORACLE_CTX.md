# ORACLE_CTX.md — YGN-SAGE project context for oracle consultations
**Date:** 2026-04-24 · **Main commit:** `820ea3e2` · **Scope:** AUDIT3.md triage

## Project identity
YGN-SAGE — Agent Development Kit with 5 cognitive pillars (Topology, Tools,
Memory, Evolution, Strategy). Rust core (`sage-core`, PyO3) + Python SDK
(`sage-python`) + knowledge pipeline (`sage-discover`). PyPI `ygn-sage` alpha;
v0.1.0. License MIT.

## Stack
- **Rust 2021**, `pyo3 = 0.25`, `tokio 1`, `serde 1`, `thiserror 2`, `dashmap 6`,
  feature-gated `oxiz` (SMT), `wasmtime` (wasm32-wasip1 RustPython sandbox
  since ADR-013 §5, 2026-04-22), `tree-sitter` (AST validator),
  `cranelift` (JIT).
- **Python 3.12/3.13**, `pydantic-ai >= 1.84` (LiteLLM → PydanticAI migrated
  2026-04-18), `httpx`, `anyio`, `aiosqlite`, `numpy`, `truststore`,
  `sentence-transformers` (embedder), `z3-solver` (formal verifier),
  `fastapi`+`uvicorn` (dashboard, optional).
- **Bench**: `evalplus`, `swebench` (offline datasets).

## Top-level directory layout
```
sage-core/        Rust orchestrator (PyO3)
  src/
    topology/     TopologyEngine, templates, controller, MAP-Elites, CMA-ME
    routing/      ModelAssigner, SystemRouter, cards.toml registry
    memory/       Arrow STM, SMMU, WriteGate
    sandbox/      tree-sitter + wasm_python (embedded RustPython)
    verification/ OxiZ SMT, LtlVerifier (graph checks)
  config/cards.toml  — single source of truth for model specs
sage-python/      Python SDK
  src/sage/
    pipeline.py         5-stage CognitiveOrchestrationPipeline
    agent_loop.py       Tool-using agent loop
    topology_controller.py  Runtime adaptation (Rust-primary since ADR-012)
    providers/          PydanticAIProvider wrapper + connector
    bench/              SWE-bench, BCB, MASBENCH, diff-verifier + repair
    tools/              typed_repo, forge (ToolForge, GapDetector)
    memory/             MemoryManager, write_gate
    phases/             Agent loop sub-phases (plan, act, learn)
  tests/              2339 collected (~2290 pass / 45 skip excl API-key deps)
sage-discover/    Knowledge pipeline (arXiv → ExoCortex RAG)
docs/adr/         ADR-009..013 (architectural decision records)
docs/audits/      Past audit triage artifacts (incl 2026-04-23-alire-verification.md)
docs/benchmarks/  Dated smoke reports + observe-mode verifier artefacts
roadmap.md        Living backlog (A0/A1/A3/A7/A8/A9/A10/A11 entries)
CLAUDE.md         Agent guidance (directives 1-6)
PROMPT.md         This audit verification protocol
```

## Dominant architectural patterns
1. **Rust-primary, Python-tolerant (Directive #1).** Performance-critical
   code lives in Rust with PyO3 exports; Python orchestrates. Topology
   runtime adaptation, model routing, memory tiers, topology evolution
   (MAP-Elites/CMA-ME), sandbox, SMT verification all Rust-owned.
2. **Single source of truth** for model specs in `sage-core/config/cards.toml`
   (Directive #6). Python mirror at `sage-python/config/cards.toml`. No
   training-data hardcodes.
3. **Deny-by-default sandbox** (ADR-013 §5, 2026-04-22). `validate_and_execute`
   uses tree-sitter AST validator + embedded RustPython wasm32-wasip1 runtime
   with WASI-p1 deny-by-default (no FS/net/proc/env inheritance, 256 MiB cap,
   epoch-interrupt timeout). No subprocess fallback on the default path.
   Escape hatch `execute_raw` gated by `SAGE_UNSAFE_RAW_EXEC=1` (bypasses
   both AST + Wasm).
4. **Contextual bandit + formal quality** (Directives #2, #4). KnnRouter
   (92% GT accuracy, primary) + Rust SystemRouter (88%) for tier routing;
   ContextualBandit (Thompson sampling) for model selection; OxiZ QF_LIA
   SMT for quality labeling (zero heuristics). ComplexityRouter heuristic
   (34%) is dead code (emergency fallback only).
5. **5-stage pipeline** (CLASSIFY → DECOMPOSE → SELECT_TOPOLOGY → ASSIGN →
   EXECUTE → LEARN). 12 templates (sequential, parallel, AVR, selfmoa,
   hierarchical, hub, debate, brainstorming, robust, horizon_pipeline,
   parallel_fanout, formal_solver) + MAP-Elites archive + LLM synthesis
   + CMA-ME mutation + MCTS search.

## Critical directives (CLAUDE.md + .claude/rules)
1. Rust first, Python tolerant
2. Minimal heuristics (learned/verified > research-backed > banned)
3. No corporate proxy (no `verify=False`)
4. kNN router is primary (92% GT)
5. Evidence before assertions (run tests+benches before claiming)
6. No training-leak model hardcodes (cards.toml + Context7/provider docs only)

## Test baseline (2026-04-24)
- Rust: **502 tests pass** with `cargo test --features smt --lib`
  (topology::controller 34, topology::templates 20 incl A7 capability-hygiene,
  sandbox 9 incl wasm_python cache_tests, plus 439 others). One parallel-
  race flake in `wasm_python::cache_tests::cold_miss` — passes in isolation.
- Python: ~2290 pass / 45 skipped / 11 pre-existing failures in
  API-key-dependent files (`test_e2e_live_providers.py`,
  `test_provider_pool_wiring.py`, `test_pydantic_ai_integration.py`
  path inherited, `test_pydantic_ai_integration.py` lives at
  `tests/providers/`).
- Recent session additions (2026-04-23/24): diff-verifier (11),
  diff-verifier repair mode (1 wire test), CRLF normalize (5), A7
  template capability hygiene (1 Rust + 0 Python equivalent),
  A10 search_repo MemoryError guard (1), A8 Phase 2 ThinkingPart
  roundtrip (2).

## Recent ships (last 7 commits)
- `820ea3e2` roadmap close A8 P2, ticket A12
- `df150a2a` A8 P2 — reasoning_content / ThinkingPart passthrough
- `debb0018` A9+A11 — gpt-5.5/5.5-pro + deepseek-v4-flash/v4-pro cards
- `62438400` A3 repair-mode implementation
- `f4df576f` A10 search_repo MemoryError guard
- `c17ffd68` A8 P1 kimi-k2.5 → k2.6 migration
- `a0fb4c97` A7 N=6 verification smoke (kimi-400 path closed, 4/6 PATCH)

## What's already audited / ticketed (reference)
- `docs/audits/2026-04-23-alire-verification.md` — AUDIT.md + AUDIT2.md
  triage (9 claims A..I; 4 confirmed-live fixed same session as A0a-A0d,
  1 ticketed as B8 shipped `4a3e0d1`, 3 orphaned post-ADR-013, 1 partial).
- `roadmap.md` — Horizons A/B/C, living backlog with A0..A12 entries.
- `docs/adr/ADR-013-wasm-sandbox-default.md` — sandbox flip rationale.

## Open runtime quirks / known-not-closed
- **kimi-k2.6 `supports_tools=false`** until live smoke validates
  A8 Phase 2 (empirical gate pending — N=20 repair smoke running).
- **ToolForge HITL gate missing** — `forge.py` `BuildLoop` persists
  forged tools without approval. Ticketed? (check AUDIT3).
- **No OpenTelemetry** — EventBus is in-process, lossy under
  backpressure. B1 in roadmap.
- **No deterministic trace replay** — B2 in roadmap.
- **cost_tracker.budget_usd=0 means unlimited** — task-level hard
  cost cap not enforced at pipeline entry (boot.py:192 passes budget
  but default is unlimited).
