# YGN-SAGE Audit Remediation Plan
**Date:** 2026-03-14
**Source:** 3 independent audits verified against codebase @ `cbea0dd`
**Approach:** 4 thematic sprints, full remediation (4-6 weeks)
**Baseline:** 1306 passed / 9 failed / 107 skipped tests, 487 commits

---

## Strategy

**Approach:** Thematic sprints — each sprint addresses one domain completely (P0-P3 items mixed by theme, not by priority). All 4 axes (integrity, security, architecture, quality) progress in parallel across sprints.

**Ablation posture:** Correct results.md immediately to match JSON artifact. Re-run at N>=100 in parallel. Replace corrected numbers with re-run results when available.

**Dependency chains:** S3-B depends on S3-A (streaming needs phase decomposition). S2-E depends on S1-D (harvest ablation results). S1-E run depends on S1-E design. All other items are independent.

---

## Sprint S1: Integrity & Evidence (Week 1)

### S1-A: Correct ablation documentation (2h)

**Problem:** `results.md` and `paper2_sage_system.md` publish ablation numbers that contradict the JSON artifact. The JSON (commit a933296) shows no-memory=100%, no-avr=100%, no-guardrails=100%. The documentation claims 90%, 90%, 95%. The original commit message says "Memory, AVR, guardrails neutral on pure code gen."

**Files to modify:**
- `docs/benchmarks/results.md` (lines 101-114)
- `docs/papers/paper2_sage_system.md` (lines 75-88)
- `CLAUDE.md` (Benchmark Results table)

**Changes:**
1. Replace ablation table with JSON-aligned values: no-memory=100%, no-avr=100%, no-guardrails=100%
2. Replace attribution text with: "Framework adds +15pp total over bare baseline. On the 20-task code benchmark, routing showed +5pp isolated contribution. Memory, AVR, and guardrails showed no isolated delta on code tasks — their value may emerge at larger scale or on non-code workloads. Re-run at N>=100 with statistical tests pending."
3. Add caveat admonition in results.md: "Small sample (N=20). Per-pillar attribution requires confirmation at larger scale."

**Acceptance criteria:**
- All 3 files align with the JSON artifact
- No claim of per-pillar contribution unsupported by data
- Commit message: "fix: align ablation docs with JSON artifact (honest attribution)"

### S1-B: Fix model="unknown" in benchmark artifacts (4h)

**Problem:** All benchmark summary JSONs have `"model": "unknown"`. Non-reproducible results.

**Files to modify:**
- `sage-python/src/sage/bench/runner.py` — add model metadata fields to `BenchReport`
- `sage-python/src/sage/bench/evalplus_bench.py` — capture model ID from AgentSystem config
- `sage-python/src/sage/boot.py` — expose resolved model ID in AgentSystem

**Changes:**
1. `BenchReport` dataclass already has `model`, `provider`, `git_sha`, `feature_flags` fields — but callers never populate them. Fix: ensure all bench adapters (evalplus, ablation, swebench, etc.) pass real values at call sites. Add `temperature: float = 0.0` as new field.
2. `boot.py`: add `AgentSystem.model_info` property returning dict with resolved model_id, provider, tier from the active config
3. `evalplus_bench.py`: read model info from AgentSystem and pass to existing BenchReport fields
4. One-shot script `scripts/fix_benchmark_artifacts.py`: patch existing JSON files to add `"model": "gemini-2.5-flash"` (the model actually used, confirmed from config/models.toml + env vars)

**Acceptance criteria:**
- Future benchmark runs produce artifacts with real model ID (existing `model` field populated, not "unknown")
- Existing artifacts patched with correct model info
- `model` field is never "unknown" in any new JSON output

### S1-C: Fix documentation drift (2h)

**Problem:** Docstrings and docs disagree with code on routing stages, blocked call counts.

**Files to modify:**
- `sage-python/src/sage/strategy/adaptive_router.py` — docstring line 1
- `CLAUDE.md` — Tool Security section
- `README.md` — if routing stage count mentioned

**Changes:**
1. `adaptive_router.py` docstring: "5-stage" → "4-stage learned routing pipeline (stage 3 reserved for future online learning)"
2. Blocked identifier counts: current docs say "23 modules + 11 calls" but actual `validator.rs` has 23 modules + 21 calls + 20 dunders. Update all references to match the actual `BLOCKED_MODULES`, `BLOCKED_CALLS`, and `BLOCKED_DUNDERS` arrays in `validator.rs` (count from source, not from audit reports)
3. Fix all "5-stage" mentions across the codebase — includes `adaptive_router.py`, `CLAUDE.md`, `README.md`, and `docs/architecture/*.md`. Use `grep -r "5-stage"` to find all occurrences

**Acceptance criteria:**
- `grep -r "5-stage"` returns zero results (or only historical references)
- Blocked call documentation matches `BLOCKED_MODULES`, `BLOCKED_CALLS`, and `BLOCKED_DUNDERS` arrays in `validator.rs` exactly (counts derived from source code, not hardcoded)

### S1-D: Launch ablation re-run N=100 (setup 2h, compute ~8h background)

**Problem:** Current ablation uses N=20 — too small for statistical significance. Need N>=100 with proper statistical tests.

**Files to modify:**
- `sage-python/src/sage/bench/ablation.py` — add statistical test output

**Changes:**
1. Add `--limit 100` support (already works, just needs a larger run)
2. Add McNemar's test (pairwise, per config pair) to the report JSON
3. Add bootstrap 95% CI (10,000 resamples) per config
4. Add Cohen's d effect size per config pair
5. Launch as background process: `python -m sage.bench --type ablation --limit 100`
6. Results consumed in S2-E

**Acceptance criteria:**
- Ablation report includes p-values, CI, effect sizes
- N>=100 tasks evaluated per configuration
- Results stored in `docs/benchmarks/2026-03-XX-ablation-study-n100.json`

**Important caveat:** This re-run will almost certainly confirm memory/AVR/guardrails are neutral on code tasks — the original commit message already says this. The N=100 run is necessary for statistical rigor, but it will NOT prove framework value on code tasks alone. See S1-E for the real proof.

### S1-E: Design heterogeneous evaluation set (3 days)

**Problem:** HumanEval/MBPP are single-turn code generation. The framework overhead (routing, memory, guardrails, topology) literally cannot help on these tasks — a raw LLM call suffices. To prove the framework adds value, we need tasks where routing, memory, and guardrails actually matter.

**Files to create:**
- `sage-python/config/heterogeneous_eval.json` — 50 human-labeled tasks
- `sage-python/src/sage/bench/heterogeneous_bench.py` — evaluation adapter

**Changes:**
1. Design a 50-task evaluation set with diversity:
   - 15 code tasks (HumanEval subset — baseline comparison)
   - 15 reasoning tasks (GSM8K-hard, logic puzzles — S2 routing should help)
   - 10 multi-turn tasks (conversations requiring episodic memory — memory tier should help)
   - 10 research/synthesis tasks (requiring ExoCortex/RAG — S3 routing + memory should help)
2. Human-label each task with expected S1/S2/S3 routing and expected benefit from each pillar
3. Run full ablation (6 configs) on this heterogeneous set
4. If memory/AVR/guardrails still show zero delta on the heterogeneous set, that is a honest finding — publish it

**Acceptance criteria:**
- 50 tasks with human labels, domain rationale, and expected pillar benefit
- Ablation results on heterogeneous set with McNemar's test
- Either proves framework value on non-code tasks, or honestly confirms framework is overhead

### S1-F: README benchmark section rewrite (2h)

**Problem:** The README SOTA comparison table compares Gemini Flash results against O1/GPT-4o — an apples-to-oranges comparison. After S1-A/S1-B corrections, the README still oversells.

**Files to modify:**
- `README.md` — benchmark section

**Changes:**
1. Remove cross-model SOTA comparison table (or clearly caveat: "Different model tiers. Not a direct comparison.")
2. Replace with honest framework-vs-baseline framing: "YGN-SAGE with Gemini 2.5 Flash achieves X% vs raw Gemini 2.5 Flash at Y% — framework delta is Zpp"
3. Lead with the ablation result (framework contribution) rather than absolute score

**Acceptance criteria:**
- No uncaveated cross-model comparisons
- Framework value is presented as delta-over-baseline, not absolute leaderboard position

---

## Sprint S2: Security & Hygiene (Week 2)

### S2-A: Sandbox hardening (4h)

**Problem:** SEC-02 (env var injection) and SEC-03 (allow_local bypass) confirmed by audit verification.

**Files to modify:**
- `sage-python/src/sage/sandbox/manager.py`

**Changes:**
1. Replace all 6 `asyncio.create_subprocess_shell()` calls with `asyncio.create_subprocess_exec()` using argument lists
2. Add env key validation: `re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key)` — reject keys with shell metacharacters
3. Add env value sanitization: `shlex.quote(value)` for values passed to Docker
4. Gate `allow_local` behind `SAGE_ALLOW_LOCAL_EXEC=1` environment variable. When not set, `_execute_local()` raises `SecurityError("Local execution disabled. Set SAGE_ALLOW_LOCAL_EXEC=1 to enable.")`
5. Log WARNING at SandboxManager init when allow_local is active

**Note:** The `docker exec` call needs `sh -c` *inside* the container. Use `create_subprocess_exec("docker", "exec", container_id, "sh", "-c", command)` — this eliminates host-side shell parsing while preserving the in-container shell intent.

**Acceptance criteria:**
- Zero `create_subprocess_shell` calls remain in sandbox/manager.py
- Env vars with metacharacters in keys are rejected with ValueError
- `allow_local=True` without env var flag raises SecurityError
- Existing tests pass (may need `SAGE_ALLOW_LOCAL_EXEC=1` in test env)

### S2-B: Dashboard auth warning (30min)

**Problem:** Dashboard runs without authentication by default. A log warning nobody reads doesn't fix the problem.

**Files to modify:**
- `ui/app.py`
- `ui/static/index.html`

**Changes:**
1. At startup (after `DASHBOARD_TOKEN` is read), if empty: `logger.warning("Dashboard running without authentication. Set SAGE_DASHBOARD_TOKEN for production use.")`
2. Inject a visible banner in the dashboard HTML when no token is configured: add a top-bar `<div>` with yellow background and text "No authentication configured — set SAGE_DASHBOARD_TOKEN". The `/api/status` endpoint already exists — add an `auth_enabled: bool` field and use it client-side.

**Acceptance criteria:**
- Starting dashboard without token produces a visible WARNING in logs
- Dashboard UI shows a prominent yellow banner when auth is disabled

### S2-C: Dead code cleanup (2h)

**Problem:** ebpf.rs (173 LOC, dep commented), simd_sort.rs (misleading name).

**Files to modify:**
- Delete `sage-core/src/sandbox/ebpf.rs`
- Rename `sage-core/src/simd_sort.rs` → `sage-core/src/sort_utils.rs`
- Update `sage-core/src/lib.rs` — remove ebpf module, update simd_sort references
- Update `sage-core/src/sandbox/mod.rs` — remove ebpf re-export

**Changes:**
1. Delete `ebpf.rs` entirely (dead code: dep commented, module commented, PyO3 commented)
2. Rename `simd_sort.rs` to `sort_utils.rs`. Update docstring: "Sorting utilities — stdlib pdqsort wrapper. No SIMD (vqsort-rs does not yet support Windows)."
3. Update all `use crate::simd_sort` to `use crate::sort_utils`
4. Verify `cargo build --no-default-features` and `cargo test --no-default-features` pass

**Acceptance criteria:**
- `ebpf.rs` no longer exists
- No file named `simd_sort` in the repo
- All Rust tests pass

### S2-D: Security automation (4h)

**Problem:** No CodeQL, Dependabot, or SBOM automation.

**Files to create:**
- `.github/dependabot.yml`
- `.github/workflows/security.yml`

**Changes:**
1. `dependabot.yml`: weekly updates for pip (sage-python), cargo (sage-core), github-actions
2. `security.yml`: CodeQL analysis job (Python + Rust), runs on push to master and weekly cron
3. Add `cyclonedx-bom` generation step in CI, upload as artifact
4. Keep this separate from main `ci.yml` to avoid slowing down the primary pipeline

**Acceptance criteria:**
- Dependabot creates PRs for dependency updates
- CodeQL runs on every push to master
- SBOM artifact available in CI

### S2-E: Harvest ablation re-run results (2h)

**Problem:** S1-D launched the re-run. Now consume results.

**Files to modify:**
- `docs/benchmarks/results.md`
- `docs/papers/paper2_sage_system.md`
- `CLAUDE.md`

**Changes:**
1. If re-run complete: replace N=20 tables with N=100 results including p-values and CI
2. If not complete: document "re-run in progress" and revisit in S3
3. Update per-pillar attribution based on actual statistical significance

**Acceptance criteria:**
- Ablation claims backed by N>=100 data with statistical tests
- Or documented as "pending" with target date

### S2-F: Fix silent degradation warnings (2h)

**Problem:** Several fallback paths are fully silent (confirmed by audit: A3-2, A3-13). Dashboard mock, S-MMU context empty return, and Rust import skips produce no warning.

**Files to modify:**
- `ui/app.py` — dashboard mock creation (lines 33-79)
- `sage-python/src/sage/memory/smmu_context.py` — empty string return path
- `sage-python/src/sage/boot.py` — silent Rust import skips (lines 26-34)

**Changes:**
1. `ui/app.py`: add `logger.warning("sage_core not available — dashboard using mock components")` when mock module is created
2. `smmu_context.py`: when returning empty string due to exception, log `logger.warning("S-MMU context retrieval failed — returning empty context")`
3. `boot.py`: add `logger.info("sage_core.SystemRouter not available — using Python fallback")` for each silent Rust import skip

**Acceptance criteria:**
- All fallback paths produce at least a WARNING or INFO log message
- No fully silent degradation remains in boot, memory, or dashboard paths

### S2-G: Document Rust as progressive enhancement (1h)

**Problem:** Audit finding A2-A1 confirmed Rust is optional everywhere. This is architecturally correct (graceful degradation) but undocumented.

**Files to modify:**
- `CLAUDE.md` — Architecture section
- `ARCHITECTURE.md` (if exists)

**Changes:**
1. Add a subsection: "Rust is a progressive enhancement, not a core dependency. All subsystems have Python fallbacks. Running without `maturin develop` gives a pure-Python system with reduced performance. Rust adds: native SIMD embeddings, sub-ms SMT verification, zero-copy Arrow memory, and compiled routing."

**Acceptance criteria:**
- The dual Python/Rust nature is explicitly documented as intentional design

---

## Sprint S3: Architecture (Weeks 3-4)

### S3-A: Decompose agent_loop.py (4-5 days)

**Problem:** 955 LOC monolith with ~528-line run() method (lines 410-937). Blocks streaming, testability.

**Files to create:**
- `sage-python/src/sage/phases/__init__.py` — `LoopContext` dataclass
- `sage-python/src/sage/phases/perceive.py` — ~120 LOC
- `sage-python/src/sage/phases/think.py` — ~150 LOC
- `sage-python/src/sage/phases/act.py` — ~100 LOC
- `sage-python/src/sage/phases/learn.py` — ~100 LOC

**Files to modify:**
- `sage-python/src/sage/agent_loop.py` — reduce to ~250-300 LOC orchestrator (includes `__init__`, `_emit`, `_compute_aio`, `_rebuild_messages` helpers)

**Changes:**
1. Define `LoopContext` dataclass:
   - `task: str`, `messages: list`, `step: int`, `done: bool`
   - `routing_decision: RoutingDecision`, `cost: float`
   - `tool_calls: list`, `has_tool_calls: bool`
   - `result_text: str`, `guardrail_results: list`
2. Extract perceive logic: routing, input guardrails, S-MMU context, code-task detection
3. Extract think logic: LLM call, AVR retry, S2/S3 escalation, stagnation detect
4. Extract act logic: tool execution, sandbox dispatch, CEGAR repair
5. Extract learn logic: output guardrails, memory write, entity extraction, cost, drift
6. `run()` becomes a thin while loop calling phase functions
7. CircuitBreakers and helper methods (`_emit`, `_compute_aio`, `_rebuild_messages`) stay in AgentLoop (cross-cutting concerns)
8. EventBus emission stays via `_emit()` in each phase (receives bus reference)

**Rollback strategy:** Keep the original `run()` as `_run_legacy()` during transition. Add a `SAGE_AGENT_LOOP_LEGACY=1` env var flag that routes to `_run_legacy()`. Remove legacy path only after integration tests confirm behavioral parity across a full benchmark run (EvalPlus smoke test N=20).

**Acceptance criteria:**
- `agent_loop.py` is <=300 LOC (orchestrator + helpers + __init__)
- Each phase module is <=200 LOC
- All 1306 existing tests pass unchanged
- `run(task) -> str` interface unchanged (backward compatible)
- No new dependencies
- `_run_legacy()` preserved behind env var flag until parity confirmed

### S3-B: Add streaming support — Phase 1: non-AVR tasks (3 days)

**Depends on:** S3-A (phases must be extractable to yield events)

**Scope limitation (deliberate):** Streaming only for tasks that do NOT trigger AVR retry loops. Code tasks with AVR continue using the non-streaming `run()` path. Rationale:
- **Tool call streaming** requires buffering tool-call JSON mid-stream, executing it, and injecting the result back. This is where every ADK's streaming gets complex.
- **AVR retry interaction**: streaming through a retry loop means either (a) silently restarting after streaming partial output (terrible UX) or (b) emitting RETRY events with client-side handling (needs protocol design).
- **Output guardrail interaction**: if guardrails reject a streamed response, we've already sent garbage to the user.

These are real engineering problems that deserve their own sprint (S5), not shortcuts in S3.

**Files to modify:**
- `sage-python/src/sage/agent_loop.py` — add `run_stream()`
- `sage-python/src/sage/llm/google.py` — add `generate_stream()`
- `sage-python/src/sage/llm/codex.py` — add `generate_stream()`
- `sage-python/src/sage/providers/openai_compat.py` — add `generate_stream()`
- `ui/app.py` — consume async generator for WebSocket push

**Changes:**
1. Add `generate_stream(messages, config) -> AsyncGenerator[str, None]` to each LLM provider
2. Add `run_stream(task) -> AsyncGenerator[AgentEvent, None]` to AgentLoop
3. `run_stream` yields `AgentEvent` at phase transitions and per-token during THINK
4. **Non-AVR path only**: if task triggers AVR (code detection), fall back to `run()` and yield a single COMPLETE event at the end
5. WebSocket endpoint in `ui/app.py` consumes the generator
6. Event types: PERCEIVE_START, THINK_DELTA, THINK_COMPLETE, LEARN_COMPLETE

**Acceptance criteria:**
- `run_stream()` yields per-token events for non-code tasks
- Code tasks with AVR gracefully fall back to batch mode (single COMPLETE event)
- `run()` still returns `str` (backward compatible)
- Dashboard shows real-time streaming for non-code tasks
- LLM providers that don't support streaming fall back to single yield

### S3-C: OpenTelemetry instrumentation (3 days)

**Independent of S3-A/B.**

**Files to modify:**
- `sage-python/pyproject.toml` — add `[otel]` extra
- `sage-python/src/sage/boot.py` — OTel init
- `sage-python/src/sage/agent_loop.py` (or phases/) — manual spans

**Files to create:**
- `sage-python/src/sage/telemetry.py` — OTel setup helper

**Changes:**
1. New optional dependency group `[otel]`: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `traceloop-sdk`
2. `telemetry.py`: `init_telemetry(service_name="ygn-sage")` — inits OTel tracer + traceloop auto-instrumentation. No-op if deps not installed.
3. `boot.py`: call `init_telemetry()` in boot sequence (try/except, graceful skip)
4. Manual spans in agent loop phases:
   - `invoke_agent {agent_name}` (top-level)
   - `gen_ai.chat {model}` (per LLM call — auto via traceloop)
   - `execute_tool {tool_name}` (per tool call)
   - `perceive`, `think`, `act`, `learn` (per phase)
5. Attributes: `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.agent.name`
6. Configure via standard env vars: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`

**Acceptance criteria:**
- `pip install -e ".[otel]"` enables instrumentation
- Without `[otel]` extra, zero overhead (graceful skip)
- LLM calls produce OTel spans with token counts
- Spans visible in any OTel-compatible backend (Jaeger, Grafana)
- EventBus continues to work for WebSocket (complementary, not replaced)

### S3-D: Rust/Python duplication resolution (2 days)

**Problem:** Routing, memory, topology, and verification each exist in both Rust and Python. The remediation plan (S3-C OTel, S4-A mypy, S4-C constrained decoding) only improves the Python path, making Rust drift further into irrelevance. S2-G "document as progressive enhancement" accepts the problem without fixing it.

**Decision:** Explicitly deprecate Rust components that are pure duplication. Keep only the Rust components that provide genuine value unavailable in Python.

**Rust components to KEEP (genuine value):**
- `verification/smt.rs` — OxiZ verifier (sub-0.1ms, pure Rust SMT, no Z3 C++ dep)
- `verification/ltl.rs` — LTL model checker (petgraph, always compiled)
- `memory/mod.rs` + `arrow_tier.rs` — Arrow-backed working memory (zero-copy, SIMD)
- `memory/embedder.rs` — ONNX embedder (native SIMD, 768-dim)
- `topology/topology_graph.rs` — TopologyGraph IR (petgraph, typed nodes/edges)
- `topology/templates.rs` — 8 topology templates
- `topology/verifier.rs` — HybridVerifier (structural + semantic + LTL)
- `topology/map_elites.rs` — MAP-Elites archive (quality-diversity)
- `sandbox/validator.rs` — tree-sitter AST validation
- `sandbox/wasm.rs` — Wasm WASI sandbox
- `sandbox/tool_executor.rs` — ToolExecutor pipeline

**Rust components to DEPRECATE (pure duplication with Python):**
- `routing/router.rs` — Rust AdaptiveRouter (Python AdaptiveRouter is the maintained version with kNN, 5-stage)
- `routing/system_router.rs` — Rust SystemRouter (Python ComplexityRouter/AdaptiveRouter is better tested)
- `routing/smmu_bridge.rs` — routing-to-SMMU bridge (low value, S-MMU reads work without it)
- `topology/executor.rs` — Rust TopologyExecutor (Python TopologyRunner is the active code path)
- `topology/engine.rs` — Rust DynamicTopologyEngine (Python boot.py Phase 6 is the active wiring)

**Changes:**
1. Mark deprecated Rust modules with `#[deprecated(since = "0.2.0", note = "Use Python equivalent")]` on PyO3 classes
2. Remove Python `try: import sage_core.SystemRouter` fallbacks for deprecated modules — use Python directly
3. Keep Python fallbacks ONLY for the "keep" list above
4. Update S2-G documentation to reflect this decision

**Acceptance criteria:**
- Deprecated Rust modules are marked with `#[deprecated]`
- Python code no longer tries to import deprecated Rust modules
- "Keep" Rust modules still work via PyO3 bindings
- No functional regression (Python paths were already the active paths)

---

## Sprint S4: Quality & SOTA (Weeks 5-6)

### S4-A: Reduce mypy ignores (7 days)

**Problem:** 27 core modules with `ignore_errors = true` in pyproject.toml. The PyO3 boundary modules will have deeply embedded `Any` types that resist clean typing.

**Files to modify:**
- `sage-python/pyproject.toml` (mypy overrides)
- 27 Python modules (type annotations)

**Changes:**
Two waves (Wave 3 deferred if time runs short):
1. **Wave 1 (critical path, 3 days):** `boot.py`, `orchestrator.py`, `agent_loop.py` (post-decomp phases/) — fix type annotations, remove from ignore list
2. **Wave 2 (memory/routing, 3 days):** `memory/embedder.py`, `memory/smmu_context.py`, `memory/working.py`, `strategy/metacognition.py`, `strategy/knn_router.py`, `strategy/adaptive_router.py`
3. **Wave 3 (best-effort, 1 day):** `llm/google.py`, `llm/codex.py`, `guardrails/builtin.py`, `evolution/engine.py`, `sandbox/manager.py`, and remaining modules

**Rules:**
- Fix real type issues (missing annotations, unhandled Optional, Any abuse)
- Do NOT add `# type: ignore` to suppress — fix the actual problem
- If a module genuinely can't be typed (e.g., dynamic PyO3 imports), keep it in ignore list with justification comment
- Target: reduce from 27 to <=8 ignored modules (Wave 1+2 complete = ~15 modules fixed)
- PyO3 boundary modules (those importing sage_core dynamically) are expected to remain ignored

**Acceptance criteria:**
- `mypy src/` passes with <=8 modules in ignore list
- Each remaining ignore has a comment explaining why (typically: "PyO3 dynamic imports")
- No regressions in pytest
- Wave 1+2 complete; Wave 3 is best-effort

### S4-B: Protocol conformance tests (2 days)

**Problem:** MCP and A2A servers have zero tests.

**Files to create:**
- `sage-python/tests/test_protocols_conformance.py`

**Changes:**
1. MCP test: start FastMCP server, list tools, call `run_task` with a simple prompt, verify non-empty response
2. A2A test: build AgentCard, verify skills and capabilities, send message via AgentExecutor, verify response
3. Use mock LLM provider (no real API calls needed)
4. Mark as `@pytest.mark.integration` (not e2e — no external services needed)

**Acceptance criteria:**
- MCP server starts, lists tools, executes a task
- A2A executor receives and processes a message
- Tests pass in CI without API keys

### S4-C: Wire constrained decoding (2 days)

**Problem:** JSON mode exists in provider code but is never used in the agent loop.

**Files to modify:**
- `sage-python/src/sage/llm/google.py`
- `sage-python/src/sage/providers/openai_compat.py`
- `sage-python/src/sage/contracts/planner.py`
- `sage-python/src/sage/topology/llm_caller.py`

**Changes:**
1. Add `response_schema: dict | None = None` to `LLMConfig`
2. `google.py`: when `config.response_schema` is set, pass `response_mime_type="application/json"` + `response_schema` to Gemini API
3. `openai_compat.py`: when set, pass `response_format={"type": "json_schema", "json_schema": schema}`
4. Use in `TaskPlanner` (DAG decomposition expects JSON) and `llm_caller.py` (topology synthesis expects JSON)
5. `SchemaGuardrail` remains as post-hoc validation safety net

**Acceptance criteria:**
- TaskPlanner produces valid JSON via constrained decoding (not post-hoc parsing)
- Topology LLM synthesis uses structured output
- Providers without JSON mode support fall back gracefully

### S4-D: Offline evolution CLI (3 days)

**Problem:** Evolution is dead code at runtime. Should be repositioned as offline tool.

**Files to modify:**
- `sage-python/src/sage/agent_loop.py` — remove `_auto_evolve` flag and evolution hooks
- `sage-python/src/sage/boot.py` — remove `loop._auto_evolve = False` line

**Files to create:**
- `sage-python/src/sage/evolution/cli.py`

**Changes:**
1. Remove from `agent_loop.py`: `self._auto_evolve` attribute, evolution stats emission block (~15 lines)
2. Remove from `boot.py`: `loop._auto_evolve = False` line
3. Create CLI entry point:
   ```
   python -m sage.evolution optimize-topology --trainset data/routing_gt.json --budget 50
   python -m sage.evolution optimize-prompts --trainset data/quality_triples.jsonl --rounds 10
   ```
4. CLI drives MAP-Elites archive (Rust) in batch mode
5. Document in CLAUDE.md: "Evolution is an offline development tool. Use `python -m sage.evolution` to optimize topologies and prompts against a training set."

**Acceptance criteria:**
- Zero evolution code in agent_loop.py runtime path
- `python -m sage.evolution --help` shows available commands
- MAP-Elites archive still functional via CLI
- All existing tests pass

### S4-E: Benchmark expansion — non-code benchmarks (3 days)

**Problem:** All current benchmarks are code tasks. The framework shows no value-add on code tasks (ablation neutral). Adding more code benchmarks (BigCodeBench) won't change this. We need benchmarks that exercise routing, memory, and guardrails.

**Files to create:**
- `sage-python/src/sage/bench/gaia_bench.py` — GAIA general assistant tasks adapter
- OR `sage-python/src/sage/bench/tau_bench.py` — τ-bench tool-use benchmark adapter

**Changes:**
1. Evaluate GAIA (general assistant tasks: reasoning, web research, multi-step planning) and τ-bench (tool-use benchmark) feasibility. Pick whichever has simpler harness integration.
2. GAIA exercises: S2/S3 routing (complex tasks), memory (multi-step context), guardrails (structured output)
3. Register in `bench/__main__.py`: `--type gaia` or `--type taubench`
4. Run chosen benchmark once and publish results with framework-vs-baseline comparison
5. BigCodeBench remains as backlog item — it's valuable but doesn't address the core problem

**Acceptance criteria:**
- At least one non-code benchmark adapter exists and runs
- Results show framework contribution (or honest negative) on non-code tasks
- Results include model ID, provider, ablation delta

### S4-F: Property-based tests (2 days)

**Problem:** Tests are mostly example-based. No property-based testing.

**Files to modify:**
- `sage-python/pyproject.toml` — add `hypothesis` to dev deps

**Files to create:**
- `sage-python/tests/test_properties.py`

**Changes:**
1. Add `hypothesis` to `[dev]` dependencies
2. Property tests for router: arbitrary string input → output is always valid S1/S2/S3 or None
3. Property tests for memory: store(key, value) then retrieve(key) → returns value (round-trip)
4. Property tests for sandbox validator: arbitrary Python code string → validate() returns bool without crashing
5. Property tests for S-MMU: register_chunk with arbitrary metadata → chunk_count increases
6. Target: 10-15 property tests

**Acceptance criteria:**
- `pytest tests/test_properties.py` passes
- Each property test runs >= 100 examples
- No crashes on adversarial inputs

### S4-G: Real API tests in CI (2 days)

**Problem:** Audit finding A2-C4 confirmed: 48.6% of tests use mocks, zero real API tests run in CI. 14 test files with `@pytest.mark.e2e` exist but are all skipped.

**Files to modify:**
- `.github/workflows/ci.yml` — add gated integration test job

**Changes:**
1. Add a new CI job `integration-smoke` that runs only when `GOOGLE_API_KEY` secret is available
2. Job runs: `pytest -m e2e --limit 5 -x` (5 tasks max, fail-fast)
3. Gate behind `if: github.event_name == 'push' && github.ref == 'refs/heads/master'` (only on master push, not on PRs)
4. Use repository secret `GOOGLE_API_KEY`. For SSL bypass: document that `REQUESTS_CA_BUNDLE=""` is required due to corporate proxy injecting self-signed certificates — this is a dev environment constraint, not a security shortcut. Ideally, provision the actual corporate CA cert via `NODE_EXTRA_CA_CERTS` / `REQUESTS_CA_BUNDLE=/path/to/corp-ca.pem` if available.

**Acceptance criteria:**
- At least 5 real API tests run on master push in CI
- Job is optional (workflow continues if it fails)
- No API keys exposed in logs
- SSL bypass is documented with rationale

### S4-H: Fuzz testing for OxiZ parser (0.5 days)

**Problem:** `smt.rs` has a hand-rolled recursive descent parser with a 100-depth recursion limit. This is one of the best pieces of code in the repo — it deserves protection. Natural target for cargo-fuzz.

**Files to create:**
- `sage-core/fuzz/Cargo.toml`
- `sage-core/fuzz/fuzz_targets/fuzz_smt_parser.rs`

**Changes:**
1. Add `cargo-fuzz` target for `verify_invariant(arbitrary_pre, arbitrary_post)`
2. Add target for `verify_arithmetic_expr(arbitrary_expr)`
3. Run for ~30 minutes during CI or locally
4. Fix any crashes found

**Acceptance criteria:**
- `cargo fuzz run fuzz_smt_parser` runs without crashes for 30 minutes
- Any found crashes are fixed in `smt.rs`

### S4-I: Dependency pinning audit (1h)

**Problem:** `Cargo.toml` uses semver ranges (`pyo3 = "0.25"`, `tokio = "1"`, `serde = "1"`). `Cargo.lock` exists for reproducibility, but `sage-python/pyproject.toml` may have loose pins.

**Changes:**
1. Verify `Cargo.lock` is committed (it should be for a binary/library crate)
2. Check `pyproject.toml` for loose version ranges on critical deps
3. Pin major+minor for critical deps (`google-genai>=1.5,<2`, `openai>=1.50,<2`, etc.)
4. Ensure `pip install -e ".[all,dev]"` still resolves cleanly

**Acceptance criteria:**
- All critical Python deps have upper-bound pins
- `Cargo.lock` is committed and up-to-date

### Explicitly deferred

**Mutation testing** (P3-6 from verification report): deferred to post-remediation. LOW priority, requires property tests (S4-F) as prerequisite. Tool: mutmut or cosmic-ray. Will evaluate after S4-F results.

**S5: Full streaming with AVR/tool-call/guardrail integration**: deferred to post-remediation. Requires protocol design for RETRY events, client-side handling of guardrail rejections on partial streams, and tool-call buffering. Depends on S3-B (Phase 1 streaming) being stable in production.

**BigCodeBench adapter**: deferred — valuable but doesn't address the framework value-add question. S4-E (GAIA/τ-bench) is higher priority.

---

## Dependency Graph

```
S1-A (ablation docs) ──────────────────────────────────────┐
S1-B (model field) ─────────────────────────────────────────┤
S1-C (doc drift) ───────────────────────────────────────────┤
S1-D (ablation re-run code) ─────────────── S2-E (harvest) ┤
S1-E (heterogeneous eval design) ─── S1-E run (in S2/S3) ──┤
S1-F (README rewrite) ─────────────────────────────────────┤
                                                            │
S2-A (sandbox hardening) ───────────────────────────────────┤
S2-B (dashboard auth + banner) ────────────────────────────┤
S2-C (dead code cleanup) ──────────────────────────────────┤
S2-D (security automation) ────────────────────────────────┤
S2-F (silent degradation) ─────────────────────────────────┤
S2-G (document Rust as progressive) ───────────────────────┤
                                                            │
S3-A (decompose agent_loop) ──── S3-B (streaming Phase 1) ─┤
S3-C (OpenTelemetry) ──────────────────────────────────────┤
S3-D (Rust/Python dedup resolution) ───────────────────────┤
                                                            │
S4-A (mypy fixes, 7d) ────────────────────────────────────┤
S4-B (protocol tests) ────────────────────────────────────┤
S4-C (constrained decoding) ──────────────────────────────┤
S4-D (evolution CLI) ─────────────────────────────────────┤
S4-E (GAIA/τ-bench, not BigCodeBench) ────────────────────┤
S4-F (property tests) ────────────────────────────────────┤
S4-G (real API tests CI) ─────────────────────────────────┤
S4-H (OxiZ fuzz testing) ────────────────────────────────┤
S4-I (dependency pinning) ───────────────────────────────┘
```

Hard dependencies:
- **S3-B depends on S3-A** (streaming needs phase decomposition)
- **S1-E heterogeneous eval run depends on S1-E design** (design task set first, run in S2/S3)
- **S2-E depends on S1-D** (harvest ablation results)

Everything else is independent.

---

## Success Criteria (Exit)

The remediation is complete when ALL of the following are true:

1. **Zero unsupported claims**: every benchmark number in docs matches a JSON artifact with model ID
2. **Zero known security vulnerabilities**: no `create_subprocess_shell`, env vars validated, allow_local gated
3. **agent_loop.py <= 300 LOC**: phases extracted, streaming functional for non-AVR tasks, legacy fallback available
4. **Ablation at N>=100**: with McNemar's test, 95% CI, honest per-pillar attribution
5. **Heterogeneous eval exists**: 50-task set (code + reasoning + multi-turn + research) with ablation results
6. **Mypy ignores <= 8**: with justification for each remaining (PyO3 boundary modules expected)
7. **Security automation active**: CodeQL + Dependabot + SBOM in CI
8. **OTel instrumented**: LLM calls produce spans with token counts
9. **Protocol tests passing**: MCP + A2A conformance tests in CI
10. **Dead code removed**: no ebpf.rs, no simd_sort.rs, no _auto_evolve in runtime, `python -m sage.evolution --help` succeeds
11. **Rust duplication resolved**: deprecated Rust modules marked `#[deprecated]`, Python fallbacks cleaned
12. **README honest**: no uncaveated cross-model SOTA comparisons, framework value as delta-over-baseline
13. **OxiZ fuzz-tested**: `cargo fuzz run` passes 30 minutes without crashes
14. **Deps pinned**: critical Python deps have upper-bound pins, `Cargo.lock` committed
15. **All existing tests still pass**: 1306+ passed, 0 new failures

---

## Estimated Effort

| Sprint | Duration | Items | Days (sequential) | Notes |
|--------|----------|-------|--------------------|-------|
| S1 | Week 1 | 6 items (A-F) | 6 | S1-E (3d) dominates; A/B/C/F are half-days; ~1.5d sprint overhead |
| S2 | Week 2 | 7 items (A-G) | 2 | Light sprint; absorbs S1-E overflow |
| S3 | Weeks 3-4 | 4 items (A-D) | 13 | S3-C/D parallel with S3-A; S3-B sequential after S3-A |
| S4 | Weeks 5-6 | 9 items (A-I) | 22 | S4-A (7d) parallel with B-I; H/I are sub-day |
| **Total** | **6 weeks** | **25 items** | **43 days seq / ~30 days parallel** |

Buffer: 5 days for unknowns (ablation compute time, mypy surprises, streaming edge cases, fuzz findings).

**Explicitly deferred:** Mutation testing (post-remediation, requires S4-F). Full streaming S5 (post S3-B stability). BigCodeBench (valuable but doesn't address framework value-add).
