# YGN-SAGE Audit Verification Report
**Date:** 2026-03-14
**Methodology:** 4-phase rigorous verification against codebase @ `cbea0dd`
**Sources audited:** Audit1.md (anonymous), Audit2.md (5.4/10), Audit3.md (overclaimed prototype)

---

## Executive Summary

38 verifiable assertions extracted from 3 independent audits. Each verified against actual source code with exact line references.

**Audit reliability scores:**
| Audit | Assertions | Confirmed | Partial | False | Reliability |
|-------|-----------|-----------|---------|-------|-------------|
| Audit 1 | 6 | 0 | 2 | 4 | **33%** — alarmist, major technical errors |
| Audit 2 | 16 | 8 | 5 | 3 | **81%** — most reliable, nuanced |
| Audit 3 | 13 | 4 | 3 | 6 | **54%** — mix of valid and stale claims |

**Critical discovery:** The ablation study data (JSON artifact) contradicts the published results (results.md, paper). The JSON shows no-memory/no-avr/no-guardrails = 100%, while documentation claims 90%/90%/95%. The original commit message (a933296) confirms: "Memory, AVR, guardrails neutral on pure code gen." The "+10pp AVR, +10pp memory" attribution is unsupported by raw data.

---

## I. COMPLETE VERIFICATION MATRIX

### Legend
- ✅ Confirmed problem (audit is correct)
- ⚠️ Partially true (nuanced reality)
- ❌ False/outdated (audit is wrong)

### Audit 1: "Fundamentally flawed architecture" (Reliability: 33%)

| ID | Assertion | Verdict | Evidence |
|----|-----------|---------|----------|
| A1-1 | LTL state space explosion (10^11 states) blocks Z3 at runtime | ❌ **FALSE** | LTL uses BFS/DFS O(V+E) on petgraph DiGraph. No state enumeration. `ltl.rs` lines 143-301: four graph-level checks, all polynomial. The auditor confused LTL model checking over state machines with LTL property checking over topology graphs. |
| A1-2 | Arrow for graph traversal = 10-15x cache thrashing | ❌ **FALSE** | Arrow stores columnar DATA (timestamps, content, embeddings). Graph structure uses native petgraph DiGraph (`smmu.rs` line 64-69: `graph: DiGraph<ChunkMetadata, MultiEdge>`). No pointer chasing on Arrow columns. |
| A1-3 | eBPF requires CAP_SYS_ADMIN, TOCTOU vulnerability | ⚠️ **THEORETICAL** | eBPF code is dead (dep commented in Cargo.toml, module commented in lib.rs, PyO3 exports commented). True if reactivated, but currently unreachable. |
| A1-4 | CMA-ME 5000 evals = 2.7h + $25 at inference time | ❌ **FALSE** | Evolution runs offline only. `engine.rs` generate() uses pre-computed archive/S-MMU lookups (6-path strategy), never triggers CMA-ME at inference time. `_auto_evolve=False` in boot.py confirms. |
| A1-5 | kNN embedding collision (string reverse vs Navier-Stokes) | ❌ **FALSE** | kNN uses snowflake-arctic-embed-m (768-dim semantic embeddings), not structural features. 92% accuracy on 50 GT tasks (vs 52% heuristic). Explicitly refuses hash embeddings (`knn_router.py` line 150). The auditor's "cosine similarity ~0.85+" claim is unsubstantiated. |
| A1-6 | MAB assumes stationary reward, collapses | ⚠️ **PARTIAL** | Decay factor exists in `bandit.rs` lines 86-96 (BetaPosterior.update with multiplicative decay). Not fully sliding-window but has exponential forgetting. Default decay value needs verification at call sites. |

**Audit 1 summary:** 0/6 assertions fully confirmed. 4/6 are technically wrong — the auditor applied theoretical arguments without reading the actual implementation. The LTL, Arrow, CMA-ME, and kNN claims are all category errors.

### Audit 2: "5.4/10" (Reliability: 81%)

| ID | Assertion | Verdict | Evidence |
|----|-----------|---------|----------|
| A2-C1 | model="unknown" in benchmark artifacts | ✅ **CONFIRMED** | Both `2026-03-10-evalplus-humaneval-v2-full-summary.json` and `2026-03-10-evalplus-mbpp-full-summary.json` have `"model": "unknown"`. Routing breakdown: S1:0 S2:156/378 S3:0 — all tasks routed to S2. |
| A2-C2 | Routing GT labels circular (derived from heuristic) | ❌ **FALSE** | `config/routing_ground_truth.json` has `labeling_method: "human_expert"`, 50 tasks with detailed domain rationale. Distribution: 10 S1 + 20 S2 + 20 S3. Labels explicitly NOT reverse-engineered from heuristic. The 30/30 self-consistency test is a DIFFERENT benchmark. |
| A2-C3 | Z3/OxiZ verifies constants, not programs | ⚠️ **PARTIAL** | `prove_memory_safety(addr, limit)` = constant bounds check (true). But `verify_invariant(pre, post)` = universal implication over ALL free variables (real). `verify_invariant_with_feedback()` provides clause-level diagnostics. `synthesize_invariant()` is simplified CEGAR (weakening-only). Mix of trivial and genuine verification. |
| A2-C4 | 39% tests use mocks, 0 real API tests in CI | ⚠️ **PARTIAL** | Actually 48.6% (69/142 files use mocks — worse than claimed). 14 API test files exist but all skipped in CI via `@pytest.mark.e2e`. 0 real API tests run in CI = TRUE. |
| A2-C5 | Wasm sandbox present but unwired | ❌ **FALSE** | `_execute_wasm()` fully implemented in `sandbox/manager.py` lines 60-108. Route path exists (line 70). Rust ToolExecutor has Wasm Component Model support (`tool_executor.rs`). The claim was outdated. |
| A2-C6 | Evolution = dead code (_auto_evolve=False) | ✅ **CONFIRMED** | `boot.py` line 401: `loop._auto_evolve = False`. `agent_loop.py` line 890: evolution stats guarded by `self._auto_evolve`. Code is functional but intentionally gated off. Original commit a933296 says "Memory, AVR, guardrails neutral." |
| A2-S1 | Shell injection via Docker container_id | ❌ **FALSE** | Container IDs are hex strings from `docker run` stdout — inherently safe from shell injection. No user-controlled input reaches container_id. |
| A2-S2 | Env var injection in config.env | ⚠️ **PARTIAL** | `sandbox/manager.py` line 258: env vars interpolated without validation. Vulnerable if attacker controls `config.env` dict. Low probability but real vector. |
| A2-S3 | allow_local=True = full host access | ✅ **CONFIRMED** | `allow_local=True` enables `asyncio.create_subprocess_shell(command)` with zero sandboxing. One boolean from RCE. Mitigated by requiring trusted command source. |
| A2-S4 | eBPF dead code (ebpf.rs) still present | ✅ **CONFIRMED** | `ebpf.rs` = 173 LOC. Cargo dep, module, and PyO3 exports all commented out. Dead weight. |
| A2-A1 | Dual Python/Rust, Rust optional everywhere | ✅ **CONFIRMED** | 5+ modules have `try: import sage_core except ImportError` fallbacks. System works fully without Rust. |
| A2-A2 | agent_loop.py = 955 LOC monolith | ✅ **CONFIRMED** | 955 LOC. `run()` = 529 lines (lines 410-938, 55% of file). 10 methods total. Classic God Object. |
| A2-A3 | simd_sort.rs has no SIMD | ✅ **CONFIRMED** | Uses `sort_unstable_by()` (pdqsort). Zero SIMD intrinsics. Comment: "When vqsort-rs supports Windows, swap." Name is aspirational, not implemented. |
| A2-G1 | No constrained decoding | ✅ **CONFIRMED** | JSON mode exists in provider code but never used in agent loop. `SchemaGuardrail` available but not default. No Outlines/LMFE integration. |
| A2-G2 | A2A/MCP = placeholders without tests | ⚠️ **PARTIAL** | Substantial implementations (A2A: AgentExecutor + AgentCard, MCP: FastMCP + tools). But ZERO conformance tests. Not stubs — incomplete in testing. |
| A2-G3 | No streaming in agent loop | ✅ **CONFIRMED** | `run()` returns `str`. Zero yield/AsyncIterator. No `run_stream()` method. |

### Audit 3: "Overclaimed research prototype" (Reliability: 54%)

| ID | Assertion | Verdict | Evidence |
|----|-----------|---------|----------|
| A3-1 | Ablation shows most subsystems = 0 delta | ⚠️ **CRITICAL** | JSON: no-memory=100%, no-avr=100%, no-guardrails=100%. results.md claims 90%, 90%, 95%. **Data contradicts documentation.** Commit a933296 confirms: "Memory, AVR, guardrails neutral." |
| A3-2 | Silent degradation everywhere | ⚠️ **PARTIAL** | Loud warnings for sage_core/memory/routing. Silent fallback for dashboard mock + S-MMU context returns empty string. Mixed behavior. |
| A3-3 | README vs code disagreement (routing stages) | ⚠️ **CONFIRMED** | Docstring says 5-stage. Code implements 0/0.5/1/2. Stage 3 = "reserved/not implemented". Documentation drift. |
| A3-4 | 50 GT labels not shipped | ❌ **FALSE** | `config/routing_ground_truth.json` has 50 complete tasks with human-expert labels and rationale. |
| A3-5 | Mypy ignores on core modules | ✅ **CONFIRMED** | 27 core modules with `ignore_errors = true` in pyproject.toml. Includes boot, orchestrator, memory, routing, guardrails, evolution. |
| A3-6 | execute_raw bypasses validation | ✅ **CONFIRMED (mitigated)** | `execute_raw()` intentionally skips tree-sitter. But only called on pre-validated code (`validate()` called first at call site). WARN log emitted. |
| A3-7 | No CodeQL/Dependabot/SBOM | ✅ **CONFIRMED** | Zero security automation. No `.github/dependabot.yml`, no CodeQL workflow, no SBOM generation. |
| A3-8 | Blocked call count stale (23+11) | ⚠️ **STALE (code is better)** | Actual counts derived from `validator.rs` source arrays (verify at implementation time). Documentation understates the actual protection. |
| A3-9 | Dashboard auth optional | ✅ **CONFIRMED** | `ui/app.py` line 126: empty `SAGE_DASHBOARD_TOKEN` = `return` (no auth). Explicit dev-mode design. |
| A3-10 | Benchmark bar obsolete | ❌ **FALSE** | SWE-Bench Lite added (swebench_bench.py). HumanEval+, MBPP+, GSM8K, SWE-Bench, ablation = reasonable diversity. Missing: BigCodeBench, GAIA, AgentDojo. |
| A3-11 | Doc/Code ratio > 1.0x | ❌ **FALSE** | 26.5K docs vs 39.3K code = 0.67x. Code exceeds docs. |
| A3-12 | 54 commits for ~100K LOC | ❌ **FALSE** | 487 commits (9x more). 77.5K LOC (not 100K). |
| A3-13 | Provider discovery silently skips failures | ⚠️ **PARTIAL** | Rust imports: silent skip. Provider discovery: warning logged. Model registry: warning logged. Dashboard mock: fully silent. |

---

## II. CRITICAL FINDING: ABLATION DATA INTEGRITY

### The Discrepancy

| Source | full | no-routing | no-guardrails | no-avr | no-memory | baseline |
|--------|------|-----------|---------------|--------|-----------|----------|
| **JSON artifact** (primary evidence) | 100% | 95% | **100%** | **100%** | **100%** | 85% |
| **results.md** (published) | 100% | 95% | 95% | 90% | 90% | 85% |
| **paper2** (published) | 100% | 95% | 95% | 90% | 90% | 85% |

### Evidence trail
- **Commit a933296** (2026-03-10 01:25:41 UTC): Created JSON with message "Memory, AVR, guardrails neutral on pure code gen (expected)"
- **Commit 089684f** (2026-03-13 20:59:10 UTC): Created results.md **3 days later** with different numbers
- **No second ablation run found** — only one JSON artifact exists

### Impact
The published claim "each pillar contributes measurably: AVR +10pp, memory +10pp, routing +5pp, guardrails +5pp" is **not supported by the raw data**. The JSON shows:
- Routing: +5pp (only confirmed contribution)
- Memory: +0pp (no measurable delta)
- AVR: +0pp (no measurable delta)
- Guardrails: +0pp (no measurable delta)
- Baseline → full: +15pp (correctly stated, but attribution is wrong)

### Required action
1. Correct results.md and paper2 to match JSON data
2. Re-run ablation at larger scale (N≥100) with statistical tests
3. If re-run confirms neutrality, update claims to honest: "Framework adds +15pp, primarily from routing and orchestration overhead reduction"

---

## III. DIVERGENCES FROM ORIGINAL AUDITS

### Where Audit 1 is WRONG

| Claim | Why it's wrong |
|-------|---------------|
| LTL state space 10^11 | Confuses LTL model checking (state enumeration) with LTL property checking (graph BFS). SAGE uses O(V+E) graph algorithms. |
| Arrow cache thrashing 10-15x | Confuses data storage format with graph traversal algorithm. Graph is petgraph DiGraph; Arrow stores the content. |
| CMA-ME 5000 evals at runtime | Evolution is offline only. generate() uses pre-computed archives. |
| kNN embedding collision | Contradicted by 92% accuracy on 50 GT tasks. arctic-embed-m is a semantic encoder, not syntactic. |
| eBPF TOCTOU vulnerability | True in theory but code is dead (dep commented). |
| Firecracker/Kùzu/vLLM recommendations | Irrelevant — SAGE is a research ADK, not a multi-tenant cloud service. |

### Where Audit 2 is WRONG

| Claim | Why it's wrong |
|-------|---------------|
| Routing GT is circular | 50-task file has `labeling_method: "human_expert"` with rationale. The 30/30 self-consistency test is a different benchmark. |
| Wasm unwired | `_execute_wasm()` fully implemented with route path. |
| Shell injection via container_id | Container IDs are hex strings from docker stdout — safe by construction. |

### Where Audit 3 is WRONG

| Claim | Why it's wrong |
|-------|---------------|
| 50 GT labels not shipped | `config/routing_ground_truth.json` has all 50 with rationale. |
| Doc/Code ratio > 1.0x | Actual: 0.67x (code > docs). |
| 54 commits | Actual: 487 commits. |
| Benchmark bar obsolete | SWE-Bench was already added. |

### Where ALL THREE audits agree (and are RIGHT)

1. **model="unknown"** in benchmark artifacts — non-reproducible
2. **Evolution is dead code** — functional but never called
3. **No streaming** — table-stakes missing
4. **Security surface** — sandbox has real vulnerabilities (env injection, allow_local)
5. **agent_loop.py is too large** — 955 LOC monolith
6. **Mypy ignores on core modules** — type safety theater
7. **No OTEL/structured observability** — custom EventBus only

---

## IV. PRIORITIZED ACTION PLAN

### P0: Trust & Security (Do Now — 3 days)

| # | Action | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| P0-1 | **Fix ablation data integrity**: correct results.md and paper2 to match JSON artifact. Add caveat: "Memory/AVR/guardrails neutral on 20-task code benchmark, re-run at scale needed" | `docs/benchmarks/results.md`, `docs/papers/paper2_sage_system.md` | 1h | CRITICAL — scientific integrity |
| P0-2 | **Set model field in benchmark artifacts**: modify bench runner to record actual model ID, provider, temperature, feature flags, git SHA | `sage-python/src/sage/bench/runner.py`, `sage-python/src/sage/bench/evalplus_bench.py` | 4h | CRITICAL — reproducibility |
| P0-3 | **Fix env var injection**: validate config.env keys against `^[A-Za-z_][A-Za-z0-9_]*$`, reject shell metacharacters in values, use `create_subprocess_exec` instead of `_shell` | `sage-python/src/sage/sandbox/manager.py` | 4h | HIGH — security |
| P0-4 | **Gate allow_local behind explicit flag**: require `SAGE_ALLOW_LOCAL_EXEC=1` env var, log WARNING when active | `sage-python/src/sage/sandbox/manager.py` | 2h | HIGH — security |

### P1: Evidence & Trust (1 week — 5 days)

| # | Action | Files | Effort | Impact | Depends |
|---|--------|-------|--------|--------|---------|
| P1-1 | **Re-run ablation at scale**: N≥100, McNemar's test, 95% CI, report p-values | `sage-python/src/sage/bench/ablation.py` | 2 days | HIGH — proves (or disproves) framework value | P0-1, P0-2 |
| P1-2 | **Delete dead evolution hooks**: remove `_auto_evolve`, evolution stats emission from agent_loop. Keep engine as offline tool. | `agent_loop.py`, `boot.py` | 4h | HIGH — reduces confusion |
| P1-3 | **Clean dead code**: delete `ebpf.rs`, rename `simd_sort.rs` to `sort_utils.rs`, update doc count (24+24+20) | `sage-core/src/sandbox/ebpf.rs`, `sage-core/src/simd_sort.rs` | 2h | MEDIUM — housekeeping |
| P1-4 | **Fix documentation drift**: align routing stage count (4 active, not 5), update blocked-call counts | `adaptive_router.py` docstring, README | 2h | MEDIUM — honesty |
| P1-5 | **Dashboard auth warning**: log WARNING on startup when SAGE_DASHBOARD_TOKEN is empty | `ui/app.py` | 30min | MEDIUM — operational safety |

### P2: Engineering Quality (2 weeks — 10 days)

| # | Action | Files | Effort | Impact | Depends |
|---|--------|-------|--------|--------|---------|
| P2-1 | **Decompose agent_loop.py**: extract perceive/think/act/learn into separate modules (~150 LOC each). Define LoopContext dataclass for shared state. | `agent_loop.py` → `phases/` | 4-5 days | HIGH — unblocks streaming, testability | — |
| P2-2 | **Add streaming**: `run_stream() -> AsyncGenerator[AgentEvent]` yielding events per phase. Add `generate_stream()` to LLM providers. | `agent_loop.py`, `llm/google.py`, `llm/codex.py` | 3-4 days | HIGH — table-stakes | P2-1 |
| P2-3 | **Add OpenTelemetry**: `traceloop-sdk` for auto-instrumentation + manual spans for PERCEIVE/THINK/ACT/LEARN. GenAI semantic conventions. | `boot.py`, new `telemetry.py` | 3 days | HIGH — observability | — |
| P2-4 | **Wire constrained decoding**: pass `response_schema` to provider APIs where JSON output expected. SchemaGuardrail for validation. | `llm/google.py`, `orchestrator.py` | 2 days | MEDIUM — reliability | — |
| P2-5 | **A2A/MCP conformance tests**: test with Claude Desktop (MCP), Google ADK client (A2A) | `tests/test_protocols_*.py` | 2 days | MEDIUM — interop | — |
| P2-6 | **Add security automation**: CodeQL workflow, Dependabot config, SBOM via CycloneDX | `.github/workflows/`, `.github/dependabot.yml` | 1 day | MEDIUM — supply chain | — |

### P3: SOTA Gaps (1 month — 20 days)

| # | Action | Files | Effort | Impact | Depends |
|---|--------|-------|--------|--------|---------|
| P3-1 | **Reduce mypy ignores**: fix type errors in core modules one by one (boot→orchestrator→memory→routing) | 27 modules | 5 days | HIGH — code quality | — |
| P3-2 | **Real API tests in CI**: add gated integration test job with secret GOOGLE_API_KEY, 5-task smoke test | CI workflow, tests/ | 2 days | MEDIUM — coverage | — |
| P3-3 | **Offline evolution CLI**: `python -m sage.evolution --optimize prompts --trainset data.json` with DSPy-compatible interface | `evolution/cli.py` | 3 days | MEDIUM — makes evolution useful | P1-2 |
| P3-4 | **Benchmark expansion**: add BigCodeBench, GAIA, or AgentDojo adapter | `bench/` | 5 days | HIGH — credibility | P0-2 |
| P3-5 | **Property-based tests**: Hypothesis for router, memory, sandbox | `tests/` | 3 days | MEDIUM — test quality | — |
| P3-6 | **Mutation testing**: mutmut/cosmic-ray to measure test suite quality | CI pipeline | 2 days | LOW — meta-quality | P3-5 |

### Total effort estimate: ~38 days
- P0 (immediate): 3 days
- P1 (week 1): 5 days
- P2 (weeks 2-3): 10 days
- P3 (weeks 4-7): 20 days

---

## V. RECOMMENDATIONS

### For the developer (Yann)

1. **Fix ablation integrity FIRST** — this is the most damaging finding. Correct docs to match data, or re-run with larger N.
2. **model="unknown" is the lowest-hanging fruit** — 4 hours to fix, massive credibility gain.
3. **Don't rename S-MMU** — the functionality is correct, nomenclature is secondary. Auditors obsess over naming; users care about behavior.
4. **Ignore Audit 1's recommendations** (Firecracker, Kùzu, vLLM PagedAttention) — they're for a different product class. SAGE is a research ADK, not a multi-tenant cloud service.
5. **Evolution is correctly gated** — the honest negative result (-10pp) is published. Convert to offline-only and document the boundary.
6. **The Rust layer IS valuable** — Audit 2's "optional Rust" finding is architecturally correct (graceful degradation), not a flaw. But document it as "progressive enhancement, not core dependency."

### For future auditors

1. **Read the actual code**, not just file names. Audit 1 made 4/6 errors by theorizing from concepts without checking implementations.
2. **Check git history** for evidence trail. The ablation discrepancy was findable via `git log --all --oneline | grep ablation`.
3. **Distinguish "dead code" from "gated code"** — `_auto_evolve=False` is intentional disablement with evidence (negative result), not abandoned code.
4. **Verify claims against artifacts**, not docs. The JSON is the source of truth, not results.md.
