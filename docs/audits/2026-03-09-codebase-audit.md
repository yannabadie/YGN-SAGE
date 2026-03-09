# YGN-SAGE — Exhaustive Codebase Audit (2026-03-09)

> **Auditor:** Claude Opus 4.6 (recursive inspection + 4 external audits + oracle consensus + ExoCortex)
> **Repository:** `github.com/yannabadie/YGN-SAGE` @ master
> **Codebase:** Python ~31K LOC | Rust ~2.9K LOC | 1079 tests passing
> **Sources:** 4 independent audit reports, deep Python SDK inspection (41 issues), ExoCortex research (5 queries), Context7 library checks

---

## Executive Summary

YGN-SAGE is a **real, substantive prototype** with genuine engineering across Rust core, Python SDK, memory tiers, guardrails, and a working dashboard. It is **not vaporware**. Cross-audit synthesis identified **68 total issues**: 16 CRITICAL, 27 HIGH, 17 MEDIUM, 8 LOW. Of the prior 7 CRITICALs from earlier audits, 6 have been remediated. The deep Python SDK audit surfaced 9 additional CRITICALs.

### Prior Audit Fix Status (Sprints 1-3)

| Finding | Severity | Status |
|---------|----------|--------|
| Z3-01: Dead z3_validator.rs | CRITICAL | **FIXED** — file deleted |
| Z3-02: check_loop_bound always False | CRITICAL | **FIXED** — documented + constrained |
| Z3-03: verify_arithmetic ignores expr | CRITICAL | **FIXED** — ast.literal_eval + _safe_z3_eval |
| Z3-05: z3_topology fabricated proofs | CRITICAL | **FIXED** — renamed to topology_verifier.py |
| Z3-06: Evolution Z3 gate validates config | CRITICAL | **FIXED** — gate removed |
| Z3-07: has_z3=False rubber stamps | CRITICAL | **FIXED** — fail-closed + logging.error |
| SEC-01: create_python_tool 6 bypass vectors | CRITICAL | **OPEN** — still registered in boot.py |
| SEC-02: create_bash_tool shell=True | HIGH | **OPEN** — still uses shell=True templates |
| SEC-03: run_bash create_subprocess_shell | HIGH | **FIXED** — no create_subprocess_shell found |
| EVO-03: DGM naming misleading | MEDIUM | **FIXED** — renamed to _sampo_ |
| RTG-01: Routing benchmark circular | HIGH | **PARTIAL** — documented but no ground-truth set |
| MEM-01: S-MMU not exposed to Python | HIGH | **OPEN** — still behind WorkingMemory |
| TST-03: CI no sandbox/ONNX tests | MEDIUM | **OPEN** |
| BLD-01: No workspace lints | MEDIUM | **OPEN** |

### New Findings from Deep Python SDK Audit (41 issues)

| Category | CRITICAL | HIGH | MEDIUM |
|----------|----------|------|--------|
| Bottlenecks | 1 | 3 | 2 |
| Blind Spots (Logic Errors) | 3 | 3 | 2 |
| Concurrency | 1 | 2 | 1 |
| Memory Leaks | 1 | 4 | 1 |
| Error Handling | 1 | 3 | 0 |
| Architecture | 1 | 3 | 2 |
| Type/Validation | 0 | 2 | 0 |
| Miscellaneous | 0 | 1 | 2 |
| **Total** | **9** | **19** | **13** |

---

## REMAINING CRITICAL: SEC-01 — create_python_tool

**File:** `sage-python/src/sage/boot.py:204-206`
**Status:** Still registered in boot sequence.

```python
from sage.tools.meta import create_python_tool, create_bash_tool
tool_registry.register(create_python_tool)
tool_registry.register(create_bash_tool)
```

**Risk:** 6 AST bypass vectors allow arbitrary code execution by LLM agent. The AST check only catches direct `exec()`/`eval()` calls. Bypasses include `__import__('os').system()`, `getattr(__builtins__, 'exec')`, `compile()`, metaclass injection, lambda + import, and `open()`.

**Mitigated by:** Rust ToolExecutor (tree-sitter validator with 23 blocked modules + 11 blocked calls) exists but is NOT wired to `create_python_tool`. The Python-side tool creation path is completely separate from the Rust sandbox.

**Fix:** Disable registration in boot.py immediately. Future: wire through ToolExecutor.

---

## Bottlenecks Identified

### 1. Python GIL vs Rust SIMD (Amdahl's Law)

The Rust data plane (Arrow memory, SIMD sort, lock-free queue) provides ~0.05ms speedups on operations where LLM latency is ~800ms. The PyO3 serialization overhead (2-5ms) can exceed the savings. The Rust core's real value is in **ToolExecutor** (security), **RustEmbedder** (ONNX native inference), and **S-MMU** (structured memory) — not raw compute speed.

**Action:** Focus Rust on security boundaries and I/O-bound operations, not compute micro-optimization.

### 2. S2 AVR Sandbox Mismatch

The S2 empirical validation path creates a `Sandbox()` but default config has `allow_local=False` and no Docker. The sandbox returns failure, causing unnecessary retries and S2→S3 escalations driven by configuration, not model quality. Fixed partially with SLF-based AVR (syntax-first validation + stagnation detection) but the core sandbox wiring remains broken.

**Action:** Wire ToolExecutor as the default execution backend for S2 AVR.

### 3. Memory Retrieval Degradation Chain

```
RustEmbedder (ONNX) → sentence-transformers → SHA-256 hash (non-semantic)
```

The 3-tier fallback silently degrades to hash-based embeddings that the code itself says are "not semantically meaningful". When this happens, S-MMU semantic retrieval returns noise, and the RelevanceGate (threshold=0.3) may still pass irrelevant context into the prompt.

**Action:** Log embedding tier at boot. If hash fallback is active, disable semantic retrieval entirely.

### 4. Mock-Heavy Test Suite

91% of Python tests use mocks. Unit test mocks are acceptable, but there's no integration test tier that validates real component interactions. The E2E proof script (25/25) partially fills this gap but is not in CI.

**Action:** Add `tests/integration/` directory with real-component tests (no mocks). Add E2E proof to CI (with `GOOGLE_API_KEY` secret).

---

## Blind Spots Identified

### 1. No Windows CI Despite Windows-Specific Code

`discover_ort_dylib()` has Windows-specific paths (`C:\Python3*`, `%APPDATA%\Python`). The GIL deadlock fix (`py.allow_threads()`) was specifically for Windows `LoadLibraryW`. CI only runs on `ubuntu-latest`.

### 2. `std::env::set_var()` Race Condition (Rust ≥1.80)

`embedder.rs:134` and `router.rs:260` call `std::env::set_var()` which is `unsafe` in Rust ≥1.80 when multiple threads exist. Currently mitigated by PyO3 GIL serialization, but technically UB.

### 3. Provider Capability Semantics Leak

`OpenAICompatProvider` reports `structured_output=False, tool_role=False` but the capability matrix claims several providers support these. The abstraction leaks: tool messages are rewritten into user messages with a warning.

### 4. ExoCortex Hardcoded Store

`DEFAULT_STORE = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"` is a single Google File Search resource. Multi-tenancy/isolation not addressed. Already has `SAGE_EXOCORTEX_STORE` env var override but the default couples to one account.

### 5. Dashboard Stale Import

Dashboard `ui/app.py` may still reference `MetacognitiveController` (renamed to `ComplexityRouter`).

---

## Context7 Findings (Latest Library Versions)

| Library | Current | Latest | Notes |
|---------|---------|--------|-------|
| PyO3 | 0.23+ | Latest | `py.allow_threads()` → `py.detach()` rename in newer versions |
| wasmtime | v36 LTS | v38 | v36 LTS supported until Aug 2027 |
| google-genai | ≥1.20 | 1.33 | No breaking changes for File Search API |
| ort | 2.0.0-rc.12 | 2.0.0-rc.12 | Still pre-release RC |

---

## Deep Python SDK Findings (Top 10 Most Impactful)

### CRITICAL: Unvalidated Tool Arguments (agent_loop.py:606)
`tool.execute(kwargs)` where kwargs = LLM-generated. No schema validation before execute. Malformed JSON → TypeError → crash.

### CRITICAL: Race Condition in ModelRegistry (registry.py:108)
`self._profiles` updated by `refresh()` without lock. Concurrent `list_available()` → TOCTOU race → partial model list.

### CRITICAL: Silent Type Coercion in DAGExecutor (executor.py:130)
Runner output type-checked as dict but keys not validated against `node.output_schema`. Silently accepts empty results.

### CRITICAL: Bare Exception Catch in EvolutionEngine (engine.py:144)
`except Exception: continue` swallows all errors during mutation. Silent failure = incorrect evolution trajectory.

### HIGH: Unbounded Messages List (agent_loop.py:301)
`messages` grows via append every retry + escalation. After S2→S3, can reach 50+ elements = 100KB+ context.

### HIGH: O(n^2) Adjacency Rebuild in SemanticMemory (semantic.py:60)
Every relation eviction rebuilds entire adjacency dict. At max_relations=10k, each eviction is O(10k).

### HIGH: EventBus Consumer Loop Hangs (bus.py:131)
`while True: await q.get()` with no timeout. Stale WebSocket consumers never cleaned up.

### HIGH: _trajectories Unbounded in EvolutionEngine (engine.py:81)
Appended every mutation, never cleared. 50 generations × 10 mutations = 500 entries.

### HIGH: God Object in AgentLoop (agent_loop.py:163)
11 injected components, 600+ line __init__+run(). Changes to any subsystem require modifying AgentLoop.

### HIGH: S2_MAX_RETRIES Mismatch (agent_loop.py:108 vs 204)
Module constant = 2, instance variable = 3. Escalation happens at retry 3, not 2. Logic diverges from docs.

---

## ExoCortex Research Correlation

Research papers from ExoCortex map directly to identified issues:

| Research Finding | YGN-SAGE Component | Gap |
|-----------------|---------------------|-----|
| MetaScaffold: cascading failures in MAS | CircuitBreaker (6 breakers) | Tracks consecutive failures but no trajectory-level anomaly detection |
| MAST: 14 error patterns taxonomy | EventBus AgentEvent schema | Could extend event types to tag MAST categories for post-hoc diagnosis |
| Metacognitive self-correction (MASC) | ComplexityRouter + AVR | SLF-based AVR is partial; full MASC adds mid-trajectory error detection |
| Memory inflation / soul erosion (BMAM) | S-MMU bounded scan + RelevanceGate | Bounded scan (128 chunks) addresses inflation; BMAM-style salience scoring could improve |
| Static vs dynamic verification | Z3 contracts + sandbox | Z3 verifies plans statically; runtime sandbox catches execution; gap = emergent multi-step behavior |
| G-Memory hierarchical 3-tier graph | S-MMU multi-view graph | Similar architecture but S-MMU lacks insight/query separation |
| Chain-of-Memory (CoM) adaptive truncation | MEM1 compressor | Compressor does pressure-triggered compression but no adaptive truncation |

---

## Remediation Priority

### P0 — This Week (Security + Correctness)
1. **SEC-01**: Disable `create_python_tool` + `create_bash_tool` in boot.py
2. Wire `ToolExecutor` as default execution backend for dynamic tools
3. Add tool argument schema validation before execute (agent_loop.py:606)
4. Add threading.Lock to ModelRegistry._profiles updates
5. Fix bare `except Exception: continue` in EvolutionEngine

### P1 — Next 2 Weeks (Stability + Performance)
6. Add Windows CI job (`runs-on: windows-latest`)
7. Add integration test directory (no mocks)
8. Wire ToolExecutor to S2 AVR sandbox path
9. Log embedding tier at boot, gate semantic retrieval on ONNX availability
10. Add timeout to EventBus consumer loop (`asyncio.wait_for`)
11. Bound messages list in agent_loop.run() (sliding window)
12. Fix S2_MAX_RETRIES constant vs instance variable mismatch
13. Refactor AgentLoop: extract SystemContext dataclass

### P2 — Month 2 (Architecture)
14. Add workspace Rust lints (`unsafe_code = "deny"`, clippy pedantic)
15. Expose `PyMultiViewMMU` directly to Python
16. Add non-circular routing benchmark with ground-truth labels
17. Fix `std::env::set_var()` race condition (use `OnceLock` pattern)
18. Optimize SemanticMemory adjacency rebuild (lazy updates)
19. Bound _trajectories in EvolutionEngine (deque maxlen=100)
20. Add MAST error categories to AgentEvent schema (from ExoCortex research)

### P3 — Continuous
21. Reduce doc/code ratio (archive stale plans)
22. Update dashboard imports
23. Provider capability conformance tests
24. Add CoM-style adaptive truncation to MEM1 compressor
