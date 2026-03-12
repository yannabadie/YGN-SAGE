# YGN-SAGE Audit Response v3 — Verified Action Plan

**Date**: 2026-03-12
**Status**: Draft — pending user approval
**Input**: Audit1.md (5.5/10), Audit2.md (7/10 cleanup, 4/10 priorities)
**Methodology**: 36 assertions extracted, 22 confirmed, 10 partially true, 2 infirmed, 2 non-verifiable

---

## CRITICAL AUDIT ERROR: Z3/SMT Is NOT Decorative

Both audits' central thesis — "formal verification is decorative/prompt theater" — is **demonstrably false**.

**Verified call chain:**
1. `agent_loop.py:505` → `self.prm.calculate_r_path(content)`
2. `kg_rlvr.py:262` → `self.kg.verify_step(step)` for each `<think>` block
3. `kg_rlvr.py:96-167` → Direct calls to Rust SmtVerifier:
   - `_rust.prove_memory_safety(addr, limit)`
   - `_rust.check_loop_bound(var_name, hard_cap)`
   - `_rust.verify_arithmetic_expr(expr, expected, tolerance)`
   - `_rust.verify_invariant_with_feedback(pre, post, max_rounds)`
4. `smt.rs` → OxiZ pure-Rust SMT solver with recursive descent parser

The verification IS real. The LLM IS told to include assertions, and those assertions ARE parsed and verified by OxiZ. The audits missed the `ProcessRewardModel` → `kg_rlvr` → `SmtVerifier` call chain.

**What IS true**: If the LLM ignores the prompt and doesn't include assertions, `verify_step()` falls back to heuristic scoring (0.0). The verification is opt-in from the LLM's perspective.

---

## Confirmed Problems — Prioritized Action Plan

### Phase 0 — Credibility Gates (MUST DO FIRST)

#### 0.1 Official EvalPlus Evaluation [CRITICAL, P1]
**Problem**: Custom `evaluate_task()` in `evalplus_bench.py` is not comparable to the official EvalPlus leaderboard.
**Evidence**: No `evalplus.sanitize`, no `evalplus.evaluate`, custom subprocess + output comparison.
**Solution**:
1. Add `evalplus` as a dev dependency
2. Generate `samples.jsonl` from SAGE code generation (capture model output only)
3. Run `evalplus.sanitize --samples samples.jsonl`
4. Run `evalplus.evaluate --dataset humaneval --samples samples.jsonl`
5. Report official scores only. Delete or clearly label custom harness results as "internal"
**Effort**: 1-2 days
**Impact**: Benchmark credibility restored

#### 0.2 Fix CI Hard-Fail [CRITICAL, P5]
**Problem**: `|| true` on integration tests and mypy masks failures.
**Evidence**: `.github/workflows/ci.yml` lines 55 and 72.
**Solution**:
1. Remove `|| true` from both lines
2. For mypy: add `--ignore-missing-imports` (already present) and fix existing type errors incrementally
3. For integration tests: mark flaky tests with `#[ignore]` + separate CI job, don't mask all failures
**Effort**: 1 day
**Impact**: CI becomes trustworthy

#### 0.3 Capture Model ID in Benchmarks [HIGH, P4]
**Problem**: 11/14 benchmark files have `"model": "unknown"`.
**Evidence**: `_last_model` only set after first LLM call, fragile capture.
**Solution**: Set `model_id` from router decision BEFORE benchmark execution, not after.
**Effort**: 0.5 day
**Impact**: Benchmark reproducibility

### Phase 1 — Architectural Fixes

#### 1.1 Unified Execution Decision [CRITICAL, P3]
**Problem**: AgentSystem.run() routes, then CognitiveOrchestrator.run() independently re-routes (confirmed B2).
**Evidence**: boot.py:131-150 (first routing), orchestrator.py:207-209 (independent re-routing).
**Solution**:
1. Define `ExecutionDecision` dataclass: system, model_id, topology_id, budget, guardrail_level
2. `AgentSystem.run()` produces the authoritative `ExecutionDecision`
3. `CognitiveOrchestrator.run()` receives and consumes it — NO re-routing
4. All telemetry (bandit, topology) records against the same decision
**Effort**: 3-5 days
**Impact**: Eliminates split-brain routing, fixes bandit feedback loop

#### 1.2 Capability Self-Test [CRITICAL, P2]
**Problem**: Static capability matrix claims structured_output=True for xAI/DeepSeek/MiniMax/Kimi, but runtime adapter returns False.
**Evidence**: capabilities.py:20-35 (static claims), openai_compat.py:66-75 (runtime contradicts).
**Solution**:
1. Remove static capability matrix from `capabilities.py`
2. Each provider adapter implements `self_test() -> dict[str, bool]` at boot
3. Self-test sends a minimal structured_output request, tool-role message, etc.
4. Only register capabilities that pass self-test
5. Routing rejects providers that don't meet task requirements (fail-closed)
**Effort**: 2-3 days
**Impact**: No more silent capability degradation

#### 1.3 Bandit Provider Diversity [HIGH, P6]
**Problem**: Bandit seeded with only 4 Gemini models (boot.py:256-258).
**Solution**: Seed bandit with ALL models from ModelRegistry, not a hardcoded Gemini list.
**Effort**: 0.5 day
**Impact**: Bandit can explore all providers

#### 1.4 Dead Code Removal [MEDIUM, P10]
**Problem**: `template_store` and `HybridVerifier` instantiated at boot but never used (confirmed B8 + bridge audit).
**Solution**: Remove both from boot.py. Internal TopologyEngine verifier suffices.
**Effort**: 0.5 day

### Phase 2 — Security Hardening

#### 2.1 tree-sitter Blocked List Expansion [HIGH, P8]
**Problem**: `bytes`, `bytearray`, `memoryview`, `ord` not blocked — enables dynamic module name construction.
**Evidence**: validator.rs BLOCKED_CALLS list missing these 4.
**Solution**: Add to BLOCKED_CALLS: `bytes`, `bytearray`, `memoryview`, `ord`, `ascii`.
**Effort**: 0.5 day
**Impact**: Closes known bypass vector

#### 2.2 Sandbox Default Warning [HIGH, P7]
**Problem**: Sandbox disabled by default, no warning to user.
**Solution**: At boot, if no sandbox (Docker/Wasm) is available, emit a prominent WARNING log: "Code execution sandbox unavailable. Tool execution will fail unless allow_local=True."
**Effort**: 0.5 day

### Phase 3 — Benchmark Rigor

#### 3.1 Benchmark Reproducibility [MEDIUM, P12]
**Problem**: 50%-100% variance on 20-task subsets.
**Solution**:
1. Fix temperature=0, top_p=1.0 for all benchmark runs
2. Run 5+ repetitions per configuration
3. Report mean + 95% CI
4. Add `--seed` parameter to benchmark CLI
**Effort**: 1 day

#### 3.2 Cost Tracking Fix [MEDIUM, P14]
**Problem**: All ablation variants report $0 cost.
**Solution**: Audit CostTracker wiring — verify that provider API cost callbacks propagate to BenchReport.
**Effort**: 1 day

#### 3.3 Fix LOC Accounting in Spec [MEDIUM, P9]
**Problem**: Rationalization spec says Python -1% but actual is +12.1%.
**Solution**: Correct the spec document with honest numbers.
**Effort**: 0.5 day

### Phase 4 — Nettoyage

#### 4.1 SubTask.depends_on [LOW, P11]
**Problem**: Dead field, never populated.
**Solution**: Either implement dependency extraction in _parse_subtasks(), or remove the field.
**Effort**: 0.5-2 days depending on choice

#### 4.2 Document Audit Corrections [LOW, P15-P18]
**Problem**: Audit numbers slightly off, design decisions undocumented.
**Solution**: Create `docs/audit-response-v3.md` with corrections and explanations.
**Effort**: 0.5 day

---

## Divergences Between Audits and Verification

| Audit Claim | Audit Verdict | Our Verification | Severity of Error |
|-------------|---------------|------------------|-------------------|
| A6: OxiZ not wired into agent_loop | "Structurally decorative" | **FULLY WIRED** via PRM → kg_rlvr → SmtVerifier | **HIGH** — central thesis invalidated |
| A4: Z3 prompt injection without verification | "Prompt-based theater" | **Verification IS real** when LLM includes assertions | **HIGH** — same thesis |
| A1: 848 mock lines | Stated as fact | **778 lines** (-8.3%) | LOW — approximate |
| A2: 46 sys.modules stubs | Stated as fact | **35 files** (-24%) | MEDIUM — significantly overstated |
| B6: All ablation 20/20 | Stated as fact | **4/6 are 20/20**, baseline=17, no-routing=19 | MEDIUM — key nuance missed |

---

## What Both Audits Got Right

1. **Custom eval harness** — confirmed, must fix (P1)
2. **Dual routing** — confirmed, must fix (P3)
3. **Capability matrix mismatch** — confirmed, must fix (P2)
4. **Bandit Gemini bias** — confirmed, must fix (P6)
5. **CI soft-fail** — confirmed, must fix (P5)
6. **Sandbox disabled by default** — confirmed (P7)
7. **model: unknown** — confirmed (P4)
8. **tree-sitter bypass vectors** — confirmed (P8)
9. **No .cwasm files** — confirmed (A9)
10. **template_store inert** — confirmed (P10)

---

## Estimated Timeline

| Phase | Items | Effort | Cumulative |
|-------|-------|--------|-----------|
| **Phase 0** (Credibility) | P1, P5, P4 | 3 days | Week 1 |
| **Phase 1** (Architecture) | P3, P2, P6, P10 | 6 days | Week 2 |
| **Phase 2** (Security) | P8, P7 | 1 day | Week 2 |
| **Phase 3** (Benchmarks) | P12, P14, P9 | 2.5 days | Week 3 |
| **Phase 4** (Cleanup) | P11, P15-18 | 2 days | Week 3 |
| **Total** | 18 items | **~15 days** | 3 weeks |

---

## Relationship to Rust/Python Rationalization Spec

The rationalization spec (2026-03-12-rust-python-rationalization-design.md) should be **revised** based on audit findings:

1. **system_router.rs migration → CANCELLED** — Audit2 correctly argues this creates a bad coupling boundary. Keep entire routing kernel (features + model_card + model_registry + system_router + bandit) in Rust as a cohesive unit.
2. **llm_synthesis.rs migration → CANCELLED** — Confirmed it uses HybridVerifier internally (C2). Can't extract without breaking topology validation.
3. **Phase 1 reduces to 3 modules**: features.rs (287 LOC), model_card.rs (469 LOC), model_registry.rs (438 LOC) = **1,194 LOC** (not 2,712).
4. **Performance projections → ALL marked TBD** pending baseline measurements.
5. **EventBus migration → CONDITIONAL** on benchmark showing actual bottleneck.

**Net effect**: Rationalization scope shrinks from 5 modules to 3. Phase 2 (Python→Rust hot paths) and Phase 3 (entity graph) are **gated on evidence** from ablation studies proving memory/routing value.
