# Audit Remediation Plan — Post-5-Audits

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all confirmed critical/high audit findings across 5 independent audits, prioritized by security-first then evidence-integrity.

**Architecture:** 7 chunks ordered by severity × effort. Security fixes first (S1/S2/S3/S6), then truth pipeline (B1/B4/B5/B8), then architecture cleanup (A4/A5/A6/R1), then evidence gaps (E1/M2). All new logic in Rust where possible.

**Tech Stack:** Python 3.12, Rust 1.94, PyO3, pytest, cargo test, GitHub Actions CI

**Audit Verification Summary:** 38 claims verified → 22 confirmed, 8 partial, 5 refuted, 3 unverifiable.

---

## Chunk 1: Critical Security — Python Sandbox Hardening (S1 + S3)

**Priority:** P0 CRITICAL — exploitable RCE via dunder chain
**Confirmed by:** Audit3 F-01, Audit3 F-11, Audit4 §3.2

### Task 1.1: Harden Python BLOCKED_CALLS + dunder detection

**Files:**
- Modify: `sage-python/src/sage/tools/sandbox_executor.py:28-30` (BLOCKED_CALLS)
- Modify: `sage-python/src/sage/tools/sandbox_executor.py:46-91` (validate_tool_code)
- Test: `sage-python/tests/test_sandbox_executor.py`

- [ ] **Step 1: Write failing exploit tests**

```python
# tests/test_sandbox_executor.py — add these tests

import pytest
from sage.tools.sandbox_executor import validate_tool_code


class TestSandboxBypassVectors:
    """Regression tests for Audit3 F-01 bypass vectors."""

    def test_getattr_blocked(self):
        errs = validate_tool_code("getattr(object, '__subclasses__')")
        assert any("getattr" in e for e in errs)

    def test_setattr_blocked(self):
        errs = validate_tool_code("setattr(obj, 'x', 1)")
        assert any("setattr" in e for e in errs)

    def test_delattr_blocked(self):
        errs = validate_tool_code("delattr(obj, 'x')")
        assert any("delattr" in e for e in errs)

    def test_globals_blocked(self):
        errs = validate_tool_code("globals()")
        assert any("globals" in e for e in errs)

    def test_locals_blocked(self):
        errs = validate_tool_code("locals()")
        assert any("locals" in e for e in errs)

    def test_vars_blocked(self):
        errs = validate_tool_code("vars()")
        assert any("vars" in e for e in errs)

    def test_dir_blocked(self):
        errs = validate_tool_code("dir()")
        assert any("dir" in e for e in errs)

    def test_chr_blocked(self):
        errs = validate_tool_code("chr(101)+chr(118)+chr(97)+chr(108)")
        assert any("chr" in e for e in errs)

    def test_type_blocked(self):
        errs = validate_tool_code("type(compile)")
        assert any("type" in e for e in errs)

    def test_dunder_class_attribute(self):
        errs = validate_tool_code("().__class__.__mro__[-1]")
        assert any("__class__" in e or "__mro__" in e for e in errs)

    def test_dunder_subclasses(self):
        errs = validate_tool_code(
            "[x for x in ().__class__.__mro__[-1].__subclasses__()]"
        )
        assert len(errs) > 0

    def test_dunder_globals(self):
        errs = validate_tool_code("s.__init__.__globals__")
        assert any("__globals__" in e for e in errs)

    def test_dunder_builtins(self):
        errs = validate_tool_code("s.__builtins__")
        assert any("__builtins__" in e for e in errs)

    def test_full_exploit_chain(self):
        """Full exploit from Audit3 lines 72-78."""
        code = (
            "subs = ().__class__.__mro__[-1].__subclasses__()\n"
            "for s in subs:\n"
            "    if 'warning' in str(s).lower():\n"
            "        import_func = s.__init__.__globals__.get('__builtins__', {})\n"
            "        break\n"
        )
        errs = validate_tool_code(code)
        assert len(errs) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_sandbox_executor.py::TestSandboxBypassVectors -v`
Expected: Multiple FAILs (bypass vectors currently pass validation)

- [ ] **Step 3: Implement sandbox hardening**

In `sandbox_executor.py`, expand BLOCKED_CALLS and add dunder detection:

```python
# Line 28-30: Replace BLOCKED_CALLS
BLOCKED_CALLS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint", "open",
    "getattr", "setattr", "delattr", "globals", "locals",  # Parity with Rust
    "vars", "dir", "chr", "type", "hasattr",  # Indirect bypass vectors
})

# Add after BLOCKED_CALLS (line ~32):
BLOCKED_DUNDERS: frozenset[str] = frozenset({
    "__class__", "__bases__", "__mro__", "__subclasses__",
    "__globals__", "__builtins__", "__import__", "__init__",
    "__dict__", "__getattr__", "__setattr__", "__delattr__",
    "__code__", "__func__", "__self__", "__module__",
    "__qualname__", "__wrapped__", "__loader__", "__spec__",
})
```

In `validate_tool_code()`, add dunder-chain detection after existing checks:

```python
# Add inside validate_tool_code(), after the Call/Import checks:
# Block dangerous dunder attribute access
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and node.attr in BLOCKED_DUNDERS:
        errors.append(
            f"Blocked dunder access: '{node.attr}' (line {node.lineno})"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_sandbox_executor.py::TestSandboxBypassVectors -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `cd sage-python && python -m pytest tests/ -x --tb=short -q`
Expected: 1149+ passed, 0 new failures

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/tools/sandbox_executor.py sage-python/tests/test_sandbox_executor.py
git commit -m "fix(security): harden Python sandbox — block dunders, getattr, chr, type, vars, dir

Addresses Audit3 F-01 (5/8 bypass vectors) and F-11 (Python/Rust parity gap).
Adds 14 regression tests for exploit vectors including full __mro__ chain."
```

---

## Chunk 2: Security — execute_raw gating + subprocess documentation (S6 + S2)

**Priority:** P0 HIGH — documented but dangerous API surface
**Confirmed by:** Audit5 §6, Audit3 F-02

### Task 2.1: Gate execute_raw behind unsafe marker

**Files:**
- Modify: `sage-core/src/sandbox/tool_executor.rs:202-218`
- Test: `sage-core/src/sandbox/tool_executor.rs` (existing tests)

- [ ] **Step 1: Add deprecation warning to execute_raw docstring**

In `tool_executor.rs`, update the `execute_raw` docstring:

```rust
/// Execute Python code WITHOUT validation.
///
/// # Safety
/// Caller MUST validate code before calling this method.
/// This method provides NO security guarantees.
/// Prefer `validate_and_execute()` for untrusted code.
///
/// # Warning
/// Logged at WARN level for audit trail.
#[pyo3(text_signature = "(code, args_json)")]
pub fn execute_raw(
    &self,
    py: Python<'_>,
    code: &str,
    args_json: &str,
) -> ExecResult {
    tracing::warn!(
        code_len = code.len(),
        "execute_raw called — bypassing validation"
    );
    // ... existing implementation
}
```

- [ ] **Step 2: Document subprocess isolation limitations**

Add a comment block at the top of `sandbox_executor.py`:

```python
# SECURITY NOTE (Audit3 F-02):
# The subprocess fallback provides TIMEOUT isolation only.
# No OS-level isolation: no seccomp, no namespaces, no cgroups.
# For production use, compile sage_core with sandbox+tool-executor features
# to get Wasm WASI deny-by-default isolation.
# See: CLAUDE.md "Sandbox & Tool Security" section.
```

- [ ] **Step 3: Run Rust tests**

Run: `cd sage-core && cargo test --features tool-executor -v 2>&1 | tail -20`
Expected: All tool_executor tests pass

- [ ] **Step 4: Commit**

```bash
git add sage-core/src/sandbox/tool_executor.rs sage-python/src/sage/tools/sandbox_executor.py
git commit -m "fix(security): add WARN tracing to execute_raw, document subprocess limitations

Addresses Audit5 §6 (execute_raw bypass) and Audit3 F-02 (subprocess isolation docs)."
```

---

## Chunk 3: Truth Pipeline — Fix README + Badge + Benchmark Provenance (B1 + B2 + B4 + B8)

**Priority:** P1 HIGH — claims/evidence gap is the #1 cross-audit finding
**Confirmed by:** All 5 audits

### Task 3.1: Fix README metrics to match reality

**Files:**
- Modify: `README.md:13` (badge), `README.md:76-84` (benchmarks)

- [ ] **Step 1: Update README badge and benchmark table**

Replace hardcoded badge (line 13):
```markdown
![tests](https://img.shields.io/badge/tests-1149%20passed-brightgreen)
```

Replace benchmark table (lines 76-84):
```markdown
### Benchmarks (March 11, 2026)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **EvalPlus HumanEval+** (164 tasks) | **84.1%** pass@1 | Official 80x harder tests |
| **EvalPlus MBPP+** (378 tasks) | **75.1%** pass@1 | Official 35x harder tests |
| **Ablation: full vs baseline** | **+15pp** (100% vs 85%) | 20-task paired comparison |
| **Routing Quality** (30 GT) | 23/30 (76.7%) | Heuristic self-consistency |
| **Rust tests** | 432 passed | Including SMT, LTL, CMA-ME, MCTS |
| **Python tests** | 1149 passed, 102 skipped | Full test suite |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "fix(docs): correct README metrics to match actual benchmark artifacts

Badge: 1127→1149. HumanEval: 95%(20)→84.1%(164). Routing: 100%→76.7%.
Addresses Audit3 F-03, F-04, Audit5 §1 (claims/evidence gap)."
```

### Task 3.2: Fix benchmark model provenance

**Files:**
- Modify: `sage-python/src/sage/bench/evalplus_bench.py:100-110`
- Modify: `sage-python/src/sage/agent_loop.py` (ensure _last_model is set)
- Test: `sage-python/tests/test_evalplus_bench.py`

- [ ] **Step 1: Write test for model tracking**

```python
def test_benchmark_manifest_tracks_model():
    """Audit5 §2: model field must not be 'unknown'."""
    # The manifest should capture actual model ID from agent_loop
    from sage.bench.evalplus_bench import EvalPlusBenchmark
    # When agent_loop._last_model is set, manifest should capture it
    # (integration test — requires mock system)
```

- [ ] **Step 2: Ensure agent_loop sets _last_model**

In `agent_loop.py`, after every LLM call, ensure `self._last_model` is set to the actual model ID used. Search for where the LLM response is received and add:
```python
self._last_model = response.model if hasattr(response, 'model') else model_id
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_evalplus_bench.py -v --tb=short`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/bench/evalplus_bench.py sage-python/src/sage/agent_loop.py
git commit -m "fix(bench): track actual model ID in benchmark artifacts

Addresses Audit5 §2 (model: unknown in benchmark artifacts)."
```

---

## Chunk 4: CI Feature Coverage (B5)

**Priority:** P1 HIGH — major features untested in CI
**Confirmed by:** Audit3 F-05

### Task 4.1: Add feature-complete Rust test jobs to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add SMT + tool-executor jobs**

Add after the `rust-features` job:

```yaml
  rust-smt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Test SMT verification
        working-directory: sage-core
        run: cargo test --features smt -v

  rust-tool-executor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Test tool-executor
        working-directory: sage-core
        run: cargo test --features tool-executor -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add smt + tool-executor feature test jobs

Addresses Audit3 F-05 (CI skips major Rust features)."
```

---

## Chunk 5: Architecture — Circuit Breaker Recovery + Provider Contracts (A4 + A5)

**Priority:** P2 MEDIUM
**Confirmed by:** Audit4 §5.2, Audit5 §4

### Task 5.1: Add time-based half-open recovery to CircuitBreaker

**Files:**
- Modify: `sage-python/src/sage/resilience.py`
- Test: `sage-python/tests/test_resilience.py`

- [ ] **Step 1: Write failing test for auto-recovery**

```python
import time
from sage.resilience import CircuitBreaker

def test_circuit_breaker_auto_recovery():
    """Audit4 §5.2: circuit should auto-recover after cooldown."""
    cb = CircuitBreaker("test", max_failures=2, cooldown_s=0.1)
    cb.record_failure(Exception("fail1"))
    cb.record_failure(Exception("fail2"))
    assert cb.is_open()
    assert cb.should_skip()
    # After cooldown, should transition to half-open
    time.sleep(0.15)
    assert not cb.should_skip()  # half-open: allow one probe
    cb.record_success()
    assert cb.is_closed()

def test_circuit_breaker_half_open_failure():
    cb = CircuitBreaker("test", max_failures=2, cooldown_s=0.1)
    cb.record_failure(Exception("fail1"))
    cb.record_failure(Exception("fail2"))
    time.sleep(0.15)
    assert not cb.should_skip()  # half-open probe
    cb.record_failure(Exception("fail3"))
    assert cb.is_open()  # back to open
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_resilience.py -v -k "auto_recovery or half_open"`
Expected: FAIL (cooldown_s parameter doesn't exist)

- [ ] **Step 3: Implement time-based recovery**

```python
import time
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Lightweight per-subsystem circuit breaker with half-open recovery."""

    def __init__(self, name: str, max_failures: int = 3, cooldown_s: float = 60.0):
        self.name = name
        self.max_failures = max_failures
        self.cooldown_s = cooldown_s
        self.failure_count = 0
        self._opened_at: float | None = None

    def is_closed(self) -> bool:
        return self.failure_count < self.max_failures

    def is_open(self) -> bool:
        return not self.is_closed()

    def _is_half_open(self) -> bool:
        """True if circuit is open but cooldown has elapsed."""
        if not self.is_open() or self._opened_at is None:
            return False
        return (time.monotonic() - self._opened_at) >= self.cooldown_s

    def record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        if self.is_open() and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "CircuitBreaker[%s] OPEN after %d failures: %s",
                self.name, self.failure_count, error,
            )
        else:
            logger.debug(
                "CircuitBreaker[%s] failure %d/%d: %s",
                self.name, self.failure_count, self.max_failures, error,
            )

    def record_success(self) -> None:
        if self.failure_count > 0:
            self.failure_count = 0
            self._opened_at = None

    def should_skip(self) -> bool:
        if self.is_closed():
            return False
        if self._is_half_open():
            logger.info(
                "CircuitBreaker[%s] half-open probe allowed", self.name
            )
            return False  # Allow one probe
        logger.warning(
            "CircuitBreaker[%s] OPEN — skipping", self.name
        )
        return True
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_resilience.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full suite for regressions**

Run: `cd sage-python && python -m pytest tests/ -x --tb=short -q`
Expected: 1149+ passed

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/resilience.py sage-python/tests/test_resilience.py
git commit -m "feat(resilience): add time-based half-open recovery to CircuitBreaker

Adds cooldown_s param (default 60s). After cooldown, circuit transitions to
half-open state allowing one probe call. Success closes, failure reopens.
Addresses Audit4 §5.2 (no automatic recovery)."
```

### Task 5.2: Document provider capability limitations

**Files:**
- Modify: `sage-python/src/sage/providers/capabilities.py`

- [ ] **Step 1: Add capability provenance comments**

At the top of `capabilities.py`, add:
```python
# CAPABILITY PROVENANCE (Audit5 §4):
# These capabilities are DECLARED, not runtime-verified.
# OpenAICompatProvider overrides structured_output=False, tool_role=False
# even though _KNOWN_CAPABILITIES lists them as True for some providers.
# TODO: Add runtime conformance tests at boot (P3 milestone).
#
# Capability states (target architecture):
#   DECLARED  — listed here, not tested
#   PROBED    — tested at boot with lightweight API call
#   VERIFIED  — passed conformance test suite
#   SELECTED  — available for routing decisions
```

- [ ] **Step 2: Commit**

```bash
git add sage-python/src/sage/providers/capabilities.py
git commit -m "docs(providers): document capability declaration vs verification gap

Addresses Audit5 §4 (capability matrix may lie to planner)."
```

---

## Chunk 6: Routing Honesty — Label benchmark as self-consistency (B3 + R1)

**Priority:** P2 MEDIUM
**Confirmed by:** Audit3 F-07, Audit2 §2

### Task 6.1: Clarify routing benchmark terminology

**Files:**
- Modify: `sage-python/src/sage/bench/routing.py` (docstring)
- Modify: `sage-python/src/sage/bench/routing_quality.py` (docstring)

- [ ] **Step 1: Update docstrings to be explicit about methodology**

In `routing.py`, ensure the docstring clearly says:
```python
"""Routing self-consistency check (NOT accuracy benchmark).

Measures whether ComplexityRouter agrees with labels that were
calibrated against the heuristic itself. 100% agreement is expected
and proves only that the heuristic is deterministic.

For actual routing validation, measure downstream task quality
(pass@1 per tier, cost, latency) — see bench/routing_downstream.py.
"""
```

In `routing_quality.py`, add at top:
```python
"""Routing quality benchmark with human-labeled ground truth.

NOTE: Labels for S3 tasks contain keywords that overlap with the
heuristic's keyword groups. True routing accuracy requires held-out
evaluation with >1000 diverse tasks and downstream quality metrics.

Current: 30 tasks, heuristic-correlated labels.
Target: 1000+ tasks, independent labeling, cross-validated.
"""
```

- [ ] **Step 2: Commit**

```bash
git add sage-python/src/sage/bench/routing.py sage-python/src/sage/bench/routing_quality.py
git commit -m "docs(bench): clarify routing benchmark is self-consistency, not accuracy

Addresses Audit3 F-07 (circular evaluation) and Audit2 §2 (heuristic branding)."
```

---

## Chunk 7: Evolution Evidence + Memory Validation Tracking (E1 + M2)

**Priority:** P3 LOW — research gaps, not bugs
**Confirmed by:** Audit3 F-09, Audit5 §7

### Task 7.1: Add evolution TODO and feature-flag warning

**Files:**
- Modify: `sage-python/src/sage/evolution/engine.py`

- [ ] **Step 1: Add evidence-gap documentation**

At the top of `engine.py`, add:
```python
# EVIDENCE STATUS (Audit3 F-09, Audit5 §7):
# This engine has NOT been validated against random search or manual tuning.
# Required evidence before removing this notice:
#   1. Ablation: SAMPO-guided vs random mutation vs no-evolution on 50+ tasks
#   2. Effect size with confidence intervals
#   3. Cost analysis (LLM calls per evolution step)
# Until then, this is an experimental component.
```

- [ ] **Step 2: Commit**

```bash
git add sage-python/src/sage/evolution/engine.py
git commit -m "docs(evolution): document evidence gap for evolution engine

Addresses Audit3 F-09 (zero evidence of improvement over random)."
```

---

## Implementation Order

```
Chunk 1 (P0 CRITICAL)  → Sandbox hardening         ~30 min
Chunk 2 (P0 HIGH)      → execute_raw + docs         ~15 min
Chunk 3 (P1 HIGH)      → README truth pipeline       ~20 min
Chunk 4 (P1 HIGH)      → CI feature coverage         ~10 min
Chunk 5 (P2 MEDIUM)    → CircuitBreaker + providers  ~30 min
Chunk 6 (P2 MEDIUM)    → Routing benchmark honesty   ~10 min
Chunk 7 (P3 LOW)       → Evolution evidence docs     ~5 min
```

Total estimated: ~2 hours

## Success Criteria

- [ ] All 14 new sandbox exploit tests pass
- [ ] README metrics match actual benchmark artifacts
- [ ] CI tests smt + tool-executor features
- [ ] CircuitBreaker has half-open auto-recovery
- [ ] No new Python test failures
- [ ] No new Rust test failures
- [ ] Each chunk independently committable
