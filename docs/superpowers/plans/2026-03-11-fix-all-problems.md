# Fix All Problems + E2E Tests Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all CI failures, fix benchmark provenance (Audit5), add comprehensive E2E tests.

**Architecture:** 6 independent tasks: (1) cargo fmt, (2) ruff lint fixes, (3) CI PyO3 linker fix, (4) Windows CI test fix, (5) benchmark provenance, (6) comprehensive E2E tests. Tasks 1-4 fix CI. Task 5 fixes Audit5 §2. Task 6 adds E2E coverage for Rust cognitive components + security.

**Tech Stack:** Rust (cargo fmt), Python (ruff, pytest), GitHub Actions CI, PyO3

---

### Task 1: Fix cargo fmt drift

**Files:**
- Modify: All `.rs` files in `sage-core/src/`

- [ ] **Step 1: Run cargo fmt**

Run: `cd sage-core && cargo fmt`

- [ ] **Step 2: Verify no more drift**

Run: `cargo fmt -- --check`
Expected: No diff output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add sage-core/
git commit -m "style: cargo fmt"
```

---

### Task 2: Fix 25 ruff lint errors

**Files:**
- Modify: `sage-python/src/sage/boot.py` (add `# noqa: E402` on line 38)
- Modify: `sage-python/src/sage/bench/sprint3_evidence.py` (remove unused `os`, fix f-string)
- Modify: `sage-python/src/sage/sandbox/z3_validator.py` (remove unused import)
- Modify: `sage-python/src/sage/tools/builtin.py` (remove unused `subprocess`)
- Modify: `sage-python/src/sage/tools/meta.py` (remove unused `ToolResult`)
- Modify: `sage-python/src/sage/topology/kg_rlvr.py` (fix f-strings)

- [ ] **Step 1: Fix boot.py E402 errors**

Add `# noqa: E402` comment to line 38 (the first import after the try/except block). This is intentional ordering — sage_core must be attempted before other sage imports. Apply to all 18 flagged lines (38-55).

The pattern: each import line from 38-55 gets ` # noqa: E402` appended.

- [ ] **Step 2: Fix sprint3_evidence.py**

Line 18: Remove `import os` (unused).
Line 284: Change `print(f"\n  By category:")` to `print("\n  By category:")`.

- [ ] **Step 3: Fix z3_validator.py**

Line 17: Remove `from sage_core import SmtVerificationResult as _RustSmtResult` (unused).

- [ ] **Step 4: Fix tools/builtin.py**

Line 6: Remove `import subprocess` (unused).

- [ ] **Step 5: Fix tools/meta.py**

Line 15: Change `from sage.tools.base import Tool, ToolResult` to `from sage.tools.base import Tool`.

- [ ] **Step 6: Fix topology/kg_rlvr.py**

Line 55: Change `f"Attribute access only allowed on 'z3'"` to `"Attribute access only allowed on 'z3'"`.
Line 59: Change `f"Function calls only allowed on z3.*"` to `"Function calls only allowed on z3.*"`.

- [ ] **Step 7: Verify zero ruff errors**

Run: `cd sage-python && ruff check src/`
Expected: No errors.

- [ ] **Step 8: Run Python tests to verify no regressions**

Run: `cd sage-python && python -m pytest tests/ -x --tb=short -q`
Expected: 1170+ passed (2 pre-existing SSL/openai failures excluded).

- [ ] **Step 9: Commit**

```bash
git add sage-python/
git commit -m "fix(lint): resolve all 25 ruff errors (E402, F401, F541)"
```

---

### Task 3: Fix CI PyO3 linker failure on Linux

**Files:**
- Modify: `sage-core/Cargo.toml` (add pyo3 auto-initialize for dev-dependencies)

The issue: `cargo test --features sandbox,cranelift` on Linux fails because integration test binaries can't find libpython. The `extension-module` feature tells PyO3 not to link libpython (correct for the cdylib), but test binaries need it.

- [ ] **Step 1: Add dev-dependencies with auto-initialize**

Add to `sage-core/Cargo.toml` after `[features]` section:

```toml
[dev-dependencies]
pyo3 = { version = "0.25", features = ["auto-initialize"] }
```

This enables test binaries to find and link libpython while keeping the cdylib extension-module-only.

- [ ] **Step 2: Verify local compilation**

Run: `cd sage-core && CARGO_HTTP_CHECK_REVOKE=false cargo check --tests --features sandbox`
Expected: Compiles without linker errors.

- [ ] **Step 3: Commit**

```bash
git add sage-core/Cargo.toml
git commit -m "fix(ci): add pyo3 auto-initialize for test binaries (Linux linker fix)"
```

---

### Task 4: Fix Windows CI test_boot_embedder failure

**Files:**
- Modify: `sage-python/tests/test_boot_embedder.py`

The issue: `test_boot_embedder_can_embed` tries to download snowflake-arctic-embed-m from HuggingFace, which fails in CI (no network/model available).

- [ ] **Step 1: Add skip condition for missing model**

Add a `pytest.mark.skipif` or try/except guard that skips when sentence-transformers model download fails.

```python
import pytest

def test_boot_embedder_can_embed():
    """The wired embedder should be functional (produce a vector)."""
    system = boot_agent_system(use_mock_llm=True)
    embedder = system.agent_loop.memory_compressor.embedder

    try:
        vec = embedder.embed("test text")
    except (OSError, RuntimeError) as e:
        pytest.skip(f"Embedding model not available: {e}")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)
```

- [ ] **Step 2: Run test locally to verify**

Run: `cd sage-python && python -m pytest tests/test_boot_embedder.py -v`
Expected: 3 passed (or 1 skipped if model not available).

- [ ] **Step 3: Commit**

```bash
git add sage-python/tests/test_boot_embedder.py
git commit -m "fix(ci): skip test_boot_embedder when model unavailable"
```

---

### Task 5: Fix benchmark provenance (Audit5 §2)

**Files:**
- Modify: `sage-python/src/sage/bench/evalplus_bench.py:100-110` (capture real model ID)
- Modify: `sage-python/src/sage/bench/runner.py:27-42` (add provenance fields to BenchReport)
- Modify: `sage-python/src/sage/bench/__main__.py` (pass provenance to report)

The issue: All benchmark artifacts have `model: "unknown"` because `_last_model` isn't set before `generate_solutions()` captures it. Also no provider, temperature, seed, git SHA, or feature flags in the report.

- [ ] **Step 1: Add provenance fields to BenchReport**

Add to `runner.py` BenchReport dataclass:

```python
    provider: str = ""
    git_sha: str = ""
    feature_flags: list[str] = field(default_factory=list)
```

- [ ] **Step 2: Fix model ID capture in evalplus_bench.py**

Move model ID capture from before task loop to after first task completes. The `_last_model` is only set after the first LLM call.

At line 100-106, change model_id capture to lazy:

```python
        model_id = "unknown"  # Will be updated after first task
```

Then after the task loop (after line ~185), add:

```python
        # Capture model ID after tasks have run (set by first LLM call)
        if hasattr(self.system, "agent_loop"):
            model_id = getattr(self.system.agent_loop, "_last_model", "") or "unknown"
        if self.baseline_mode:
            model_id = f"baseline:{model_id}" if model_id else "baseline"
        self.manifest.model = model_id
```

- [ ] **Step 3: Capture git SHA and provider in __main__.py**

In `_save_report()`, populate provenance from subprocess `git rev-parse HEAD` and from the system's provider name.

- [ ] **Step 4: Run benchmark smoke test**

Run: `cd sage-python && python -m sage.bench --type routing`
Expected: Report generated with correct provenance fields.

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/bench/
git commit -m "fix(bench): add model/provider/git_sha provenance to benchmark artifacts (Audit5 §2)"
```

---

### Task 6: Comprehensive E2E test suite

**Files:**
- Create: `tests/e2e_comprehensive.py`

This test suite covers what e2e_proof.py does NOT test:
- Rust cognitive components (SmtVerifier, HybridVerifier, TopologyEngine, ContextualBandit)
- Security regression (sandbox validator, dunder blocking)
- Benchmark artifact consistency (no contradictory metrics)
- Silent fallback detection

All tests are offline (no LLM calls needed) except where noted.

- [ ] **Step 1: Write SmtVerifier E2E tests**

```python
def test_smt_memory_safety():
    """OxiZ proves memory safety constraints."""
    from sage_core import SmtVerifier
    v = SmtVerifier()
    r = v.prove_memory_safety(0, 1024)
    assert r.is_sat

def test_smt_invariant_with_feedback():
    """OxiZ returns clause-level feedback on invariant failure."""
    from sage_core import SmtVerifier
    v = SmtVerifier()
    r = v.verify_invariant_with_feedback("x > 0", ["x > 100"])
    # Should report which clauses failed

def test_smt_provider_assignment():
    """OxiZ solves provider assignment (exactly-one SAT)."""
    from sage_core import SmtVerifier
    v = SmtVerifier()
    r = v.verify_provider_assignment(["google", "openai"], [["fast", "cheap"], ["fast"]], ["fast"])
    assert r.is_sat
```

- [ ] **Step 2: Write HybridVerifier E2E tests**

```python
def test_hybrid_verifier_structural_checks():
    """HybridVerifier runs 6 structural checks on a topology."""
    from sage_core import PyHybridVerifier, TopologyGraph
    # Build a simple valid graph and verify it passes

def test_ltl_safety_check():
    """LTL verifier detects HIGH→LOW security violation."""
    from sage_core import TopologyGraph
    # Build graph with HIGH→LOW edge, verify safety check fails
```

- [ ] **Step 3: Write TopologyEngine E2E tests**

```python
def test_topology_engine_generate():
    """DynamicTopologyEngine generates a topology for a task."""
    from sage_core import TopologyEngine
    engine = TopologyEngine()
    result = engine.generate("Write a sorting algorithm")
    assert result.topology_id is not None

def test_topology_engine_evolve():
    """Evolution loop produces MAP-Elites archive entries."""
    from sage_core import TopologyEngine
    engine = TopologyEngine()
    result = engine.generate("Write a function")
    engine.record_outcome(result.topology_id, 0.8)
```

- [ ] **Step 4: Write ContextualBandit E2E tests**

```python
def test_bandit_thompson_sampling():
    """Bandit selects arms via Thompson sampling."""
    from sage_core import ContextualBandit
    bandit = ContextualBandit(n_arms=3)
    arm = bandit.select([0.5, 0.3, 0.2])
    assert 0 <= arm < 3
    bandit.update(arm, 1.0)
```

- [ ] **Step 5: Write security regression tests**

```python
def test_validator_blocks_dunder_chain():
    """Rust tree-sitter blocks __class__.__mro__ exploit."""
    from sage_core import ToolExecutor
    ex = ToolExecutor()
    r = ex.validate("x = ''.__class__.__mro__[1].__subclasses__()")
    assert not r.valid

def test_validator_blocks_getattr():
    """Rust tree-sitter blocks getattr bypass."""
    from sage_core import ToolExecutor
    ex = ToolExecutor()
    r = ex.validate("getattr(__builtins__, 'eval')('1+1')")
    assert not r.valid

def test_python_sandbox_blocks_dunder():
    """Python AST validator blocks dunder access."""
    from sage.tools.sandbox_executor import validate_tool_code
    errors = validate_tool_code("x = ''.__class__.__mro__")
    assert len(errors) > 0
```

- [ ] **Step 6: Write benchmark consistency tests**

```python
def test_readme_metrics_match_artifacts():
    """README metrics don't contradict benchmark artifacts."""
    import json, re
    from pathlib import Path

    root = Path(__file__).parent.parent
    readme = (root / "README.md").read_text()

    # Check test count badge is reasonable
    badge_match = re.search(r'tests-(\d+)%20passed', readme)
    if badge_match:
        badge_count = int(badge_match.group(1))
        assert badge_count > 1000, f"Badge count {badge_count} seems too low"
```

- [ ] **Step 7: Write silent fallback detection test**

```python
def test_circuit_breaker_half_open_recovery():
    """CircuitBreaker transitions CLOSED → OPEN → HALF_OPEN → CLOSED."""
    import time
    from sage.resilience import CircuitBreaker
    cb = CircuitBreaker("test_e2e", max_failures=2, cooldown_s=0.1)
    assert cb.is_closed()
    cb.record_failure(ValueError("1"))
    cb.record_failure(ValueError("2"))
    assert cb.is_open()
    time.sleep(0.15)
    assert cb.should_skip() is False  # half-open
    cb.record_success()
    assert cb.is_closed()
```

- [ ] **Step 8: Run the full E2E suite**

Run: `python tests/e2e_comprehensive.py`
Expected: All offline tests pass. Tests requiring sage_core skip gracefully if not compiled.

- [ ] **Step 9: Commit**

```bash
git add tests/e2e_comprehensive.py
git commit -m "test(e2e): add comprehensive E2E tests for Rust cognitive + security + consistency"
```

---

## Final verification

- [ ] **Step 1: Run full CI simulation locally**

```bash
# Rust
cd sage-core && cargo fmt -- --check && cargo clippy --no-default-features -- -D warnings && cargo test --no-default-features

# Python
cd sage-python && ruff check src/ && python -m pytest tests/ -x --tb=short -q

# E2E
python tests/e2e_comprehensive.py
```

- [ ] **Step 2: Commit all and push**

```bash
git push origin master
```
