# Audit Remediation Plan — V3

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all 44 confirmed audit findings from 4 independent audits (Audit1-4.md), prioritized by security impact then correctness.

**Architecture:** Sequential fixes in 4 sprints, security-first. Each task has tests.

**Tech Stack:** Python 3.12+, Rust (sage-core), Z3, pytest, shlex

---

## Sprint 0: Security (P0) — IMMEDIATE

### Task 1: Disable create_python_tool + create_bash_tool

**Files:**
- Modify: `sage-python/src/sage/boot.py:201-204`
- Modify: `sage-python/src/sage/tools/meta.py`
- Create: `sage-python/tests/test_meta_security.py`

**Step 1: Comment out tool registration in boot.py**

Replace lines 201-204:
```python
# Runtime tool synthesis (Agent0/AutoTool pattern)
from sage.tools.meta import create_python_tool, create_bash_tool
tool_registry.register(create_python_tool)
tool_registry.register(create_bash_tool)
```
With:
```python
# Runtime tool synthesis DISABLED — SEC-01/SEC-02 security audit findings.
# create_python_tool has 6 AST bypass vectors (arbitrary code execution).
# create_bash_tool uses shell=True (shell injection).
# Re-enable only after Wasm sandbox pipeline is wired (see Task 1b).
# from sage.tools.meta import create_python_tool, create_bash_tool
# tool_registry.register(create_python_tool)
# tool_registry.register(create_bash_tool)
```

**Step 2: Add deprecation warnings to meta.py functions**

At the top of `create_python_tool` and `create_bash_tool`:
```python
import warnings
warnings.warn(
    "create_python_tool is disabled due to security vulnerabilities (SEC-01). "
    "Use statically defined tools or Wasm sandbox instead.",
    DeprecationWarning,
    stacklevel=2,
)
return "DISABLED: create_python_tool is not available due to security constraints."
```

**Step 3: Fix run_bash in builtin.py**

Replace `create_subprocess_shell` with `create_subprocess_exec`:
```python
# BEFORE (line 25):
proc = await asyncio.create_subprocess_shell(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
# AFTER:
import shlex
BLOCKED_PATTERNS = re.compile(r'rm\s+-rf|mkfs|dd\s+if=|:\(\)\{|fork|/dev/sd')
if BLOCKED_PATTERNS.search(command):
    return "BLOCKED: Potentially destructive command"
proc = await asyncio.create_subprocess_exec(
    "/bin/bash", "-c", command,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

**Step 4: Write test**
```python
# test_meta_security.py
def test_create_python_tool_disabled():
    """SEC-01: create_python_tool must be disabled."""
    from sage.tools.meta import create_python_tool
    import asyncio
    result = asyncio.run(create_python_tool.fn(name="test", code="x=1"))
    assert "DISABLED" in result

def test_create_bash_tool_disabled():
    """SEC-02: create_bash_tool must be disabled."""
    from sage.tools.meta import create_bash_tool
    import asyncio
    result = asyncio.run(create_bash_tool.fn(name="test", script="echo hi"))
    assert "DISABLED" in result
```

**Step 5: Run tests**
```bash
cd sage-python && python -m pytest tests/ -v --tb=short
```

**Step 6: Commit**
```bash
git add sage-python/src/sage/boot.py sage-python/src/sage/tools/meta.py sage-python/src/sage/tools/builtin.py sage-python/tests/test_meta_security.py
git commit -m "fix(security): disable create_python_tool and create_bash_tool (SEC-01/SEC-02)

Both tools had critical security vulnerabilities:
- create_python_tool: 6 AST bypass vectors for arbitrary code execution
- create_bash_tool: shell=True enabling shell injection
- run_bash: replaced create_subprocess_shell with create_subprocess_exec

Tools disabled until Wasm sandbox pipeline is wired."
```

---

### Task 2: Z3 fail-closed fallback (Z3-07)

**Files:**
- Modify: `sage-python/src/sage/topology/kg_rlvr.py:65-107`
- Modify: `sage-python/tests/test_kg_rlvr.py` (if exists)

**Step 1: Change all 4 fallback methods to return False**

In `FormalKnowledgeGraph`:
```python
def prove_memory_safety(self, addr_expr: int, limit: int) -> bool:
    if not self.has_z3:
        return 0 <= addr_expr < limit  # Trivial Python check
    # ... Z3 path unchanged

def check_loop_bound(self, var_name: str, hard_cap: int) -> bool:
    if not self.has_z3:
        return False  # Cannot prove without Z3

def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
    if not self.has_z3:
        return False  # Cannot verify without Z3

def verify_invariant(self, pre: str, post: str) -> bool:
    if not self.has_z3:
        return False  # Cannot verify without Z3
```

**Step 2: Upgrade warning to error level**
```python
if not self.has_z3:
    logging.error(
        "z3-solver not installed. ALL formal verification disabled — "
        "returning unverified (fail-closed)."
    )
```

**Step 3: Update tests, run, commit**

---

### Task 3: Fix verify_arithmetic (Z3-03)

**Files:**
- Modify: `sage-python/src/sage/topology/kg_rlvr.py:91-98`

**Step 1: Replace with safe evaluation**
```python
def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
    """Evaluate arithmetic expr and verify result is within tolerance.

    Uses ast.literal_eval for simple constants, _safe_z3_eval for
    expressions. Returns False on parse failure (fail-closed).
    """
    if not self.has_z3:
        return False
    try:
        import ast
        # Try simple constant first
        actual = ast.literal_eval(expr)
        return expected - tolerance <= actual <= expected + tolerance
    except (ValueError, SyntaxError):
        pass
    try:
        # Try safe Z3 eval for complex expressions
        result = _safe_z3_eval(expr, {"z3": z3})
        if isinstance(result, (int, float)):
            return expected - tolerance <= result <= expected + tolerance
        return False  # Can't evaluate to number
    except Exception:
        return False  # Fail-closed
```

**Step 2: Write test, run, commit**

---

### Task 4: Delete dead z3_validator.rs (Z3-01)

**Files:**
- Delete: `sage-core/src/sandbox/z3_validator.rs`
- Modify: `sage-core/src/sandbox/mod.rs` (update comment)

**Step 1: Delete the file**
```bash
rm sage-core/src/sandbox/z3_validator.rs
```

**Step 2: Update mod.rs comment**
```rust
// z3_validator: implemented in Python (sage.sandbox.z3_validator) using z3-solver package.
// Rust implementation removed — z3 crate was never added to Cargo.toml dependencies.
```

**Step 3: Verify Rust builds, commit**
```bash
cd sage-core && cargo build --no-default-features && cargo test --no-default-features
```

---

## Sprint 1: Correctness (P1)

### Task 5: Rename z3_topology → topology_verifier (Z3-05)

**Files:**
- Rename: `sage-python/src/sage/topology/z3_topology.py` → `topology_verifier.py`
- Rename: `sage-python/tests/test_z3_topology.py` → `test_topology_verifier.py`
- Modify: All imports referencing z3_topology
- Modify: Proof strings

**Step 1: Find all references**
```bash
grep -rn "z3_topology" sage-python/ --include="*.py"
```

**Step 2: Rename files**
**Step 3: Fix imports and proof strings**

Replace:
```python
result.proof = (
    f"PROVED: Topology is a valid DAG with {len(spec.agents)} agents, "
    f"{len(spec.edges)} edges, max depth {depth}. "
    f"Terminates: sat. No cycles: sat."
)
```
With:
```python
result.proof = (
    f"VERIFIED (graph analysis): DAG with {len(spec.agents)} agents, "
    f"{len(spec.edges)} edges, max depth {depth}. "
    f"Acyclic: yes. Terminates: yes. Method: Kahn's algorithm."
)
```

**Step 4: Run tests, commit**

---

### Task 6: Remove evolution Z3 safety gate (Z3-06)

**Files:**
- Modify: `sage-python/src/sage/evolution/engine.py:166-172`

**Step 1: Remove the gate (Option A — honest)**

Replace lines 166-172:
```python
# Z3 safety gate removed — was validating static config constraints,
# not the actual mutated code. See audit finding Z3-06.
```

**Step 2: Remove z3_constraints from EvolutionConfig if unused**
**Step 3: Run tests, commit**

---

### Task 7: Add non-circular routing benchmark (RTG-01)

**Files:**
- Create: `sage-python/src/sage/bench/routing_quality.py`
- Create: `sage-python/tests/test_routing_quality.py`

**Step 1: Create ground-truth benchmark with 50+ manually labeled tasks**
**Step 2: Measure accuracy, over-routing, under-routing rates**
**Step 3: Run, commit**

---

### Task 8: Fix README memory claims (A1-14)

**Files:**
- Modify: `README.md` (memory section)

**Step 1: Change "all persistent" to accurate description**
```
Tier 0 — Working Memory: per-session Arrow buffer (persistent via compressor → Tier 1)
Tier 1 — Episodic: SQLite (~/.sage/episodic.db), persistent across sessions
Tier 2 — Semantic: SQLite (~/.sage/semantic.db), persistent across sessions
Tier 3 — ExoCortex: Google File Search API, persistent (cloud-hosted)
```

**Step 2: Commit**

---

### Task 9: Document check_loop_bound limitation (Z3-02)

**Files:**
- Modify: `sage-python/src/sage/sandbox/z3_validator.py:55-67`
- Modify: `sage-python/src/sage/topology/kg_rlvr.py:81-89`

**Step 1: Add docstring explaining the behavior**
```python
def check_loop_bound(self, var_name: str, hard_cap: int) -> bool:
    """Check if a loop variable is provably bounded.

    Returns True only if, given the registered constraints, it's impossible
    for the variable to exceed hard_cap. For an unconstrained symbolic
    variable, this correctly returns False — we cannot prove boundedness
    without domain-specific constraints.

    Callers must add constraints via solver before calling this for
    meaningful results. Without constraints, this is a conservative
    "cannot prove" answer, not a "loop is infinite" answer.
    """
```

**Step 2: Run tests, commit**

---

## Sprint 2: Build & Tests (P2)

### Task 10: Add integration test tier (TST-01)

**Files:**
- Create: `sage-python/tests/integration/test_memory_pipeline.py`
- Create: `sage-python/tests/integration/test_guardrail_pipeline.py`
- Create: `sage-python/tests/integration/conftest.py`

**Step 1: Create integration tests using real implementations (no mocks)**
**Step 2: Run, commit**

---

### Task 11: Add CI job for sandbox+ONNX features (TST-03)

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add feature-gated CI job**
```yaml
rust-features:
  name: Rust (sandbox + ONNX features)
  runs-on: ubuntu-latest
  defaults:
    run:
      working-directory: sage-core
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    - run: pip install maturin
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo test --features sandbox,cranelift
    - name: ONNX compile check
      run: cargo check --features onnx
```

**Step 2: Commit**

---

### Task 12: Rust workspace lints + edition (BLD-01)

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `sage-core/Cargo.toml`

**Step 1: Add workspace lints**
```toml
[workspace.lints.rust]
unsafe_code = "deny"

[workspace.lints.clippy]
all = "warn"
```

**Step 2: In sage-core/Cargo.toml**
```toml
[lints]
workspace = true
```

**Step 3: Run clippy, fix warnings, commit**

---

### Task 13: Rename DGM → SAMPO (EVO-03)

**Files:**
- Modify: `sage-python/src/sage/evolution/engine.py`
- Modify: `sage-python/tests/test_evolution*.py`

**Step 1: Rename all _dgm_ → _sampo_, DGM_ → SAMPO_**
**Step 2: Run tests, commit**

---

## Sprint 3: Polish (P3)

### Task 14: Update README badge + governance files

**Files:**
- Modify: `README.md` (badge line)
- Modify: `.gitignore` (add Cert/)

**Step 1: Update badge to 846 or link to CI**
**Step 2: Add Cert/ to .gitignore, git rm --cached**
**Step 3: Update dashboard import (MetacognitiveController → ComplexityRouter)**
**Step 4: Commit**

---

## Dependencies

```
Task 1 (security) → independent
Task 2 (Z3 fallback) → independent
Task 3 (verify_arithmetic) → depends on Task 2
Task 4 (delete dead code) → independent
Task 5 (rename topology) → independent
Task 6 (remove evo gate) → independent
Task 7 (routing benchmark) → independent
Task 8 (README) → independent
Task 9 (document loop_bound) → independent
Task 10 (integration tests) → after Tasks 1-4
Task 11 (CI features) → independent
Task 12 (Rust lints) → independent
Task 13 (rename DGM) → independent
Task 14 (polish) → after all others
```

## Success Criteria

After all tasks:
- 0 CRITICAL findings remaining
- All tests pass (target: 860+ tests)
- `cargo clippy -- -D warnings` clean
- No shell=True in production code paths
- Z3 fallback is fail-closed
- No fabricated proof strings
- README claims match code behavior
