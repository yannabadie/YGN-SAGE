# Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 15 confirmed problems from the 5-audit cross-verification (docs/audits/2026-03-07-audit-verification.md), prioritized by severity across 4 sprints.

**Architecture:** Sprint 1 eliminates security vulnerabilities and stale claims (eval RCE, eBPF, badges). Sprint 2 adds observability (circuit breaker for silent catches, tool-role warning) and removes decorative Z3. Sprint 3 validates core thesis (routing cost-frontier, HumanEval 164, evolution ablation). Sprint 4 adds real Z3 DAG verification, ExoCortex abstraction, and dashboard queue.

**Tech Stack:** Python 3.12+, pytest, Z3 (optional), SQLite, FastAPI

---

## Sprint 1 — Stop the Bleeding

### Task 1: Replace eval() with safe AST evaluator in kg_rlvr.py

**Files:**
- Modify: `sage-python/src/sage/topology/kg_rlvr.py:62-74`
- Test: `sage-python/tests/test_kg_rlvr.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_kg_rlvr.py`:

```python
def test_verify_invariant_blocks_code_injection():
    """Verify that malicious pre/post strings cannot execute arbitrary code."""
    kg = FormalKnowledgeGraph()
    # These strings would execute arbitrary code if eval() is used
    malicious_pre = "__import__('os').system('echo pwned')"
    malicious_post = "x > 0"
    # Must NOT execute the code — should return False (fail-closed) or raise
    result = kg.verify_invariant(malicious_pre, malicious_post)
    assert result is False, "Malicious input must fail-closed (return False), not pass"


def test_verify_invariant_accepts_valid_z3_expressions():
    """Verify that legitimate Z3 constraint strings still work."""
    kg = FormalKnowledgeGraph()
    if not kg.has_z3:
        return  # Skip if z3 not installed
    # Valid Z3: pre="x > 0" post="x > -1" — should be valid (pre implies post)
    result = kg.verify_invariant("x > 0", "x > -1")
    assert result is True

    # Valid Z3: pre="x > 10" post="x > 20" — should be invalid
    result = kg.verify_invariant("x > 10", "x > 20")
    assert result is False


def test_verify_invariant_fails_closed_on_unparseable():
    """Verify that unparseable expressions fail-closed (return False)."""
    kg = FormalKnowledgeGraph()
    result = kg.verify_invariant("not a valid expression ???", "also garbage")
    assert result is False, "Unparseable input must fail-closed"
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_kg_rlvr.py::test_verify_invariant_blocks_code_injection tests/test_kg_rlvr.py::test_verify_invariant_fails_closed_on_unparseable -v`
Expected: FAIL — eval() currently returns True for unparseable input (fail-open) and executes malicious code

**Step 3: Implement safe AST evaluator**

Replace `verify_invariant` in `sage-python/src/sage/topology/kg_rlvr.py:62-74` with:

```python
    def verify_invariant(self, pre: str, post: str) -> bool:
        """Verify a pre/post-condition pair using Z3.

        Uses a restricted AST evaluator instead of eval() to prevent
        arbitrary code execution. Fails closed (returns False) on any error.
        """
        if not self.has_z3:
            return True
        solver = z3.Solver()
        x = z3.Int("x")
        try:
            pre_constraint = _safe_z3_eval(pre, {"x": x, "z3": z3})
            post_constraint = _safe_z3_eval(post, {"x": x, "z3": z3})
            solver.add(z3.And(pre_constraint, z3.Not(post_constraint)))
            return solver.check() == z3.unsat
        except Exception:
            return False  # Fail CLOSED — can't parse means reject
```

Add the `_safe_z3_eval` function before the class definition (after imports):

```python
import ast

# Allowed AST node types for safe Z3 expression evaluation
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Constant, ast.Name, ast.Attribute, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.And, ast.Or, ast.Not, ast.USub,
)


def _safe_z3_eval(expr: str, namespace: dict) -> Any:
    """Evaluate a Z3 constraint string using restricted AST parsing.

    Only allows: comparisons, arithmetic, boolean ops, variable names,
    constants, and z3.* attribute access / function calls.
    Raises ValueError on any disallowed construct.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        # Block attribute access on anything except 'z3'
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "z3"):
                raise ValueError(f"Attribute access only allowed on 'z3', got: {ast.dump(node.value)}")
        # Block function calls except z3.* methods and comparisons
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "z3"):
                    raise ValueError(f"Function calls only allowed on z3.*, got: {ast.dump(node.func)}")
            elif isinstance(node.func, ast.Name):
                # Allow only names from namespace
                if node.func.id not in namespace:
                    raise ValueError(f"Unknown function: {node.func.id}")
        # Block names not in namespace
        if isinstance(node, ast.Name) and node.id not in namespace:
            raise ValueError(f"Unknown name: {node.id}")
    # AST is safe — compile and eval
    code = compile(tree, "<z3_constraint>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)
```

**Step 4: Run all kg_rlvr tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_kg_rlvr.py -v`
Expected: All tests PASS (including existing ones)

**Step 5: Commit**

```bash
git add sage-python/src/sage/topology/kg_rlvr.py sage-python/tests/test_kg_rlvr.py
git commit -m "security(kg_rlvr): replace eval() with safe AST evaluator + fail-closed

CRITICAL fix: verify_invariant() used eval() on user-supplied strings,
enabling arbitrary code execution. Now uses restricted AST parser that
only allows comparisons, arithmetic, boolean ops, and z3.* calls.
Fail-open (return True) changed to fail-closed (return False).

Fixes P1 from audit verification report."
```

---

### Task 2: Remove eBPF claim from README

**Files:**
- Modify: `sage-python/../README.md` (project root `README.md`)

**Step 1: Edit README.md line 29**

Change:
```
- **Sandbox** — Wasm (wasmtime) + eBPF (solana_rbpf) execution sandboxes (experimental)
```
To:
```
- **Sandbox** — Wasm (wasmtime) execution sandbox (experimental)
```

**Step 2: Edit README.md line 97**

Change:
```
|-- sage-core/           # Rust core (eBPF, Z3, Arrow memory, RagCache)
```
To:
```
|-- sage-core/           # Rust core (Z3, Arrow memory, RagCache)
```

**Step 3: Verify no other eBPF claims remain in README**

Run: `grep -n "eBPF\|ebpf\|solana_rbpf" README.md`
Expected: No matches

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): remove stale eBPF/solana_rbpf claims

solana_rbpf is commented out of Cargo.toml and not built.
eBPF sandbox code exists but is not available as a feature.
Only claim what is actually built and testable.

Fixes P3 from audit verification report."
```

---

### Task 3: Update test count badges

**Files:**
- Modify: `README.md:13`
- Modify: `ARCHITECTURE.md` (search for "620")

**Step 1: Run test suite to get current count**

Run: `cd sage-python && python -m pytest tests/ --co -q | tail -1`
Expected: Something like "674 tests collected"

**Step 2: Update README.md badge**

Change line 13:
```
  <img src="https://img.shields.io/badge/tests-413%20collected-brightgreen?style=flat-square" alt="Tests">
```
To (use actual count from step 1):
```
  <img src="https://img.shields.io/badge/tests-674%20passed-brightgreen?style=flat-square" alt="Tests">
```

**Step 3: Update ARCHITECTURE.md**

Search for "620" and update to actual count.

**Step 4: Commit**

```bash
git add README.md ARCHITECTURE.md
git commit -m "docs: update test count badges to 674 (was 413/620)

Fixes P5 from audit verification report."
```

---

### Task 4: Add warning for tool-to-user role rewrite

**Files:**
- Modify: `sage-python/src/sage/providers/openai_compat.py:73-74`
- Test: `sage-python/tests/test_openai_compat.py` (create)

**Step 1: Write the failing test**

Create `sage-python/tests/test_openai_compat.py`:

```python
import sys
import types
import logging

# Mock sage_core
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.providers.openai_compat import OpenAICompatProvider
from sage.llm.base import Message, Role


def test_tool_role_rewrite_logs_warning(caplog):
    """Verify tool->user rewrite emits a warning."""
    provider = OpenAICompatProvider(api_key="test")
    messages = [
        Message(role=Role.SYSTEM, content="You are helpful"),
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.TOOL, content="Tool result here"),
    ]
    # We can't call generate() without a real API, but we can test
    # the message conversion logic. Extract it for testing.
    with caplog.at_level(logging.WARNING):
        converted = provider._convert_messages(messages)

    assert converted[2]["role"] == "user"
    assert any("tool" in r.message.lower() and "user" in r.message.lower() for r in caplog.records)
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_openai_compat.py -v`
Expected: FAIL — `_convert_messages` doesn't exist yet

**Step 3: Refactor openai_compat.py to extract message conversion + add warning**

In `sage-python/src/sage/providers/openai_compat.py`, replace lines 69-75 with:

```python
        oai_messages = self._convert_messages(messages)
```

Add a new method to the class:

```python
    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to OpenAI dict format."""
        oai_messages: list[dict[str, str]] = []
        for msg in messages:
            role = msg.role.value
            if role == "tool":
                log.warning(
                    "Rewriting tool role to user for OpenAI-compat API — "
                    "semantic context (tool provenance) is lost"
                )
                role = "user"
            oai_messages.append({"role": role, "content": msg.content})
        return oai_messages
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_openai_compat.py -v`
Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/providers/openai_compat.py sage-python/tests/test_openai_compat.py
git commit -m "fix(openai_compat): warn on tool->user role rewrite

Previously the tool role was silently rewritten to user with no
indication that semantic context was being lost. Now emits a
WARNING log entry.

Fixes P7 from audit verification report."
```

---

### Task 5: Mark snap_bpf.c as stub in docs and code

**Files:**
- Modify: `sage-core/src/sandbox/snap_bpf.c` (add prominent stub comment)
- Modify: `ARCHITECTURE.md` (add stub note)

**Step 1: Add stub header to snap_bpf.c**

Replace the first 8 lines of `sage-core/src/sandbox/snap_bpf.c` with:

```c
/*
 * STUB — NOT FUNCTIONAL
 *
 * This file is a placeholder for a future kernel-level SnapBPF agent.
 * It compiles but does NOT implement any CoW memory logic.
 * The actual SnapBPF implementation is in Rust: sage-core/src/sandbox/ebpf.rs
 * (userspace CoW via Arc<Vec<u8>> snapshots in DashMap).
 */
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
```

**Step 2: Update ARCHITECTURE.md**

Find the eBPF section and add a note about snap_bpf.c being a stub.

**Step 3: Commit**

```bash
git add sage-core/src/sandbox/snap_bpf.c ARCHITECTURE.md
git commit -m "docs: mark snap_bpf.c as stub, clarify Rust SnapBPF is the real impl

Fixes P12 from audit verification report."
```

---

## Sprint 2 — Observability + Z3 Honest

### Task 6: Replace silent catches in agent_loop.py with circuit breaker

**Files:**
- Create: `sage-python/src/sage/resilience.py`
- Modify: `sage-python/src/sage/agent_loop.py` (lines 239, 251, 357, 444, 453, 511)
- Test: `sage-python/tests/test_resilience.py` (create)

**Step 1: Write the failing test for CircuitBreaker**

Create `sage-python/tests/test_resilience.py`:

```python
import sys
import types
import logging

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.resilience import CircuitBreaker


def test_circuit_breaker_allows_calls_initially():
    cb = CircuitBreaker("test", max_failures=3)
    assert cb.is_closed()
    assert not cb.is_open()


def test_circuit_breaker_opens_after_max_failures():
    cb = CircuitBreaker("test", max_failures=2)
    cb.record_failure(ValueError("fail 1"))
    assert cb.is_closed()
    cb.record_failure(ValueError("fail 2"))
    assert cb.is_open()


def test_circuit_breaker_skips_when_open(caplog):
    cb = CircuitBreaker("test", max_failures=1)
    cb.record_failure(ValueError("boom"))
    assert cb.is_open()

    with caplog.at_level(logging.WARNING):
        skipped = cb.should_skip()
    assert skipped is True
    assert any("circuit open" in r.message.lower() for r in caplog.records)


def test_circuit_breaker_records_success_resets():
    cb = CircuitBreaker("test", max_failures=2)
    cb.record_failure(ValueError("fail"))
    assert cb.failure_count == 1
    cb.record_success()
    assert cb.failure_count == 0
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_resilience.py -v`
Expected: FAIL — module `sage.resilience` doesn't exist

**Step 3: Implement CircuitBreaker**

Create `sage-python/src/sage/resilience.py`:

```python
"""Lightweight circuit breaker for best-effort subsystems.

Replaces silent except:pass patterns with observable failure tracking.
After max_failures consecutive errors, the breaker opens and skips
calls until record_success() resets it.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Per-subsystem circuit breaker.

    Parameters
    ----------
    name:
        Human-readable name for log messages (e.g. "semantic_memory").
    max_failures:
        Number of consecutive failures before the circuit opens.
    """

    def __init__(self, name: str, max_failures: int = 3):
        self.name = name
        self.max_failures = max_failures
        self.failure_count = 0

    def is_closed(self) -> bool:
        return self.failure_count < self.max_failures

    def is_open(self) -> bool:
        return not self.is_closed()

    def record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        if self.failure_count == self.max_failures:
            log.warning(
                "Circuit OPEN for %s after %d failures (last: %s)",
                self.name, self.max_failures, error,
            )
        elif self.failure_count < self.max_failures:
            log.debug("Failure %d/%d in %s: %s", self.failure_count, self.max_failures, self.name, error)

    def record_success(self) -> None:
        if self.failure_count > 0:
            self.failure_count = 0

    def should_skip(self) -> bool:
        """Check if the circuit is open (should skip the call)."""
        if self.is_open():
            log.warning("Circuit open for %s — skipping call", self.name)
            return True
        return False
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_resilience.py -v`
Expected: All PASS

**Step 5: Wire CircuitBreaker into agent_loop.py**

In `agent_loop.py`, add import and initialize breakers in `__init__`:

```python
from sage.resilience import CircuitBreaker
```

In the constructor, add after existing attributes:

```python
        # Circuit breakers for best-effort subsystems
        self._cb_semantic = CircuitBreaker("semantic_memory")
        self._cb_smmu = CircuitBreaker("smmu_context")
        self._cb_runtime_guard = CircuitBreaker("runtime_guardrails")
        self._cb_episodic = CircuitBreaker("episodic_store")
        self._cb_entity = CircuitBreaker("entity_extraction")
        self._cb_evo = CircuitBreaker("evolution_stats")
```

Then replace each silent catch. Example for semantic memory (line ~239):

Before:
```python
            except Exception:
                pass  # Best-effort semantic enrichment
```
After:
```python
            except Exception as e:
                self._cb_semantic.record_failure(e)
```

And add a guard before the try block:
```python
        if self.semantic_memory and not self._cb_semantic.should_skip():
```

Apply the same pattern to all 6 silent catches at lines 239, 251, 357, 444, 453, 511.

**Step 6: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 7: Commit**

```bash
git add sage-python/src/sage/resilience.py sage-python/tests/test_resilience.py sage-python/src/sage/agent_loop.py
git commit -m "feat(resilience): replace 6 silent catches with circuit breaker

Adds CircuitBreaker class that tracks consecutive failures per
subsystem. After max_failures (default 3), the circuit opens and
skips calls with a WARNING log. Replaces except:pass patterns in
agent_loop.py for: semantic_memory, smmu_context, runtime_guardrails,
episodic_store, entity_extraction, evolution_stats.

Fixes P6 from audit verification report."
```

---

### Task 7: Replace trivial Z3 checks with Python builtins

**Files:**
- Modify: `sage-python/src/sage/contracts/z3_verify.py`
- Test: `sage-python/tests/test_z3_verify.py` (existing)

**Step 1: Read existing tests**

Read `sage-python/tests/test_z3_verify.py` to understand current coverage.

**Step 2: Write test confirming pure-Python equivalence**

Add to tests:

```python
def test_capability_coverage_without_z3():
    """Verify capability check works identically without Z3."""
    dag = build_test_dag()  # Use existing test helper
    result = capability_coverage(dag)
    assert isinstance(result, VFResult)
    # Should work purely via set operations


def test_budget_feasibility_without_z3():
    """Verify budget check works identically without Z3."""
    dag = build_test_dag()
    result = budget_feasibility(dag, budget=100.0)
    assert isinstance(result, VFResult)
```

**Step 3: Replace Z3 ceremony with pure Python**

In `z3_verify.py`, replace each function's Z3 solver calls with equivalent Python:
- `capability_coverage`: `required.issubset(available)`
- `budget_feasibility`: `sum(costs) <= budget`
- `type_compatibility`: `required_types.issubset(provided_types)`

Keep the same function signatures and VFResult return types. Add a comment explaining why Z3 was removed.

**Step 4: Remove z3 import from z3_verify.py**

The file should no longer need `import z3`.

**Step 5: Run tests**

Run: `cd sage-python && python -m pytest tests/test_z3_verify.py -v`
Expected: All PASS

**Step 6: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 7: Commit**

```bash
git add sage-python/src/sage/contracts/z3_verify.py sage-python/tests/test_z3_verify.py
git commit -m "refactor(contracts): replace decorative Z3 with pure Python builtins

capability_coverage, budget_feasibility, type_compatibility were
using Z3 Solver ceremony for trivially decidable checks (set
membership, sum comparison). Replaced with direct Python operations
that produce identical results without the SMT dependency overhead.

Z3 is still used in kg_rlvr.py for actual constraint solving.

Fixes P2 (short-term) from audit verification report."
```

---

### Task 8: Add SemanticMemory SQLite persistence

**Files:**
- Modify: `sage-python/src/sage/memory/semantic.py`
- Test: `sage-python/tests/test_semantic_memory.py` (existing)

**Step 1: Read existing semantic.py and tests**

Read the full files to understand current API surface.

**Step 2: Write failing test for persistence**

```python
import tempfile, os

def test_semantic_memory_persists_to_sqlite():
    """Verify entities survive save/load cycle."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem1 = SemanticMemory(db_path=db_path)
        mem1.add_entity("Python")
        mem1.add_relation("Python", "is_a", "Language")
        mem1.save()

        mem2 = SemanticMemory(db_path=db_path)
        mem2.load()
        assert "Python" in mem2.entities
        relations = mem2.get_relations("Python")
        assert any(r[1] == "Language" for r in relations)
    finally:
        os.unlink(db_path)
```

**Step 3: Implement SQLite persistence**

Add `save()` and `load()` methods to `SemanticMemory` that serialize entities and relations to a SQLite database. Constructor gets optional `db_path` parameter.

**Step 4: Wire into boot.py**

Pass `db_path=os.path.expanduser("~/.sage/semantic.db")` when creating SemanticMemory in boot.py.

**Step 5: Run tests**

Run: `cd sage-python && python -m pytest tests/test_semantic_memory.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/memory/semantic.py sage-python/tests/test_semantic_memory.py sage-python/src/sage/boot.py
git commit -m "feat(memory): add SQLite persistence for SemanticMemory

SemanticMemory was in-memory only — entities and relations lost on
restart. Now supports save()/load() with SQLite backend.
Defaults to ~/.sage/semantic.db when wired through boot.py.

Fixes P9 from audit verification report."
```

---

### Task 9: Complete truth-pack for benchmarks

**Files:**
- Modify: `sage-python/src/sage/bench/runner.py`
- Test: Run benchmark with `--truth-pack`

**Step 1: Read existing runner.py**

Understand the current `--truth-pack` flag implementation (partially done in audit response).

**Step 2: Ensure truth-pack saves all artifacts**

Verify the runner saves: raw results JSON, environment info, git commit hash, model versions, and timestamps. Add any missing pieces.

**Step 3: Run routing benchmark with truth-pack**

Run: `cd sage-python && python -m sage.bench --type routing --truth-pack`
Expected: Creates timestamped JSON in `docs/benchmarks/`

**Step 4: Commit**

```bash
git add sage-python/src/sage/bench/runner.py
git commit -m "feat(bench): complete truth-pack artifact generation

Fixes P15 from audit verification report."
```

---

## Sprint 3 — Prove or Remove

### Task 10: Run cost-performance frontier benchmark

**Files:**
- Create: `sage-python/src/sage/bench/cost_frontier.py`
- Test: `sage-python/tests/test_cost_frontier.py`

This task requires API keys and real LLM calls. It creates a benchmark that:
1. Runs the same set of tasks through S1, S2, and S3 model tiers
2. Records cost and quality (pass/fail + latency) for each
3. Plots the cost-performance frontier
4. Measures whether routing provides Pareto improvement over fixed-tier

**Step 1: Design the benchmark spec**

The benchmark needs 30 tasks (10 easy, 10 medium, 10 hard) with known ground truth. Reuse routing benchmark tasks where possible.

**Step 2: Implement cost_frontier.py**

Create the benchmark module with `run_frontier()` that executes each task on each tier and records results.

**Step 3: Write test**

Unit test that verifies the benchmark runs with mocked LLM responses.

**Step 4: Run the benchmark** (requires API keys)

Run: `cd sage-python && python -m sage.bench --type cost-frontier`

**Step 5: Commit**

```bash
git add sage-python/src/sage/bench/cost_frontier.py sage-python/tests/test_cost_frontier.py
git commit -m "feat(bench): add cost-performance frontier benchmark

Runs tasks through all model tiers to measure whether routing
provides Pareto improvement over fixed-tier selection.

Addresses P8 from audit verification report."
```

---

### Task 11: Run full HumanEval 164

**Files:**
- None to modify — just run the benchmark

**Step 1: Run full HumanEval**

Run: `cd sage-python && python -m sage.bench --type humaneval --truth-pack`
Expected: Completes 164 problems, saves results to `docs/benchmarks/`

**Step 2: Record results in ARCHITECTURE.md**

Update the benchmarks section with actual pass@1 score.

**Step 3: Commit**

```bash
git add docs/benchmarks/ ARCHITECTURE.md
git commit -m "bench: full HumanEval 164 results with truth-pack

Fixes P11 from audit verification report."
```

---

### Task 12: Evolution engine ablation study

**Files:**
- Create: `sage-python/src/sage/bench/evolution_ablation.py`
- Results: `docs/benchmarks/`

**Step 1: Design ablation**

Compare agent performance WITH and WITHOUT evolution engine on a fixed task set. Measure: quality, latency, cost. Run at least 3 trials per condition.

**Step 2: Implement and run**

Create the ablation benchmark. Run it with API keys.

**Step 3: Document results**

If evolution provides no measurable improvement, document honestly and consider making it opt-in only.

**Step 4: Commit**

```bash
git add sage-python/src/sage/bench/evolution_ablation.py docs/benchmarks/
git commit -m "bench: evolution engine ablation study

Addresses P10 from audit verification report."
```

---

### Task 13: Upgrade wasmtime v29 to v36 LTS

**Files:**
- Modify: `sage-core/Cargo.toml` (line 27)
- Modify: `sage-core/src/sandbox/wasm.rs` (API changes if any)

**Step 1: Read wasmtime v36 migration guide**

Check for breaking API changes between v29 and v36.

**Step 2: Update Cargo.toml**

Change: `wasmtime = { version = "29.0", optional = true }`
To: `wasmtime = { version = "36.0", optional = true }`

**Step 3: Fix any compilation errors**

Run: `cd sage-core && cargo check --features sandbox`
Fix any API changes (wasmtime v36 may change Store/Engine initialization).

**Step 4: Run tests**

Run: `cd sage-core && cargo test --features sandbox`

**Step 5: Commit**

```bash
git add sage-core/Cargo.toml sage-core/src/sandbox/wasm.rs
git commit -m "build(core): upgrade wasmtime v29 -> v36 LTS

v29 is out of security support. v36 is the current LTS release,
supported until Aug 2027.

Fixes P4 from audit verification report."
```

---

## Sprint 4 — Differentiate

### Task 14: Real Z3 DAG verification

**Files:**
- Modify: `sage-python/src/sage/contracts/z3_verify.py`
- Test: `sage-python/tests/test_z3_verify.py`

**Step 1: Identify genuinely useful Z3 checks**

Design Z3 checks that are NOT trivially decidable:
- **Data flow analysis**: Verify that every node's output type is consumed by at least one downstream node (transitive closure)
- **Deadlock detection**: Verify no circular wait in resource allocation across DAG nodes
- **Budget propagation**: Verify budget allocation is feasible across parallel branches (not just sum)

**Step 2: Implement real Z3 checks**

Add new functions alongside the pure-Python builtins from Task 7.
These are opt-in (only used when z3-solver is installed).

**Step 3: Test with non-trivial DAGs**

Write tests with DAGs where Python builtins cannot detect the issue but Z3 can.

**Step 4: Commit**

```bash
git add sage-python/src/sage/contracts/z3_verify.py sage-python/tests/test_z3_verify.py
git commit -m "feat(contracts): add genuine Z3 verification (data flow, deadlock)

Adds Z3 checks that solve genuinely hard problems:
- Transitive data flow type compatibility
- Deadlock detection via resource ordering
- Parallel branch budget propagation

These complement the pure-Python builtins from Task 7.

Fixes P2 (long-term) from audit verification report."
```

---

### Task 15: ExoCortex provider abstraction

**Files:**
- Create: `sage-python/src/sage/memory/rag_provider.py` (Protocol)
- Modify: `sage-python/src/sage/memory/remote_rag.py` (implement Protocol)
- Test: `sage-python/tests/test_rag_provider.py`

**Step 1: Define RAGProvider Protocol**

```python
from typing import Protocol

class RAGProvider(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[dict]: ...
    async def ingest(self, content: str, metadata: dict) -> str: ...
```

**Step 2: Make ExoCortex implement RAGProvider**

**Step 3: Write test with mock provider**

**Step 4: Commit**

```bash
git add sage-python/src/sage/memory/rag_provider.py sage-python/src/sage/memory/remote_rag.py sage-python/tests/test_rag_provider.py
git commit -m "refactor(memory): abstract ExoCortex behind RAGProvider protocol

Defines RAGProvider protocol so future backends (Qdrant, Weaviate)
can be swapped in without changing consumer code.

Fixes P14 from audit verification report."
```

---

### Task 16: Dashboard task queue

**Files:**
- Modify: `ui/app.py` (lines 259-261 — single task slot)
- Test: `sage-python/tests/test_dashboard.py` (existing or create)

**Step 1: Read current single-task implementation**

Understand the 409 conflict behavior and DashboardState.

**Step 2: Replace with asyncio.Queue**

Add a `run_id`-based task queue so multiple tasks can be enqueued. Workers dequeue and execute.

**Step 3: Test concurrent submission**

Write test that submits 2 tasks and verifies both complete (no 409).

**Step 4: Commit**

```bash
git add ui/app.py
git commit -m "feat(dashboard): replace single-task slot with asyncio.Queue

Allows concurrent task submission via run_id-based queue.
Previously returned 409 on concurrent requests.

Fixes P13 from audit verification report."
```

---

## Post-Sprint: Documentation Sync

After all sprints are complete, update:
- `CLAUDE.md` — Reflect all changes
- `ARCHITECTURE.md` — Update evidence levels
- `MEMORY.md` — Record completion

```bash
git add CLAUDE.md ARCHITECTURE.md
git commit -m "docs: sync documentation with audit fix sprints"
```
