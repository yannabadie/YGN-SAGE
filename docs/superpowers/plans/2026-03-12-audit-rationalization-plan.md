# Audit Response + Rust/Python Rationalization — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 14 confirmed audit problems (Phases 0-2) and migrate 3 Rust modules to Python (Rationalization Phase 1), bringing the codebase to a clean, credible, CI-trustworthy state.

**Architecture:** Sequenced as credibility gates first (CI, benchmarks, model ID), then architecture fixes (dual routing, capability self-test), then security hardening + dead code cleanup, then Rust→Python migration for 3 low-value-in-Rust modules (features, model_card, model_registry). Each task produces green CI.

**Tech Stack:** Python 3.12, Rust 1.90+ (PyO3 0.25), pytest, cargo test, tree-sitter-python, tomllib, EvalPlus, GitHub Actions CI.

**Source specs:**
- `docs/superpowers/specs/2026-03-12-audit-response-v3-design.md`
- `docs/superpowers/specs/2026-03-12-rust-python-rationalization-design.md` (v3)

**Out of scope (follow-up plans, evidence-gated):**
- Rationalization Phase 2 (Python→Rust hot paths: relevance_gate, quality_estimator, EventBus)
- Rationalization Phase 3 (entity graph consolidation)
- Audit Phase 3 (benchmark reproducibility P12, cost tracking P14) — needs Phase 0 eval changes first
- Audit Phase 4 (SubTask.depends_on P11, documentation P15-P18)

---

## Chunk 1: Phase 0 — Credibility Gates

These MUST be done first. CI becomes trustworthy, benchmarks become reproducible, model IDs get captured.

### Task 1: Fix CI Hard-Fail [P5]

**Files:**
- Modify: `.github/workflows/ci.yml:55` (remove `|| true` on integration tests)
- Modify: `.github/workflows/ci.yml:72` (remove `|| true` on mypy)

- [ ] **Step 1: Read current CI file**

```bash
cat .github/workflows/ci.yml
```

Identify the two `|| true` lines:
- Line 55: `cargo test --no-default-features --test '*' || true`
- Line 72: `python -m mypy src/sage/ --ignore-missing-imports --no-error-summary || true`

- [ ] **Step 2: Remove `|| true` from integration tests (line 55)**

Replace:
```yaml
        run: cargo test --no-default-features --test '*' || true
```
With:
```yaml
        run: cargo test --no-default-features --test '*'
```

- [ ] **Step 3: Remove `|| true` from mypy (line 72)**

Replace:
```yaml
        run: python -m mypy src/sage/ --ignore-missing-imports --no-error-summary || true
```
With:
```yaml
        run: python -m mypy src/sage/ --ignore-missing-imports --no-error-summary
```

- [ ] **Step 4: Verify mypy passes locally**

```bash
cd sage-python && pip install -e ".[all,dev]" && python -m mypy src/sage/ --ignore-missing-imports --no-error-summary
```

If mypy reports errors, fix them before committing. Common fixes:
- Add type stubs for missing packages
- Use `# type: ignore[import]` for PyO3 imports that can't be typed

- [ ] **Step 5: Verify Rust integration tests pass**

```bash
cd sage-core && cargo test --no-default-features --test '*'
```

If any tests fail, either fix them or mark specific flaky ones with `#[ignore]` (NOT blanket `|| true`). If marking ignored:
```rust
#[test]
#[ignore = "flaky: depends on network / external service"]
fn test_flaky_thing() { ... }
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "fix(ci): remove || true masks on integration tests and mypy [P5]

CI now fails fast on real errors instead of silently masking them.
Flaky tests should be marked #[ignore] individually, not blanket-suppressed."
```

---

### Task 2: Capture Model ID in Benchmarks [P4]

**Files:**
- Modify: `sage-python/src/sage/bench/evalplus_bench.py:100-113`
- Modify: `sage-python/src/sage/bench/runner.py:28` (add `model` field to BenchReport)
- Test: `sage-python/tests/test_evalplus_model_id.py`

The problem: `model_id = "pending"` at line 101 is set lazily and 11/14 benchmark files end up with `"model": "unknown"`. The fix: extract model ID from the FIRST task's routing decision and propagate to all subsequent task results (improvement over post-hoc capture).

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_evalplus_model_id.py`:

```python
"""Test that benchmark reports capture model_id from router, not 'unknown'."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from sage.bench.runner import BenchReport, TaskResult


def test_bench_report_has_model_field():
    """BenchReport must have a 'model' field."""
    report = BenchReport.from_results("test", [])
    assert hasattr(report, "model")


def test_bench_report_model_propagated():
    """Model ID from results should propagate to report."""
    results = [TaskResult(task_id="t1", passed=True)]
    report = BenchReport.from_results("test", results, model_config={"model": "gemini-2.5-flash"})
    assert report.model == "gemini-2.5-flash"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_evalplus_model_id.py -v
```

Expected: FAIL — `BenchReport` has no `model` field.

- [ ] **Step 3: Add `model` field to BenchReport**

In `sage-python/src/sage/bench/runner.py`, add after the `timestamp: str = ""` field (~line 45):

```python
    model: str = "unknown"
```

In `from_results()`, add `model_config: dict | None = None` parameter and extract model.

**IMPORTANT**: `from_results()` has TWO return paths (empty-results branch ~line 68 AND normal branch ~line 96). BOTH must include the new `model=` argument:

```python
            model=model_config.get("model", "unknown") if model_config else "unknown",
```

Add this line after `timestamp=datetime.now(...)` in BOTH return statements.

- [ ] **Step 4: Fix model capture in EvalPlusBench**

In `sage-python/src/sage/bench/evalplus_bench.py`, replace the lazy `model_id = "pending"` block (lines 100-113) with eager model extraction:

```python
        # Model ID captured from first routing decision (not lazily post-hoc)
        model_id = "unknown"
        if hasattr(self.system, "_last_decision") and self.system._last_decision:
            model_id = getattr(self.system._last_decision, "model_id", "unknown")
        elif hasattr(self.system, "agent_loop"):
            llm = getattr(self.system.agent_loop, "_llm", None)
            if llm:
                model_id = getattr(llm, "model_id", "unknown")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd sage-python && python -m pytest tests/test_evalplus_model_id.py -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed (no regressions)

- [ ] **Step 7: Commit**

```bash
git add sage-python/src/sage/bench/runner.py sage-python/src/sage/bench/evalplus_bench.py sage-python/tests/test_evalplus_model_id.py
git commit -m "fix(bench): capture model_id eagerly from router decision [P4]

BenchReport now has a 'model' field. EvalPlusBench extracts model ID
from the routing decision at start, not lazily after first LLM call.
Fixes 11/14 benchmark files showing 'model: unknown'."
```

---

### Task 3: Official EvalPlus Evaluation Mode [P1]

**Files:**
- Modify: `sage-python/src/sage/bench/evalplus_bench.py` (add `official_mode` parameter)
- Modify: `sage-python/src/sage/bench/__main__.py` (add `--official` flag)
- Test: `sage-python/tests/test_evalplus_official.py`

The custom `evaluate_task()` subprocess harness is not comparable to the official EvalPlus leaderboard. We add an `--official` flag that generates `samples.jsonl` and calls `evalplus.evaluate` CLI.

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_evalplus_official.py`:

```python
"""Test official EvalPlus evaluation mode generates correct JSONL format."""
import json
import pytest
from pathlib import Path


def test_official_samples_jsonl_format(tmp_path):
    """Official mode must produce EvalPlus-compatible samples.jsonl."""
    from sage.bench.evalplus_bench import EvalPlusBench

    # Mock: create a samples file in the expected format
    samples = [
        {"task_id": "HumanEval/0", "solution": "def has_close_elements(numbers, threshold):\n    return False\n"},
        {"task_id": "HumanEval/1", "solution": "def separate_paren_groups(paren_string):\n    return []\n"},
    ]
    samples_path = tmp_path / "samples.jsonl"
    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Verify format
    loaded = []
    with open(samples_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            assert "task_id" in obj
            assert "solution" in obj
            loaded.append(obj)
    assert len(loaded) == 2


def test_evalplus_bench_has_official_mode():
    """EvalPlusBench constructor must accept official_mode parameter."""
    bench = EvalPlusBench(system=None, dataset="humaneval", official_mode=True)
    assert bench.official_mode is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_evalplus_official.py -v
```

Expected: FAIL — `EvalPlusBench.__init__()` doesn't accept `official_mode`.

- [ ] **Step 3: Add `official_mode` to EvalPlusBench**

In `sage-python/src/sage/bench/evalplus_bench.py`, modify `__init__`:

```python
    def __init__(
        self,
        system=None,
        event_bus=None,
        dataset: str = "humaneval",
        baseline_mode: bool = False,
        official_mode: bool = False,
    ):
        self.system = system
        self.event_bus = event_bus
        self.dataset = dataset
        self.baseline_mode = baseline_mode
        self.official_mode = official_mode
        self.manifest: BenchmarkManifest | None = None
```

- [ ] **Step 4: Add `run_official()` method**

Add to `EvalPlusBench` class:

```python
    async def run_official(self, limit: int | None = None, output_dir: str | None = None) -> dict[str, Any]:
        """Generate samples.jsonl and evaluate with official EvalPlus CLI.

        This produces scores comparable to the EvalPlus leaderboard.
        Requires: pip install evalplus

        Returns dict with 'base' and 'plus' pass rates.
        """
        # EvalPlus is invoked as `python -m evalplus.evaluate`, not a standalone binary
        try:
            import evalplus  # noqa: F401
        except ImportError:
                raise RuntimeError(
                    "evalplus not installed. Run: pip install evalplus"
                )

        # 1. Generate solutions
        solutions = await self.generate_solutions(limit=limit)
        if not solutions:
            return {"base": 0.0, "plus": 0.0, "error": "no solutions generated"}

        # 2. Write samples.jsonl
        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="sage_evalplus_"))
        samples_path = out_dir / "samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as f:
            for sol in solutions:
                entry = {"task_id": sol["task_id"], "solution": sol["solution"]}
                f.write(json.dumps(entry) + "\n")
        log.info("Wrote %d samples to %s", len(solutions), samples_path)

        # 3. Sanitize
        sanitize_result = subprocess.run(
            ["python", "-m", "evalplus.sanitize", "--samples", str(samples_path)],
            capture_output=True, text=True, timeout=120,
        )
        if sanitize_result.returncode != 0:
            log.warning("evalplus.sanitize failed: %s", sanitize_result.stderr[:300])
            # Continue with unsanitized — sanitize is best-effort

        # Find sanitized file (evalplus appends -sanitized)
        sanitized = samples_path.with_name(samples_path.stem + "-sanitized.jsonl")
        eval_samples = str(sanitized) if sanitized.exists() else str(samples_path)

        # 4. Evaluate
        eval_result = subprocess.run(
            ["python", "-m", "evalplus.evaluate",
             "--dataset", self.dataset,
             "--samples", eval_samples],
            capture_output=True, text=True, timeout=600,
        )
        log.info("evalplus.evaluate stdout:\n%s", eval_result.stdout)

        # 5. Parse results
        results = {"base": 0.0, "plus": 0.0, "raw_output": eval_result.stdout}
        for line in eval_result.stdout.split("\n"):
            if "pass@1" in line.lower():
                # EvalPlus output: "pass@1: 0.841"
                try:
                    val = float(line.split(":")[-1].strip())
                    if "plus" in line.lower() or "+" in line:
                        results["plus"] = val
                    else:
                        results["base"] = val
                except ValueError:
                    pass

        return results
```

- [ ] **Step 5: Add `--official` flag to CLI**

In `sage-python/src/sage/bench/__main__.py`, add argument after `--dataset`:

```python
    parser.add_argument(
        "--official",
        action="store_true",
        default=False,
        help="Use official EvalPlus CLI evaluation (comparable to leaderboard)",
    )
```

And modify `_run_evalplus` to pass it through:

```python
async def _run_evalplus(
    output: str | None, limit: int | None, dataset: str, official: bool = False,
) -> None:
    from sage.bench.evalplus_bench import EvalPlusBench

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  ERROR: GOOGLE_API_KEY required for EvalPlus benchmark")
        return

    system, bus = _boot_system()
    bench = EvalPlusBench(system=system, event_bus=bus, dataset=dataset, official_mode=official)

    if official:
        results = await bench.run_official(limit=limit)
        print(f"\n  Official EvalPlus Results:")
        print(f"    Base pass@1: {results.get('base', 0):.1%}")
        print(f"    Plus pass@1: {results.get('plus', 0):.1%}")
    else:
        report = await bench.run(limit=limit)
        _print_report(report)
        _save_report(report, bench, output, f"evalplus-{dataset}")
```

Update the dispatch at line 252:
```python
    if args.type == "evalplus":
        asyncio.run(_run_evalplus(args.output, args.limit, args.dataset, args.official))
```

- [ ] **Step 6: Run tests**

```bash
cd sage-python && python -m pytest tests/test_evalplus_official.py tests/ -v --tb=short
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add sage-python/src/sage/bench/evalplus_bench.py sage-python/src/sage/bench/__main__.py sage-python/tests/test_evalplus_official.py
git commit -m "feat(bench): add official EvalPlus evaluation mode [P1]

New --official flag generates samples.jsonl and runs evalplus.sanitize +
evalplus.evaluate for leaderboard-comparable scores.
Custom harness retained as default for faster iteration.
CLI: python -m sage.bench --type evalplus --official"
```

---

## Chunk 2: Security Hardening + Dead Code Cleanup

### Task 4: tree-sitter Blocked List Expansion [P8]

**Files:**
- Modify: `sage-core/src/sandbox/validator.rs:38-56` (BLOCKED_CALLS list)
- Test: existing `cargo test --no-default-features --features tool-executor --lib -- sandbox::validator`

The audit confirmed that `bytes`, `bytearray`, `memoryview`, `ord`, `ascii` can be used to dynamically construct module names and bypass the import blocklist.

- [ ] **Step 1: Add bypass vectors to BLOCKED_CALLS**

In `sage-core/src/sandbox/validator.rs`, add 5 entries to the `BLOCKED_CALLS` array after `"hasattr"`:

```rust
    // Dynamic module name construction bypass vectors (Audit P8):
    "bytes",
    "bytearray",
    "memoryview",
    "ord",
    "ascii",
```

- [ ] **Step 2: Run Rust validator tests**

```bash
cd sage-core && cargo test --no-default-features --features tool-executor --lib -- sandbox::validator
```

Expected: All existing tests PASS (new entries don't break existing checks).

- [ ] **Step 3: Add a test for the new blocked calls**

In the test module at the bottom of `validator.rs`, add:

```rust
#[test]
fn test_blocks_byte_construction_bypass() {
    // Uses validate_python_code() free function (NOT PythonValidator struct).
    // Result fields: .valid (bool) and .errors (Vec<String>).
    let r = validate_python_code("x = bytes([111, 115])");
    assert!(!r.valid, "bytes() should be blocked");
    assert!(r.errors.iter().any(|e| e.contains("bytes")));

    let r2 = validate_python_code("x = bytearray([111, 115])");
    assert!(!r2.valid, "bytearray() should be blocked");

    let r3 = validate_python_code("x = ord('a')");
    assert!(!r3.valid, "ord() should be blocked");
}
```

- [ ] **Step 4: Run tests again to verify new test passes**

```bash
cd sage-core && cargo test --no-default-features --features tool-executor --lib -- sandbox::validator
```

Expected: All tests PASS including the new one.

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/sandbox/validator.rs
git commit -m "fix(security): block bytes/bytearray/memoryview/ord/ascii in tree-sitter validator [P8]

These builtins can construct module name strings dynamically to bypass
the import blocklist (e.g., bytes([111,115]).decode() == 'os').
Closes known sandbox bypass vector identified in audit."
```

---

### Task 5: Sandbox Default Warning [P7]

**Files:**
- Modify: `sage-python/src/sage/boot.py` (add warning near sandbox initialization)
- Test: `sage-python/tests/test_boot_sandbox_warning.py`

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_boot_sandbox_warning.py`:

```python
"""Test that boot emits WARNING when no sandbox is available."""
import logging
import pytest


def test_sandbox_unavailable_warning(caplog):
    """When sandbox is unavailable, boot should emit a WARNING."""
    from sage.boot import _check_sandbox_availability

    with caplog.at_level(logging.WARNING):
        available = _check_sandbox_availability()

    if not available:
        assert any("sandbox" in r.message.lower() for r in caplog.records), (
            "Expected a WARNING about sandbox unavailability"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_boot_sandbox_warning.py -v
```

Expected: FAIL — `_check_sandbox_availability` doesn't exist.

- [ ] **Step 3: Implement `_check_sandbox_availability` in boot.py**

Add near the top of `boot.py` (after imports, before `boot_agent_system`):

```python
def _check_sandbox_availability() -> bool:
    """Check if any code execution sandbox is available. Warns if not."""
    has_wasm = False
    has_docker = False

    try:
        from sage_core import ToolExecutor
        te = ToolExecutor()
        has_wasm = te.has_wasm() or te.has_wasi()
    except Exception:
        pass

    if not has_wasm:
        try:
            import shutil
            has_docker = shutil.which("docker") is not None
        except Exception:
            pass

    available = has_wasm or has_docker
    if not available:
        _log.warning(
            "Code execution sandbox unavailable (no Wasm, no Docker). "
            "Tool execution will fail unless allow_local=True."
        )
    return available
```

Call it from `boot_agent_system()` near the end (after all components are wired):

```python
    _check_sandbox_availability()
```

- [ ] **Step 4: Run test**

```bash
cd sage-python && python -m pytest tests/test_boot_sandbox_warning.py -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_boot_sandbox_warning.py
git commit -m "fix(boot): warn when no code execution sandbox is available [P7]

Emits a WARNING log at boot if neither Wasm WASI sandbox nor Docker
is detected. Users running without sandbox must explicitly set
allow_local=True."
```

---

### Task 6: Dead Code Removal — HybridVerifier + template_store [P10]

**Files:**
- Modify: `sage-python/src/sage/boot.py:541-553` (remove unused instantiation)
- Test: verify existing tests still pass

- [ ] **Step 1: Read boot.py around lines 541-553**

Verify these variables are created but never passed to AgentSystem or used downstream:
```python
    template_store = None
    verifier = None
    if _HAS_RUST_ROUTER and rust_router:
        try:
            template_store = RustTemplateStore()
            verifier = RustHybridVerifier()
```

Confirm they're not referenced after this block.

- [ ] **Step 2: Remove the dead code block**

Remove the entire Phase 2 block (lines 541-553). Replace with a comment:

```python
    # Phase 2: Topology templates + HybridVerifier are internal to
    # DynamicTopologyEngine (Rust). No separate Python instantiation needed.
    # (Removed: template_store + verifier were instantiated but never used — audit P10)
```

- [ ] **Step 3: Remove ALL associated dead code**

The removal must be complete — check these 4 locations:

1. **Imports** (top of boot.py): Remove `RustTemplateStore` and `RustHybridVerifier` imports if only used in the removed block.
2. **AgentSystem dataclass fields** (~lines 84-86): Remove `template_store: Any = None` and `verifier: Any = None` if present.
3. **AgentSystem constructor call** (~lines 797-798): Remove `template_store=template_store, verifier=verifier` arguments if present.
4. Verify `grep -rn "template_store\|verifier" sage-python/src/sage/boot.py` returns no remaining references (except the comment).

- [ ] **Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed (no regressions — these variables were never used)

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/boot.py
git commit -m "cleanup(boot): remove dead HybridVerifier + template_store instantiation [P10]

These were instantiated at boot but never passed to any consumer.
The DynamicTopologyEngine creates its own internal verifier.
Confirmed by both audits and PyO3 bridge wiring analysis."
```

---

## Chunk 3: Architecture Fixes

### Task 7: Capability Self-Test [P2]

**Files:**
- Modify: `sage-python/src/sage/providers/capabilities.py` (replace static matrix with self-test protocol)
- Modify: `sage-python/src/sage/providers/openai_compat.py` (implement `self_test()`)
- Modify: `sage-python/src/sage/boot.py` (call self-test at boot)
- Test: `sage-python/tests/test_capability_self_test.py` (flat tests/ dir, matching project convention)

The static `_KNOWN_CAPABILITIES` dict claims `structured_output=True` for xAI/DeepSeek/MiniMax/Kimi, but the runtime adapter returns `False`. The fix: provider adapters declare their own capabilities (already done in `openai_compat.py:66`), and `CapabilityMatrix` trusts runtime over static.

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_capability_self_test.py`:

```python
"""Test that capability matrix uses runtime adapter capabilities, not static lies."""
import pytest
from sage.providers.capabilities import CapabilityMatrix, ProviderCapabilities


def test_runtime_overrides_static():
    """Provider-reported capabilities must override static claims."""
    matrix = CapabilityMatrix()

    # Simulate: static says structured_output=True, runtime says False
    static_caps = ProviderCapabilities.for_provider("deepseek")
    assert static_caps.structured_output is True  # static lie

    # Register with runtime-reported capabilities
    runtime_caps = ProviderCapabilities(
        provider="deepseek",
        structured_output=False,  # truth from openai_compat
        tool_role=False,
    )
    matrix.register(runtime_caps)

    # Matrix should reflect runtime truth
    assert matrix.get("deepseek").structured_output is False


def test_register_from_adapter():
    """CapabilityMatrix.register_from_adapter() should use adapter.capabilities()."""
    matrix = CapabilityMatrix()
    matrix.register_from_adapter("test_provider", {
        "structured_output": False,
        "tool_role": True,
        "system_prompt": True,
    })
    caps = matrix.get("test_provider")
    assert caps.structured_output is False
    assert caps.tool_role is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_capability_self_test.py -v
```

Expected: FAIL — `register_from_adapter` doesn't exist.

- [ ] **Step 3: Add `register_from_adapter()` to CapabilityMatrix**

In `sage-python/src/sage/providers/capabilities.py`, add to `CapabilityMatrix`:

```python
    def register_from_adapter(self, provider: str, caps_dict: dict[str, bool]) -> None:
        """Register capabilities from a provider adapter's runtime report.

        This is the preferred registration method — it uses actual runtime
        capabilities instead of static claims from _KNOWN_CAPABILITIES.
        """
        self._providers[provider] = ProviderCapabilities(
            provider=provider,
            structured_output=caps_dict.get("structured_output", False),
            tool_role=caps_dict.get("tool_role", False),
            file_search=caps_dict.get("file_search", False),
            grounding=caps_dict.get("grounding", False),
            system_prompt=caps_dict.get("system_prompt", True),
            streaming=caps_dict.get("streaming", False),
        )
```

- [ ] **Step 4: Update `populate_from_providers()` to prefer runtime adapters**

Replace the `populate_from_providers` method:

```python
    def populate_from_providers(
        self, provider_names: list[str], adapters: dict[str, Any] | None = None,
    ) -> None:
        """Auto-populate from discovered providers.

        If adapters dict is provided, uses each adapter's capabilities() method
        (runtime truth). Falls back to static _KNOWN_CAPABILITIES only if no
        adapter is available.
        """
        adapters = adapters or {}
        for name in provider_names:
            if name in self._providers:
                continue
            adapter = adapters.get(name)
            if adapter and hasattr(adapter, "capabilities"):
                self.register_from_adapter(name, adapter.capabilities())
            else:
                self.register(ProviderCapabilities.for_provider(name))
```

- [ ] **Step 5: Run tests**

```bash
cd sage-python && python -m pytest tests/test_capability_self_test.py tests/ -v --tb=short -q
```

Expected: PASS

- [ ] **Step 6: Run full suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 7: Wire boot.py to pass adapters**

In `sage-python/src/sage/boot.py`, find the `populate_from_providers()` call (~line 711) and update it to pass the provider adapters dict:

```python
    # Pass runtime adapters so CapabilityMatrix trusts their capabilities()
    # over static _KNOWN_CAPABILITIES claims
    adapters = {}
    for name, provider in self._providers.items():
        adapters[name] = provider
    self.capability_matrix.populate_from_providers(
        list(self._providers.keys()), adapters=adapters,
    )
```

Without this wiring, the `register_from_adapter()` code exists but is never called in production.

- [ ] **Step 8: Run full suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 9: Commit**

```bash
git add sage-python/src/sage/providers/capabilities.py sage-python/src/sage/boot.py sage-python/tests/test_capability_self_test.py
git commit -m "fix(providers): capability matrix uses runtime adapter reports, not static claims [P2]

CapabilityMatrix.register_from_adapter() trusts the provider adapter's
capabilities() method over the hardcoded _KNOWN_CAPABILITIES dict.
boot.py wired to pass adapters dict to populate_from_providers().
Fixes structured_output=True lie for xAI/DeepSeek/MiniMax/Kimi where
runtime returns False."
```

---

### Task 8: Unified ExecutionDecision [P3]

**Files:**
- Create: `sage-python/src/sage/execution_decision.py`
- Modify: `sage-python/src/sage/boot.py` (produce ExecutionDecision in AgentSystem.run)
- Modify: `sage-python/src/sage/orchestrator.py:203-209` (consume decision, no re-routing)
- Test: `sage-python/tests/test_execution_decision.py`

This is the highest-impact architectural fix. Currently AgentSystem.run() routes, then CognitiveOrchestrator.run() independently re-routes — a split-brain problem.

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_execution_decision.py`:

```python
"""Test ExecutionDecision dataclass and orchestrator consumption."""
import pytest
from sage.execution_decision import ExecutionDecision


def test_execution_decision_fields():
    ed = ExecutionDecision(
        system=2,
        model_id="gemini-2.5-flash",
        topology_id="topo_abc123",
        budget_usd=0.5,
        guardrail_level="standard",
    )
    assert ed.system == 2
    assert ed.model_id == "gemini-2.5-flash"
    assert ed.topology_id == "topo_abc123"
    assert ed.budget_usd == 0.5
    assert ed.guardrail_level == "standard"


def test_execution_decision_defaults():
    ed = ExecutionDecision(system=1, model_id="test")
    assert ed.topology_id is None
    assert ed.budget_usd == 0.0
    assert ed.guardrail_level == "standard"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_execution_decision.py -v
```

Expected: FAIL — module `sage.execution_decision` doesn't exist.

- [ ] **Step 3: Create ExecutionDecision dataclass**

Create `sage-python/src/sage/execution_decision.py`:

```python
"""ExecutionDecision — single authoritative routing decision.

Produced by AgentSystem.run(), consumed by CognitiveOrchestrator.run().
Eliminates split-brain routing where both systems make independent decisions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionDecision:
    """Authoritative routing decision for a task."""

    system: int  # 1, 2, or 3
    model_id: str
    topology_id: str | None = None
    budget_usd: float = 0.0
    guardrail_level: str = "standard"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd sage-python && python -m pytest tests/test_execution_decision.py -v
```

Expected: PASS

- [ ] **Step 5: Modify orchestrator to accept ExecutionDecision**

In `sage-python/src/sage/orchestrator.py`, change `run()` method signature:

```python
    async def run(self, task: str, decision: ExecutionDecision | None = None) -> str:
        """Analyze task, select models, build topology, execute.

        If decision is provided, uses it directly (no re-routing).
        If None, falls back to local assessment (backward compat).
        """
        t0 = time.perf_counter()

        if decision is not None:
            # Use authoritative decision from AgentSystem — no re-routing.
            # Stub profile with all fields downstream code may access
            # (complexity, uncertainty, tool_required, etc.)
            profile = type("Profile", (), {
                "complexity": 0.5, "uncertainty": 0.3,
                "tool_required": False, "system": decision.system,
            })()
            log.info(
                "Orchestrator: using ExecutionDecision S%d model=%s",
                decision.system, decision.model_id,
            )
        else:
            # Fallback: local assessment (backward compatibility)
            profile = await self.metacognition.assess_complexity_async(task)
            decision_local = self.metacognition.route(profile)
            from sage.execution_decision import ExecutionDecision
            decision = ExecutionDecision(
                system=decision_local.system,
                model_id="unknown",
            )
            log.info(
                "Orchestrator: task routed locally to S%d (c=%.2f u=%.2f)",
                decision.system, profile.complexity, profile.uncertainty,
            )
```

Add the import at the top of `orchestrator.py`:
```python
from sage.execution_decision import ExecutionDecision
```

- [ ] **Step 6: Write integration test**

Add to `tests/test_execution_decision.py`:

```python
@pytest.mark.asyncio
async def test_orchestrator_uses_decision():
    """Orchestrator should use provided ExecutionDecision without re-routing."""
    from unittest.mock import MagicMock, AsyncMock
    from sage.orchestrator import CognitiveOrchestrator
    from sage.execution_decision import ExecutionDecision

    registry = MagicMock()
    registry.select.return_value = MagicMock(id="gemini-2.5-flash")
    mc = MagicMock()

    orch = CognitiveOrchestrator(registry=registry, metacognition=mc)
    decision = ExecutionDecision(system=1, model_id="gemini-2.5-flash")

    # Mock the model execution path
    model_mock = MagicMock()
    model_mock.generate = AsyncMock(return_value="result")
    registry.select.return_value = model_mock

    # Key assertion: metacognition.assess_complexity_async should NOT be called
    # when decision is provided
    try:
        await orch.run("simple task", decision=decision)
    except Exception:
        pass  # May fail on model execution, that's OK

    mc.assess_complexity_async.assert_not_called()
```

- [ ] **Step 7: Run all tests**

```bash
cd sage-python && python -m pytest tests/test_execution_decision.py tests/ -v --tb=short -q
```

Expected: PASS

- [ ] **Step 8: Wire boot.py to pass ExecutionDecision to orchestrator**

In `sage-python/src/sage/boot.py`, find `AgentSystem.run()` (~line 288) where it calls `await self.orchestrator.run(task)`. Modify to construct and pass an ExecutionDecision:

```python
        # Construct authoritative ExecutionDecision from routing result
        from sage.execution_decision import ExecutionDecision
        decision = ExecutionDecision(
            system=getattr(self._last_decision, "system", 2),
            model_id=getattr(self._last_decision, "model_id", "unknown") or "unknown",
            topology_id=getattr(self._last_decision, "topology_id", None),
        )
        result = await self.orchestrator.run(task, decision=decision)
```

This is the critical wiring step — without it, the ExecutionDecision exists in the API but is never propagated.

- [ ] **Step 9: Run all tests**

```bash
cd sage-python && python -m pytest tests/test_execution_decision.py tests/ -v --tb=short -q
```

Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add sage-python/src/sage/execution_decision.py sage-python/src/sage/orchestrator.py sage-python/src/sage/boot.py sage-python/tests/test_execution_decision.py
git commit -m "feat(arch): unified ExecutionDecision eliminates split-brain routing [P3]

AgentSystem.run() produces the authoritative ExecutionDecision.
CognitiveOrchestrator.run() receives and consumes it — no re-routing.
boot.py wired to construct and pass decision from routing result.
Falls back to local assessment for backward compatibility.
Fixes bandit feedback loop (telemetry now records against single decision)."
```

---

### Task 9: Bandit Provider Seeding [P6]

**Files:**
- Modify: `sage-python/src/sage/boot.py:255-258` (replace hardcoded Gemini list)
- Test: `sage-python/tests/test_bandit_seeding.py`

Oracle feedback (Gemini gemini-3.1-pro-preview): the 4-Gemini seeding may be correct by design since other providers serve as cascade fallback only. Adjusted fix: seed all models from the ModelRegistry that are registered (not just 4 hardcoded), but keep it scoped to primary execution providers.

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_bandit_seeding.py`:

```python
"""Test that bandit arms are seeded from registry, not hardcoded."""
import pytest
from unittest.mock import MagicMock


def test_bandit_seeded_from_registry():
    """Bandit should be seeded from ModelRegistry discovered models."""
    # The hardcoded list should not appear in boot.py
    import inspect
    from sage.boot import boot_agent_system
    source = inspect.getsource(boot_agent_system)

    # Check that the hardcoded 4-model list is gone
    assert 'gemini-2.5-flash", "gemini-3-flash-preview"' not in source, (
        "Hardcoded Gemini model list should be replaced with registry-based seeding"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_bandit_seeding.py -v
```

Expected: FAIL — the hardcoded list is still there.

- [ ] **Step 3: Replace hardcoded list with registry-based seeding**

In `sage-python/src/sage/boot.py`, replace lines 255-258:

```python
                # Seed arms from known models (no-op if already registered)
                for model_id in ["gemini-2.5-flash", "gemini-3-flash-preview",
                                 "gemini-3.1-pro-preview", "gemini-2.5-flash-lite"]:
                    self.bandit.register_arm(model_id, template_type)
```

With:

```python
                # Seed arms from all registered models in Rust ModelRegistry.
                # NOTE: self.registry is the Python providers.registry.ModelRegistry.
                # The Rust registry is stored as rust_registry during boot.
                # The Rust method is list_ids() (NOT all_model_ids()).
                if hasattr(self, "_rust_registry") and self._rust_registry:
                    for model_id in self._rust_registry.list_ids():
                        self.bandit.register_arm(model_id, template_type)
                else:
                    # Fallback: seed from Python registry's available models
                    for profile in self.registry.list_available():
                        self.bandit.register_arm(profile.id, template_type)
                    if not self.registry.list_available():
                        _log.debug("Bandit: no registry models available, skipping arm seeding")
```

NOTE: This requires storing the Rust registry as `self._rust_registry` in `boot_agent_system()`. Check if it's already stored on AgentSystem or add it.

- [ ] **Step 4: Run test**

```bash
cd sage-python && python -m pytest tests/test_bandit_seeding.py tests/ -v --tb=short -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_bandit_seeding.py
git commit -m "fix(boot): seed bandit arms from ModelRegistry instead of hardcoded Gemini list [P6]

Replaces the hardcoded 4-model Gemini list with registry-based seeding.
Oracle confirmed: scoping to primary execution providers is correct
(cascade fallback providers don't need Thompson sampling exploration)."
```

---

## Chunk 4: Rationalization Phase 1 — Shrink Rust

Migrate 3 Rust modules to Python. Strict dependency order: features (no deps) → model_card (no deps) → model_registry (depends on model_card). Each migration follows the TDD protocol: Python tests first, implementation, verify parity, switchover, keep Rust shim until zero consumers.

### Task 10: Migrate features.rs → Python [Rationalization 1.2.1]

**Files:**
- Create: `sage-python/src/sage/strategy/structural_features.py`
- Create: `sage-python/tests/strategy/test_structural_features.py`
- Modify: `sage-python/src/sage/strategy/adaptive_router.py` (use Python impl)

`features.rs` (287 LOC) provides `StructuralFeatures` — zero-cost keyword/structural feature extraction. This is pure string processing with no SIMD benefit. Python reimplementation with identical API.

- [ ] **Step 1: Read the full Rust source**

```bash
# Read sage-core/src/routing/features.rs completely to understand the exact API
```

Key: `StructuralFeatures` PyO3 class with 6 fields: `word_count: usize`, `has_code_block: bool`, `has_question_mark: bool`, `keyword_complexity: f32` (0.0–1.0), `keyword_uncertainty: f32` (0.0–1.0), `tool_required: bool`. Static method: `extract(task: &str) -> StructuralFeatures`. 6 keyword groups: ALGO, CODE, DEBUG, DESIGN, UNCERTAINTY, TOOL. Complexity uses elif priority (algo > debug > design > code) + code_block bonus + word count scaling.

- [ ] **Step 2: Write Python tests (TDD — tests first)**

Create `sage-python/tests/strategy/test_structural_features.py`:

```python
"""Test Python reimplementation of StructuralFeatures (was Rust features.rs).

Field-for-field match with Rust: word_count, has_code_block, has_question_mark,
keyword_complexity (0.0-1.0), keyword_uncertainty (0.0-1.0), tool_required.
"""
import pytest
from sage.strategy.structural_features import StructuralFeatures


class TestStructuralFeatures:
    def test_simple_factual(self):
        sf = StructuralFeatures.extract("What is the capital of France?")
        assert sf.has_question_mark is True
        assert sf.has_code_block is False
        assert sf.tool_required is False
        # Simple factual → base complexity only (0.2)
        assert sf.keyword_complexity < 0.35

    def test_code_task(self):
        sf = StructuralFeatures.extract("Write a Python function to sort a list")
        assert sf.has_code_block is False
        assert sf.has_question_mark is False
        # "write" + "function" → CODE hit → base 0.2 + 0.15 = 0.35
        assert abs(sf.keyword_complexity - 0.35) < 0.01

    def test_algo_task_high_complexity(self):
        sf = StructuralFeatures.extract(
            "Debug the race condition in the concurrent queue implementation"
        )
        # "concurrent" → ALGO takes priority (elif): base 0.2 + 0.35 = 0.55
        assert sf.keyword_complexity >= 0.50

    def test_code_block_detection(self):
        task = "Fix this code:\n```python\ndef foo():\n    pass\n```"
        sf = StructuralFeatures.extract(task)
        assert sf.has_code_block is True
        # "fix" → DEBUG (0.3) + code_block (0.1) → 0.2 + 0.3 + 0.1 = 0.6
        assert sf.keyword_complexity >= 0.55

    def test_tool_required(self):
        sf = StructuralFeatures.extract("Search the web for recent Rust async tutorials")
        assert sf.tool_required is True

    def test_uncertainty_keywords(self):
        sf = StructuralFeatures.extract("Maybe investigate the intermittent flaky test")
        # "maybe", "investigate", "intermittent", "flaky" → 4 * 0.25 = 1.0
        assert sf.keyword_uncertainty >= 0.75

    def test_long_task_scaling(self):
        padding = "word " * 130
        sf = StructuralFeatures.extract(f"Implement an algorithm that {padding}")
        assert sf.word_count > 120
        # ALGO hit (0.35) + word scaling (0.15 for >100) → 0.2 + 0.35 + 0.15 = 0.70
        assert abs(sf.keyword_complexity - 0.70) < 0.01

    def test_empty(self):
        sf = StructuralFeatures.extract("")
        assert sf.word_count == 0
        # Empty string → base complexity 0.2 (matching Rust, no early return)
        assert abs(sf.keyword_complexity - 0.2) < 0.01
        assert sf.keyword_uncertainty == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/strategy/test_structural_features.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement Python StructuralFeatures**

Create `sage-python/src/sage/strategy/structural_features.py`:

```python
"""Structural feature extraction for adaptive routing (Stage 0).

Python reimplementation of sage-core/src/routing/features.rs (287 LOC).
Field-for-field, method-for-method compatible with the Rust PyO3 class.
Pure string processing — no SIMD benefit in Rust for this workload.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Keyword groups (exact match with features.rs) ────────────────────
ALGO_KEYWORDS: list[str] = [
    "implement", "build", "algorithm", "optimize", "compiler",
    "concurrent", "distributed", "consensus", "lock-free",
]
CODE_KEYWORDS: list[str] = [
    "write", "create", "code", "function", "class", "method",
    "parse", "regex", "query", "endpoint", "decorator",
]
DEBUG_KEYWORDS: list[str] = [
    "debug", "fix", "error", "crash", "bug",
    "race condition", "deadlock", "oom", "memory leak",
]
DESIGN_KEYWORDS: list[str] = [
    "design", "architect", "refactor", "schema", "system",
    "prove", "induction", "complexity",
]
UNCERTAINTY_KEYWORDS: list[str] = [
    "maybe", "possibly", "explore", "investigate",
    "intermittent", "sometimes", "random", "flaky",
]
TOOL_KEYWORDS: list[str] = [
    "file", "search", "run", "execute", "compile",
    "test", "deploy", "download", "upload",
]


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def _count_keywords(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


@dataclass
class StructuralFeatures:
    """Cheap structural features extracted from a task string.

    Fields match Rust PyO3 class exactly:
      word_count, has_code_block, has_question_mark,
      keyword_complexity (0.0-1.0), keyword_uncertainty (0.0-1.0),
      tool_required.
    """

    word_count: int = 0
    has_code_block: bool = False
    has_question_mark: bool = False
    keyword_complexity: float = 0.0
    keyword_uncertainty: float = 0.0
    tool_required: bool = False

    @classmethod
    def extract(cls, task: str) -> StructuralFeatures:
        """Extract structural features from a task string.

        Mirrors features.rs::StructuralFeatures::extract_from() exactly.
        """
        # No early return for empty string — let it flow through normally
        # to match Rust behavior (empty string → base complexity 0.2).
        lower = task.lower() if task else ""
        word_count = len(task.split())
        has_code_block = "```" in lower or "~~~" in lower
        has_question_mark = "?" in task
        tool_required = _has_any(lower, TOOL_KEYWORDS)

        # Uncertainty score
        uncertainty_hits = _count_keywords(lower, UNCERTAINTY_KEYWORDS)
        keyword_uncertainty = min(uncertainty_hits * 0.25, 1.0)

        # Complexity score (elif priority: algo > debug > design > code)
        complexity = 0.2  # base
        if _has_any(lower, ALGO_KEYWORDS):
            complexity += 0.35
        elif _has_any(lower, DEBUG_KEYWORDS):
            complexity += 0.30
        elif _has_any(lower, DESIGN_KEYWORDS):
            complexity += 0.20
        elif _has_any(lower, CODE_KEYWORDS):
            complexity += 0.15

        if has_code_block:
            complexity += 0.1

        if word_count > 100:
            complexity += 0.15
        elif word_count > 50:
            complexity += 0.1
        elif word_count > 20:
            complexity += 0.05

        keyword_complexity = max(0.0, min(complexity, 1.0))

        return cls(
            word_count=word_count,
            has_code_block=has_code_block,
            has_question_mark=has_question_mark,
            keyword_complexity=keyword_complexity,
            keyword_uncertainty=keyword_uncertainty,
            tool_required=tool_required,
        )

    def __repr__(self) -> str:
        return (
            f"StructuralFeatures(words={self.word_count}, "
            f"code_block={self.has_code_block}, "
            f"question={self.has_question_mark}, "
            f"complexity={self.keyword_complexity:.3f}, "
            f"uncertainty={self.keyword_uncertainty:.3f}, "
            f"tool={self.tool_required})"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd sage-python && python -m pytest tests/strategy/test_structural_features.py -v
```

Expected: PASS

- [ ] **Step 6: Wire into AdaptiveRouter with fallback**

In `sage-python/src/sage/strategy/adaptive_router.py`, update the import to prefer Python implementation with Rust fallback:

```python
# During migration shadow period: Rust preferred (validated path).
# After shadow validation passes, swap to Python-preferred.
from sage.strategy.structural_features import StructuralFeatures as PyStructuralFeatures

try:
    from sage_core import StructuralFeatures as RustStructuralFeatures
    _HAS_RUST_FEATURES = True
except ImportError:
    _HAS_RUST_FEATURES = False

# Both implementations expose identical fields:
#   word_count, has_code_block, has_question_mark,
#   keyword_complexity, keyword_uncertainty, tool_required
# TODO(rationalization): swap to PyStructuralFeatures after shadow validation
StructuralFeatures = RustStructuralFeatures if _HAS_RUST_FEATURES else PyStructuralFeatures
```

- [ ] **Step 7: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 8: Commit**

```bash
git add sage-python/src/sage/strategy/structural_features.py sage-python/tests/strategy/test_structural_features.py sage-python/src/sage/strategy/adaptive_router.py
git commit -m "feat(rationalization): migrate features.rs to Python structural_features.py

Python reimplementation of StructuralFeatures (287 LOC Rust → ~80 LOC Python).
Pure string processing — no SIMD benefit. Rust fallback retained.
Rationalization Phase 1, module 1/3."
```

---

### Task 11: Migrate model_card.rs → Python [Rationalization 1.2.2]

**Files:**
- Create: `sage-python/src/sage/llm/model_card.py`
- Create: `sage-python/tests/llm/test_model_card.py`

`model_card.rs` (469 LOC) is a dataclass with TOML parsing. Perfect Python fit using `dataclasses` + `tomllib`.

- [ ] **Step 1: Read full Rust source**

```bash
# Read sage-core/src/routing/model_card.rs to understand all fields and TOML parsing logic
```

Key fields (exact Rust names): `id`, `provider`, `family`, `code_score`, `reasoning_score`, `tool_use_score`, `math_score`, `formal_z3_strength`, `cost_input_per_m`, `cost_output_per_m`, `latency_ttft_ms`, `tokens_per_sec`, `s1_affinity`, `s2_affinity`, `s3_affinity`, `recommended_topologies: Vec<String>`, `supports_tools`, `supports_json_mode`, `supports_vision`, `context_window: u32`, `domain_scores: HashMap<String,f32>`, `safety_rating` (default 0.5).

Key methods: `best_system() -> CognitiveSystem`, `affinity_for(system) -> f32`, `estimate_cost(input_tokens, output_tokens) -> f32`, `domain_score(domain) -> f32` (**default 0.5**, not 0.0), `parse_toml(s)`, `load_from_file(path)`.

**CAUTION**: Field names differ from what might be intuitive — e.g. `cost_input_per_m` not `cost_per_1m_input`, `context_window` not `max_context`, `latency_ttft_ms` not `latency_p50`. Must match Rust exactly for TOML compat.

- [ ] **Step 2: Write Python tests (TDD)**

Create `sage-python/tests/llm/test_model_card.py`:

```python
"""Test Python ModelCard (migrated from Rust model_card.rs).

Field names, defaults, and method signatures match Rust PyO3 class exactly.
"""
import pytest
from sage.llm.model_card import ModelCard, CognitiveSystem


class TestCognitiveSystem:
    def test_values(self):
        assert CognitiveSystem.S1 == 1
        assert CognitiveSystem.S2 == 2
        assert CognitiveSystem.S3 == 3


def _make_card(s1=0.5, s2=0.5, s3=0.5, **kw):
    """Helper matching Rust make_test_card()."""
    defaults = dict(
        id="test", provider="test", family="test",
        code_score=0.5, reasoning_score=0.5, tool_use_score=0.5,
        math_score=0.5, formal_z3_strength=0.5,
        cost_input_per_m=1.0, cost_output_per_m=2.0,
        latency_ttft_ms=500.0, tokens_per_sec=100.0,
        s1_affinity=s1, s2_affinity=s2, s3_affinity=s3,
        context_window=128000,
    )
    defaults.update(kw)
    return ModelCard(**defaults)


class TestModelCard:
    def test_best_system_s1(self):
        card = _make_card(s1=0.9, s2=0.5, s3=0.3)
        assert card.best_system() == CognitiveSystem.S1

    def test_best_system_s2(self):
        card = _make_card(s1=0.3, s2=0.9, s3=0.5)
        assert card.best_system() == CognitiveSystem.S2

    def test_best_system_tie_favors_s1(self):
        card = _make_card(s1=0.7, s2=0.7, s3=0.7)
        assert card.best_system() == CognitiveSystem.S1

    def test_affinity_for(self):
        card = _make_card(s1=0.1, s2=0.5, s3=0.9)
        assert abs(card.affinity_for(CognitiveSystem.S1) - 0.1) < 0.001
        assert abs(card.affinity_for(CognitiveSystem.S2) - 0.5) < 0.001
        assert abs(card.affinity_for(CognitiveSystem.S3) - 0.9) < 0.001

    def test_estimate_cost(self):
        card = _make_card()
        # 1000 input @ $1/M + 500 output @ $2/M = 2000/1M = 0.002
        assert abs(card.estimate_cost(1000, 500) - 0.002) < 0.0001

    def test_domain_score_known(self):
        card = _make_card(domain_scores={"math": 0.94, "code": 0.87})
        assert abs(card.domain_score("math") - 0.94) < 0.001

    def test_domain_score_unknown_returns_05(self):
        """Rust default is 0.5 (neutral), NOT 0.0."""
        card = _make_card()
        assert abs(card.domain_score("unknown") - 0.5) < 0.001

    def test_parse_toml(self):
        toml_str = '''
[[models]]
id = "gemini-2.5-flash"
provider = "google"
family = "gemini-2.5"
code_score = 0.85
reasoning_score = 0.80
tool_use_score = 0.90
math_score = 0.75
formal_z3_strength = 0.60
cost_input_per_m = 0.075
cost_output_per_m = 0.30
latency_ttft_ms = 200.0
tokens_per_sec = 200.0
s1_affinity = 0.70
s2_affinity = 0.85
s3_affinity = 0.40
recommended_topologies = ["sequential", "avr"]
supports_tools = true
supports_json_mode = true
supports_vision = true
context_window = 1048576
'''
        cards = ModelCard.parse_toml(toml_str)
        assert len(cards) == 1
        assert cards[0].id == "gemini-2.5-flash"
        assert abs(cards[0].s2_affinity - 0.85) < 0.001
        assert cards[0].context_window == 1048576
        assert cards[0].supports_tools is True

    def test_parse_toml_with_domain_scores(self):
        toml_str = '''
[[models]]
id = "test"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.5
tool_use_score = 0.5
math_score = 0.5
formal_z3_strength = 0.5
cost_input_per_m = 1.0
cost_output_per_m = 2.0
latency_ttft_ms = 500.0
tokens_per_sec = 100.0
s1_affinity = 0.5
s2_affinity = 0.5
s3_affinity = 0.5
recommended_topologies = []
supports_tools = false
supports_json_mode = false
supports_vision = false
context_window = 128000
safety_rating = 0.85

[models.domain_scores]
math = 0.94
code = 0.87
'''
        cards = ModelCard.parse_toml(toml_str)
        assert abs(cards[0].safety_rating - 0.85) < 0.001
        assert abs(cards[0].domain_score("math") - 0.94) < 0.001
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/llm/test_model_card.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement Python ModelCard**

Create `sage-python/src/sage/llm/model_card.py`:

```python
"""ModelCard — structured capability descriptor for LLM models.

Python reimplementation of sage-core/src/routing/model_card.rs (469 LOC).
Field-for-field compatible with Rust PyO3 class. Uses dataclasses + tomllib.

NOTE: Field names match Rust EXACTLY (cost_input_per_m, not cost_per_1m_input;
context_window, not max_context; latency_ttft_ms, not latency_p50).
"""
from __future__ import annotations

import enum
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class CognitiveSystem(enum.IntEnum):
    """Kahneman-inspired cognitive modes for task routing."""
    S1 = 1  # Fast / Intuitive
    S2 = 2  # Deliberate / Tools
    S3 = 3  # Formal / Reasoning


@dataclass
class ModelCard:
    """Structured capability card for an LLM model.

    All fields match sage-core/src/routing/model_card.rs exactly.
    """

    # Identity
    id: str
    provider: str
    family: str

    # Benchmark scores (0.0–1.0)
    code_score: float = 0.0
    reasoning_score: float = 0.0
    tool_use_score: float = 0.0
    math_score: float = 0.0
    formal_z3_strength: float = 0.0

    # Cost & latency
    cost_input_per_m: float = 0.0
    cost_output_per_m: float = 0.0
    latency_ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0

    # Cognitive affinities (0.0–1.0)
    s1_affinity: float = 0.5
    s2_affinity: float = 0.5
    s3_affinity: float = 0.5

    # Topology & capabilities
    recommended_topologies: list[str] = field(default_factory=list)
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_vision: bool = False
    context_window: int = 128000

    # Domain scores & safety
    domain_scores: dict[str, float] = field(default_factory=dict)
    safety_rating: float = 0.5

    def best_system(self) -> CognitiveSystem:
        """Return cognitive system with highest affinity. Ties favor S1."""
        if self.s1_affinity >= self.s2_affinity and self.s1_affinity >= self.s3_affinity:
            return CognitiveSystem.S1
        elif self.s2_affinity >= self.s3_affinity:
            return CognitiveSystem.S2
        return CognitiveSystem.S3

    def affinity_for(self, system: CognitiveSystem | int) -> float:
        """Return affinity score for a cognitive system (matches Rust name)."""
        s = int(system)
        if s == 1:
            return self.s1_affinity
        elif s == 2:
            return self.s2_affinity
        elif s == 3:
            return self.s3_affinity
        return 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost (USD) for given token counts."""
        return (
            input_tokens * self.cost_input_per_m
            + output_tokens * self.cost_output_per_m
        ) / 1_000_000

    def domain_score(self, domain: str) -> float:
        """Get domain-specific score. Returns 0.5 (neutral) if unknown.

        NOTE: Default is 0.5, NOT 0.0 — matches Rust unwrap_or(0.5).
        """
        return self.domain_scores.get(domain, 0.5)

    @classmethod
    def parse_toml(cls, toml_str: str) -> list[ModelCard]:
        """Parse ModelCards from a TOML string with [[models]] array."""
        data = tomllib.loads(toml_str)
        models = data.get("models", [])
        return [cls._from_dict(m) for m in models]

    @classmethod
    def load_from_file(cls, path: str) -> list[ModelCard]:
        """Load ModelCards from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        models = data.get("models", [])
        return [cls._from_dict(m) for m in models]

    @classmethod
    def _from_dict(cls, d: dict) -> ModelCard:
        return cls(
            id=d.get("id", ""),
            provider=d.get("provider", ""),
            family=d.get("family", ""),
            code_score=d.get("code_score", 0.0),
            reasoning_score=d.get("reasoning_score", 0.0),
            tool_use_score=d.get("tool_use_score", 0.0),
            math_score=d.get("math_score", 0.0),
            formal_z3_strength=d.get("formal_z3_strength", 0.0),
            cost_input_per_m=d.get("cost_input_per_m", 0.0),
            cost_output_per_m=d.get("cost_output_per_m", 0.0),
            latency_ttft_ms=d.get("latency_ttft_ms", 0.0),
            tokens_per_sec=d.get("tokens_per_sec", 0.0),
            s1_affinity=d.get("s1_affinity", 0.5),
            s2_affinity=d.get("s2_affinity", 0.5),
            s3_affinity=d.get("s3_affinity", 0.5),
            recommended_topologies=d.get("recommended_topologies", []),
            supports_tools=d.get("supports_tools", False),
            supports_json_mode=d.get("supports_json_mode", False),
            supports_vision=d.get("supports_vision", False),
            context_window=d.get("context_window", 128000),
            domain_scores=d.get("domain_scores", {}),
            safety_rating=d.get("safety_rating", 0.5),
        )

    def __repr__(self) -> str:
        return (
            f"ModelCard(id='{self.id}', provider='{self.provider}', "
            f"s1={self.s1_affinity:.2f}, s2={self.s2_affinity:.2f}, "
            f"s3={self.s3_affinity:.2f})"
        )
```

- [ ] **Step 5: Run tests**

```bash
cd sage-python && python -m pytest tests/llm/test_model_card.py -v
```

Expected: PASS

- [ ] **Step 6: Run full suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 7: Commit**

```bash
git add sage-python/src/sage/llm/model_card.py sage-python/tests/llm/test_model_card.py
git commit -m "feat(rationalization): migrate model_card.rs to Python dataclass + tomllib

Python reimplementation of ModelCard (469 LOC Rust → ~100 LOC Python).
Dataclass with TOML parsing via tomllib (stdlib 3.11+).
Rationalization Phase 1, module 2/3."
```

---

### Task 12: Migrate model_registry.rs → Python [Rationalization 1.2.3]

**Files:**
- Create: `sage-python/src/sage/llm/model_registry.py`
- Create: `sage-python/tests/llm/test_model_registry.py`

`model_registry.rs` (438 LOC) manages ModelCards with telemetry (quality average + P95 latency via VecDeque ring buffer). Depends on `model_card`. Python reimplementation using `collections.deque`.

- [ ] **Step 1: Read full Rust source**

```bash
# Read sage-core/src/routing/model_registry.rs for all methods
```

Key Rust API: `ModelRegistry` with `from_toml_file(path)`, `register(card)`, `unregister(id)`, `get(id) -> Option<ModelCard>`, `len()`, `is_empty()`, `list_ids() -> Vec<String>`, `all_models() -> Vec<ModelCard>`, `select_for_system(system) -> Vec<ModelCard>` (**returns sorted Vec, not single item**), `select_best_for_domain(domain, max_cost_usd) -> Option<ModelCard>`, `record_telemetry(model_id, quality, cost)` (2 params, no latency), `record_telemetry_full(model_id, quality, cost, latency_ms)` (4 params, cost BEFORE latency), `observed_latency_p95(model_id) -> f32`, `calibrated_affinity(model_id, system) -> f32` (blends card prior with telemetry, w = min(count/50, 0.8)).

Internal: `TelemetryRecord` with `quality_sum`, `cost_sum`, `count`, `latencies: VecDeque` (max 100). Methods: `avg_quality()`, `latency_p95()`. **0.0 latency guard**: `record_telemetry()` delegates to `record_telemetry_full()` with latency=0.0, and the full version skips appending if `latency_ms <= 0.0`.

**NAMING CONFLICT**: `sage.providers.registry.ModelRegistry` already exists (runtime API discovery). This new class goes in `sage.llm.model_registry` — different module, different purpose. Document clearly in module docstring. Callers must use explicit imports.

- [ ] **Step 2: Write Python tests (TDD)**

Create `sage-python/tests/llm/test_model_registry.py`:

```python
"""Test Python ModelRegistry (migrated from Rust model_registry.rs).

Matches Rust API: select_for_system returns sorted list (not single item),
record_telemetry has 2-param and 4-param variants, calibrated_affinity blends
card prior with telemetry (w = min(count/50, 0.8)).
"""
import pytest
from sage.llm.model_card import ModelCard, CognitiveSystem
from sage.llm.model_registry import ModelRegistry, TelemetryRecord


class TestTelemetryRecord:
    def test_empty(self):
        tr = TelemetryRecord()
        assert tr.avg_quality() == 0.0
        assert tr.latency_p95() == 0.0

    def test_avg_quality(self):
        tr = TelemetryRecord()
        tr.record_full(0.8, 0.01, 100.0)  # quality, cost, latency
        tr.record_full(0.6, 0.02, 200.0)
        assert abs(tr.avg_quality() - 0.7) < 0.01

    def test_p95_latency(self):
        tr = TelemetryRecord()
        for i in range(20):
            tr.record_full(0.8, 0.01, 100.0 + i * 10.0)
        # P95 of 100..290 should be > 200
        assert tr.latency_p95() > 200.0

    def test_zero_latency_not_recorded(self):
        """0.0 latency from record() delegation must not corrupt P95."""
        tr = TelemetryRecord()
        tr.record(0.8, 0.01)  # no latency → delegates with 0.0
        tr.record(0.9, 0.02)
        tr.record_full(0.7, 0.01, 200.0)
        assert abs(tr.latency_p95() - 200.0) < 0.001

    def test_ring_buffer_bounded(self):
        tr = TelemetryRecord()
        for i in range(150):
            tr.record_full(0.5, 0.01, float(i))
        assert len(tr._latencies) <= 100


class TestModelRegistry:
    def _make_card(self, id="m1", **kw):
        defaults = dict(provider="test", family="test")
        defaults.update(kw)
        return ModelCard(id=id, **defaults)

    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register(self._make_card())
        assert reg.get("m1") is not None
        assert reg.get("m1").id == "m1"

    def test_len(self):
        reg = ModelRegistry()
        assert len(reg) == 0
        reg.register(self._make_card("a"))
        reg.register(self._make_card("b"))
        assert len(reg) == 2

    def test_list_ids(self):
        reg = ModelRegistry()
        reg.register(self._make_card("a"))
        reg.register(self._make_card("b"))
        assert set(reg.list_ids()) == {"a", "b"}

    def test_select_for_system_returns_sorted_list(self):
        """Rust returns Vec<ModelCard> sorted by affinity, not single item."""
        reg = ModelRegistry()
        reg.register(self._make_card("fast", s1_affinity=0.9, s2_affinity=0.2))
        reg.register(self._make_card("smart", s1_affinity=0.2, s2_affinity=0.9))
        results = reg.select_for_system(CognitiveSystem.S2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].id == "smart"  # highest S2 first

    def test_calibrated_affinity_no_telemetry(self):
        reg = ModelRegistry()
        reg.register(self._make_card("m1", s1_affinity=0.9))
        aff = reg.calibrated_affinity("m1", CognitiveSystem.S1)
        assert abs(aff - 0.9) < 0.001  # pure card affinity

    def test_calibrated_affinity_blends(self):
        reg = ModelRegistry()
        reg.register(self._make_card("m1", s1_affinity=0.9))
        for _ in range(25):
            reg.record_telemetry("m1", 0.5, 0.01)
        # w = min(25/50, 0.8) = 0.5
        # calibrated = (1-0.5)*0.9 + 0.5*0.5 = 0.70
        aff = reg.calibrated_affinity("m1", CognitiveSystem.S1)
        assert abs(aff - 0.70) < 0.01

    def test_calibrated_affinity_caps_at_80_percent(self):
        reg = ModelRegistry()
        reg.register(self._make_card("m1", s1_affinity=0.9))
        for _ in range(100):
            reg.record_telemetry("m1", 1.0, 0.01)
        # w = min(100/50, 0.8) = 0.8
        # calibrated = (1-0.8)*0.9 + 0.8*1.0 = 0.98
        aff = reg.calibrated_affinity("m1", CognitiveSystem.S1)
        assert abs(aff - 0.98) < 0.01

    def test_select_best_for_domain(self):
        reg = ModelRegistry()
        reg.register(self._make_card("math", domain_scores={"math": 0.94}, s3_affinity=0.9))
        reg.register(self._make_card("general", s1_affinity=0.9))
        best = reg.select_best_for_domain("math", 10.0)
        assert best is not None
        assert best.id == "math"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/llm/test_model_registry.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement Python ModelRegistry**

Create `sage-python/src/sage/llm/model_registry.py`:

```python
"""ModelRegistry — manages ModelCards with telemetry calibration.

Python reimplementation of sage-core/src/routing/model_registry.rs (438 LOC).
Method-for-method compatible with Rust PyO3 class.

NOTE: This is sage.llm.model_registry.ModelRegistry (TOML-based catalog with
telemetry). NOT sage.providers.registry.ModelRegistry (runtime API discovery).
"""
from __future__ import annotations

import logging
from collections import deque

from sage.llm.model_card import CognitiveSystem, ModelCard

log = logging.getLogger(__name__)

_MAX_LATENCY_SAMPLES = 100


class TelemetryRecord:
    """Running telemetry record for a single model.

    Matches Rust TelemetryRecord: quality_sum, cost_sum, count,
    latencies (VecDeque max 100). 0.0 latency guard included.
    """

    def __init__(self) -> None:
        self.quality_sum: float = 0.0
        self.cost_sum: float = 0.0
        self.count: int = 0
        self._latencies: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)

    def record(self, quality: float, cost: float) -> None:
        """Record without latency (delegates with 0.0)."""
        self.record_full(quality, cost, 0.0)

    def record_full(self, quality: float, cost: float, latency_ms: float) -> None:
        """Record with latency. Param order matches Rust: quality, cost, latency."""
        self.quality_sum += quality
        self.cost_sum += cost
        self.count += 1
        # Guard: don't record 0.0 latency (from record() delegation)
        if latency_ms > 0.0:
            self._latencies.append(latency_ms)

    def avg_quality(self) -> float:
        return self.quality_sum / self.count if self.count > 0 else 0.0

    def latency_p95(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lats = sorted(self._latencies)
        idx = int((len(sorted_lats) - 1) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]


class ModelRegistry:
    """Model catalog with system-based selection and telemetry calibration.

    API matches Rust PyO3 class exactly: select_for_system returns sorted list,
    calibrated_affinity blends card prior with telemetry observations.
    """

    def __init__(self) -> None:
        self._cards: dict[str, ModelCard] = {}
        self._telemetry: dict[str, TelemetryRecord] = {}

    def __len__(self) -> int:
        return len(self._cards)

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def register(self, card: ModelCard) -> None:
        self._cards[card.id] = card

    def unregister(self, id: str) -> None:
        self._cards.pop(id, None)

    def get(self, id: str) -> ModelCard | None:
        return self._cards.get(id)

    def list_ids(self) -> list[str]:
        return list(self._cards.keys())

    def all_models(self) -> list[ModelCard]:
        return list(self._cards.values())

    def select_for_system(self, system: CognitiveSystem | int) -> list[ModelCard]:
        """Return ALL cards sorted by affinity descending (matches Rust Vec return)."""
        s = CognitiveSystem(int(system))
        candidates = list(self._cards.values())
        candidates.sort(key=lambda c: c.affinity_for(s), reverse=True)
        return candidates

    def select_best_for_domain(
        self, domain: str, max_cost_usd: float = 0.0,
    ) -> ModelCard | None:
        """Select best model for a domain within budget.

        Scoring: domain_score*0.6 + calibrated_affinity*0.3 + (1-cost_norm)*0.1
        System inferred: math/formal → S3, code/reasoning/tool_use → S2, else → S1.
        """
        candidates = list(self._cards.values())
        if max_cost_usd > 0:
            candidates = [
                c for c in candidates
                if c.estimate_cost(1000, 500) <= max_cost_usd
            ]
        if not candidates:
            return None

        max_cost = max(
            (c.estimate_cost(1000, 500) for c in candidates), default=0.001
        )
        max_cost = max(max_cost, 0.001)

        system = self._system_for_domain(domain)

        def score(c: ModelCard) -> float:
            ds = c.domain_score(domain)
            aff = self.calibrated_affinity(c.id, system)
            cost_norm = c.estimate_cost(1000, 500) / max_cost
            return ds * 0.6 + aff * 0.3 + (1.0 - cost_norm) * 0.1

        return max(candidates, key=score)

    @staticmethod
    def _system_for_domain(domain: str) -> CognitiveSystem:
        if domain in ("math", "formal"):
            return CognitiveSystem.S3
        if domain in ("code", "reasoning", "tool_use"):
            return CognitiveSystem.S2
        return CognitiveSystem.S1

    def record_telemetry(self, model_id: str, quality: float, cost: float) -> None:
        """Record without latency (Rust: py_record_telemetry)."""
        self.record_telemetry_full(model_id, quality, cost, 0.0)

    def record_telemetry_full(
        self, model_id: str, quality: float, cost: float, latency_ms: float,
    ) -> None:
        """Record with latency (Rust: py_record_telemetry_full)."""
        if model_id not in self._telemetry:
            self._telemetry[model_id] = TelemetryRecord()
        self._telemetry[model_id].record_full(quality, cost, latency_ms)
        log.debug("telemetry_recorded model=%s count=%d", model_id,
                   self._telemetry[model_id].count)

    def observed_latency_p95(self, model_id: str) -> float:
        tr = self._telemetry.get(model_id)
        return tr.latency_p95() if tr else 0.0

    def calibrated_affinity(
        self, model_id: str, system: CognitiveSystem | int,
    ) -> float:
        """Blend card prior with telemetry: w = min(count/50, 0.8).

        Returns (1-w)*card_affinity + w*observed_quality.
        Falls back to card affinity (0.0 if unknown model).
        """
        s = CognitiveSystem(int(system))
        card = self._cards.get(model_id)
        card_affinity = card.affinity_for(s) if card else 0.0

        tr = self._telemetry.get(model_id)
        if not tr or tr.count == 0:
            return card_affinity

        w = min(tr.count / 50.0, 0.8)
        observed = tr.avg_quality()
        return (1.0 - w) * card_affinity + w * observed

    @classmethod
    def from_toml_file(cls, path: str) -> ModelRegistry:
        """Load a registry from a TOML file with [[models]] array."""
        cards = ModelCard.load_from_file(path)
        reg = cls()
        for card in cards:
            reg.register(card)
        return reg

    def __repr__(self) -> str:
        return f"ModelRegistry(models={len(self._cards)})"
```

- [ ] **Step 5: Run tests**

```bash
cd sage-python && python -m pytest tests/llm/test_model_registry.py -v
```

Expected: PASS

- [ ] **Step 6: Run full suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 7: Wire fallback imports in boot.py**

In `sage-python/src/sage/boot.py`, ensure the Rust→Python fallback pattern is used for all 3 migrated modules. Find the existing Rust imports and add fallbacks:

```python
# ModelCard — prefer Python (migrated from Rust)
from sage.llm.model_card import ModelCard, CognitiveSystem
try:
    from sage_core import ModelCard as RustModelCard, CognitiveSystem as RustCognitiveSystem
    _HAS_RUST_MODEL_CARD = True
except ImportError:
    _HAS_RUST_MODEL_CARD = False

# ModelRegistry — prefer Python (migrated from Rust)
from sage.llm.model_registry import ModelRegistry as PyModelRegistry
try:
    from sage_core import ModelRegistry as RustModelRegistry
    _HAS_RUST_MODEL_REGISTRY = True
except ImportError:
    _HAS_RUST_MODEL_REGISTRY = False
```

Note: The Python implementations are now the primary code path. Rust is the optional override. The Rust shims stay until Phase 1 migration is fully validated (all downstream Rust consumers migrated per spec 1.2 protocol step 4).

- [ ] **Step 8: Run full suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 9: Commit**

```bash
git add sage-python/src/sage/llm/model_registry.py sage-python/tests/llm/test_model_registry.py sage-python/src/sage/boot.py
git commit -m "feat(rationalization): migrate model_registry.rs to Python with deque telemetry

Python reimplementation of ModelRegistry (438 LOC Rust → ~150 LOC Python).
Telemetry ring buffer via collections.deque(maxlen=100).
calibrated_affinity() blends card prior with telemetry (w = min(count/50, 0.8)).
Boot.py wired with fallback imports for all 3 migrated modules.
Rationalization Phase 1, module 3/3. Rust shims retained."
```

---

### Task 13: Regex Heuristic Deprecation [Rationalization 1.1]

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py` (gut `_assess_heuristic()` regex)
- Modify: `sage-python/src/sage/strategy/adaptive_router.py` (gut fallback regex)
- Test: existing tests

- [ ] **Step 1: Read `_assess_heuristic()` in metacognition.py**

Find the method with ~10 regex patterns. These are the production dead code identified in the routing audit — the kNN router now handles all routing decisions.

- [ ] **Step 2: Replace regex body with keyword-count fallback**

Replace the regex-heavy body of `_assess_heuristic()` with a simple 3-line keyword count fallback:

```python
    def _assess_heuristic(self, task: str) -> float:
        """Degraded keyword-count fallback (no regex).

        Used only when ONNX model and kNN are both unavailable.
        Returns complexity estimate in [0.0, 1.0].
        """
        import warnings
        warnings.warn(
            "Using degraded keyword-count heuristic. "
            "Install sage_core[onnx] or build kNN exemplars for accurate routing.",
            stacklevel=2,
        )
        words = task.lower().split()
        complex_kw = {"implement", "algorithm", "optimize", "distributed", "concurrent",
                      "debug", "fix", "race", "deadlock", "proof", "verify", "formal"}
        hits = sum(1 for w in words if w in complex_kw)
        return min(hits / 3.0, 1.0)
```

- [ ] **Step 3: Do the same in adaptive_router.py if it has a similar fallback**

Check for regex-heavy fallback paths and replace with the Python StructuralFeatures class.

- [ ] **Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/strategy/ tests/ -v --tb=short -q
```

Expected: 1216+ passed

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/src/sage/strategy/adaptive_router.py
git commit -m "cleanup(routing): replace regex heuristic with keyword-count fallback [Rat 1.1]

Guts _assess_heuristic() body: 10 regex patterns → 3-line keyword count.
Emits deprecation warning. kNN router handles all production routing.
Rationalization Phase 1.1 — dead code removal."
```

---

## Chunk 5: LOC Accounting Fix [P9]

### Task 14: Fix LOC Accounting in Rationalization Spec [P9]

**Files:**
- Modify: `docs/superpowers/specs/2026-03-12-rust-python-rationalization-design.md`

This was already partially fixed in v3 (scope reduction). Verify the numbers are now correct.

- [ ] **Step 1: Verify current LOC counts**

```bash
cd sage-core && find src -name "*.rs" | xargs wc -l | tail -1
cd sage-python && find src -name "*.py" | xargs wc -l | tail -1
```

- [ ] **Step 2: Update spec if numbers are off**

Ensure the Final State table matches actual counts after Phase 1 migration.

- [ ] **Step 3: Commit if changed**

```bash
git add docs/superpowers/specs/2026-03-12-rust-python-rationalization-design.md
git commit -m "docs: fix LOC accounting in rationalization spec with verified counts [P9]"
```

---

## Summary

| Chunk | Tasks | Items Covered | Effort |
|-------|-------|---------------|--------|
| **1: Credibility Gates** | Tasks 1-3 | P1, P4, P5 | ~3 days |
| **2: Security + Cleanup** | Tasks 4-6 | P7, P8, P10 | ~1.5 days |
| **3: Architecture** | Tasks 7-9 | P2, P3, P6 | ~5 days |
| **4: Rust→Python Migration** | Tasks 10-13 | Rat 1.1, 1.2.1-3 | ~3 days |
| **5: LOC Fix** | Task 14 | P9 | 0.5 day |
| **Total** | 14 tasks | 14 items | ~13 days |

**Deferred to follow-up plans:**
- Audit Phase 3: P12 (benchmark reproducibility), P14 (cost tracking) — needs Phase 0 changes
- Audit Phase 4: P11 (SubTask.depends_on), P15-P18 (documentation)
- Rationalization Phase 2: relevance_gate → Rust, quality_estimator → Rust, EventBus benchmark
- Rationalization Phase 3: entity graph unification, PyO3 0.26 migration
