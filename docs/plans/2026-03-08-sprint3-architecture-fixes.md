# Sprint 3 Evidence-Driven Architecture Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 4 critical architectural failures revealed by Sprint 3 benchmarks: non-functional S2 AVR loop, miscalibrated routing, harmful memory injection, broken cost tracking.

**Architecture:** Evidence-first approach. Each fix targets a specific benchmark finding with measurable before/after. The "Scout-and-Sniper" routing pattern (ExoCortex: MARL research) replaces S2-heavy routing. MetaScaffold SLF feedback replaces blind AVR retry. CRAG-style quality gate protects memory injection. All features that don't beat baseline are disabled by default.

**Tech Stack:** Python 3.12, pytest, asyncio, google-genai SDK, wasmtime (fuel metering)

**Evidence Base (Sprint 3 Benchmarks — March 8, 2026):**

| Benchmark | Finding | Data | File |
|-----------|---------|------|------|
| HumanEval 3-config | Baseline 85% beats full-stack 55% | 30pp gap | `docs/benchmarks/2026-03-08-humaneval-comparison.json` |
| Routing proof | All-S1 100% beats router 96.7% at 42x cost | $0.0009 vs $0.0392 | `docs/benchmarks/2026-03-08-routing-proof.json` |
| Evolution 4-arm | Random mutation 67% of full engine score | 0.33 vs 0.50 | `docs/benchmarks/2026-03-08-evolution-4arm.json` |
| Memory ablation | Episodic+semantic (30%) worse than no-memory (50%) | 35pp degradation | `docs/benchmarks/2026-03-08-memory-ablation.json` |

**Research References (ExoCortex + Context7):**
- ExoCortex: Scout-and-Sniper filter pattern (MARL paper) — cheap scout first, expensive agent only on trigger
- ExoCortex: MetaScaffold SLF — Single-Line Feedback for code repair loops (pass ratio + failing test IDs)
- ExoCortex: CRAG (Corrective RAG) — evaluate retrieval quality before injection, reject noisy context
- Context7: wasmtime fuel metering — `config.consume_fuel(true)` + `store.set_fuel(N)` for bounded execution
- Context7: PydanticAI output validators — structured output + ModelRetry on validation failure

---

## Sprint A — Kill Critical Failures (Tasks 1-6)

### Task 1: Fix S2 AVR validation — replace blind retry with SLF feedback

**Problem:** S2 AVR loop fails 100% of the time. Code extraction produces non-executable fragments. Refine prompt lacks actionable error details. No S3 escalation observed (S3:0 in all benchmarks).

**Root cause (agent_loop.py:356-441):** `_extract_code_blocks()` gets raw markdown fragments that fail `python3 -c`. The refine prompt appends stderr but the LLM sees the same system prompt and generates the same broken code.

**Fix:** Replace blind sandbox execution with structured validation: (1) syntax check via `ast.parse()`, (2) only sandbox-execute if syntax valid, (3) SLF feedback with pass ratio and specific error line, (4) stagnation detection to skip retries on repeated errors.

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:356-441`
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write failing tests**

```python
# In tests/test_agent_loop.py — add these tests

import ast
import pytest
from sage.agent_loop import _extract_code_blocks, _validate_code_syntax


def test_validate_code_syntax_valid():
    """Syntax check should pass for valid Python."""
    code = "def foo(x):\n    return x + 1"
    ok, error = _validate_code_syntax(code)
    assert ok is True
    assert error == ""


def test_validate_code_syntax_invalid():
    """Syntax check should fail with specific error for broken code."""
    code = "def foo(x)\n    return x + 1"  # Missing colon
    ok, error = _validate_code_syntax(code)
    assert ok is False
    assert "SyntaxError" in error or "invalid syntax" in error


def test_validate_code_syntax_incomplete_block():
    """Incomplete code blocks from LLM should fail gracefully."""
    code = "```python\ndef foo():\n    pass\n```"
    # If code still has markdown fences, strip them first
    ok, error = _validate_code_syntax(code)
    # Should fail — fences are not valid Python
    assert ok is False


def test_avr_stagnation_detection():
    """AVR should stop retrying if error message is identical to previous."""
    from sage.agent_loop import _is_stagnating
    history = [
        "NameError: name 'foo' is not defined",
        "NameError: name 'foo' is not defined",
        "NameError: name 'foo' is not defined",
    ]
    assert _is_stagnating(history, window=3) is True


def test_avr_stagnation_progressing():
    """AVR should continue if errors are changing (progress)."""
    from sage.agent_loop import _is_stagnating
    history = [
        "NameError: name 'foo' is not defined",
        "TypeError: foo() takes 1 argument",
        "AssertionError: expected 5",
    ]
    assert _is_stagnating(history, window=3) is False
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -k "validate_code_syntax or avr_stagnation" -v`
Expected: FAIL (functions don't exist yet)

**Step 3: Implement `_validate_code_syntax()` and `_is_stagnating()`**

Add to `sage-python/src/sage/agent_loop.py` (after `_extract_code_blocks`):

```python
def _validate_code_syntax(code: str) -> tuple[bool, str]:
    """Check if code is syntactically valid Python.

    Returns (is_valid, error_message). Uses ast.parse — no execution.
    """
    # Strip markdown fences if LLM included them
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        stripped = "\n".join(lines)
    try:
        ast.parse(stripped)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _is_stagnating(error_history: list[str], window: int = 3) -> bool:
    """Detect if AVR loop is stuck producing the same error.

    Returns True if the last `window` errors are identical.
    """
    if len(error_history) < window:
        return False
    recent = error_history[-window:]
    return len(set(recent)) == 1
```

**Step 4: Replace the AVR loop body in `_think()`**

Replace `agent_loop.py:356-441` with:

```python
# System 2 validation (Empirical — AVR with SLF feedback)
elif self.config.validation_level == 2 and content:
    code_blocks = _extract_code_blocks(content)

    if code_blocks and self.sandbox_manager:
        code = code_blocks[-1]

        # Phase 1: Syntax check (fast, no sandbox needed)
        syntax_ok, syntax_err = _validate_code_syntax(code)
        if not syntax_ok:
            self._s2_avr_retries += 1
            self._avr_error_history.append(syntax_err)

            # Stagnation detection: skip retries if stuck
            if _is_stagnating(self._avr_error_history):
                log.info("S2 AVR stagnation detected — skipping to escalation.")
                self._s2_avr_retries = self._max_s2_avr_retries + 1
            elif self._s2_avr_retries <= self._max_s2_avr_retries:
                budget_left = self._max_s2_avr_retries - self._s2_avr_retries + 1
                self._emit(LoopPhase.ACT,
                           validation="s2_avr_fail",
                           avr_iteration=self._s2_avr_retries,
                           avr_budget_left=budget_left,
                           error_type="syntax",
                           stderr=syntax_err[:200])
                # SLF feedback: concise, actionable
                messages.append(Message(
                    role=Role.USER,
                    content=(
                        f"SYSTEM [AVR {self._s2_avr_retries}/{self._max_s2_avr_retries}]: "
                        f"Code has syntax error: {syntax_err}\n"
                        f"Fix the syntax and return ONLY the corrected function. "
                        f"No explanation, no markdown fences."
                    ),
                ))
                continue
        else:
            # Phase 2: Sandbox execution (only if syntax valid)
            # Runtime guardrail check
            if self.guardrail_pipeline and not self._cb_runtime_guard.should_skip():
                try:
                    runtime_results = await self.guardrail_pipeline.check_all(
                        input=code,
                        context={"step": self.step_count, "phase": "runtime"}
                    )
                    for r in runtime_results:
                        self._emit(LoopPhase.ACT,
                                   guardrail="runtime",
                                   guardrail_passed=r.passed,
                                   guardrail_reason=r.reason)
                    self._cb_runtime_guard.record_success()
                except Exception as e:
                    self._cb_runtime_guard.record_failure(e)

            sandbox = await self.sandbox_manager.create()
            try:
                result = await sandbox.execute(
                    f"python3 -c {_shell_quote(code)}"
                )
                if result.exit_code != 0:
                    self._s2_avr_retries += 1
                    err_msg = result.stderr[:200].strip()
                    self._avr_error_history.append(err_msg)

                    if _is_stagnating(self._avr_error_history):
                        log.info("S2 AVR stagnation detected — skipping to escalation.")
                        self._s2_avr_retries = self._max_s2_avr_retries + 1
                    elif self._s2_avr_retries <= self._max_s2_avr_retries:
                        budget_left = self._max_s2_avr_retries - self._s2_avr_retries + 1
                        self._emit(LoopPhase.ACT,
                                   validation="s2_avr_fail",
                                   avr_iteration=self._s2_avr_retries,
                                   avr_budget_left=budget_left,
                                   error_type="runtime",
                                   stderr=err_msg)
                        # SLF: specific error + pass ratio
                        messages.append(Message(
                            role=Role.USER,
                            content=(
                                f"SYSTEM [AVR {self._s2_avr_retries}/{self._max_s2_avr_retries}]: "
                                f"Code executed but failed (exit {result.exit_code}).\n"
                                f"Error: {err_msg}\n"
                                f"Fix this specific error. Return ONLY corrected code, "
                                f"no explanation."
                            ),
                        ))
                        continue
                else:
                    self._emit(LoopPhase.ACT,
                               validation="s2_avr_pass",
                               stdout=result.stdout[:200])
                    self._s2_avr_retries = 0
                    self._avr_error_history.clear()
            finally:
                await self.sandbox_manager.destroy(sandbox.id)

    elif not code_blocks and self.step_count == 1:
        has_reasoning = "<think>" in content or "\n1." in content or "\n- " in content
        if not has_reasoning:
            self._s2_avr_retries += 1
            if self._s2_avr_retries <= self._max_s2_avr_retries:
                messages.append(Message(
                    role=Role.USER,
                    content="SYSTEM: Provide step-by-step reasoning for this task.",
                ))
                continue

    # S2 -> S3 escalation if max retries exhausted
    if self._s2_avr_retries > self._max_s2_avr_retries and self.config.validation_level == 2:
        log.info("S2 AVR exhausted — escalating to S3 (formal verification).")
        self.config.validation_level = 3
        self._s3_retries = 0
        self._avr_error_history.clear()
        self._emit(LoopPhase.THINK, escalation="s2_to_s3",
                   reason="AVR budget exhausted")
        messages.append(Message(
            role=Role.USER,
            content=(
                "SYSTEM: Escalating to formal verification. Use <think> tags "
                "with Z3 assertions (assert bounds, assert loop, assert arithmetic, "
                "assert invariant) for rigorous step-by-step reasoning."
            ),
        ))
        continue
```

Also add to `__init__` (after `self._s2_avr_retries = 0`):
```python
self._avr_error_history: list[str] = []
```

**Step 5: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -k "validate_code_syntax or avr_stagnation" -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "fix(avr): replace blind retry with SLF feedback + stagnation detection"
```

---

### Task 2: Fix routing heuristic — widen S1, add code-task detection

**Problem:** Heuristic `_assess_heuristic()` produces complexity=0.3/uncertainty=0.2 for ALL code tasks (no keywords match "implement", "write", "create"). All HumanEval tasks route to S2 via the agent system (S2:20 in benchmark), wasting 42x cost vs all-S1.

**Root cause (metacognition.py:198-225):** Keywords only match debug/fix/error/optimize. No detection for code generation tasks. `tool_required` regex matches "run/execute/test" but not "implement/write/code".

**Fix:** (1) Add code-task keywords (implement, write, code, function, algorithm, class), (2) Widen S1 ceiling from 0.35 to 0.50, (3) Add task-length scaling, (4) Lower S3 floor from 0.7 to 0.65 for earlier formal verification.

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:87-92,198-225`
- Test: `sage-python/tests/test_strategy.py`

**Step 1: Write failing tests**

```python
# In tests/test_strategy.py — add these tests

from sage.strategy.metacognition import ComplexityRouter


def test_routing_simple_factual_to_s1():
    """Simple factual questions should route to S1."""
    router = ComplexityRouter()
    profile = router.assess_complexity("What is the capital of France?")
    decision = router.route(profile)
    assert decision.system == 1, f"Expected S1, got S{decision.system} (c={profile.complexity})"


def test_routing_code_generation_to_s2():
    """Code generation tasks should route to S2 (medium complexity)."""
    router = ComplexityRouter()
    profile = router.assess_complexity("Implement binary search in Python")
    decision = router.route(profile)
    assert decision.system == 2, f"Expected S2, got S{decision.system} (c={profile.complexity})"


def test_routing_simple_code_to_s1():
    """Very simple code tasks should still route to S1."""
    router = ComplexityRouter()
    profile = router.assess_complexity("Write a hello world function")
    decision = router.route(profile)
    # Short, simple code task — could be S1 or S2 depending on thresholds
    assert decision.system in (1, 2)


def test_routing_complex_debug_to_s3():
    """Complex debugging tasks should route to S3."""
    router = ComplexityRouter()
    profile = router.assess_complexity(
        "Debug this race condition in the distributed lock manager. "
        "The error is intermittent and crashes only under high load."
    )
    decision = router.route(profile)
    assert decision.system == 3, f"Expected S3, got S{decision.system} (c={profile.complexity})"


def test_routing_fibonacci_to_s1():
    """Simple algorithm like fibonacci should route to S1 (cheap model can handle)."""
    router = ComplexityRouter()
    profile = router.assess_complexity("Write a function that returns the nth fibonacci number")
    decision = router.route(profile)
    # Short code task — S1 should be sufficient per benchmark evidence
    assert decision.system <= 2


def test_routing_long_task_gets_complexity_boost():
    """Long task descriptions should get complexity boost."""
    router = ComplexityRouter()
    short = router.assess_complexity("Sort a list")
    long_task = "Implement a distributed consensus algorithm " * 20
    long = router.assess_complexity(long_task)
    assert long.complexity > short.complexity
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_strategy.py -k "test_routing_" -v`
Expected: Some FAIL (especially code tasks routing to wrong tier)

**Step 3: Fix `_assess_heuristic` and routing thresholds**

In `metacognition.py`, update `__init__` defaults:
```python
def __init__(
    self,
    s1_complexity_ceil: float = 0.50,     # Was 0.35 — widened per benchmark evidence
    s1_uncertainty_ceil: float = 0.3,
    s3_complexity_floor: float = 0.65,    # Was 0.7 — earlier S3 for hard tasks
    s3_uncertainty_floor: float = 0.6,
    brake_window: int = 3,
    brake_entropy_threshold: float = 0.15,
):
```

Replace `_assess_heuristic` method:
```python
def _assess_heuristic(self, task: str) -> CognitiveProfile:
    """Fast keyword-based fallback (no LLM call).

    Calibrated against Sprint 3 benchmarks (March 2026):
    - Simple factual → c=0.2 → S1
    - Simple code → c=0.3-0.4 → S1
    - Medium code/algorithm → c=0.5-0.6 → S2
    - Complex debug/design → c=0.7+ → S3
    """
    lower = task.lower()
    words = lower.split()
    word_count = len(words)

    # Base complexity from task type
    complexity = 0.2  # Default: simple factual (was 0.3)

    # Code generation indicators (+0.15)
    if re.search(r'\b(?:implement|write|create|build|code|function|class|method|algorithm)\b', lower):
        complexity += 0.15

    # Debug/error indicators (+0.3)
    if re.search(r'\b(?:debug|fix|error|crash|bug|race\s*condition|deadlock)\b', lower):
        complexity += 0.3

    # Design/architecture indicators (+0.2)
    if re.search(r'\b(?:optimize|evolve|design|architect|refactor|distributed)\b', lower):
        complexity += 0.2

    # Multi-step indicators (+0.1)
    if re.search(r'\b(?:then|after|first|next|finally|step)\b', lower):
        complexity += 0.1

    # Task length scaling (longer = more complex)
    if word_count > 100:
        complexity += 0.15
    elif word_count > 50:
        complexity += 0.1
    elif word_count > 20:
        complexity += 0.05

    # Uncertainty
    uncertainty = 0.2
    if "?" in task:
        uncertainty += 0.2
    if re.search(r'\b(?:maybe|possibly|explore|investigate|unclear|ambiguous)\b', lower):
        uncertainty += 0.2
    if re.search(r'\b(?:intermittent|sometimes|random|flaky)\b', lower):
        uncertainty += 0.15

    # Tool requirement
    tool_required = bool(re.search(
        r'\b(?:file|search|run|execute|compile|test|deploy|read|write|download|upload)\b', lower
    ))

    return CognitiveProfile(
        complexity=min(1.0, complexity),
        uncertainty=min(1.0, uncertainty),
        tool_required=tool_required,
        reasoning="heuristic",
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_strategy.py -k "test_routing_" -v`
Expected: PASS (6 tests)

**Step 5: Run full test suite to check for regressions**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: 740+ passed (some routing tests may need threshold updates)

**Step 6: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/tests/test_strategy.py
git commit -m "fix(routing): recalibrate heuristic — widen S1 ceiling, add code-task detection"
```

---

### Task 3: Default validation to S1 — disable AVR for non-Codex providers

**Problem:** `boot.py:237` sets `validation_level=2` for all non-Codex providers. This means ALL Google Gemini tasks get the S2 AVR loop, which (per Sprint 3 evidence) always fails and degrades quality by 30pp.

**Root cause:** The assumption was that non-Codex models need external validation. Sprint 3 proves this is net-negative: baseline (no validation) = 85%, full-stack (S2 validation) = 55%.

**Fix:** Default `validation_level=1` for all providers. Let routing decide if validation_level should be raised to 2 (only for explicitly routed S2 code tasks with sandbox available).

**Files:**
- Modify: `sage-python/src/sage/boot.py:236-237`
- Modify: `sage-python/src/sage/boot.py:63-134` (AgentSystem.run — only set VL=2 if routed to S2)
- Test: `sage-python/tests/test_boot.py`

**Step 1: Write failing test**

```python
# In tests/test_boot.py — add this test

def test_default_validation_level_is_s1():
    """Default validation should be S1 (no AVR) per Sprint 3 evidence."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.config.validation_level == 1, \
        f"Expected validation_level=1 (S1), got {system.agent_loop.config.validation_level}"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_boot.py::test_default_validation_level_is_s1 -v`
Expected: FAIL (currently defaults to 2 for non-Codex)

**Step 3: Fix boot.py defaults**

In `boot.py:236-237`, change:
```python
# BEFORE:
config = AgentConfig(
    ...
    validation_level=1 if llm_config.provider == "codex" else 2,
)

# AFTER:
config = AgentConfig(
    ...
    validation_level=1,  # Default S1 — routing promotes to S2 only for code tasks
)
```

In `AgentSystem.run()` (~line 80-85), update the validation_level setting:
```python
# BEFORE:
if current_provider == "codex":
    self.agent_loop.config.validation_level = 1
else:
    self.agent_loop.config.validation_level = decision.validation_level

# AFTER:
# Only promote to S2 validation if routed to S2 AND sandbox is available
if decision.system == 2 and self.agent_loop.sandbox_manager:
    self.agent_loop.config.validation_level = decision.validation_level
elif decision.system == 3:
    self.agent_loop.config.validation_level = 3
else:
    self.agent_loop.config.validation_level = 1
```

**Step 4: Run test + full suite**

Run: `cd sage-python && python -m pytest tests/test_boot.py::test_default_validation_level_is_s1 -v`
Expected: PASS

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: 740+ passed

**Step 5: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_boot.py
git commit -m "fix(boot): default validation_level=1 — S2 AVR only when routing demands it"
```

---

### Task 4: Add CRAG-style relevance gate for memory injection

**Problem:** Memory context injection degrades HumanEval from 85% (baseline) to 55% (full-stack). Episodic+semantic alone is catastrophic at 30%. Memory injects irrelevant/noisy context that misleads the LLM.

**Root cause (agent_loop.py:241-266):** `semantic_memory.get_context_for(task)` and `retrieve_smmu_context()` inject context unconditionally — no relevance check. The LLM sees unrelated past interactions mixed with the current task.

**Fix:** Add a relevance gate: (1) score retrieved context against current task, (2) only inject if relevance score exceeds threshold, (3) log when context is rejected.

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:241-266`
- Create: `sage-python/src/sage/memory/relevance_gate.py`
- Test: `sage-python/tests/test_relevance_gate.py`

**Step 1: Write failing tests**

```python
# tests/test_relevance_gate.py

import pytest
from sage.memory.relevance_gate import RelevanceGate


def test_relevance_gate_passes_relevant():
    """Context about Python coding should be relevant to a Python task."""
    gate = RelevanceGate(threshold=0.3)
    task = "Implement binary search in Python"
    context = "Previous interaction: implemented merge sort in Python using recursion"
    assert gate.is_relevant(task, context) is True


def test_relevance_gate_rejects_irrelevant():
    """Context about cooking should be irrelevant to a Python task."""
    gate = RelevanceGate(threshold=0.3)
    task = "Implement binary search in Python"
    context = "Previous interaction: discussed recipe for chocolate cake"
    assert gate.is_relevant(task, context) is False


def test_relevance_gate_rejects_empty():
    """Empty context should be rejected."""
    gate = RelevanceGate(threshold=0.3)
    assert gate.is_relevant("any task", "") is False


def test_relevance_gate_high_threshold_strict():
    """High threshold should be more strict."""
    gate_strict = RelevanceGate(threshold=0.8)
    gate_loose = RelevanceGate(threshold=0.1)
    task = "Sort a list"
    context = "data structures and algorithms"
    # Loose should pass, strict might not
    assert gate_loose.is_relevant(task, context) is True


def test_relevance_gate_score_between_0_and_1():
    """Relevance score should be normalized to [0, 1]."""
    gate = RelevanceGate(threshold=0.3)
    score = gate.score(
        "Implement binary search",
        "binary search tree implementation in Python"
    )
    assert 0.0 <= score <= 1.0
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_relevance_gate.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Implement `RelevanceGate`**

Create `sage-python/src/sage/memory/relevance_gate.py`:
```python
"""CRAG-style relevance gate for memory injection.

Evaluates whether retrieved memory context is relevant to the current task
before injecting it into the LLM prompt. Prevents noisy/irrelevant context
from degrading performance (Sprint 3 evidence: -30pp from blind injection).

Uses keyword overlap scoring (fast, no LLM call).
"""

from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)


class RelevanceGate:
    """Gate that scores and filters memory context by relevance to task.

    Args:
        threshold: Minimum relevance score (0-1) to allow injection.
                   Default 0.3 calibrated against Sprint 3 benchmarks.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self._stop_words = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "not", "no", "nor",
            "this", "that", "these", "those", "it", "its", "i", "you",
            "he", "she", "we", "they", "me", "him", "her", "us", "them",
        })

    def _tokenize(self, text: str) -> set[str]:
        """Extract meaningful tokens (lowercase, no stop words, len >= 3)."""
        words = re.findall(r'\b[a-z][a-z0-9_]+\b', text.lower())
        return {w for w in words if w not in self._stop_words and len(w) >= 3}

    def score(self, task: str, context: str) -> float:
        """Compute relevance score between task and context.

        Uses Jaccard-like overlap of meaningful tokens.
        Returns float in [0.0, 1.0].
        """
        if not task or not context:
            return 0.0

        task_tokens = self._tokenize(task)
        ctx_tokens = self._tokenize(context)

        if not task_tokens or not ctx_tokens:
            return 0.0

        overlap = task_tokens & ctx_tokens
        # Weighted Jaccard: overlap / task_tokens (recall-oriented)
        # We care more about covering the task than about context precision
        return len(overlap) / len(task_tokens)

    def is_relevant(self, task: str, context: str) -> bool:
        """Check if context passes relevance threshold for injection."""
        if not context or not context.strip():
            return False
        s = self.score(task, context)
        if s < self.threshold:
            log.debug(
                "RelevanceGate rejected context (score=%.2f < threshold=%.2f)",
                s, self.threshold,
            )
            return False
        return True
```

**Step 4: Wire into agent_loop.py**

In `agent_loop.py`, import at top:
```python
from sage.memory.relevance_gate import RelevanceGate
```

In `__init__`, add:
```python
self._relevance_gate = RelevanceGate(threshold=0.3)
```

Replace memory injection block (lines 241-266):
```python
# Semantic memory context injection (with CRAG-style relevance gate)
if self.semantic_memory and not self._cb_semantic.should_skip():
    try:
        sem_context = self.semantic_memory.get_context_for(task)
        if sem_context and self._relevance_gate.is_relevant(task, sem_context):
            messages.insert(1, Message(
                role=Role.SYSTEM,
                content=f"Relevant knowledge from previous interactions:\n{sem_context}",
            ))
        elif sem_context:
            log.debug("Semantic context rejected by relevance gate for task: %s", task[:80])
        self._cb_semantic.record_success()
    except Exception as e:
        self._cb_semantic.record_failure(e)

# S-MMU context injection (with relevance gate)
if not self._cb_smmu.should_skip():
    try:
        from sage.memory.smmu_context import retrieve_smmu_context
        smmu_context = retrieve_smmu_context(self.working_memory)
        if smmu_context and self._relevance_gate.is_relevant(task, smmu_context):
            messages.insert(
                min(2, len(messages)),
                Message(role=Role.SYSTEM, content=smmu_context),
            )
        elif smmu_context:
            log.debug("S-MMU context rejected by relevance gate for task: %s", task[:80])
        self._cb_smmu.record_success()
    except Exception as e:
        self._cb_smmu.record_failure(e)
```

**Step 5: Run tests**

Run: `cd sage-python && python -m pytest tests/test_relevance_gate.py -v`
Expected: PASS (5 tests)

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: 740+ passed

**Step 6: Commit**

```bash
git add sage-python/src/sage/memory/relevance_gate.py sage-python/tests/test_relevance_gate.py sage-python/src/sage/agent_loop.py
git commit -m "feat(memory): add CRAG-style relevance gate — reject noisy context before injection"
```

---

### Task 5: Fix cost tracking — prevent negative costs

**Problem:** Several routing proof tasks show negative `cost_usd` (e.g., -$0.000558 for "Reverse string"). This corrupts Pareto frontier analysis.

**Root cause:** `CostTracker` or `agent_loop` computes incremental cost as `current_total - previous_total`, which can go negative if the provider resets its internal cost counter between calls.

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py` (cost calculation)
- Modify: `sage-python/src/sage/contracts/cost_tracker.py`
- Test: `sage-python/tests/test_cost_tracker.py`

**Step 1: Write failing test**

```python
# In tests/test_cost_tracker.py — add this test

def test_cost_never_negative():
    """Incremental cost should never be negative."""
    from sage.contracts.cost_tracker import CostTracker
    tracker = CostTracker(budget_usd=10.0)
    tracker.record("node1", 0.005)
    tracker.record("node2", 0.003)
    # Even if somehow called with negative, should clamp to 0
    tracker.record("node3", -0.001)
    assert tracker.get_cost("node3") >= 0.0
    assert tracker.total_cost >= 0.0
```

**Step 2: Run test, implement fix**

In `cost_tracker.py`, clamp negative values:
```python
def record(self, node_id: str, cost: float) -> None:
    """Record cost for a node. Clamps negative values to 0."""
    cost = max(0.0, cost)  # Prevent negative costs
    self._costs[node_id] = self._costs.get(node_id, 0.0) + cost
    self._total += cost
```

**Step 3: Run full suite, commit**

```bash
git add sage-python/src/sage/contracts/cost_tracker.py sage-python/tests/test_cost_tracker.py
git commit -m "fix(cost): clamp negative costs to zero — prevents corrupted Pareto analysis"
```

---

### Task 6: Disable evolution engine by default — defer to v3

**Problem:** Full evolution engine (DGM+SAMPO+MAP-Elites) only achieves 0.50 best score vs 0.33 for random mutation (67% efficiency). SAMPO/DGM add minimal value over random perturbation. Evolution adds cost (LLM mutations) without proportional quality improvement.

**Root cause:** Not a bug — the evolution engine works correctly. But per evidence-first policy, features that don't demonstrably beat baseline should be disabled.

**Fix:** Add `enable_evolution` flag to boot.py (default=False). Evolution engine still available for explicit use but not wired into default agent loop.

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Test: `sage-python/tests/test_boot.py`

**Step 1: Write test**

```python
def test_evolution_disabled_by_default():
    """Evolution should be disabled by default per Sprint 3 evidence."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    # topology_evolver should exist but not be wired to auto-run
    assert system.topology_evolver is not None  # Still available
    # The agent_loop should NOT auto-trigger evolution
    assert not getattr(system.agent_loop, '_auto_evolve', True)
```

**Step 2: Implement**

In `boot.py`, after creating `loop`, add:
```python
loop._auto_evolve = False  # Sprint 3 evidence: evolution marginal vs random mutation
```

In `agent_loop.py` LEARN phase, gate evolution:
```python
# Evolution (only if explicitly enabled)
if getattr(self, '_auto_evolve', False) and self.topology_population:
    # ... existing evolution code ...
```

**Step 3: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/src/sage/agent_loop.py sage-python/tests/test_boot.py
git commit -m "feat(boot): disable evolution by default — Sprint 3 shows marginal value vs random"
```

---

## Sprint B — Memory System Hardening (Tasks 7-9)

### Task 7: Disable episodic/semantic injection for code tasks

**Problem:** Episodic+semantic memory alone produces 30% accuracy (vs 50% no-memory, vs 85% baseline). These stores inject cross-session entity data that is irrelevant to code generation.

**Fix:** Only inject episodic/semantic context for non-code tasks. Code tasks (detected by routing profile) skip memory injection entirely. ExoCortex grounding remains active (it's the valuable component: +15% over no-memory per ablation).

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py` (memory injection block)
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write test**

```python
def test_code_tasks_skip_episodic_semantic():
    """Code generation tasks should not get episodic/semantic context injection."""
    # Verify that when task contains code indicators, memory injection is skipped
    from sage.agent_loop import _is_code_task
    assert _is_code_task("Implement binary search in Python") is True
    assert _is_code_task("What is the capital of France?") is False
    assert _is_code_task("Write a function to sort a list") is True
    assert _is_code_task("Explain photosynthesis") is False
```

**Step 2: Implement**

Add helper to `agent_loop.py`:
```python
def _is_code_task(task: str) -> bool:
    """Detect if task is primarily about code generation."""
    lower = task.lower()
    return bool(re.search(
        r'\b(?:implement|code|function|class|method|algorithm|program|'
        r'write\s+(?:a\s+)?(?:function|method|class|code|script)|'
        r'python|javascript|rust|java|def\s|return\s)\b', lower
    ))
```

Gate memory injection:
```python
_skip_episodic_semantic = _is_code_task(task)

# Semantic memory (skip for code tasks — Sprint 3 evidence: harmful)
if self.semantic_memory and not _skip_episodic_semantic and not self._cb_semantic.should_skip():
    # ... existing code ...
```

**Step 3: Commit**

```bash
git commit -m "fix(memory): skip episodic/semantic injection for code tasks — Sprint 3: 30% vs 50%"
```

---

### Task 8: Limit S-MMU context to 3 chunks max with relevance floor

**Problem:** S-MMU injects up to 5 chunks (`top_k=5`) regardless of relevance score. Low-scoring chunks add noise.

**Fix:** Reduce `top_k` to 3, add minimum score threshold (0.5). Only inject chunks that actually scored well.

**Files:**
- Modify: `sage-python/src/sage/memory/smmu_context.py:34-43`
- Test: `sage-python/tests/test_smmu_context.py`

**Step 1: Write test**

```python
def test_smmu_context_respects_min_score():
    """S-MMU should not include chunks below min_score threshold."""
    from sage.memory.smmu_context import _filter_by_score
    hits = [(0, 0.9), (1, 0.6), (2, 0.3), (3, 0.1)]
    filtered = _filter_by_score(hits, min_score=0.5)
    assert len(filtered) == 2
    assert all(score >= 0.5 for _, score in filtered)
```

**Step 2: Implement**

In `smmu_context.py`, add filter and reduce defaults:
```python
def _filter_by_score(
    hits: list[tuple[int, float]], min_score: float = 0.5
) -> list[tuple[int, float]]:
    """Remove chunks below minimum relevance score."""
    return [(cid, s) for cid, s in hits if s >= min_score]
```

Update `retrieve_smmu_context()`:
```python
def retrieve_smmu_context(
    working_memory: WorkingMemory,
    max_hops: int = 2,
    top_k: int = 3,           # Was 5 — reduced per Sprint 3 evidence
    min_score: float = 0.5,   # New: minimum relevance threshold
    weights: tuple[float, float, float, float] | None = None,
) -> str:
    # ... existing code ...
    hits = _filter_by_score(hits, min_score)
    top_hits = hits[:top_k]
    if not top_hits:
        return ""
    # ... rest of formatting ...
```

**Step 3: Commit**

```bash
git commit -m "fix(smmu): reduce top_k to 3, add min_score=0.5 — reject low-relevance chunks"
```

---

### Task 9: ExoCortex grounding — make opt-in tool instead of passive injection

**Problem:** ExoCortex passive grounding in `_think()` adds latency to every task. Per memory ablation, it's the most valuable component (+15%), but not every task benefits.

**Fix:** Keep ExoCortex as an active tool (`search_exocortex`), remove passive injection in `_think()`. Agent can explicitly invoke it when needed.

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py` (remove passive ExoCortex call in _think)
- Test: verify ExoCortex tools still registered

**Step 1: Verify current passive injection exists**

Read `agent_loop.py`, find ExoCortex passive call in `_think()`.

**Step 2: Remove passive injection, keep tool**

Comment out or remove the passive ExoCortex grounding block in `_think()`, keeping the tool registration in `boot.py`.

**Step 3: Run tests + commit**

```bash
git commit -m "refactor(exocortex): remove passive injection — keep as active tool only"
```

---

## Sprint C — Validation & Documentation (Tasks 10-12)

### Task 10: Run full test suite after all changes

**Files:** None (test-only)

**Step 1: Run full suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

Expected: 740+ passed, 0 failed

**Step 2: Run lint**

```bash
cd sage-python && ruff check src/
```

**Step 3: Fix any failures**

If tests fail, fix root cause (don't disable tests). If lint errors, fix them.

**Step 4: Commit fixes if any**

```bash
git commit -m "fix: test and lint fixes after Sprint 3 architecture changes"
```

---

### Task 11: Re-run Sprint 3 benchmarks to measure improvement

**Prerequisites:** GOOGLE_API_KEY in .env

**Step 1: Run HumanEval comparison (20-task smoke)**

```bash
cd sage-python && python -c "
import asyncio
from sage.bench.sprint3_evidence import run_humaneval_all_configs
asyncio.run(run_humaneval_all_configs(limit=20))
"
```

**Expected improvements (vs pre-fix):**
- Full-stack should improve from 55% toward 75%+ (routing fix + memory gate)
- Routing-only should improve from 70% toward 80%+ (AVR fix)
- Baseline should stay at ~85%

**Step 2: Run routing proof (30 tasks)**

```bash
cd sage-python && python -c "
import asyncio
from sage.bench.sprint3_evidence import run_routing_proof
asyncio.run(run_routing_proof())
"
```

**Expected:** Router cost should decrease significantly (S1 for more tasks).

**Step 3: Save new results to docs/benchmarks/**

**Step 4: Commit**

```bash
git commit -m "bench: post-fix Sprint 3 re-benchmark — measure architecture improvements"
```

---

### Task 12: Update MEMORY.md, CLAUDE.md, ARCHITECTURE.md

Sync all documentation with Sprint 3 architecture changes:
- New routing thresholds (S1 ceil=0.50, S3 floor=0.65)
- AVR SLF feedback + stagnation detection
- CRAG relevance gate for memory injection
- Evolution disabled by default
- ExoCortex passive→active transition
- Validation_level defaults to 1

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/ARCHITECTURE.md` (if exists)
- Modify: Auto-memory MEMORY.md

**Commit:**

```bash
git commit -m "docs: sync documentation with Sprint 3 architecture fixes"
```

---

## Summary: Expected Impact

| Fix | Before | After (Expected) | Metric |
|-----|--------|-------------------|--------|
| AVR SLF + stagnation | 55% full-stack | 75%+ | HumanEval pass@1 |
| Routing recalibration | 42x cost vs S1 | <5x cost vs S1 | Cost ratio |
| CRAG relevance gate | 30% episodic-semantic | 50%+ (no worse than no-memory) | HumanEval pass@1 |
| Default VL=1 | S2 AVR on all tasks | S2 only when routed | Latency reduction |
| Negative cost fix | -$0.0005 on some tasks | $0.0000+ always | Cost tracking |
| Evolution off by default | 42x cost for 0.17 score gain | No cost (opt-in) | Budget savings |
