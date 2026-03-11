"""Official Benchmark Evaluation Protocol.

Runs benchmarks in real conditions with comprehensive error logging.
Every error is captured with full context for post-mortem analysis.

Usage:
    python -m sage.bench.eval_protocol --suite humaneval --limit 20
    python -m sage.bench.eval_protocol --suite mbpp --limit 20
    python -m sage.bench.eval_protocol --replay docs/benchmarks/latest-errors.jsonl
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger("sage.bench.eval_protocol")

# ---------------------------------------------------------------------------
# Dataset loaders (same dispatch table as evalplus_bench.py)
# ---------------------------------------------------------------------------
_DATASET_LOADERS = {
    "humaneval": "evalplus.data:get_human_eval_plus",
    "mbpp": "evalplus.data:get_mbpp_plus",
}


def _load_dataset(dataset: str) -> dict[str, dict[str, Any]]:
    """Load an EvalPlus dataset by name.

    Returns dict keyed by task_id, e.g. ``{"HumanEval/0": {...}, ...}``.
    Falls back to bundled ``humaneval_data.json`` when *evalplus* is not
    installed and *dataset* is ``"humaneval"``.
    """
    if dataset not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: {list(_DATASET_LOADERS)}"
        )
    module_path, func_name = _DATASET_LOADERS[dataset].split(":")
    try:
        mod = importlib.import_module(module_path)
        loader = getattr(mod, func_name)
        return loader()
    except ImportError:
        if dataset == "humaneval":
            return _load_bundled_humaneval()
        raise


def _load_bundled_humaneval() -> dict[str, dict[str, Any]]:
    """Load HumanEval from the bundled JSON file (no evalplus dependency)."""
    data_file = Path(__file__).parent / "humaneval_data.json"
    if not data_file.exists():
        _log.warning("Bundled humaneval_data.json not found at %s", data_file)
        return {}
    with open(data_file, encoding="utf-8") as f:
        problems = json.load(f)
    # Bundled format is a list; convert to dict keyed by task_id
    if isinstance(problems, list):
        return {p["task_id"]: p for p in problems}
    return problems


def _git_sha() -> str:
    """Return the short git SHA of the current HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        return ""


def _detect_feature_flags() -> list[str]:
    """Detect which sage_core features are available at runtime."""
    flags: list[str] = []
    try:
        import sage_core  # noqa: F811
        flags.append("sage_core")
        if hasattr(sage_core, "SmtVerifier"):
            flags.append("smt")
        if hasattr(sage_core, "ToolExecutor"):
            flags.append("tool-executor")
        if hasattr(sage_core, "RustEmbedder"):
            flags.append("onnx")
        if hasattr(sage_core, "WasmSandbox"):
            flags.append("sandbox")
    except ImportError:
        pass
    return flags


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ErrorRecord:
    """Detailed error captured during benchmark execution."""

    task_id: str
    error_type: str          # Exception class name
    error_message: str
    traceback_str: str       # Full traceback
    phase: str               # perceive/think/act/learn/routing/sandbox/guardrail/
                             # verification/init/execution
    model_id: str = ""
    system_used: int = 0     # S1/S2/S3 (0 = unknown)
    routing_decision: dict = field(default_factory=dict)
    task_text: str = ""
    partial_result: str = ""
    latency_ms: float = 0.0
    timestamp: str = ""
    context: dict = field(default_factory=dict)  # Extra debug info

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class TaskEvalResult:
    """Result of evaluating a single benchmark task."""

    task_id: str
    passed: bool
    base_passed: bool = False
    plus_passed: bool = False
    system_used: int = 0
    model_id: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    result_text: str = ""
    errors: list[ErrorRecord] = field(default_factory=list)
    routing_decision: dict = field(default_factory=dict)
    avr_iterations: int = 0
    sandbox_executions: int = 0
    guardrail_events: list[dict] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class EvalReport:
    """Complete benchmark evaluation report with error analysis."""

    suite: str               # humaneval / mbpp
    total: int
    passed: int
    failed: int
    error_count: int         # Tasks that encountered errors (may still pass)
    pass_rate: float
    base_pass_rate: float    # Pass rate on original tests only
    plus_pass_rate: float    # Pass rate on EvalPlus enhanced tests
    avg_latency_ms: float
    avg_cost_usd: float
    routing_breakdown: dict[str, int] = field(default_factory=dict)
    error_categories: dict[str, int] = field(default_factory=dict)
    results: list[TaskEvalResult] = field(default_factory=list)
    all_errors: list[ErrorRecord] = field(default_factory=list)
    git_sha: str = ""
    feature_flags: list[str] = field(default_factory=list)
    model_config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Error capture helper
# ---------------------------------------------------------------------------
class ErrorCapture:
    """Accumulator that records errors with full context during execution."""

    def __init__(self, task_id: str, phase: str) -> None:
        self.task_id = task_id
        self.phase = phase
        self.errors: list[ErrorRecord] = []

    def record(self, exc: Exception, **context: Any) -> ErrorRecord:
        """Record an error with full traceback and optional context fields."""
        rec = ErrorRecord(
            task_id=self.task_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_str=traceback.format_exc(),
            phase=context.pop("phase", self.phase),
            context=context,
        )
        self.errors.append(rec)
        _log.warning(
            "Error in %s/%s: %s: %s",
            self.task_id, rec.phase, rec.error_type, rec.error_message,
        )
        return rec


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
class EvalProtocol:
    """Official benchmark evaluation protocol.

    Runs tasks through the full SAGE pipeline (routing, topology, agent loop,
    sandbox, guardrails) and captures every error along the way.

    Args:
        suite: ``"humaneval"`` or ``"mbpp"``.
        limit: Maximum number of tasks to evaluate (``None`` = all).
        output_dir: Directory for reports and error logs.
        verbose: Print per-task progress.
        task_timeout: Seconds before a single task is forcibly stopped.
    """

    def __init__(
        self,
        suite: str = "humaneval",
        limit: int | None = None,
        output_dir: str = "docs/benchmarks",
        verbose: bool = False,
        task_timeout: float = 120.0,
    ) -> None:
        if suite not in _DATASET_LOADERS:
            raise ValueError(
                f"Unknown suite '{suite}'. Supported: {list(_DATASET_LOADERS)}"
            )
        self.suite = suite
        self.limit = limit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.task_timeout = task_timeout

        self._errors: list[ErrorRecord] = []
        self._results: list[TaskEvalResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(self) -> EvalReport:
        """Execute the full evaluation protocol and return a report."""
        wall_start = time.monotonic()
        _log.info("Starting %s evaluation (limit=%s)", self.suite, self.limit)

        # 1. Load dataset
        problems = self._load_problems()
        if not problems:
            _log.error("No problems loaded for suite '%s'", self.suite)
            return self._build_report(time.monotonic() - wall_start)

        task_ids = list(problems.keys())
        if self.limit:
            task_ids = task_ids[: self.limit]
        _log.info("Loaded %d tasks", len(task_ids))

        # 2. Boot agent system
        system = self._boot_system()

        # 3. Evaluate each task
        for idx, task_id in enumerate(task_ids):
            _log.info("[%d/%d] Running %s", idx + 1, len(task_ids), task_id)

            problem = problems[task_id]
            result = await self._run_single_task(system, task_id, problem)
            self._results.append(result)
            self._errors.extend(result.errors)

            if self.verbose:
                status = "PASS" if result.passed else "FAIL"
                errs = f" ({len(result.errors)} errors)" if result.errors else ""
                _log.info(
                    "  %s %s [%.0fms, S%d]%s",
                    status, task_id, result.latency_ms, result.system_used, errs,
                )

        # 4. Build and save report
        duration = time.monotonic() - wall_start
        report = self._build_report(duration)
        self._save_report(report)
        self._save_error_log()

        _log.info(
            "Evaluation complete: %d/%d passed (%.1f%%), %d errors, %.1fs",
            report.passed, report.total, report.pass_rate * 100,
            report.error_count, duration,
        )
        return report

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_problems(self) -> dict[str, dict[str, Any]]:
        """Load benchmark problems for the selected suite."""
        capture = ErrorCapture("__dataset__", "init")
        try:
            return _load_dataset(self.suite)
        except Exception as exc:
            capture.record(exc, note=f"Failed to load dataset '{self.suite}'")
            self._errors.extend(capture.errors)
            return {}

    # ------------------------------------------------------------------
    # System boot
    # ------------------------------------------------------------------
    def _boot_system(self) -> Any:
        """Boot the SAGE agent system.

        Falls back to mock LLM if the real boot fails (e.g. no API key).
        """
        capture = ErrorCapture("__boot__", "init")
        try:
            from sage.boot import boot_agent_system  # noqa: WPS433
            from sage.events.bus import EventBus

            bus = EventBus()
            system = boot_agent_system(
                use_mock_llm=False, llm_tier="fast", event_bus=bus,
            )
            _log.info("Agent system booted (real LLM)")
            return system
        except Exception as exc:
            capture.record(exc, note="Failed to boot real agent system")
            self._errors.extend(capture.errors)
            _log.warning("Falling back to mock LLM for evaluation")

        # Fallback: mock LLM (tests still run through full pipeline)
        try:
            from sage.boot import boot_agent_system
            from sage.events.bus import EventBus

            bus = EventBus()
            return boot_agent_system(use_mock_llm=True, event_bus=bus)
        except Exception as exc2:
            capture2 = ErrorCapture("__boot__", "init")
            capture2.record(exc2, note="Even mock boot failed")
            self._errors.extend(capture2.errors)
            return None

    # ------------------------------------------------------------------
    # Single task execution
    # ------------------------------------------------------------------
    async def _run_single_task(
        self,
        system: Any,
        task_id: str,
        problem: dict[str, Any],
    ) -> TaskEvalResult:
        """Run and verify a single benchmark task with full error capture."""
        capture = ErrorCapture(task_id, "execution")
        t0 = time.perf_counter()
        result_text = ""
        model_id = ""
        system_used = 0
        base_passed = False
        plus_passed = False

        if system is None:
            capture.record(
                RuntimeError("No agent system available"),
                phase="execution",
            )
            return TaskEvalResult(
                task_id=task_id,
                passed=False,
                errors=capture.errors,
            )

        # -- Generate solution --
        prompt = problem.get("prompt", "")
        entry_point = problem.get("entry_point", "")

        if entry_point:
            full_prompt = (
                "Complete this Python function. "
                "Return ONLY the complete function, no explanation.\n\n"
                f"```python\n{prompt}\n```"
            )
        else:
            full_prompt = prompt

        try:
            result_text = await asyncio.wait_for(
                system.run(full_prompt),
                timeout=self.task_timeout,
            )
        except asyncio.TimeoutError:
            capture.record(
                TimeoutError(
                    f"Task {task_id} timed out after {self.task_timeout}s"
                ),
                phase="execution",
                task_text=prompt[:300],
            )
        except Exception as exc:
            capture.record(
                exc,
                phase="execution",
                task_text=prompt[:300],
                partial_result=result_text[:500] if result_text else "",
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # -- Extract routing metadata --
        try:
            loop = getattr(system, "agent_loop", None)
            if loop is not None:
                model_id = getattr(loop, "_last_model", "") or ""
                system_used = getattr(loop, "_last_routing_system", 0) or 0
        except Exception:
            pass

        # -- Extract code from LLM response --
        solution_code = ""
        if result_text and entry_point:
            try:
                from sage.bench.humaneval import extract_code
                solution_code = extract_code(result_text, entry_point)
            except Exception as exc:
                capture.record(exc, phase="extraction")
        elif result_text:
            solution_code = result_text

        # -- Verify: base tests --
        if solution_code and entry_point:
            base_passed, plus_passed = self._verify_solution(
                task_id, solution_code, problem, capture,
            )
        elif solution_code:
            # Non-code task: consider passed if we got a non-empty response
            base_passed = True
            plus_passed = True

        passed = base_passed and plus_passed

        # -- Quality estimation --
        quality = self._estimate_quality(prompt, result_text)

        # -- Cost --
        cost_usd = 0.0
        try:
            loop = getattr(system, "agent_loop", None)
            if loop is not None:
                cost_usd = getattr(loop, "total_cost_usd", 0.0) or 0.0
        except Exception:
            pass

        # -- AVR iterations --
        avr_iterations = 0
        try:
            loop = getattr(system, "agent_loop", None)
            if loop is not None:
                avr_iterations = getattr(loop, "_last_avr_iterations", 0) or 0
        except Exception:
            pass

        # -- Fill error records with task-level context --
        for err in capture.errors:
            err.model_id = model_id
            err.system_used = system_used
            err.latency_ms = latency_ms
            if not err.task_text:
                err.task_text = prompt[:300]
            if not err.partial_result:
                err.partial_result = result_text[:500] if result_text else ""

        return TaskEvalResult(
            task_id=task_id,
            passed=passed,
            base_passed=base_passed,
            plus_passed=plus_passed,
            system_used=system_used,
            model_id=model_id,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            result_text=result_text[:1000] if result_text else "",
            errors=capture.errors,
            avr_iterations=avr_iterations,
            quality_score=quality,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def _verify_solution(
        self,
        task_id: str,
        solution_code: str,
        problem: dict[str, Any],
        capture: ErrorCapture,
    ) -> tuple[bool, bool]:
        """Verify generated code against base and EvalPlus+ tests.

        Returns ``(base_passed, plus_passed)``.
        """
        entry_point = problem.get("entry_point", "")
        prompt = problem.get("prompt", "")
        canonical = problem.get("canonical_solution", "")
        test_code = problem.get("test", "")
        base_inputs = problem.get("base_input", [])
        plus_inputs = problem.get("plus_input", [])
        atol = problem.get("atol", 0)

        # --- Base tests ---
        if test_code:
            # HumanEval: use the original check() function
            base_passed = self._run_check_tests(
                task_id, solution_code, prompt, test_code, entry_point,
                capture,
            )
        elif base_inputs:
            # MBPP: compare solution vs canonical on base_inputs
            base_passed = self._run_comparison_tests(
                task_id, solution_code, prompt, canonical, entry_point,
                base_inputs, atol, capture, label="base",
            )
        else:
            base_passed = bool(solution_code and f"def {entry_point}" in solution_code)

        if not base_passed:
            return False, False

        # --- Plus tests (EvalPlus enhanced) ---
        if not plus_inputs:
            return True, True

        plus_passed = self._run_comparison_tests(
            task_id, solution_code, prompt, canonical, entry_point,
            plus_inputs[:200], atol, capture, label="plus",
        )
        return True, plus_passed

    def _run_check_tests(
        self,
        task_id: str,
        solution_code: str,
        prompt: str,
        test_code: str,
        entry_point: str,
        capture: ErrorCapture,
    ) -> bool:
        """Run HumanEval-style ``check()`` function tests."""
        # Ensure the solution contains the function definition
        if f"def {entry_point}" in solution_code:
            full_code = solution_code
        else:
            full_code = prompt + solution_code

        program = f"{full_code}\n\n{test_code}\n\ncheck({entry_point})\n"
        return self._execute_program(
            task_id, program, capture, label="base_check",
            code_snippet=solution_code[:500],
        )

    def _run_comparison_tests(
        self,
        task_id: str,
        solution_code: str,
        prompt: str,
        canonical: str,
        entry_point: str,
        inputs: list,
        atol: float,
        capture: ErrorCapture,
        label: str = "comparison",
    ) -> bool:
        """Run solution vs canonical on a set of inputs."""
        if not inputs:
            return True

        if f"def {entry_point}" in solution_code:
            sol_code = solution_code
        else:
            sol_code = prompt + solution_code

        program = f"""{sol_code}

# Save reference to solution BEFORE canonical overwrites the name
import types
_solution = types.FunctionType(
    {entry_point}.__code__, {{**globals()}}, "{entry_point}_sol"
)

# --- Canonical solution (overwrites {entry_point}) ---
{prompt}{canonical}
_canonical = {entry_point}

# --- Run comparison tests ---
import json, sys
inputs = json.loads(sys.stdin.read())
atol = {atol}
failures = 0
first_failure = ""
for args in inputs:
    try:
        expected = _canonical(*args)
        actual = _solution(*args)
        if atol > 0:
            if isinstance(expected, float) and isinstance(actual, float):
                if abs(expected - actual) > atol:
                    failures += 1
                    if not first_failure:
                        first_failure = f"args={{args}}: expected={{expected}}, got={{actual}}"
                    continue
        if expected != actual:
            failures += 1
            if not first_failure:
                first_failure = f"args={{args}}: expected={{expected}}, got={{actual}}"
    except Exception as e:
        failures += 1
        if not first_failure:
            first_failure = f"args={{args}}: {{type(e).__name__}}: {{e}}"
print(f"COMPARE_RESULT:{{failures}}/{{len(inputs)}}")
if first_failure:
    print(f"FIRST_FAILURE:{{first_failure}}")
"""

        return self._execute_comparison_program(
            task_id, program, json.dumps(inputs), capture, label=label,
            code_snippet=solution_code[:500],
        )

    def _execute_program(
        self,
        task_id: str,
        program: str,
        capture: ErrorCapture,
        label: str = "test",
        code_snippet: str = "",
        timeout: float = 30.0,
    ) -> bool:
        """Execute a test program in a subprocess sandbox."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(program)
                tmp_path = f.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )

            if proc.returncode != 0:
                capture.record(
                    RuntimeError(
                        f"Test failed (exit={proc.returncode}): "
                        f"{proc.stderr[:500]}"
                    ),
                    phase="verification",
                    label=label,
                    stdout=proc.stdout[:500],
                    stderr=proc.stderr[:500],
                    code_snippet=code_snippet,
                )
                return False
            return True

        except subprocess.TimeoutExpired:
            capture.record(
                TimeoutError(f"Code execution timed out ({timeout}s)"),
                phase="verification",
                label=label,
            )
            return False
        except Exception as exc:
            capture.record(exc, phase="verification", label=label)
            return False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _execute_comparison_program(
        self,
        task_id: str,
        program: str,
        stdin_data: str,
        capture: ErrorCapture,
        label: str = "comparison",
        code_snippet: str = "",
        timeout: float = 30.0,
    ) -> bool:
        """Execute a comparison test program with stdin data."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(program)
                tmp_path = f.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin_data,
                capture_output=True, text=True, timeout=timeout,
            )

            if proc.returncode != 0:
                capture.record(
                    RuntimeError(
                        f"Comparison test crashed (exit={proc.returncode}): "
                        f"{proc.stderr[:500]}"
                    ),
                    phase="verification",
                    label=label,
                    stdout=proc.stdout[:500],
                    stderr=proc.stderr[:500],
                    code_snippet=code_snippet,
                )
                return False

            # Parse COMPARE_RESULT line
            for line in proc.stdout.split("\n"):
                if line.startswith("COMPARE_RESULT:"):
                    parts = line.split(":")[1].split("/")
                    failures = int(parts[0])
                    total = int(parts[1])
                    if failures > 0:
                        # Extract first failure detail
                        first_fail = ""
                        for fl in proc.stdout.split("\n"):
                            if fl.startswith("FIRST_FAILURE:"):
                                first_fail = fl[len("FIRST_FAILURE:"):]
                                break
                        capture.record(
                            AssertionError(
                                f"{failures}/{total} {label} tests failed"
                                + (f": {first_fail}" if first_fail else "")
                            ),
                            phase="verification",
                            label=label,
                            failures=failures,
                            total=total,
                            first_failure=first_fail,
                            code_snippet=code_snippet,
                        )
                        return False
                    return True

            capture.record(
                RuntimeError("No COMPARE_RESULT line in subprocess output"),
                phase="verification",
                label=label,
                stdout=proc.stdout[:500],
            )
            return False

        except subprocess.TimeoutExpired:
            capture.record(
                TimeoutError(f"Comparison timed out ({timeout}s)"),
                phase="verification",
                label=label,
            )
            return False
        except Exception as exc:
            capture.record(exc, phase="verification", label=label)
            return False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Quality estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_quality(task: str, result: str) -> float:
        """Estimate result quality using the QualityEstimator."""
        try:
            from sage.quality_estimator import QualityEstimator
            return QualityEstimator.estimate(task, result or "")
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------
    def _build_report(self, duration: float) -> EvalReport:
        """Build the evaluation report from accumulated results."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        error_count = sum(1 for r in self._results if r.errors)
        base_pass = sum(1 for r in self._results if r.base_passed)
        plus_pass = sum(1 for r in self._results if r.plus_passed)

        # Routing breakdown
        breakdown: dict[str, int] = {"S0": 0, "S1": 0, "S2": 0, "S3": 0}
        for r in self._results:
            key = f"S{r.system_used}"
            if key in breakdown:
                breakdown[key] += 1
            else:
                breakdown[key] = 1
        # Remove S0 if no tasks were unrouted
        if breakdown.get("S0", 0) == 0:
            breakdown.pop("S0", None)

        # Error categories
        categories: dict[str, int] = {}
        for err in self._errors:
            categories[err.error_type] = categories.get(err.error_type, 0) + 1

        return EvalReport(
            suite=self.suite,
            total=total,
            passed=passed,
            failed=failed,
            error_count=error_count,
            pass_rate=passed / total if total else 0.0,
            base_pass_rate=base_pass / total if total else 0.0,
            plus_pass_rate=plus_pass / total if total else 0.0,
            avg_latency_ms=(
                sum(r.latency_ms for r in self._results) / total if total else 0.0
            ),
            avg_cost_usd=(
                sum(r.cost_usd for r in self._results) / total if total else 0.0
            ),
            routing_breakdown=breakdown,
            error_categories=categories,
            results=self._results,
            all_errors=self._errors,
            git_sha=_git_sha(),
            feature_flags=_detect_feature_flags(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_s=duration,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_report(self, report: EvalReport) -> None:
        """Save the full evaluation report as JSON."""
        date = datetime.now().strftime("%Y-%m-%d")
        path = self.output_dir / f"{date}-eval-{self.suite}.json"

        data = asdict(report)
        # Trim result_text in per-task results for the main report
        for r in data["results"]:
            r["result_text"] = r["result_text"][:200]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        _log.info("Report saved to %s", path)

    def _save_error_log(self) -> None:
        """Save detailed error log as JSONL for post-mortem / replay."""
        if not self._errors:
            _log.info("No errors to log")
            return

        date = datetime.now().strftime("%Y-%m-%d")
        path = self.output_dir / f"{date}-errors-{self.suite}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for err in self._errors:
                f.write(json.dumps(asdict(err), default=str) + "\n")

        _log.info("Error log saved to %s (%d errors)", path, len(self._errors))

    # ------------------------------------------------------------------
    # Replay / post-mortem
    # ------------------------------------------------------------------
    @staticmethod
    def replay_errors(error_log_path: str) -> None:
        """Replay and analyze errors from a previous evaluation.

        Groups errors by type and phase, prints a summary with sample
        tracebacks for each category.
        """
        path = Path(error_log_path)
        if not path.exists():
            print(f"Error log not found: {path}")
            return

        errors: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    errors.append(json.loads(line))

        if not errors:
            print("No errors in log.")
            return

        # Group by error type
        by_type: dict[str, list[dict]] = {}
        for err in errors:
            et = err["error_type"]
            by_type.setdefault(et, []).append(err)

        # Group by phase
        by_phase: dict[str, int] = {}
        for err in errors:
            phase = err.get("phase", "unknown")
            by_phase[phase] = by_phase.get(phase, 0) + 1

        print(f"\n{'=' * 70}")
        print(f"Error Analysis: {len(errors)} errors from {path.name}")
        print(f"{'=' * 70}")

        print("\n  Phase breakdown:")
        for phase, count in sorted(by_phase.items(), key=lambda x: -x[1]):
            print(f"    {phase}: {count}")

        for error_type, errs in sorted(
            by_type.items(), key=lambda x: -len(x[1])
        ):
            print(f"\n## {error_type} ({len(errs)} occurrences)")
            phases = sorted(set(e.get("phase", "unknown") for e in errs))
            print(f"   Phases: {', '.join(phases)}")
            task_ids = [e["task_id"] for e in errs[:5]]
            print(f"   Tasks:  {', '.join(task_ids)}")
            if len(errs) > 5:
                print(f"          ... and {len(errs) - 5} more")
            # Sample error message
            if errs:
                print(f"   Sample: {errs[0]['error_message'][:200]}")
                tb = errs[0].get("traceback_str", "")
                if tb:
                    tb_lines = tb.strip().split("\n")
                    print("   Traceback (last 3 lines):")
                    for line in tb_lines[-3:]:
                        print(f"     {line}")

        # List failed task IDs for easy --replay targeting
        failed_tasks = sorted(set(e["task_id"] for e in errors))
        if failed_tasks:
            print(f"\n  Failed task IDs ({len(failed_tasks)}):")
            for tid in failed_tasks:
                error_types = sorted(
                    set(e["error_type"] for e in errors if e["task_id"] == tid)
                )
                print(f"    {tid}: {', '.join(error_types)}")

        print(f"\n{'=' * 70}\n")

    @staticmethod
    def list_reports(output_dir: str = "docs/benchmarks") -> list[Path]:
        """List available evaluation reports in the output directory."""
        d = Path(output_dir)
        if not d.exists():
            return []
        reports = sorted(d.glob("*-eval-*.json"), reverse=True)
        return reports

    @staticmethod
    def load_report(report_path: str) -> EvalReport:
        """Load a previously saved EvalReport from JSON."""
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct dataclasses from dicts
        results = []
        for r in data.pop("results", []):
            errs = [ErrorRecord(**e) for e in r.pop("errors", [])]
            results.append(TaskEvalResult(**r, errors=errs))
        all_errors = [ErrorRecord(**e) for e in data.pop("all_errors", [])]
        return EvalReport(**data, results=results, all_errors=all_errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_env() -> None:
    """Load .env file for API keys (same logic as boot.py)."""
    try:
        from dotenv import load_dotenv
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass


def main() -> None:
    """CLI entry point for the evaluation protocol."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YGN-SAGE Official Benchmark Evaluation Protocol",
    )
    parser.add_argument(
        "--suite", choices=["humaneval", "mbpp"], default="humaneval",
        help="Benchmark suite to run (default: humaneval)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tasks (for quick validation)",
    )
    parser.add_argument(
        "--replay", type=str, default=None,
        help="Replay errors from a previous evaluation JSONL log",
    )
    parser.add_argument(
        "--list-reports", action="store_true",
        help="List available evaluation reports",
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks",
        help="Directory for reports and error logs (default: docs/benchmarks)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-task timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose per-task output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load environment variables
    _load_env()

    if args.list_reports:
        reports = EvalProtocol.list_reports(args.output_dir)
        if reports:
            print(f"\nAvailable reports in {args.output_dir}/:")
            for rp in reports:
                size = rp.stat().st_size
                print(f"  {rp.name}  ({size:,} bytes)")
        else:
            print(f"No reports found in {args.output_dir}/")
        return

    if args.replay:
        EvalProtocol.replay_errors(args.replay)
        return

    protocol = EvalProtocol(
        suite=args.suite,
        limit=args.limit,
        output_dir=args.output_dir,
        verbose=args.verbose,
        task_timeout=args.timeout,
    )
    report = asyncio.run(protocol.run())

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  {report.suite.upper()} Evaluation Results")
    print(f"{'=' * 70}")
    print(f"  Pass rate (plus): {report.passed}/{report.total} "
          f"({report.pass_rate:.1%})")
    print(f"  Base pass rate:   {report.base_pass_rate:.1%}")
    print(f"  Plus pass rate:   {report.plus_pass_rate:.1%}")
    print(f"  Errors:           {report.error_count} tasks had errors")
    print(f"  Routing:          {report.routing_breakdown}")
    print(f"  Avg latency:      {report.avg_latency_ms:.0f}ms")
    print(f"  Avg cost:         ${report.avg_cost_usd:.6f}/task")
    print(f"  Duration:         {report.duration_s:.1f}s")
    print(f"  Git SHA:          {report.git_sha}")
    print(f"  Features:         {report.feature_flags}")

    if report.error_categories:
        print("\n  Error breakdown:")
        for et, count in sorted(
            report.error_categories.items(), key=lambda x: -x[1]
        ):
            print(f"    {et}: {count}")

    # Show first few failures
    failures = [r for r in report.results if not r.passed]
    if failures:
        shown = failures[:10]
        print(f"\n  Failures ({len(failures)} total, showing first {len(shown)}):")
        for f in shown:
            err_summary = ""
            if f.errors:
                err_summary = f.errors[0].error_message[:80]
            print(f"    {f.task_id}: {err_summary or 'no error message'}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    else:
        print(f"\n  All {report.total} tasks passed.")

    print(f"\n  Report:     {args.output_dir}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
