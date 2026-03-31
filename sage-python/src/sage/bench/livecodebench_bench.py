"""LiveCodeBench adapter: contamination-free coding benchmark.

Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite
Rolling updates prevent data contamination.

Evaluation: generates code via AgentSystem, runs against input/output test
pairs in a subprocess sandbox. AVR retry on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage._python import PYTHON
from sage.bench.runner import BenchReport, TaskResult

log = logging.getLogger(__name__)


def _load_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    """Load LiveCodeBench code generation dataset from HuggingFace.

    Args:
        limit: Maximum number of problems to return.

    Returns:
        List of problem dicts.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        trust_remote_code=True,
    )
    problems: list[dict[str, Any]] = []
    for row in ds:
        problems.append(dict(row))
        if limit and len(problems) >= limit:
            break
    return problems


def _parse_test_cases(problem: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (input, expected_output) test case pairs from a problem.

    LiveCodeBench stores test cases in different formats depending on version.
    This handles the common patterns:
    - 'public_test_cases': JSON string or list of {input, output}
    - 'private_test_cases': same format (used for eval)
    - 'input_output': JSON string with inputs/outputs lists (APPS-style)
    """
    pairs: list[tuple[str, str]] = []

    # Try public_test_cases first, then private
    for field in ("public_test_cases", "private_test_cases", "input_output"):
        raw = problem.get(field)
        if not raw:
            continue

        # Parse JSON string if needed
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            data = raw

        # Format 1: list of {input, output} dicts
        if isinstance(data, list):
            for tc in data:
                if isinstance(tc, dict) and "input" in tc and "output" in tc:
                    pairs.append((str(tc["input"]), str(tc["output"])))
                elif isinstance(tc, dict) and "expected_output" in tc:
                    pairs.append((
                        str(tc.get("input", "")),
                        str(tc["expected_output"]),
                    ))

        # Format 2: {inputs: [...], outputs: [...]} (APPS-style)
        elif isinstance(data, dict):
            inputs = data.get("inputs", [])
            outputs = data.get("outputs", [])
            pairs.extend(zip(
                (str(i) for i in inputs),
                (str(o) for o in outputs),
            ))

        if pairs:
            break  # Use the first field that yields results

    return pairs


class LiveCodeBenchBench:
    """LiveCodeBench adapter for SAGE.

    Args:
        system: AgentSystem to benchmark.
        event_bus: EventBus for BENCH_RESULT events.
        task_timeout: Max seconds for LLM generation per attempt.
        eval_timeout: Max seconds for test evaluation per test case.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        task_timeout: float = 120.0,
        eval_timeout: float = 30.0,
    ):
        self.system = system
        self.event_bus = event_bus
        self.task_timeout = task_timeout
        self.eval_timeout = eval_timeout

    async def run(self, limit: int | None = None) -> BenchReport:
        """Run LiveCodeBench benchmark."""
        problems = _load_dataset(limit=limit)
        if not problems:
            log.warning("No LiveCodeBench problems loaded")
            return BenchReport(
                benchmark="livecodebench",
                total=0, passed=0, failed=0, errors=0,
                pass_rate=0.0, avg_latency_ms=0.0, avg_cost_usd=0.0,
                routing_breakdown={}, results=[],
            )

        results: list[TaskResult] = []
        passed_count = 0

        for i, problem in enumerate(problems):
            task_id = problem.get("question_id", f"LCB/{i}")
            question = problem.get("question_content", "")
            code_context = problem.get("code_context", "")
            test_cases = _parse_test_cases(problem)

            prompt = self._build_prompt(question, code_context)

            t0 = time.time()
            solution = ""
            error = ""
            eval_stderr = ""

            if self.system:
                try:
                    raw = await asyncio.wait_for(
                        self.system.run(prompt),
                        timeout=self.task_timeout,
                    )
                    solution = _extract_program(raw)
                except asyncio.TimeoutError:
                    error = "TIMEOUT"
                except Exception as exc:
                    error = str(exc)[:200]

            latency_ms = (time.time() - t0) * 1000

            if solution and not error:
                if test_cases:
                    task_passed, eval_stderr = self._evaluate_io(
                        solution=solution,
                        test_cases=test_cases,
                        task_id=str(task_id),
                        timeout=self.eval_timeout,
                    )
                else:
                    # No test cases available — mark as error
                    task_passed = False
                    error = "no test cases"

                # AVR retry: send error back to LLM for self-correction
                avr_retries = 2
                for avr_attempt in range(avr_retries):
                    if task_passed or not eval_stderr:
                        break
                    retry_prompt = (
                        f"Your previous code for this task failed with this error:\n"
                        f"```\n{eval_stderr[-500:]}\n```\n\n"
                        f"Original task:\n{prompt}\n\n"
                        f"Please fix the code. Return ONLY the corrected Python program."
                    )
                    try:
                        raw = await asyncio.wait_for(
                            self.system.run(retry_prompt),
                            timeout=self.task_timeout,
                        )
                        code = _extract_program(raw)
                        if code.strip():
                            task_passed, eval_stderr = self._evaluate_io(
                                solution=code,
                                test_cases=test_cases,
                                task_id=str(task_id),
                                timeout=self.eval_timeout,
                            )
                            if task_passed:
                                log.info(
                                    "  AVR retry %d succeeded for %s",
                                    avr_attempt + 1,
                                    task_id,
                                )
                    except Exception:
                        break
            else:
                task_passed = False
                if not error:
                    error = "no solution generated"

            if task_passed:
                passed_count += 1

            results.append(TaskResult(
                task_id=str(task_id),
                passed=task_passed,
                latency_ms=latency_ms,
                error=error,
            ))

            status = "PASS" if task_passed else "FAIL"
            log.info(
                "[%d/%d] %s %s (%.0fms)",
                i + 1, len(problems), status, task_id, latency_ms,
            )

        total = len(results)
        return BenchReport(
            benchmark="livecodebench",
            total=total,
            passed=passed_count,
            failed=total - passed_count,
            errors=sum(1 for r in results if r.error),
            pass_rate=passed_count / total if total else 0.0,
            avg_latency_ms=sum(r.latency_ms for r in results) / total if total else 0.0,
            avg_cost_usd=0.0,
            routing_breakdown={},
            results=results,
        )

    @staticmethod
    def _build_prompt(question: str, code_context: str) -> str:
        """Build the LLM prompt from problem fields."""
        parts = [
            "Solve this competitive programming problem.",
        ]
        if code_context and code_context.strip():
            parts.append(
                f"\nCode context:\n```python\n{code_context}\n```"
            )
        parts.append(f"\n{question}")
        parts.append(
            "\nWrite a complete Python program that reads from stdin and "
            "writes to stdout. Return ONLY the Python code."
        )
        return "\n".join(parts)

    @staticmethod
    def _evaluate_io(
        solution: str,
        test_cases: list[tuple[str, str]],
        task_id: str,
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """Evaluate solution against input/output test cases.

        Returns:
            (all_passed, last_stderr) for AVR retry.
        """
        all_passed = True
        last_stderr = ""

        for idx, (inp, expected) in enumerate(test_cases):
            passed, stderr = LiveCodeBenchBench._run_single_io(
                solution, inp, expected, task_id, idx, timeout,
            )
            if not passed:
                all_passed = False
                last_stderr = stderr
                break  # Fail fast on first failing test case

        return all_passed, last_stderr

    @staticmethod
    def _run_single_io(
        solution: str,
        inp: str,
        expected: str,
        task_id: str,
        case_idx: int,
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """Run solution with given stdin input and check stdout matches expected."""
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(solution)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                [PYTHON, tmp_path],
                input=inp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            actual = result.stdout.strip()
            exp = expected.strip()

            if result.returncode != 0:
                stderr = result.stderr or ""
                return False, (
                    f"Case {case_idx}: returncode={result.returncode}\n"
                    f"{stderr[-300:]}"
                )

            if actual == exp:
                return True, ""

            # Fuzzy match: normalize whitespace
            if " ".join(actual.split()) == " ".join(exp.split()):
                return True, ""

            return False, (
                f"Case {case_idx}: output mismatch\n"
                f"Expected:\n{exp[:200]}\n"
                f"Got:\n{actual[:200]}"
            )

        except subprocess.TimeoutExpired:
            log.debug("Eval timeout for %s case %d", task_id, case_idx)
            return False, f"TIMEOUT: case {case_idx} exceeded time limit"
        except Exception as exc:
            log.debug("Eval error for %s case %d: %s", task_id, case_idx, exc)
            return False, str(exc)[:500]
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _extract_program(response: str) -> str:
    """Extract a complete Python program from LLM response.

    Competitive programming problems expect stdin/stdout programs.
    Tries fenced code blocks first, then falls back to raw response.
    """
    # Strategy 1: fenced code block
    pattern = r"```(?:python)?\s*\n(.*?)```"
    blocks = re.findall(pattern, response, re.DOTALL)
    if blocks:
        # Prefer the longest block (most likely the full solution)
        return max(blocks, key=len)

    # Strategy 2: if the response looks like code, use it directly
    code_indicators = (
        "import ", "def ", "for ", "while ", "input(", "print(", "sys.stdin",
    )
    lines = response.strip().split("\n")
    if any(
        line.strip().startswith(ind)
        for line in lines
        for ind in code_indicators
    ):
        return response.strip()

    return response.strip()
