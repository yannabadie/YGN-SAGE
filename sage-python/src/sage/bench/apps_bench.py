"""APPS benchmark adapter: 10,000 competitive programming problems.

Dataset: https://huggingface.co/datasets/codeparrot/apps
Difficulty levels: introductory (3,639), interview (5,000), competition (1,361)

Evaluation: generates code via AgentSystem, runs against input/output test
pairs in a subprocess sandbox. AVR retry on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage.bench.runner import BenchReport, TaskResult

log = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ("introductory", "interview", "competition")


def _load_dataset(
    difficulty: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load APPS dataset from HuggingFace.

    Args:
        difficulty: Filter by level — "introductory", "interview", or "competition".
                    None returns all problems.
        limit: Maximum number of problems to return.

    Returns:
        List of problem dicts with keys: question, solutions, input_output, difficulty.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    problems: list[dict[str, Any]] = []
    for row in ds:
        row_diff = row.get("difficulty", "")
        if difficulty and row_diff != difficulty:
            continue
        problems.append(dict(row))
        if limit and len(problems) >= limit:
            break
    return problems


def _parse_input_output(raw: str) -> list[tuple[str, str]]:
    """Parse the input_output JSON field into (input, expected_output) pairs.

    APPS stores test cases as a JSON string with 'inputs' and 'outputs' lists.
    Returns empty list if parsing fails or field is empty.
    """
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    inputs = data.get("inputs", [])
    outputs = data.get("outputs", [])
    # Pair them up — truncate to shortest list
    return list(zip(inputs, outputs))


class APPSBench:
    """APPS benchmark adapter for SAGE.

    Args:
        system: AgentSystem to benchmark.
        event_bus: EventBus for BENCH_RESULT events.
        difficulty: Filter problems by difficulty level.
        task_timeout: Max seconds for LLM generation per attempt.
        eval_timeout: Max seconds for test evaluation per test case.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        difficulty: str | None = None,
        task_timeout: float = 120.0,
        eval_timeout: float = 30.0,
    ):
        self.system = system
        self.event_bus = event_bus
        self.difficulty = difficulty
        self.task_timeout = task_timeout
        self.eval_timeout = eval_timeout

    async def run(self, limit: int | None = None) -> BenchReport:
        """Run APPS benchmark."""
        problems = _load_dataset(difficulty=self.difficulty, limit=limit)
        if not problems:
            log.warning("No APPS problems loaded (difficulty=%s)", self.difficulty)
            return BenchReport(
                benchmark=self._bench_name(),
                total=0, passed=0, failed=0, errors=0,
                pass_rate=0.0, avg_latency_ms=0.0, avg_cost_usd=0.0,
                routing_breakdown={}, results=[],
            )

        results: list[TaskResult] = []
        passed_count = 0

        for i, problem in enumerate(problems):
            task_id = f"APPS/{i}"
            question = problem.get("question", "")
            difficulty_label = problem.get("difficulty", "unknown")
            io_raw = problem.get("input_output", "")
            test_cases = _parse_input_output(io_raw)

            prompt = (
                f"Solve this competitive programming problem.\n"
                f"Difficulty: {difficulty_label}\n\n"
                f"{question}\n\n"
                f"Write a complete Python program that reads from stdin and "
                f"writes to stdout. Return ONLY the Python code."
            )

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
                        task_id=task_id,
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
                                task_id=task_id,
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
                task_id=task_id,
                passed=task_passed,
                latency_ms=latency_ms,
                error=error,
            ))

            status = "PASS" if task_passed else "FAIL"
            log.info(
                "[%d/%d] %s %s (%s, %.0fms)",
                i + 1, len(problems), status, task_id, difficulty_label, latency_ms,
            )

        total = len(results)
        return BenchReport(
            benchmark=self._bench_name(),
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

    def _bench_name(self) -> str:
        diff = self.difficulty or "all"
        return f"apps-{diff}"

    @staticmethod
    def _evaluate_io(
        solution: str,
        test_cases: list[tuple[str, str]],
        task_id: str,
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """Evaluate solution against input/output test cases.

        Runs each test case as a subprocess, feeding input via stdin and
        comparing stdout to expected output.

        Returns:
            (all_passed, last_stderr) for AVR retry.
        """
        all_passed = True
        last_stderr = ""

        for idx, (inp, expected) in enumerate(test_cases):
            passed, stderr = APPSBench._run_single_io(
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
                ["python", tmp_path],
                input=inp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            actual = result.stdout.strip()
            exp = expected.strip()

            if result.returncode != 0:
                stderr = result.stderr or ""
                return False, f"Case {case_idx}: returncode={result.returncode}\n{stderr[-300:]}"

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

    APPS problems expect stdin/stdout programs, not functions.
    Tries fenced code blocks first, then falls back to raw response.
    """
    import re

    # Strategy 1: fenced code block
    pattern = r"```(?:python)?\s*\n(.*?)```"
    blocks = re.findall(pattern, response, re.DOTALL)
    if blocks:
        # Prefer the longest block (most likely the full solution)
        return max(blocks, key=len)

    # Strategy 2: if the response looks like code (has import/def/for/while/input),
    # use it directly
    code_indicators = ("import ", "def ", "for ", "while ", "input(", "print(", "sys.stdin")
    lines = response.strip().split("\n")
    if any(line.strip().startswith(ind) for line in lines for ind in code_indicators):
        return response.strip()

    return response.strip()
