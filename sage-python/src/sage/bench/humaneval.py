"""HumanEval benchmark: 164 Python code generation problems."""

from __future__ import annotations

import json
import re
import subprocess
import time
import tempfile
import logging
from pathlib import Path
from typing import Any

from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

log = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "humaneval_data.json"


def load_problems(limit: int | None = None) -> list[dict[str, Any]]:
    """Load HumanEval problems from bundled JSON."""
    with open(DATA_FILE, encoding="utf-8") as f:
        problems = json.load(f)
    if limit:
        problems = problems[:limit]
    return problems


def extract_code(response: str, entry_point: str) -> str:
    """Extract the function body from LLM response.

    Tries multiple strategies:
    1. Look for ```python code blocks
    2. Look for def entry_point
    3. Use the raw response
    """
    # Strategy 1: fenced code block
    pattern = r"```(?:python)?\s*\n(.*?)```"
    blocks = re.findall(pattern, response, re.DOTALL)
    if blocks:
        # Find the block containing the entry_point function
        for block in blocks:
            if f"def {entry_point}" in block:
                return block
        return blocks[-1]  # Fallback: last code block

    # Strategy 2: find the function definition
    lines = response.split("\n")
    in_func = False
    func_lines: list[str] = []
    for line in lines:
        if f"def {entry_point}" in line:
            in_func = True
        if in_func:
            func_lines.append(line)
    if func_lines:
        return "\n".join(func_lines)

    # Strategy 3: raw response
    return response


def run_test(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Execute the generated code + tests in a subprocess sandbox.

    Returns (passed: bool, error_message: str).
    """
    # Build the full program: if completion contains the full function def,
    # use it standalone; otherwise prepend the prompt (signature).
    if f"def {entry_point}" in completion:
        # Completion has the full function — don't duplicate the prompt
        full_code = completion + "\n" + test_code + f"\ncheck({entry_point})\n"
    else:
        # Completion is just the body — needs the prompt's signature
        full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class HumanEvalBench:
    """Run HumanEval benchmark against an AgentSystem."""

    def __init__(self, system: Any = None, event_bus: Any = None):
        self.system = system
        self.event_bus = event_bus
        self.manifest: BenchmarkManifest | None = None

    async def run(self, limit: int | None = None) -> BenchReport:
        """Run HumanEval pass@1 benchmark.

        If system is None, runs in "direct code extraction" mode (no agent).
        If system is provided, submits each task to AgentSystem.run().
        """
        problems = load_problems(limit)
        results: list[TaskResult] = []

        model_id = ""
        if self.system and hasattr(self.system, "agent_loop"):
            model_id = getattr(self.system.agent_loop, "_last_model", "") or "unknown"
        self.manifest = BenchmarkManifest(benchmark="humaneval", model=model_id)

        for i, problem in enumerate(problems):
            task_id = problem["task_id"]
            prompt = problem["prompt"]
            test_code = problem["test"]
            entry_point = problem["entry_point"]

            t0 = time.perf_counter()
            error = ""
            system_used = 0

            try:
                if self.system:
                    # Full agent mode
                    task = (
                        "Complete this Python function. "
                        "Return ONLY the function body, no explanation.\n\n"
                        f"```python\n{prompt}\n```"
                    )
                    response = await self.system.run(task)
                    system_used = (
                        getattr(
                            self.system.agent_loop, "_last_routing_system", 0
                        )
                        or 2  # Default S2 for code tasks
                    )
                else:
                    response = ""

                completion = extract_code(response, entry_point)
                passed, error = run_test(
                    prompt, completion, test_code, entry_point
                )

            except Exception as e:
                passed = False
                error = str(e)[:200]

            latency = (time.perf_counter() - t0) * 1000
            cost = (
                getattr(self.system.agent_loop, "total_cost_usd", 0.0)
                if self.system
                else 0.0
            )

            results.append(
                TaskResult(
                    task_id=task_id,
                    passed=passed,
                    system_used=system_used,
                    latency_ms=round(latency, 1),
                    cost_usd=round(cost, 6),
                    error=error,
                )
            )

            self.manifest.add(TaskTrace(
                task_id=task_id,
                passed=passed,
                latency_ms=round(latency, 1),
                cost_usd=round(cost, 6),
                model=model_id,
                routing=f"S{system_used}",
                error=error[:200] if error else "",
            ))

            status = "PASS" if passed else "FAIL"
            log.info(
                f"[{i + 1}/{len(problems)}] {task_id}: {status} ({latency:.0f}ms)"
            )

            # Emit benchmark event
            if self.event_bus:
                from sage.agent_loop import AgentEvent

                self.event_bus.emit(
                    AgentEvent(
                        type="BENCH_RESULT",
                        step=i + 1,
                        timestamp=time.time(),
                        meta={
                            "task_id": task_id,
                            "passed": passed,
                            "system_used": system_used,
                            "latency_ms": round(latency, 1),
                            "progress": f"{i + 1}/{len(problems)}",
                        },
                    )
                )

        return BenchReport.from_results("humaneval", results)
