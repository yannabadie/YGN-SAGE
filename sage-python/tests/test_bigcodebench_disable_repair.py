from __future__ import annotations

from typing import Any

import pytest

from sage.bench.bigcodebench_bench import BigCodeBenchBench


class _FailThenRepairSystem:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def run(self, task_input: Any) -> str:
        self.calls.append(task_input)
        return "```python\ndef solve():\n    return 'bad'\n```"


class _RepairAwareBench(BigCodeBenchBench):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reasoner_calls = 0

    async def _run_with_reasoner(self, prompt: str) -> str:
        self.reasoner_calls += 1
        return "```python\ndef solve():\n    return 'fixed'\n```"


def _one_failing_problem() -> dict[str, dict[str, Any]]:
    return {
        "BigCodeBench/1": {
            "entry_point": "solve",
            "instruct_prompt": "Return the expected string.",
            "code_prompt": "def solve():\n    pass",
            "test": "unused by patched evaluator",
            "libs": "[]",
        }
    }


def _patched_eval(
    *,
    solution: str,
    test_code: str,
    entry_point: str,
    task_id: str,
    timeout: float,
) -> tuple[bool, str]:
    del test_code, entry_point, task_id, timeout
    if "fixed" in solution:
        return True, ""
    return False, "assertion failed"


@pytest.mark.asyncio
async def test_disable_repair_reports_first_attempt_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sage.bench.bigcodebench_bench._load_dataset",
        lambda subset: _one_failing_problem(),
    )
    monkeypatch.setattr(
        BigCodeBenchBench,
        "_evaluate_solution_with_stderr",
        staticmethod(_patched_eval),
    )
    monkeypatch.setenv("SAGE_BENCH_DISABLE_REPAIR", "1")

    bench = _RepairAwareBench(system=_FailThenRepairSystem(), subset="hard")

    report = await bench.run(limit=1)

    assert bench.reasoner_calls == 0
    assert report.passed == 0
    assert report.failed == 1


@pytest.mark.asyncio
async def test_repair_branch_stays_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sage.bench.bigcodebench_bench._load_dataset",
        lambda subset: _one_failing_problem(),
    )
    monkeypatch.setattr(
        BigCodeBenchBench,
        "_evaluate_solution_with_stderr",
        staticmethod(_patched_eval),
    )
    monkeypatch.delenv("SAGE_BENCH_DISABLE_REPAIR", raising=False)

    bench = _RepairAwareBench(system=_FailThenRepairSystem(), subset="hard")

    report = await bench.run(limit=1)

    assert bench.reasoner_calls == 1
    assert report.passed == 1
    assert report.failed == 0
