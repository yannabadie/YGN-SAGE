"""End-to-end integration test for host-suspend detection in the bench loop.

Cycle-9 recovery Step γ.2 (cgpro 2026-05-04). The unit tests in
``test_bench_watchdog.py`` cover the wallclock primitive in isolation;
``test_event_ledger.py`` covers the ledger schema in isolation. This
test connects the two through ``BigCodeBenchBench.run`` to prove the
wired path actually catches a sleep-poisoned task end-to-end.

Pattern: mock-based, no LLM / no dataset. We patch:

- ``BigCodeBenchBench._load_dataset`` to return a single fake task,
- ``self.system.run`` to return correct code in <1s of real time,
- ``BigCodeBenchBench._evaluate_solution_with_stderr`` to deterministically
  return (passed=True, stderr=""),
- ``sage.bench.bigcodebench_bench.time.time`` so the wall-clock
  computation looks like the OS just resumed from a suspend (or didn't).

Verifies:
1. Normal completion → TASK_END with status=PASS, no TASK_ABORT.
2. Simulated 200s wall-elapsed with task_timeout=60 (grace 2.0 → 120s
   threshold) → TASK_ABORT, status=ABORT, passed_count not incremented.
3. Just-under-grace 119s → TASK_END (negative case, threshold not crossed).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sage.bench.bigcodebench_bench import BigCodeBenchBench
from sage.bench.event_ledger import BenchEventLedger, build_run_meta


def _read_lines(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _FakeSystem:
    """Minimal stand-in for AgentSystem; satisfies the attribute reads
    in BigCodeBenchBench.run + _capture_control_surface.
    """

    def __init__(self, response: str = "def task_func():\n    return 0\n") -> None:
        self._response = response
        self.pipeline = None
        self.agent_loop = None
        self.tool_registry = None

    async def run(self, task_input: Any) -> str:
        # Yield once so asyncio.wait_for has something to schedule.
        await asyncio.sleep(0)
        return self._response


def _fake_dataset() -> dict[str, dict]:
    """Return a single deterministic task that the mock evaluator will
    grade as passed=True so the test can distinguish "task ran cleanly"
    from "task was aborted by suspend"."""
    return {
        "BigCodeBench/0": {
            "task_id": "BigCodeBench/0",
            "instruct_prompt": "Return zero.",
            "complete_prompt": "def task_func():\n    pass\n",
            "test": (
                "import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_zero(self):\n"
                "        self.assertEqual(task_func(), 0)\n"
            ),
            "entry_point": "task_func",
            "code_prompt": "def task_func():\n",
        },
    }


import time as _stdlib_time

# Cycle-11 cgpro VERIFY follow-up (2026-05-05): capture the real
# ``time.time`` FUNCTION REFERENCE at module-import time, NOT the
# ``time`` module. The previous implementation captured the module
# inside ``_make_time_seq``; ``monkeypatch.setattr("sage.bench.
# bigcodebench_bench.time.time", fake_time)`` patches the same
# module-level binding (``time`` is a singleton), so when the
# fake_time fallback called ``_real_time.time()`` it re-entered
# the patched fake_time → infinite recursion → RecursionError on
# Windows where one extra ``time.time()`` call exhausts the seq.
# Linux passed by accident because the seq sized exactly to its
# call count.
_REAL_TIME_FUNC = _stdlib_time.time


def _make_time_seq(values: list[float]):
    """Return a fake_time() that yields ``values`` then falls back to real time().

    Real fall-through is needed because ``BigCodeBenchBench`` and its
    subprocess evaluator may call ``time.time()`` more than twice
    (e.g. inside the eval subprocess timeout machinery).
    """
    seq = list(values)

    def fake_time() -> float:
        if seq:
            return seq.pop(0)
        # Use the captured function reference so this never
        # re-enters fake_time even if ``time.time`` is monkey-
        # patched module-wide.
        return _REAL_TIME_FUNC()

    return fake_time


@pytest.mark.asyncio
async def test_normal_task_emits_task_end(tmp_path: Path) -> None:
    """Happy path: clean completion → TASK_END status=PASS, no abort."""
    ledger_path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="ablation-test", tier="budget", timeout_s=60.0)
    ledger = BenchEventLedger(ledger_path, meta)

    bench = BigCodeBenchBench(
        system=_FakeSystem(),
        subset="hard",
        split="instruct",
        task_timeout=60.0,
        eval_timeout=10.0,
        event_ledger=ledger,
        config_label="full",
        wallclock_grace_factor=2.0,
    )

    with patch(
        "sage.bench.bigcodebench_bench._load_dataset",
        return_value=_fake_dataset(),
    ), patch.object(
        BigCodeBenchBench, "_evaluate_solution_with_stderr",
        return_value=(True, ""),
    ):
        ledger.emit_run_start()
        report = await bench.run(limit=1)
        ledger.emit_run_end()
        ledger.close()

    assert report.total == 1
    assert report.passed == 1
    lines = _read_lines(ledger_path)
    events = [r["event"] for r in lines]
    assert "TASK_END" in events
    assert "TASK_ABORT" not in events
    end = next(r for r in lines if r["event"] == "TASK_END")
    assert end["status"] == "PASS"
    assert end["passed"] is True
    assert end["host_suspend_or_event_loop_stall"] is False
    # Cycle-9 α post-mortem: control_surface must carry omega/delta/gamma
    # so future replays can diagnose BCB/82-style topology coupling
    # (5-node robust → 3-node sequential when _skip_guardrails toggled)
    # without re-running the whole ablation. Fields can be None when no
    # pipeline ctx (e.g. mock system here) — but they MUST be present.
    cs = end["control_surface"]
    assert "dag_omega" in cs, "control_surface must include AdaptOrch dag_omega"
    assert "dag_delta" in cs, "control_surface must include AdaptOrch dag_delta"
    assert "dag_gamma" in cs, "control_surface must include AdaptOrch dag_gamma"
    # Cycle-9 α post-mortem 2026-05-04 (cgpro): the executed_template field
    # used to be the ULID topology_id, not the template name. Pin the
    # contract: control_surface MUST have separate topology_id + selected_template
    # + executed_template fields (values may be empty when no pipeline ctx).
    assert "topology_id" in cs, "control_surface must include ULID topology_id"
    assert "selected_template" in cs, "control_surface must include selected_template"
    assert "executed_template" in cs, "control_surface must include executed_template"


@pytest.mark.asyncio
async def test_simulated_suspend_emits_task_abort_and_excludes_task(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """200s wall-elapsed with timeout=60, grace=2.0 → ABORT not PASS.

    Even though the mock evaluator says passed=True, the host-suspend
    detector flips the result to FAIL/aborted. ``passed_count`` does
    NOT increment. Ledger emits TASK_ABORT with reason=host_suspend_detected.
    """
    ledger_path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="ablation-test", tier="budget", timeout_s=60.0)
    ledger = BenchEventLedger(ledger_path, meta)

    bench = BigCodeBenchBench(
        system=_FakeSystem(),
        subset="hard",
        split="instruct",
        task_timeout=60.0,
        eval_timeout=10.0,
        event_ledger=ledger,
        config_label="full",
        wallclock_grace_factor=2.0,
    )

    fake_time = _make_time_seq([1000.0, 1200.0])  # 200s elapsed wall

    with patch(
        "sage.bench.bigcodebench_bench._load_dataset",
        return_value=_fake_dataset(),
    ), patch.object(
        BigCodeBenchBench, "_evaluate_solution_with_stderr",
        return_value=(True, ""),
    ), patch(
        "sage.bench.bigcodebench_bench.time.time",
        side_effect=fake_time,
    ):
        with caplog.at_level("WARNING"):
            ledger.emit_run_start()
            report = await bench.run(limit=1)
            ledger.emit_run_end()
            ledger.close()

    assert report.total == 1
    # Critical: passed_count must be 0 even though the mock evaluator
    # said the solution passed. The wall-clock detection MUST mask
    # success that happened across a suspend.
    assert report.passed == 0, (
        "host_suspend_detected must exclude the task from pass_rate "
        "regardless of mock evaluator outcome"
    )

    lines = _read_lines(ledger_path)
    events = [r["event"] for r in lines]
    assert "TASK_ABORT" in events
    assert "TASK_END" not in events  # mutually exclusive on this path
    abort = next(r for r in lines if r["event"] == "TASK_ABORT")
    assert abort["reason"] == "host_suspend_detected"
    assert abort["elapsed_wall_ms"] == pytest.approx(200_000.0, rel=1e-3)
    assert abort["timeout_s"] == 60.0
    assert abort["grace_factor"] == 2.0
    cs = abort["control_surface"]
    assert cs["error"] == "host_suspend_detected"
    # ABORT status surfaced in logs
    abort_logs = [r for r in caplog.records if "ABORT" in r.getMessage()]
    assert abort_logs, f"expected ABORT in logs; got {[r.getMessage() for r in caplog.records]}"


@pytest.mark.asyncio
async def test_just_under_grace_factor_emits_task_end(tmp_path: Path) -> None:
    """119s elapsed with timeout=60, grace=2.0 → 119 < 120 → TASK_END not abort.

    Pins the threshold from the inequality side (negative test).
    """
    ledger_path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="ablation-test", tier="budget", timeout_s=60.0)
    ledger = BenchEventLedger(ledger_path, meta)

    bench = BigCodeBenchBench(
        system=_FakeSystem(),
        subset="hard",
        split="instruct",
        task_timeout=60.0,
        eval_timeout=10.0,
        event_ledger=ledger,
        config_label="full",
        wallclock_grace_factor=2.0,
    )

    fake_time = _make_time_seq([1000.0, 1119.0])  # 119s elapsed wall, just under 120s threshold

    with patch(
        "sage.bench.bigcodebench_bench._load_dataset",
        return_value=_fake_dataset(),
    ), patch.object(
        BigCodeBenchBench, "_evaluate_solution_with_stderr",
        return_value=(True, ""),
    ), patch(
        "sage.bench.bigcodebench_bench.time.time",
        side_effect=fake_time,
    ):
        ledger.emit_run_start()
        report = await bench.run(limit=1)
        ledger.emit_run_end()
        ledger.close()

    assert report.total == 1
    assert report.passed == 1, "119s < 120s grace threshold → must not trip suspend detection"
    lines = _read_lines(ledger_path)
    events = [r["event"] for r in lines]
    assert "TASK_END" in events
    assert "TASK_ABORT" not in events
    end = next(r for r in lines if r["event"] == "TASK_END")
    assert end["passed"] is True
    assert end["elapsed_wall_ms"] == pytest.approx(119_000.0, rel=1e-3)
