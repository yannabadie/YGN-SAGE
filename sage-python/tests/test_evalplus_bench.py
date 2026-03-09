"""Tests for the EvalPlus benchmark adapter (sage.bench.evalplus_bench)."""

import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sage.bench.evalplus_bench import EvalPlusBench, _load_dataset
from sage.bench.runner import BenchReport


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------


def test_evalplus_bench_imports():
    """EvalPlusBench and helpers can be imported."""
    from sage.bench.evalplus_bench import EvalPlusBench, _load_dataset

    assert EvalPlusBench is not None
    assert callable(_load_dataset)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_evalplus_bench_init():
    """Default init uses humaneval dataset."""
    bench = EvalPlusBench()
    assert bench.dataset == "humaneval"
    assert bench.system is None
    assert bench.event_bus is None
    assert bench.baseline_mode is False
    assert bench.manifest is None


def test_evalplus_bench_init_mbpp():
    """MBPP dataset can be selected."""
    bench = EvalPlusBench(dataset="mbpp")
    assert bench.dataset == "mbpp"
    assert bench.system is None
    assert bench.baseline_mode is False


def test_evalplus_bench_init_baseline():
    """baseline_mode flag is stored."""
    bench = EvalPlusBench(baseline_mode=True)
    assert bench.baseline_mode is True


def test_evalplus_bench_init_invalid_dataset():
    """Unknown dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        EvalPlusBench(dataset="nonexistent")


# ---------------------------------------------------------------------------
# generate_solutions
# ---------------------------------------------------------------------------


async def test_generate_solutions_no_system():
    """Without a system, generate_solutions returns empty list."""
    bench = EvalPlusBench()
    solutions = await bench.generate_solutions(limit=5)
    assert solutions == []


async def test_generate_solutions_with_mock_system():
    """Mock system generates solutions in the expected format."""
    # Build a mock AgentSystem
    mock_system = MagicMock()
    mock_system.agent_loop._last_model = "test-model"
    mock_system.agent_loop._last_routing_system = 2
    mock_system.agent_loop.total_cost_usd = 0.001

    # system.run() returns a code response
    mock_system.run = AsyncMock(
        return_value=(
            "```python\n"
            "def has_close_elements(numbers, threshold):\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
            "```"
        )
    )

    bench = EvalPlusBench(system=mock_system)
    solutions = await bench.generate_solutions(limit=3)

    # Should have 3 solutions
    assert len(solutions) == 3

    # Check structure of each solution
    for sol in solutions:
        assert "task_id" in sol
        assert "solution" in sol
        assert "_latency_ms" in sol
        assert "_cost_usd" in sol
        assert "_system_used" in sol
        assert "_error" in sol
        assert isinstance(sol["task_id"], str)
        assert isinstance(sol["solution"], str)
        assert sol["_latency_ms"] >= 0
        assert sol["_system_used"] == 2

    # system.run() should have been called 3 times
    assert mock_system.run.call_count == 3

    # Manifest should be populated
    assert bench.manifest is not None
    assert len(bench.manifest.traces) == 3
    assert bench.manifest.benchmark == "evalplus_humaneval"


async def test_generate_solutions_baseline_mode():
    """Baseline mode calls LLM directly, bypassing routing."""
    mock_llm_response = MagicMock()
    mock_llm_response.content = "def foo():\n    return 42\n"

    mock_system = MagicMock()
    mock_system.agent_loop._last_model = "test-model"
    mock_system.agent_loop._llm.generate = AsyncMock(
        return_value=mock_llm_response
    )
    mock_system.agent_loop.total_cost_usd = 0.0

    bench = EvalPlusBench(system=mock_system, baseline_mode=True)
    solutions = await bench.generate_solutions(limit=2)

    assert len(solutions) == 2
    # Baseline mode: system_used should be 0
    for sol in solutions:
        assert sol["_system_used"] == 0

    # system.run() should NOT have been called (baseline bypasses it)
    mock_system.run.assert_not_called()
    # LLM generate should have been called directly
    assert mock_system.agent_loop._llm.generate.call_count == 2

    # Manifest model should reflect baseline
    assert bench.manifest is not None
    assert bench.manifest.model.startswith("baseline:")


async def test_generate_solutions_with_event_bus():
    """Event bus receives BENCH_RESULT events during generation."""
    mock_system = MagicMock()
    mock_system.agent_loop._last_model = "test-model"
    mock_system.agent_loop._last_routing_system = 1
    mock_system.agent_loop.total_cost_usd = 0.0
    mock_system.run = AsyncMock(return_value="def foo():\n    return 1\n")

    mock_bus = MagicMock()

    bench = EvalPlusBench(system=mock_system, event_bus=mock_bus)
    solutions = await bench.generate_solutions(limit=2)

    assert len(solutions) == 2
    # Event bus should have emitted 2 events
    assert mock_bus.emit.call_count == 2
    # Each emitted event should be an AgentEvent with BENCH_RESULT type
    for call in mock_bus.emit.call_args_list:
        event = call[0][0]
        assert event.type == "BENCH_RESULT"
        assert "evalplus_humaneval" in event.meta["benchmark"]


async def test_generate_solutions_handles_errors():
    """Errors during generation are captured, not raised."""
    mock_system = MagicMock()
    mock_system.agent_loop._last_model = "test-model"
    mock_system.agent_loop._last_routing_system = 2
    mock_system.agent_loop.total_cost_usd = 0.0
    mock_system.run = AsyncMock(side_effect=RuntimeError("LLM exploded"))

    bench = EvalPlusBench(system=mock_system)
    solutions = await bench.generate_solutions(limit=2)

    # Should still return 2 solutions (with errors)
    assert len(solutions) == 2
    for sol in solutions:
        assert sol["solution"] == ""
        assert "LLM exploded" in sol["_error"]


# ---------------------------------------------------------------------------
# write_solutions
# ---------------------------------------------------------------------------


def test_write_solutions(tmp_path):
    """write_solutions creates a valid JSONL file."""
    bench = EvalPlusBench()
    solutions = [
        {"task_id": "HumanEval/0", "solution": "def foo():\n    return 1\n"},
        {"task_id": "HumanEval/1", "solution": "def bar():\n    return 2\n"},
    ]
    out_path = tmp_path / "solutions.jsonl"
    bench.write_solutions(solutions, out_path)

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    import json

    for line in lines:
        obj = json.loads(line)
        assert "task_id" in obj
        assert "solution" in obj


# ---------------------------------------------------------------------------
# _load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_humaneval():
    """Loading humaneval dataset returns 164 tasks."""
    data = _load_dataset("humaneval")
    assert len(data) == 164
    first_key = list(data.keys())[0]
    assert first_key.startswith("HumanEval/")
    assert "prompt" in data[first_key]
    assert "entry_point" in data[first_key]


def test_load_dataset_invalid():
    """Unknown dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        _load_dataset("nonexistent")


# ---------------------------------------------------------------------------
# run (full pipeline, mocked evaluate)
# ---------------------------------------------------------------------------


async def test_run_no_system():
    """run() with no system returns empty report."""
    bench = EvalPlusBench()
    report = await bench.run(limit=3)
    assert isinstance(report, BenchReport)
    assert report.total == 0
    assert report.benchmark == "evalplus_humaneval"


async def test_run_with_mock_system():
    """run() with mock system returns a BenchReport."""
    mock_system = MagicMock()
    mock_system.agent_loop._last_model = "test-model"
    mock_system.agent_loop._last_routing_system = 2
    mock_system.agent_loop.total_cost_usd = 0.001
    mock_system.run = AsyncMock(
        return_value="def has_close_elements(numbers, threshold):\n    return False\n"
    )

    bench = EvalPlusBench(system=mock_system)

    # Mock evaluate to avoid running the real CLI
    with patch.object(bench, "evaluate", return_value={"eval_details": {"eval": {}}}):
        report = await bench.run(limit=2)

    assert isinstance(report, BenchReport)
    assert report.total == 2
    assert report.benchmark == "evalplus_humaneval"
    assert len(report.results) == 2
