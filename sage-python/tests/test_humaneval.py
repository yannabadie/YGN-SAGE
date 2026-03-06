"""Tests for the HumanEval benchmark module."""

import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest

from sage.bench.humaneval import (
    load_problems,
    extract_code,
    run_test,
    HumanEvalBench,
)


# ---------------------------------------------------------------------------
# load_problems
# ---------------------------------------------------------------------------


def test_load_problems():
    problems = load_problems()
    assert len(problems) == 164
    assert "task_id" in problems[0]
    assert "prompt" in problems[0]
    assert "test" in problems[0]
    assert "entry_point" in problems[0]


def test_load_problems_with_limit():
    problems = load_problems(limit=10)
    assert len(problems) == 10


# ---------------------------------------------------------------------------
# extract_code
# ---------------------------------------------------------------------------


def test_extract_code_from_fenced_block():
    response = (
        "Here's the solution:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
        "Done!"
    )
    code = extract_code(response, "add")
    assert "def add" in code
    assert "return a + b" in code


def test_extract_code_from_raw():
    response = "def add(a, b):\n    return a + b"
    code = extract_code(response, "add")
    assert "return a + b" in code


def test_extract_code_fenced_block_without_entry_point():
    """Falls back to the last code block when entry_point not found."""
    response = (
        "```python\n"
        "x = 42\n"
        "```\n"
    )
    code = extract_code(response, "nonexistent")
    assert "x = 42" in code


def test_extract_code_empty_response():
    """Empty response returns empty string."""
    code = extract_code("", "add")
    assert code == ""


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def test_run_test_passes():
    prompt = "def add(a, b):\n"
    completion = "    return a + b\n"
    test_code = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
    )
    passed, error = run_test(prompt, completion, test_code, "add")
    assert passed
    assert error == ""


def test_run_test_fails():
    prompt = "def add(a, b):\n"
    completion = "    return a - b\n"  # Wrong!
    test_code = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    passed, error = run_test(prompt, completion, test_code, "add")
    assert not passed
    assert error  # Should have error message


def test_run_test_timeout():
    prompt = "def hang():\n"
    completion = "    while True: pass\n"
    test_code = "def check(candidate):\n    candidate()\n"
    passed, error = run_test(prompt, completion, test_code, "hang", timeout=2.0)
    assert not passed
    assert "Timeout" in error


# ---------------------------------------------------------------------------
# HumanEvalBench
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_humaneval_bench_no_system():
    """HumanEval bench runs in direct mode (no agent, no LLM)."""
    bench = HumanEvalBench()
    report = await bench.run(limit=3)
    assert report.total == 3
    assert report.benchmark == "humaneval"
    assert 0.0 <= report.pass_rate <= 1.0
    assert report.avg_latency_ms >= 0
