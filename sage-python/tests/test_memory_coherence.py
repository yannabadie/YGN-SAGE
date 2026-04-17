"""Tests for the memory coherence benchmark.

Covers the LLM-free invariants: scoring, code extraction, report aggregation.
The live runner is exercised in a minimal mock-LLM smoke test below.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.bench.memory_coherence import (
    _extract_code,
    _score_solution,
    run_memory_coherence,
    TASK_PAIRS,
)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def test_score_empty_is_zero():
    assert _score_solution("") == 0.0
    assert _score_solution("   ") == 0.0


def test_score_trivial_string_no_code():
    assert _score_solution("x = 1") == 0.3  # only AST parses


def test_score_valid_function_full():
    solution = (
        "def is_palindrome(s):\n"
        "    s = s.lower()\n"
        "    return s == s[::-1]\n"
    )
    # > 30 chars (0.4) + has 'def ' (0.3) + parses (0.3) = 1.0
    assert _score_solution(solution) == pytest.approx(1.0, abs=1e-6)


def test_score_syntax_error_no_parse_bonus():
    # Long enough (> 30 chars) + has "def ", but broken syntax
    solution = (
        "def broken_function_name_here(\n"  # long, def, unterminated paren
        "    return 1\n"
    )
    assert len(solution.strip()) > 30
    # Has len>30 (0.4) + has "def " (0.3), no parse bonus
    assert _score_solution(solution) == pytest.approx(0.7, abs=1e-6)


# ----------------------------------------------------------------------
# Code extraction
# ----------------------------------------------------------------------

def test_extract_code_from_python_fence():
    response = (
        "Here you go:\n"
        "```python\n"
        "def foo():\n"
        "    return 42\n"
        "```\n"
        "Hope that helps."
    )
    code = _extract_code(response)
    assert "def foo():" in code
    assert "Here you go" not in code


def test_extract_code_no_fence_returns_raw():
    response = "def bar():\n    return 1"
    assert _extract_code(response) == response


def test_extract_code_handles_plain_fence():
    response = "```\nprint('hi')\n```"
    assert _extract_code(response).strip() == "print('hi')"


# ----------------------------------------------------------------------
# Task pairs sanity
# ----------------------------------------------------------------------

def test_task_pairs_nonempty_and_distinct():
    assert len(TASK_PAIRS) >= 3, "bench must cover >= 3 pairs"
    for anchor, probe in TASK_PAIRS:
        assert anchor and probe
        assert anchor != probe, "anchor and probe must differ"


# ----------------------------------------------------------------------
# Mock-LLM end-to-end smoke
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_coherence_mock_system_smoke():
    """End-to-end smoke: mock systems return canned code, bench aggregates.

    The bench calls boot_fn twice per pair:
      - first boot  = cold run
      - second boot = primed run (makes 2 system.run calls: anchor + probe)
    """

    # 22 chars, 'def ', parses → 0.6
    mock_cold = "def probe():\n    pass\n"
    # > 30 chars, 'def ', parses → 1.0
    mock_primed = (
        "def probe():\n"
        "    # adapted from prior anchor\n"
        "    return 42\n"
    )

    # boot_counter is 0,2,4,... for cold boots and 1,3,5,... for primed boots
    boot_counter = {"n": 0}

    async def boot_fn():
        idx = boot_counter["n"]
        boot_counter["n"] += 1
        is_primed = idx % 2 == 1
        sys = MagicMock()
        sys.run = AsyncMock(return_value=mock_primed if is_primed else mock_cold)
        sys.agent_loop = MagicMock(episodic_memory=None)
        return sys

    # Use only 2 pairs for speed
    report = await run_memory_coherence(boot_fn, limit=2)

    assert report.total == 2
    # Cold score = 0.6, Primed = 1.0, Δ = +0.4
    assert report.avg_cold_quality == pytest.approx(0.6, abs=1e-6)
    assert report.avg_primed_quality == pytest.approx(1.0, abs=1e-6)
    assert report.quality_gain == pytest.approx(0.4, abs=1e-6)
    assert report.cold_pass == 0
    assert report.primed_pass == 2
    assert len(report.probes) == 2
    for p in report.probes:
        assert p["cold_error"] is None
        assert p["primed_error"] is None
