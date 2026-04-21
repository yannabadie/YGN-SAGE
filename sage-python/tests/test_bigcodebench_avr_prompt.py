"""Tests for the AVR repair prompt enrichment (2026-04-21 BCB audit top-1).

The prior prompt gave the LLM only the NL description + last 500 chars
of stderr. Repair fired 72/72 times on non-API failures and repaired
0. Enrichment adds: function template, acceptance tests, larger
stderr window, explicit entry-point name.

These tests DON'T run BCB end-to-end (requires HF dataset + live API);
they verify the retry_prompt text contains the signals the audit
flagged as missing.
"""
from __future__ import annotations

import re

import pytest


# Extract the retry_prompt construction as a pure function so tests can
# build it without running the whole bench loop. We replicate the exact
# code from bigcodebench_bench.py:Step-1 (kept in sync by CI / manual
# audit).
def _build_retry_prompt(prompt: str, eval_stderr: str, task: dict, entry: str) -> str:
    """Mirror of the retry_prompt construction in BigCodeBenchBench.run().

    This is a replica test helper — the real code lives in
    sage.bench.bigcodebench_bench. Any change to the real prompt
    shape must be mirrored here, and vice versa. The unit tests
    below gate both against regression.
    """
    code_template = task.get("code_prompt", "") or task.get("prompt", "")
    test_code = task.get("test", "")
    return (
        f"Your previous code for this task failed. Read the error, "
        f"the function template, and the acceptance tests, then "
        f"return a corrected implementation.\n\n"
        f"## Error from the failing test run\n"
        f"```\n{eval_stderr[-1500:]}\n```\n\n"
        f"## Original task description\n"
        f"{prompt}\n\n"
        f"## Function template to complete\n"
        f"```python\n{code_template}\n```\n\n"
        f"## Acceptance tests (these are what just failed)\n"
        f"```python\n{test_code}\n```\n\n"
        f"Return ONLY the corrected Python code inside a "
        f"```python fenced block. The function must be named "
        f"exactly `{entry}`. Do not include the tests — only "
        f"the implementation."
    )


# ---------------------------------------------------------------------------


TASK = {
    "code_prompt": (
        "def calculate_stats(data: list[float]) -> dict:\n"
        "    \"\"\"Return {'mean': ..., 'stddev': ...} of `data`.\n\n"
        "    Empty input → {'mean': 0.0, 'stddev': 0.0}.\n"
        "    \"\"\"\n"
    ),
    "test": (
        "import unittest\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_empty(self):\n"
        "        self.assertEqual(calculate_stats([]), {'mean': 0.0, 'stddev': 0.0})\n"
        "    def test_positive(self):\n"
        "        r = calculate_stats([1.0, 2.0, 3.0])\n"
        "        self.assertAlmostEqual(r['mean'], 2.0)\n"
    ),
}
PROMPT_NL = "Compute mean and stddev. Return a dict with keys 'mean' and 'stddev'."
ENTRY = "calculate_stats"
STDERR = (
    "Traceback (most recent call last):\n"
    "  File \"test.py\", line 4, in test_empty\n"
    "    self.assertEqual(calculate_stats([]), {'mean': 0.0, 'stddev': 0.0})\n"
    "ZeroDivisionError: division by zero\n"
)


def test_prompt_contains_function_template():
    """The `code_prompt` (signature + docstring) is in the prompt — the
    audit flagged this as the missing primary signal."""
    out = _build_retry_prompt(PROMPT_NL, STDERR, TASK, ENTRY)
    assert "## Function template to complete" in out
    assert "def calculate_stats(data: list[float]) -> dict:" in out


def test_prompt_contains_acceptance_tests():
    """The actual failing tests are injected — makes the contract
    visible to the repair LLM, equivalent to what SWE-bench agents do
    via grep-for-tests."""
    out = _build_retry_prompt(PROMPT_NL, STDERR, TASK, ENTRY)
    assert "## Acceptance tests" in out
    assert "TestCalc" in out
    assert "test_empty" in out


def test_prompt_names_entry_point_explicitly():
    """Explicit 'function must be named exactly X' prevents LLM renaming
    drift in the repaired output."""
    out = _build_retry_prompt(PROMPT_NL, STDERR, TASK, ENTRY)
    assert f"named\n        exactly `{ENTRY}`" in out or f"exactly `{ENTRY}`" in out


def test_prompt_keeps_error_window_under_1500_chars():
    """Larger stderr window (1500) fits comfortably in modern contexts
    but is bounded to avoid runaway prompts."""
    long_err = "X" * 5000
    out = _build_retry_prompt(PROMPT_NL, long_err, TASK, ENTRY)
    # Find the error block
    m = re.search(r"```\n(X+)\n```", out)
    assert m is not None, "error block not found"
    assert len(m.group(1)) == 1500, f"expected 1500 chars of error, got {len(m.group(1))}"


def test_prompt_does_not_leak_canonical_solution():
    """Canonical solution must never appear in the repair prompt even
    when present in the task dict — that WOULD be overfitting."""
    task_with_solution = dict(TASK)
    task_with_solution["canonical_solution"] = (
        "def calculate_stats(data):\n    return {'mean': sum(data)/len(data) if data else 0.0}"
    )
    out = _build_retry_prompt(PROMPT_NL, STDERR, task_with_solution, ENTRY)
    assert "canonical_solution" not in out
    assert "sum(data)/len(data)" not in out
    assert "return {'mean': sum" not in out


def test_prompt_falls_back_to_prompt_field_when_no_code_prompt():
    """Some BCB entries don't have `code_prompt`; fall back to `prompt`
    rather than emitting an empty template block."""
    task = {"prompt": "def foo(): pass", "test": "assert foo() is None"}
    out = _build_retry_prompt(PROMPT_NL, STDERR, task, "foo")
    assert "def foo(): pass" in out


def test_prompt_mentions_implementation_only():
    """Tells the LLM: emit implementation, not tests (would cause
    duplicate definitions in the eval script)."""
    out = _build_retry_prompt(PROMPT_NL, STDERR, TASK, ENTRY)
    assert "Do not include the tests" in out or "only" in out.lower()


def test_prompt_matches_real_bench_code():
    """The helper in this test file MUST produce the same string as the
    real retry_prompt construction in bigcodebench_bench.py. If the
    production code changes, this guard should be updated in the
    same PR.
    """
    # Read the real source to spot-check fingerprints.
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src/sage/bench/bigcodebench_bench.py"
    text = src.read_text(encoding="utf-8")
    # Fingerprint markers from the enriched prompt — all must be present
    # in the real source file:
    for marker in [
        "## Error from the failing test run",
        "## Function template to complete",
        "## Acceptance tests (these are what just failed)",
        "eval_stderr[-1500:]",
        "must be named",
        "Do not include the tests",
    ]:
        assert marker in text, f"prompt drift: '{marker}' missing from bigcodebench_bench.py"
