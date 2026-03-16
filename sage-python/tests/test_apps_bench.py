"""Tests for APPS benchmark adapter."""
from __future__ import annotations

import json

import pytest


class TestAPPSImport:
    def test_import(self):
        from sage.bench.apps_bench import APPSBench
        assert callable(APPSBench)

    def test_extract_program(self):
        from sage.bench.apps_bench import _extract_program

        # Fenced code block
        response = "Here is my solution:\n```python\nx = int(input())\nprint(x * 2)\n```"
        code = _extract_program(response)
        assert "int(input())" in code
        assert "print(x * 2)" in code

        # Raw code (no fence)
        response2 = "x = int(input())\nprint(x * 2)"
        code2 = _extract_program(response2)
        assert "int(input())" in code2

    def test_parse_input_output(self):
        from sage.bench.apps_bench import _parse_input_output

        # Valid JSON with inputs/outputs
        raw = json.dumps({"inputs": ["1\n", "2\n"], "outputs": ["2\n", "4\n"]})
        pairs = _parse_input_output(raw)
        assert len(pairs) == 2
        assert pairs[0] == ("1\n", "2\n")
        assert pairs[1] == ("2\n", "4\n")

        # Empty string
        assert _parse_input_output("") == []

        # Invalid JSON
        assert _parse_input_output("not json") == []

    def test_parse_input_output_none(self):
        from sage.bench.apps_bench import _parse_input_output
        assert _parse_input_output(None) == []

    def test_difficulty_levels(self):
        from sage.bench.apps_bench import DIFFICULTY_LEVELS
        assert "introductory" in DIFFICULTY_LEVELS
        assert "interview" in DIFFICULTY_LEVELS
        assert "competition" in DIFFICULTY_LEVELS


class TestAPPSDataset:
    def test_dataset_loadable(self):
        """Verify APPS dataset can be loaded (skips if not available)."""
        try:
            from datasets import load_dataset
        except ImportError:
            pytest.skip("datasets library not installed")
        try:
            ds = load_dataset(
                "codeparrot/apps", split="test", streaming=True, trust_remote_code=True,
            )
            first = next(iter(ds))
            assert "question" in first
            assert "solutions" in first or "input_output" in first
        except Exception as exc:
            pytest.skip(f"APPS dataset not accessible: {exc}")


class TestAPPSEval:
    def test_eval_simple_io(self):
        """Test subprocess I/O evaluation with a trivial program."""
        from sage.bench.apps_bench import APPSBench

        solution = "x = int(input())\nprint(x * 2)"
        test_cases = [("3\n", "6"), ("5\n", "10")]
        passed, stderr = APPSBench._evaluate_io(
            solution=solution,
            test_cases=test_cases,
            task_id="test/0",
            timeout=10,
        )
        assert passed
        assert stderr == ""

    def test_eval_wrong_output(self):
        """Test that wrong output is detected."""
        from sage.bench.apps_bench import APPSBench

        solution = "x = int(input())\nprint(x + 1)"  # Wrong: adds instead of doubles
        test_cases = [("3\n", "6")]
        passed, stderr = APPSBench._evaluate_io(
            solution=solution,
            test_cases=test_cases,
            task_id="test/1",
            timeout=10,
        )
        assert not passed
        assert "mismatch" in stderr

    def test_eval_syntax_error(self):
        """Test that syntax errors are caught."""
        from sage.bench.apps_bench import APPSBench

        solution = "def broken(\n"  # Syntax error
        test_cases = [("1\n", "1")]
        passed, stderr = APPSBench._evaluate_io(
            solution=solution,
            test_cases=test_cases,
            task_id="test/2",
            timeout=10,
        )
        assert not passed
        assert stderr  # Should have error info
