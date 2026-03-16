"""Tests for LiveCodeBench benchmark adapter."""
from __future__ import annotations

import json

import pytest


class TestLiveCodeBenchImport:
    def test_import(self):
        from sage.bench.livecodebench_bench import LiveCodeBenchBench
        assert callable(LiveCodeBenchBench)

    def test_extract_program(self):
        from sage.bench.livecodebench_bench import _extract_program

        # Fenced code block
        response = "Here is my solution:\n```python\nn = int(input())\nprint(n)\n```"
        code = _extract_program(response)
        assert "int(input())" in code
        assert "print(n)" in code

        # Raw code (no fence)
        response2 = "import sys\nfor line in sys.stdin:\n    print(line)"
        code2 = _extract_program(response2)
        assert "sys.stdin" in code2

    def test_parse_test_cases_list_format(self):
        from sage.bench.livecodebench_bench import _parse_test_cases

        # Format: list of {input, output} dicts
        problem = {
            "public_test_cases": json.dumps([
                {"input": "1\n", "output": "2\n"},
                {"input": "3\n", "output": "4\n"},
            ]),
        }
        pairs = _parse_test_cases(problem)
        assert len(pairs) == 2
        assert pairs[0] == ("1\n", "2\n")

    def test_parse_test_cases_apps_format(self):
        from sage.bench.livecodebench_bench import _parse_test_cases

        # Format: {inputs: [...], outputs: [...]}
        problem = {
            "input_output": json.dumps({
                "inputs": ["5\n", "10\n"],
                "outputs": ["25\n", "100\n"],
            }),
        }
        pairs = _parse_test_cases(problem)
        assert len(pairs) == 2
        assert pairs[0] == ("5\n", "25\n")

    def test_parse_test_cases_empty(self):
        from sage.bench.livecodebench_bench import _parse_test_cases
        assert _parse_test_cases({}) == []
        assert _parse_test_cases({"public_test_cases": ""}) == []
        assert _parse_test_cases({"public_test_cases": "not json"}) == []

    def test_build_prompt(self):
        from sage.bench.livecodebench_bench import LiveCodeBenchBench

        prompt = LiveCodeBenchBench._build_prompt("Sum two numbers", "")
        assert "Sum two numbers" in prompt
        assert "stdin" in prompt

        prompt_with_ctx = LiveCodeBenchBench._build_prompt(
            "Sum two numbers", "from typing import List"
        )
        assert "from typing import List" in prompt_with_ctx


class TestLiveCodeBenchDataset:
    def test_dataset_loadable(self):
        """Verify LiveCodeBench dataset can be loaded (skips if not available)."""
        try:
            from datasets import load_dataset
        except ImportError:
            pytest.skip("datasets library not installed")
        try:
            ds = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                streaming=True,
                trust_remote_code=True,
            )
            first = next(iter(ds))
            assert "question_content" in first or "question" in first
        except Exception as exc:
            pytest.skip(f"LiveCodeBench dataset not accessible: {exc}")


class TestLiveCodeBenchEval:
    def test_eval_simple_io(self):
        """Test subprocess I/O evaluation with a trivial program."""
        from sage.bench.livecodebench_bench import LiveCodeBenchBench

        solution = "n = int(input())\nprint(n ** 2)"
        test_cases = [("3\n", "9"), ("5\n", "25")]
        passed, stderr = LiveCodeBenchBench._evaluate_io(
            solution=solution,
            test_cases=test_cases,
            task_id="test/0",
            timeout=10,
        )
        assert passed
        assert stderr == ""

    def test_eval_wrong_output(self):
        """Test that wrong output is detected."""
        from sage.bench.livecodebench_bench import LiveCodeBenchBench

        solution = "n = int(input())\nprint(n + 1)"
        test_cases = [("3\n", "9")]
        passed, stderr = LiveCodeBenchBench._evaluate_io(
            solution=solution,
            test_cases=test_cases,
            task_id="test/1",
            timeout=10,
        )
        assert not passed
        assert "mismatch" in stderr

    def test_eval_runtime_error(self):
        """Test that runtime errors are caught."""
        from sage.bench.livecodebench_bench import LiveCodeBenchBench

        solution = "raise ValueError('boom')"
        test_cases = [("1\n", "1")]
        passed, stderr = LiveCodeBenchBench._evaluate_io(
            solution=solution,
            test_cases=test_cases,
            task_id="test/2",
            timeout=10,
        )
        assert not passed
        assert stderr  # Should have error info
