"""Tests for GRPO v4 execution reward — real sandbox execution."""
import asyncio
import pytest


class TestExtractPythonCode:
    def test_fenced_python_block(self):
        from sage.grpo.execution_reward import extract_python_code
        text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        assert extract_python_code(text) == "def add(a, b):\n    return a + b"

    def test_fenced_block_no_lang(self):
        from sage.grpo.execution_reward import extract_python_code
        text = "```\nimport math\nprint(math.pi)\n```"
        assert "import math" in extract_python_code(text)

    def test_raw_code(self):
        from sage.grpo.execution_reward import extract_python_code
        assert extract_python_code("def foo():\n    pass") == "def foo():\n    pass"

    def test_think_tags_stripped(self):
        from sage.grpo.execution_reward import extract_python_code
        text = "<think>reasoning here</think>\n```python\nprint(42)\n```"
        code = extract_python_code(text)
        assert "<think>" not in code
        assert "print(42)" in code

    def test_no_code_returns_none(self):
        from sage.grpo.execution_reward import extract_python_code
        assert extract_python_code("This is just text with no code.") is None

    def test_empty_string(self):
        from sage.grpo.execution_reward import extract_python_code
        assert extract_python_code("") is None


class TestRunCodeSandbox:
    @pytest.mark.asyncio
    async def test_simple_print(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = await run_code_sandbox("print(42)", timeout=10)
        assert result.exit_code == 0
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = await run_code_sandbox("def (", timeout=10)
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_timeout(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = await run_code_sandbox("import time; time.sleep(60)", timeout=2)
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_runtime_error(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = await run_code_sandbox("raise ValueError('boom')", timeout=10)
        assert result.exit_code != 0
        assert "ValueError" in result.stderr


class TestComputeExecutionScore:
    @pytest.mark.asyncio
    async def test_code_runs_ok(self):
        from sage.grpo.execution_reward import compute_execution_score
        score, status = await compute_execution_score("print(42)", "")
        assert score == 0.3
        assert status == "RUNS_OK"

    @pytest.mark.asyncio
    async def test_code_crash(self):
        from sage.grpo.execution_reward import compute_execution_score
        score, status = await compute_execution_score("raise Exception()", "")
        assert score == 0.0
        assert status == "CRASH"

    @pytest.mark.asyncio
    async def test_code_timeout(self):
        from sage.grpo.execution_reward import compute_execution_score
        score, status = await compute_execution_score(
            "import time; time.sleep(60)", "", timeout=2
        )
        assert score == 0.5
        assert status == "TIMEOUT"


class TestStdinTests:
    @pytest.mark.asyncio
    async def test_stdin_passed(self):
        from sage.grpo.execution_reward import _run_stdin_tests
        code = "n = int(input())\nprint(n * 2)"
        pairs = [("3\n", "6\n"), ("5\n", "10\n")]
        score, status = await _run_stdin_tests(code, pairs)
        assert score == 1.5
        assert status == "PASSED"

    @pytest.mark.asyncio
    async def test_stdin_wrong_answer(self):
        from sage.grpo.execution_reward import _run_stdin_tests
        code = "n = int(input())\nprint(n + 1)"  # wrong: adds 1 instead of doubling
        pairs = [("3\n", "6\n")]
        score, status = await _run_stdin_tests(code, pairs)
        assert score == 1.0
        assert status == "WRONG_ANSWER"

    @pytest.mark.asyncio
    async def test_stdin_crash(self):
        from sage.grpo.execution_reward import _run_stdin_tests
        code = "raise ValueError()"
        pairs = [("3\n", "6\n")]
        score, status = await _run_stdin_tests(code, pairs)
        # Crash produces empty stdout → classified as WRONG_ANSWER (no output matched)
        assert status == "WRONG_ANSWER"


class TestBuildTopologyGraph:
    def test_nodes_and_edges_built(self):
        from sage.grpo.execution_reward import _build_topology_graph
        topo = {
            "difficulty": "moderate",
            "nodes": [
                {"role": "planner", "prompt": "Plan the task"},
                {"role": "coder", "prompt": "Write the code"},
            ],
            "edges": [{"from_idx": 0, "to_idx": 1, "flow_type": "message"}],
        }
        graph = _build_topology_graph(topo)
        if graph is None:
            pytest.skip("sage_core not available")
        assert graph.node_count() == 2
        assert graph.edge_count() == 1

    def test_system_from_difficulty(self):
        from sage.grpo.execution_reward import _build_topology_graph
        try:
            from sage_core import TopologyDensity
        except ImportError:
            pytest.skip("sage_core not available")
        for diff, expected_nmax in [("simple", 4), ("moderate", 7), ("complex", 10)]:
            topo = {"difficulty": diff, "nodes": [{"role": "coder"}]}
            graph = _build_topology_graph(topo)
            sys_val = {"simple": 1, "moderate": 2, "complex": 3}[diff]
            density = TopologyDensity()
            score = density.compute(graph, sys_val)
            assert score.n_max == expected_nmax

    def test_fallback_sequential_edges(self):
        from sage.grpo.execution_reward import _build_topology_graph
        topo = {"nodes": [{"role": "a"}, {"role": "b"}, {"role": "c"}]}
        graph = _build_topology_graph(topo)
        if graph is None:
            pytest.skip("sage_core not available")
        assert graph.edge_count() == 2

    def test_invalid_edge_indices_skipped(self):
        from sage.grpo.execution_reward import _build_topology_graph
        topo = {
            "nodes": [{"role": "a"}],
            "edges": [{"from_idx": 0, "to_idx": 99, "flow_type": "message"}],
        }
        graph = _build_topology_graph(topo)
        if graph is None:
            pytest.skip("sage_core not available")
        assert graph.edge_count() == 0
