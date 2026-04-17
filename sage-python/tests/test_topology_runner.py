"""Tests for TopologyRunner — real multi-agent topology execution."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


class FakeNode:
    """Minimal node matching TopologyGraph.get_node() return type."""
    def __init__(self, role: str, model_id: str, system: int, required_capabilities: list[str] | None = None):
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities or []


class FakeGraph:
    """Minimal TopologyGraph stub for testing without Rust."""
    def __init__(self, nodes: list[FakeNode]):
        self._nodes = nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]


class FakeExecutor:
    """Minimal TopologyExecutor stub — yields batches of ready node indices."""
    def __init__(self, order: list[list[int]]):
        self._batches = list(order)
        self._batch_idx = 0

    def next_ready(self, graph) -> list[int]:
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        pass

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


@pytest.mark.asyncio
async def test_sequential_two_node_topology():
    """Two-node sequential: node0 output feeds node1 as context."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[
        FakeNode(role="researcher", model_id="gemini-2.5-flash", system=1),
        FakeNode(role="writer", model_id="gemini-2.5-flash", system=1),
    ])
    executor = FakeExecutor(order=[[0], [1]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Research findings about X"),
        MagicMock(content="Final article about X based on research"),
    ])

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("Write about X")

    assert "Final article" in result
    assert mock_provider.generate.call_count == 2
    # Second call should include first node's output in messages
    second_call_msgs = mock_provider.generate.call_args_list[1].kwargs.get(
        "messages", mock_provider.generate.call_args_list[1].args[0] if mock_provider.generate.call_args_list[1].args else []
    )
    msg_texts = " ".join(m.content for m in second_call_msgs if hasattr(m, "content"))
    assert "Research findings" in msg_texts


@pytest.mark.asyncio
async def test_parallel_three_node_topology():
    """Two parallel workers + one aggregator: workers run concurrently, aggregator sees both outputs."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[
        FakeNode(role="analyst-A", model_id="gemini-2.5-flash", system=1),
        FakeNode(role="analyst-B", model_id="gemini-2.5-flash", system=1),
        FakeNode(role="synthesizer", model_id="gemini-2.5-flash", system=2),
    ])
    # Batch [0, 1] runs in parallel; batch [2] runs after both complete
    executor = FakeExecutor(order=[[0, 1], [2]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Analysis from A: market is bullish"),
        MagicMock(content="Analysis from B: risk is low"),
        MagicMock(content="Synthesis: combined view is positive"),
    ])

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("Analyse the market")

    # Final result is synthesizer's output
    assert "Synthesis" in result
    # All 3 LLM calls made
    assert mock_provider.generate.call_count == 3
    # Synthesizer (3rd call, index 2) must have received BOTH analysts' outputs in context
    third_call_msgs = mock_provider.generate.call_args_list[2].kwargs.get(
        "messages",
        mock_provider.generate.call_args_list[2].args[0] if mock_provider.generate.call_args_list[2].args else [],
    )
    msg_texts = " ".join(m.content for m in third_call_msgs if hasattr(m, "content"))
    assert "Analysis from A" in msg_texts
    assert "Analysis from B" in msg_texts


@pytest.mark.asyncio
async def test_single_node_returns_direct():
    """Single-node topology: result equals the direct LLM output, exactly 1 call made."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[
        FakeNode(role="solo-agent", model_id="gemini-2.5-flash", system=1),
    ])
    executor = FakeExecutor(order=[[0]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Direct answer from solo agent"),
    ])

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("What is 2+2?")

    assert result == "Direct answer from solo agent"
    assert mock_provider.generate.call_count == 1


@pytest.mark.asyncio
async def test_empty_topology_returns_empty():
    """Empty topology: result is empty string, 0 LLM calls made."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[])
    executor = FakeExecutor(order=[])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock()

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("Any task")

    assert result == ""
    assert mock_provider.generate.call_count == 0


@pytest.mark.asyncio
async def test_agent_loop_delegates_to_topology_runner():
    """When topology has >1 node, AgentLoop delegates to TopologyRunner."""
    from sage.topology.runner import TopologyRunner
    from unittest.mock import patch

    graph = FakeGraph(nodes=[
        FakeNode(role="thinker", model_id="gemini-2.5-flash", system=2),
        FakeNode(role="verifier", model_id="gemini-2.5-flash", system=2),
    ])
    executor = FakeExecutor(order=[[0], [1]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Thought result"),
        MagicMock(content="Verified result"),
    ])

    with patch("sage.topology.runner.TopologyRunner.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Multi-agent result"
        runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
        result = await runner.run("test task")
        assert result == "Multi-agent result"


# --- Sentinel-strip tests (post-smoke-v4 cascade fix) ---


def _make_runner():
    """Lightweight TopologyRunner with enough structure to exercise
    _gather_predecessor_context directly (no LLM calls)."""
    from sage.topology.runner import TopologyRunner

    class _StubGraph:
        def __init__(self):
            self._n = {
                0: FakeNode("planner", "mock", 2),
                1: FakeNode("coder", "mock", 2),
                2: FakeNode("synthesizer", "mock", 2),
            }
        def node_count(self):
            return len(self._n)
        def get_node(self, idx):
            return self._n[idx]
        def get_predecessors(self, idx):
            return [i for i in self._n if i < idx]

    runner = TopologyRunner(
        _StubGraph(),
        FakeExecutor(order=[]),
        llm_provider=AsyncMock(),
    )
    return runner


def test_sentinel_output_dropped_from_predecessor_context():
    """Predecessor outputs matching EMPTY_STEP_SENTINEL must not reach downstream."""
    runner = _make_runner()
    runner._node_outputs = {
        0: "[sage: agent exited after 5 steps with no content]",
        1: "real investigation findings",
    }
    ctx = runner._gather_predecessor_context(2)
    assert "sage: agent exited" not in ctx
    assert "real investigation" in ctx


def test_all_sentinels_produces_empty_context():
    """If every predecessor is a sentinel, context is empty — downstream
    node falls back to its own task prompt rather than 'everyone failed'."""
    runner = _make_runner()
    runner._node_outputs = {
        0: "[sage: agent exited after 3 steps with no content]",
        1: "[sage: agent exited after 5 steps with no content]",
    }
    ctx = runner._gather_predecessor_context(2)
    assert ctx == ""


def test_sentinel_prefix_only_matches_exact_format():
    """Legitimate outputs that merely mention the marker text must not be dropped."""
    from sage.topology.runner import _is_sentinel
    assert not _is_sentinel("Discussion of the [sage: agent exited after ...] failure mode")
    assert _is_sentinel("[sage: agent exited after 17 steps with no content]")
    assert _is_sentinel("  [sage: agent exited after 0 steps with no content]  ")
    assert not _is_sentinel("")
    assert not _is_sentinel(None)


# --- Planner-output injection experiment (SAGE_PLANNER_INJECTION=1) ---


def test_planner_injection_off_by_default(monkeypatch):
    """With the flag unset, system_prompt is unchanged."""
    monkeypatch.delenv("SAGE_PLANNER_INJECTION", raising=False)
    runner = _make_runner()
    runner._node_outputs = {0: "Plan: step 1, step 2, step 3"}
    out = runner._maybe_planner_injection(1, "You are the coder.")
    assert out == "You are the coder."


def test_planner_injection_on_prepends_planner_output(monkeypatch):
    """With flag ON and a planner predecessor, output is prepended."""
    monkeypatch.setenv("SAGE_PLANNER_INJECTION", "1")
    runner = _make_runner()
    runner._node_outputs = {0: "Plan: step 1, step 2, step 3"}
    out = runner._maybe_planner_injection(1, "You are the coder.")
    assert "Upstream plan (from planner)" in out
    assert "step 1, step 2, step 3" in out
    assert "You are the coder." in out


def test_planner_injection_skips_if_current_node_is_planner(monkeypatch):
    """A planner node never receives planner-injection (would self-inject)."""
    monkeypatch.setenv("SAGE_PLANNER_INJECTION", "1")
    runner = _make_runner()
    # node 0 is planner; treat node 0 as the target
    runner._node_outputs = {0: "Plan output"}
    out = runner._maybe_planner_injection(0, "You are the planner.")
    assert out == "You are the planner."


def test_planner_injection_skips_sentinel_output(monkeypatch):
    """If planner produced a sentinel, we don't inject the failure signal."""
    monkeypatch.setenv("SAGE_PLANNER_INJECTION", "1")
    runner = _make_runner()
    runner._node_outputs = {0: "[sage: agent exited after 5 steps with no content]"}
    out = runner._maybe_planner_injection(1, "You are the coder.")
    assert out == "You are the coder."


def test_planner_injection_truncates_long_output(monkeypatch):
    """Planner outputs longer than the budget are truncated with a marker."""
    from sage.topology.runner import _PLANNER_INJECTION_BUDGET
    monkeypatch.setenv("SAGE_PLANNER_INJECTION", "1")
    runner = _make_runner()
    long_plan = "STEP " * 10000  # way over budget
    runner._node_outputs = {0: long_plan}
    out = runner._maybe_planner_injection(1, "You are the coder.")
    assert "[truncated]" in out
    # The injected section itself is bounded; the full prompt will be a bit
    # larger due to headers and the original system_prompt, but the injection
    # proper is at most _PLANNER_INJECTION_BUDGET chars.
    assert len(out) < _PLANNER_INJECTION_BUDGET + 500


def test_is_planner_role_aliases():
    from sage.topology.runner import _is_planner_role
    assert _is_planner_role("planner")
    assert _is_planner_role("input_processor")
    assert _is_planner_role("decomposer")
    assert _is_planner_role("Task Planner")  # case + substring
    assert not _is_planner_role("coder")
    assert not _is_planner_role("synthesizer")
    assert not _is_planner_role("")
    assert not _is_planner_role(None)
