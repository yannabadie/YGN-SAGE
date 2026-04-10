"""Tests for TopologyRunner dispatching LLM nodes to agent_loop.

Phase 2: each LLM topology node calls agent_loop.run() instead of
provider.generate(), gaining tools + validation + guardrails.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_graph(n_nodes=2):
    """Create a mock topology graph."""
    graph = MagicMock()
    graph.node_count.return_value = n_nodes
    nodes = []
    for i in range(n_nodes):
        node = MagicMock()
        node.role = "actor" if i == 0 else "verifier"
        node.model_id = ""
        node.prompt = ""
        node.node_type = "llm"
        node.required_capabilities = []
        nodes.append(node)
    graph.get_node = lambda idx: nodes[idx]
    graph.get_predecessors = lambda idx: list(range(idx))
    return graph


def _make_executor(ready_sequence):
    """Create a mock executor that yields nodes in sequence."""
    executor = MagicMock()
    call_count = [0]
    def _next_ready(graph):
        if call_count[0] < len(ready_sequence):
            result = ready_sequence[call_count[0]]
            call_count[0] += 1
            return result
        return []
    executor.next_ready = _next_ready
    executor.is_done = lambda: call_count[0] >= len(ready_sequence)
    return executor


@pytest.mark.asyncio
async def test_llm_node_uses_agent_loop_when_factory_set():
    """LLM nodes should call agent_loop.run() when factory is provided."""
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="agent result")
    mock_loop.total_cost_usd = 0.0

    factory = MagicMock(return_value=mock_loop)

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    result = await runner.run("test task")

    factory.assert_called_once()
    mock_loop.run.assert_called_once()
    assert "agent result" in result


@pytest.mark.asyncio
async def test_code_node_skips_agent_loop():
    """Code nodes should NOT use agent_loop, even when factory is set."""
    factory = MagicMock()

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    node = graph.get_node(0)
    node.node_type = "code"
    node.code_spec = "print('hello')"
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    with patch.object(runner, '_execute_code_node', new_callable=AsyncMock, return_value="hello"):
        await runner.run("task")

    factory.assert_not_called()


@pytest.mark.asyncio
async def test_predecessor_context_in_task():
    """H7: predecessor output should be injected in the task passed to agent_loop."""
    captured_tasks = []

    async def _capture_run(task):
        captured_tasks.append(task)
        return "result"

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(2)
    executor = _make_executor([[0], [1]])

    # Two loops: first produces output, second captures what it receives
    loop_0 = MagicMock()
    loop_0.run = AsyncMock(return_value="first node output")
    loop_0.total_cost_usd = 0.0

    loop_1 = MagicMock()
    loop_1.run = _capture_run
    loop_1.total_cost_usd = 0.0

    factory = MagicMock(side_effect=[loop_0, loop_1])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    await runner.run("original task")

    # Second node should receive predecessor context + original task
    assert len(captured_tasks) == 1
    assert "first node output" in captured_tasks[0]
    assert "original task" in captured_tasks[0]


@pytest.mark.asyncio
async def test_no_factory_uses_provider_directly():
    """Without factory, runner should use existing provider.generate() path."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "provider result"
    mock_provider.generate = AsyncMock(return_value=mock_response)

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=mock_provider,
        # No agent_loop_factory
    )

    result = await runner.run("task")
    mock_provider.generate.assert_called()
    assert "provider result" in result


@pytest.mark.asyncio
async def test_factory_receives_node_role():
    """Factory should receive the node role for tool filtering."""
    factory = MagicMock()
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="result")
    mock_loop.total_cost_usd = 0.0
    factory.return_value = mock_loop

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    graph.get_node(0).role = "verifier"
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    await runner.run("task")

    call_kwargs = factory.call_args.kwargs
    assert call_kwargs.get("node_role") == "verifier"
