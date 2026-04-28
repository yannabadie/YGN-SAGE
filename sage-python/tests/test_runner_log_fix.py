"""Tests for TopologyRunner code node execution (Axis 1: _log fix)."""
from unittest.mock import AsyncMock, MagicMock

import pytest
import sage.topology.runner as runner_mod

from sage.topology.runner import TopologyRunner


def _make_runner(graph=None, executor=None):
    """Create a minimal TopologyRunner for testing."""
    if graph is None:
        graph = MagicMock()
    if executor is None:
        executor = MagicMock()
    llm = AsyncMock()
    return TopologyRunner(graph=graph, executor=executor, llm_provider=llm)


def _patch_isolated_executor(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    stderr: str,
    exit_code: int,
) -> None:
    """Pretend the code-node isolation backend is available for log tests."""

    def fake_execute_isolated(code: str, timeout: int = 30) -> tuple[str, str, int]:
        return stdout, stderr, exit_code

    monkeypatch.setattr(runner_mod, "_SANDBOX_IMPORT_OK", True)
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMPORT_ERR", None)
    monkeypatch.setattr(runner_mod, "BWRAP_AVAILABLE", True)
    monkeypatch.setattr(runner_mod, "execute_isolated", fake_execute_isolated)


@pytest.mark.asyncio
async def test_code_node_no_code_spec_logs_error():
    """_execute_code_node with empty code_spec should log error, not raise NameError."""
    node = MagicMock()
    node.role = "validator"
    node.code_spec = ""
    node.prompt = ""
    node.node_type = "code"

    graph = MagicMock()
    graph.get_node.return_value = node

    runner = _make_runner(graph=graph)
    result = await runner._execute_code_node(0, "test task")
    assert "ERROR" in result or "no code_spec" in result


@pytest.mark.asyncio
async def test_code_node_success_logs_info(monkeypatch: pytest.MonkeyPatch):
    """Successful code node execution should not raise NameError."""
    node = MagicMock()
    node.role = "compute"
    node.code_spec = "print('hello')"
    node.prompt = ""
    node.node_type = "code"

    graph = MagicMock()
    graph.get_node.return_value = node
    graph.get_predecessors.return_value = []

    _patch_isolated_executor(monkeypatch, stdout="hello\n", stderr="", exit_code=0)
    runner = _make_runner(graph=graph)
    result = await runner._execute_code_node(0, "test task")
    # Should complete without NameError (the bug was _log.info crashing)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_code_node_failure_logs_warning(monkeypatch: pytest.MonkeyPatch):
    """Failed code node should log warning, not raise NameError."""
    node = MagicMock()
    node.role = "validator"
    node.code_spec = "raise ValueError('test error')"
    node.prompt = ""
    node.node_type = "code"

    graph = MagicMock()
    graph.get_node.return_value = node
    graph.get_predecessors.return_value = []

    _patch_isolated_executor(
        monkeypatch,
        stdout="",
        stderr="ValueError: test error",
        exit_code=1,
    )
    runner = _make_runner(graph=graph)
    result = await runner._execute_code_node(0, "test task")
    # Should complete without NameError (the bug was _log.warning crashing)
    assert isinstance(result, str)
