"""R4: code-node sandbox fail-closed contract tests."""
from __future__ import annotations

import builtins
import logging
import subprocess
import types
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest


class FakeNode:
    def __init__(
        self,
        role: str,
        model_id: str,
        system: int,
        required_capabilities: list[str] | None = None,
        node_type: str = "code",
        code_spec: str = "print('hello')",
    ) -> None:
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities or []
        self.node_type = node_type
        self.code_spec = code_spec
        self.prompt = ""


class FakeGraph:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]

    def get_predecessors(self, idx: int) -> list[int]:
        return list(range(idx))


class FakeExecutor:
    def __init__(self, ready_sequence: list[list[int]]) -> None:
        self._batches = list(ready_sequence)
        self._batch_idx = 0

    def next_ready(self, graph: FakeGraph) -> list[int]:
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        pass

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


def _make_runner() -> Any:
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph([FakeNode(role="coder", model_id="x", system=1)])
    executor = FakeExecutor([[0]])
    return TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=None,
    )


def _force_runner_sandbox_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    import_ok: bool,
    bwrap_available: bool,
    import_error: ImportError | None,
) -> None:
    import sage.topology.runner as runner_mod

    monkeypatch.setattr(runner_mod, "_SANDBOX_IMPORT_OK", import_ok, raising=False)
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMPORT_ERR", import_error, raising=False)
    monkeypatch.setattr(runner_mod, "BWRAP_AVAILABLE", bwrap_available, raising=False)


def _patch_isolated_executor_import(
    monkeypatch: pytest.MonkeyPatch,
    module: types.ModuleType | None,
    exc: ImportError | None = None,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "sage.sandbox.isolated_executor":
            if exc is not None:
                raise exc
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.asyncio
async def test_code_node_fails_closed_when_sandbox_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No unsafe env + no isolation backend raises and never calls subprocess."""
    from sage.sandbox.errors import SandboxUnavailable

    monkeypatch.delenv("SAGE_UNSAFE_RAW_EXEC", raising=False)

    _force_runner_sandbox_state(
        monkeypatch,
        import_ok=False,
        bwrap_available=False,
        import_error=ImportError("isolated executor intentionally unavailable"),
    )
    with patch("subprocess.run") as mock_run:
        with pytest.raises(SandboxUnavailable):
            await _make_runner().run("any task")
        mock_run.assert_not_called()

    _force_runner_sandbox_state(
        monkeypatch,
        import_ok=True,
        bwrap_available=False,
        import_error=None,
    )
    with patch("subprocess.run") as mock_run:
        with pytest.raises(SandboxUnavailable):
            await _make_runner().run("any task")
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_code_node_subprocess_requires_explicit_unsafe_env(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SAGE_UNSAFE_RAW_EXEC=1 + no bwrap is the only raw subprocess fallback."""
    monkeypatch.setenv("SAGE_UNSAFE_RAW_EXEC", "1")
    _force_runner_sandbox_state(
        monkeypatch,
        import_ok=False,
        bwrap_available=False,
        import_error=ImportError("isolated executor intentionally unavailable"),
    )

    fake_proc = MagicMock(stdout="raw output", stderr="", returncode=0)
    with patch("subprocess.run", return_value=fake_proc) as mock_run:
        with caplog.at_level(logging.WARNING, logger="sage.topology.runner"):
            output = await _make_runner().run("any task")

    assert output == "raw output"
    mock_run.assert_called_once()
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("SAGE_UNSAFE_RAW_EXEC=1" in message for message in warnings)
    assert any("DO NOT USE IN PRODUCTION" in message for message in warnings)


@pytest.mark.asyncio
async def test_toolforge_gate2_stays_closed_without_toolforge_subprocess_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAGE_UNSAFE_RAW_EXEC must not unlock ToolForge Gate 2."""
    from sage.tools.forge import ToolForge

    _patch_isolated_executor_import(
        monkeypatch,
        module=None,
        exc=ImportError("isolated executor intentionally unavailable"),
    )
    monkeypatch.setenv("SAGE_UNSAFE_RAW_EXEC", "1")
    monkeypatch.delenv("SAGE_UNSAFE_TOOLFORGE_SUBPROCESS", raising=False)
    subprocess_run = Mock(side_effect=AssertionError("subprocess fallback forbidden"))
    monkeypatch.setattr(subprocess, "run", subprocess_run)

    ok, error = await ToolForge._run_tests("x = 1", "assert x == 1")

    assert ok is False
    assert "SAGE_UNSAFE_TOOLFORGE_SUBPROCESS" in error
    assert "plain subprocess" in error
    subprocess_run.assert_not_called()
