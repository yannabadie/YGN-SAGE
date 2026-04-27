from __future__ import annotations

import builtins
import logging
import subprocess
import types
from typing import Any
from unittest.mock import Mock

import pytest


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
async def test_run_tests_fails_closed_when_isolated_executor_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.tools.forge import ToolForge

    _patch_isolated_executor_import(
        monkeypatch,
        module=None,
        exc=ImportError("isolated executor intentionally unavailable"),
    )
    monkeypatch.delenv("SAGE_UNSAFE_TOOLFORGE_SUBPROCESS", raising=False)
    subprocess_run = Mock(side_effect=AssertionError("subprocess fallback forbidden"))
    monkeypatch.setattr(subprocess, "run", subprocess_run)

    ok, error = await ToolForge._run_tests("x = 1", "assert x == 1")

    assert ok is False
    assert "SAGE_UNSAFE_TOOLFORGE_SUBPROCESS" in error
    assert "plain subprocess" in error
    subprocess_run.assert_not_called()


@pytest.mark.asyncio
async def test_run_tests_unsafe_env_unlocks_subprocess_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sage.tools.forge import ToolForge

    _patch_isolated_executor_import(
        monkeypatch,
        module=None,
        exc=ImportError("isolated executor intentionally unavailable"),
    )
    monkeypatch.setenv("SAGE_UNSAFE_TOOLFORGE_SUBPROCESS", "1")
    real_run = subprocess.run
    subprocess_run = Mock(wraps=real_run)
    monkeypatch.setattr(subprocess, "run", subprocess_run)

    with caplog.at_level(logging.WARNING, logger="sage.tools.forge"):
        ok, error = await ToolForge._run_tests("x = 1", "assert x == 1")

    assert (ok, error) == (True, "")
    subprocess_run.assert_called_once()
    assert any(
        "SAGE_UNSAFE_TOOLFORGE_SUBPROCESS=1" in record.message
        and "plain Python subprocess" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_run_tests_refuses_isolated_executor_runtime_failure_without_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.tools.forge import ToolForge

    isolated_executor = types.ModuleType("sage.sandbox.isolated_executor")

    def execute_isolated(code: str, timeout: int) -> tuple[str, str, int]:
        raise RuntimeError("bwrap runner crashed")

    isolated_executor.execute_isolated = execute_isolated
    _patch_isolated_executor_import(monkeypatch, module=isolated_executor)
    monkeypatch.setenv("SAGE_UNSAFE_TOOLFORGE_SUBPROCESS", "1")
    subprocess_run = Mock(side_effect=AssertionError("subprocess fallback forbidden"))
    monkeypatch.setattr(subprocess, "run", subprocess_run)

    ok, error = await ToolForge._run_tests("x = 1", "assert x == 1")

    assert ok is False
    assert "Gate 2: isolated executor failed" in error
    assert "refusing to downgrade" in error
    subprocess_run.assert_not_called()


@pytest.mark.asyncio
async def test_run_tests_no_tests_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.tools.forge import ToolForge

    _patch_isolated_executor_import(
        monkeypatch,
        module=None,
        exc=ImportError("isolated executor should not be imported"),
    )
    subprocess_run = Mock(side_effect=AssertionError("subprocess fallback forbidden"))
    monkeypatch.setattr(subprocess, "run", subprocess_run)

    assert await ToolForge._run_tests("x = 1", "") == (True, "")
    subprocess_run.assert_not_called()
