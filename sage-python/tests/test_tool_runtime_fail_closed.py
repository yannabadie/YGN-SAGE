from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import tomllib
from types import SimpleNamespace
from typing import Any

import pytest

from sage.tools.runtime_safety import (
    UNSAFE_PY_SUBPROCESS_ENV,
    load_tool_executor_or_raise,
)


class RecordingRegistry:
    def __init__(self) -> None:
        self.registered: list[Any] = []

    def register(self, tool: Any) -> None:
        self.registered.append(tool)


def _block_sage_core_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "sage_core":
            raise ImportError("sage_core intentionally unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def test_pyproject_declares_sage_core_runtime_dependency() -> None:
    pyproject = pathlib.Path(__file__).parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert any(
        dep.split(";")[0].strip().lower().replace("_", "-").startswith("sage-core")
        for dep in deps
    )


def test_load_tool_executor_fails_closed_when_sage_core_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_sage_core_import(monkeypatch)
    monkeypatch.delenv(UNSAFE_PY_SUBPROCESS_ENV, raising=False)

    with pytest.raises(ImportError) as exc_info:
        load_tool_executor_or_raise()

    message = str(exc_info.value)
    assert "sage_core.ToolExecutor" in message
    assert "disabled by default" in message
    assert "SAGE_UNSAFE_PY_SUBPROCESS=1" in message


@pytest.mark.asyncio
async def test_create_python_tool_fails_closed_when_sage_core_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.tools.meta import create_python_tool

    _block_sage_core_import(monkeypatch)
    monkeypatch.delenv(UNSAFE_PY_SUBPROCESS_ENV, raising=False)
    registry = RecordingRegistry()

    result = await create_python_tool.execute(
        {"name": "blocked_tool", "code": "print('x')", "registry": registry}
    )

    assert result.is_error is True
    assert "sage_core.ToolExecutor" in result.output
    assert registry.registered == []


@pytest.mark.asyncio
async def test_unsafe_py_subprocess_env_unlocks_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sage.tools.meta as meta_module
    from sage.tools.meta import create_python_tool

    _block_sage_core_import(monkeypatch)
    monkeypatch.setenv(UNSAFE_PY_SUBPROCESS_ENV, "1")
    code = 'import json\nprint(json.dumps({"output": args.get("value", "ok")}))'

    from sage.tools.sandbox_executor import SandboxResult

    async def fake_execute_python_in_sandbox(
        saved_code: str,
        args: dict[str, Any],
    ) -> SandboxResult:
        assert saved_code == code
        return SandboxResult(
            stdout=json.dumps({"output": args.get("value", "ok")}),
            stderr="",
            exit_code=0,
        )

    monkeypatch.setattr(
        meta_module,
        "execute_python_in_sandbox",
        fake_execute_python_in_sandbox,
    )
    tools_workspace = pathlib.Path(__file__).parent / ".tmp_tool_runtime"
    shutil.rmtree(tools_workspace, ignore_errors=True)
    tools_workspace.mkdir()
    monkeypatch.setattr(meta_module, "TOOLS_WORKSPACE", str(tools_workspace))
    registry = RecordingRegistry()

    try:
        result = await create_python_tool._handler(
            name="unsafe_fallback_tool",
            code=code,
            registry=registry,
        )

        assert result.startswith("Success:")
        assert len(registry.registered) == 1

        execution = await registry.registered[0].execute({"value": "from subprocess"})
        assert execution.is_error is False
        assert execution.output == "from subprocess"
    finally:
        shutil.rmtree(tools_workspace, ignore_errors=True)


@pytest.mark.asyncio
async def test_create_python_tool_refuses_rust_validator_failure_without_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.tools.meta import create_python_tool

    class CrashingToolExecutor:
        def validate(self, code: str) -> SimpleNamespace:
            raise RuntimeError("validator crashed")

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "sage_core":
            return SimpleNamespace(ToolExecutor=CrashingToolExecutor)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setenv(UNSAFE_PY_SUBPROCESS_ENV, "1")
    registry = RecordingRegistry()

    result = await create_python_tool.execute(
        {"name": "bad_rust_validator", "code": "print('x')", "registry": registry}
    )

    assert result.is_error is True
    assert "refusing to downgrade" in result.output
    assert registry.registered == []
