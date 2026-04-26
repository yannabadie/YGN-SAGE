from __future__ import annotations

import sys
import types

import pytest


def _install_sage_core_without_tool_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sage_core", types.ModuleType("sage_core"))


def test_toolforge_validate_ast_fails_closed_when_rust_validator_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.tools.forge import ToolForge

    _install_sage_core_without_tool_executor(monkeypatch)
    monkeypatch.delenv("SAGE_TOOLFORGE_STRICT", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        ToolForge._validate_ast("print('x')")

    assert "SAGE_TOOLFORGE_STRICT=1" in str(exc_info.value)


def test_toolforge_validate_ast_strict_zero_allows_ast_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sage.tools.forge as forge_module
    from sage.tools.forge import ToolForge

    _install_sage_core_without_tool_executor(monkeypatch)
    monkeypatch.setenv("SAGE_TOOLFORGE_STRICT", "0")
    monkeypatch.setattr(forge_module, "_AST_FALLBACK_WARNED", False, raising=False)

    assert ToolForge._validate_ast("print('x')") == (True, "")
