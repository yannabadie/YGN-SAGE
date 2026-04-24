from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _install_tool_executor(monkeypatch: pytest.MonkeyPatch, executor: Mock) -> Mock:
    sage_core = types.ModuleType("sage_core")
    tool_executor = Mock(return_value=executor)
    sage_core.ToolExecutor = tool_executor
    monkeypatch.setitem(sys.modules, "sage_core", sage_core)
    return tool_executor


def test_rust_validator_success_ignores_strict(monkeypatch):
    from sage.tools.forge import ToolForge

    monkeypatch.setenv("SAGE_TOOLFORGE_STRICT", "1")
    executor = Mock()
    executor.validate.return_value = SimpleNamespace(valid=True, errors=[])
    tool_executor = _install_tool_executor(monkeypatch, executor)

    ok, err = ToolForge._validate_ast("x = 1 + 2")

    assert ok is True
    assert err == ""
    tool_executor.assert_called_once_with()
    executor.validate.assert_called_once_with("x = 1 + 2")


def test_rust_validator_error_strict_default_raises(monkeypatch):
    from sage.tools.forge import ToolForge

    monkeypatch.delenv("SAGE_TOOLFORGE_STRICT", raising=False)
    executor = Mock()
    executor.validate.side_effect = RuntimeError("validator unavailable")
    _install_tool_executor(monkeypatch, executor)

    with pytest.raises(RuntimeError, match="SAGE_TOOLFORGE_STRICT"):
        ToolForge._validate_ast("x = 1")


def test_rust_validator_error_strict_zero_warns_once_and_falls_back(
    monkeypatch,
    caplog,
):
    import sage.tools.forge as forge_module
    from sage.tools.forge import ToolForge

    monkeypatch.setenv("SAGE_TOOLFORGE_STRICT", "0")
    monkeypatch.setattr(forge_module, "_AST_FALLBACK_WARNED", False, raising=False)
    executor = Mock()
    executor.validate.side_effect = RuntimeError("validator unavailable")
    _install_tool_executor(monkeypatch, executor)

    with caplog.at_level(logging.WARNING, logger="sage.tools.forge"):
        assert ToolForge._validate_ast("x = 1") == (True, "")
        assert ToolForge._validate_ast("y = 2") == (True, "")

    fallback_warnings = [
        record
        for record in caplog.records
        if "SAGE_TOOLFORGE_STRICT=0" in record.message
        and "ast.parse-only validation" in record.message
    ]
    assert len(fallback_warnings) == 1


def test_rust_validator_error_strict_zero_propagates_ast_syntax_error(
    monkeypatch,
):
    import sage.tools.forge as forge_module
    from sage.tools.forge import ToolForge

    monkeypatch.setenv("SAGE_TOOLFORGE_STRICT", "0")
    monkeypatch.setattr(forge_module, "_AST_FALLBACK_WARNED", False, raising=False)
    executor = Mock()
    executor.validate.side_effect = RuntimeError("validator unavailable")
    _install_tool_executor(monkeypatch, executor)

    with patch("sage.tools.forge.ast.parse", side_effect=SyntaxError("bad syntax")):
        with pytest.raises(SyntaxError, match="bad syntax"):
            ToolForge._validate_ast("def broken(:\n    pass")
