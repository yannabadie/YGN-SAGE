"""Tests for security audit findings SEC-01, SEC-02, SEC-03."""
import pytest
import asyncio
from unittest.mock import patch


class TestMetaToolsDisabled:
    """SEC-01/SEC-02: Dynamic tool creation must be disabled."""

    def test_create_python_tool_returns_disabled(self):
        from sage.tools.meta import create_python_tool
        # The tool handler is the _handler attribute of the Tool object
        result = asyncio.run(create_python_tool._handler(name="test", code="x=1"))
        assert "DISABLED" in result

    def test_create_bash_tool_returns_disabled(self):
        from sage.tools.meta import create_bash_tool
        result = asyncio.run(create_bash_tool._handler(name="test", description="test tool", script="echo hi"))
        assert "DISABLED" in result

    def test_tools_not_registered_in_boot(self):
        """Verify create_python_tool and create_bash_tool are NOT in the default tool registry."""
        from sage.boot import boot_agent_system
        system = boot_agent_system(use_mock_llm=True)
        tool_names = system.tool_registry.list_tools()
        assert "create_python_tool" not in tool_names
        assert "create_bash_tool" not in tool_names


class TestRunBashSecurity:
    """SEC-03: run_bash must use create_subprocess_exec and block destructive commands."""

    def test_destructive_command_blocked(self):
        from sage.tools.builtin import _run_bash
        result = asyncio.run(_run_bash("rm -rf /"))
        assert "BLOCKED" in result

    def test_fork_bomb_blocked(self):
        from sage.tools.builtin import _run_bash
        result = asyncio.run(_run_bash(":(){ :|:& };:"))
        assert "BLOCKED" in result
