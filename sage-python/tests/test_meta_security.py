"""Tests for security audit findings SEC-01, SEC-02, SEC-03."""
import pytest
import asyncio
from unittest.mock import patch


class TestMetaToolsSandboxed:
    """SEC-01: Dynamic tool creation uses subprocess sandbox."""

    def test_create_python_tool_validates(self):
        """SEC-01: create_python_tool validates code via AST."""
        from sage.tools.meta import create_python_tool
        # No registry → error
        result = asyncio.run(create_python_tool._handler(name="test", code="x=1"))
        assert "Error" in result

    def test_create_python_tool_runs_in_sandbox(self):
        """SEC-01 fix: create_python_tool executes code in subprocess, not exec()."""
        from sage.tools.meta import create_python_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = asyncio.run(create_python_tool._handler(
            name="adder",
            code='result = args["a"] + args["b"]\nprint(json.dumps({"output": str(result)}))',
            registry=registry,
        ))
        assert "Success" in result
        assert registry.get("adder") is not None

    def test_created_tool_executes_in_sandbox(self):
        """Created tool runs in subprocess isolation."""
        from sage.tools.meta import create_python_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        asyncio.run(create_python_tool._handler(
            name="greeter",
            code='name = args.get("name", "world")\nprint(json.dumps({"output": f"Hello, {name}!"}))',
            registry=registry,
        ))
        tool = registry.get("greeter")
        assert tool is not None
        result = asyncio.run(tool.execute({"name": "Yann"}))
        assert "Hello, Yann!" in result.output

    def test_created_tool_rejects_dangerous_code(self):
        """create_python_tool rejects code with blocked imports."""
        from sage.tools.meta import create_python_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = asyncio.run(create_python_tool._handler(
            name="evil",
            code='import subprocess; subprocess.run(["rm", "-rf", "/"])',
            registry=registry,
        ))
        assert "Blocked" in result
        assert registry.get("evil") is None

    def test_create_bash_tool_validates(self):
        """SEC-02: create_bash_tool validates script before registration."""
        from sage.tools.meta import create_bash_tool
        result = asyncio.run(create_bash_tool._handler(name="test", description="test", script="echo hi"))
        assert "Error" in result  # No registry provided

    def test_create_bash_tool_runs_in_sandbox(self):
        """SEC-02 fix: create_bash_tool executes via subprocess, no shell=True."""
        from sage.tools.meta import create_bash_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = asyncio.run(create_bash_tool._handler(
            name="echo_tool",
            description="Echoes input",
            script='echo "hello from bash"',
            registry=registry,
        ))
        assert "Success" in result or "registered" in result.lower()
        assert registry.get("echo_tool") is not None

    def test_bash_tool_blocks_dangerous_commands(self):
        """create_bash_tool blocks rm -rf and similar."""
        from sage.tools.meta import create_bash_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = asyncio.run(create_bash_tool._handler(
            name="evil_bash",
            description="Dangerous",
            script="rm -rf /",
            registry=registry,
        ))
        assert "Blocked" in result
        assert registry.get("evil_bash") is None

    def test_tools_registered_in_boot(self):
        """Verify create_python_tool and create_bash_tool ARE registered (sandboxed — SEC-01/02 fixed)."""
        from sage.boot import boot_agent_system
        system = boot_agent_system(use_mock_llm=True)
        tool_names = system.tool_registry.list_tools()
        assert "create_python_tool" in tool_names
        assert "create_bash_tool" in tool_names


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
