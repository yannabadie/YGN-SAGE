"""Integration tests for dynamic tool creation pipeline.

Full flow: create tool -> register -> execute -> get result.
No mocks. Real subprocess execution.
"""
from __future__ import annotations

import asyncio
import platform
import shutil

import pytest

from sage.tools.meta import create_python_tool, create_bash_tool
from sage.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine synchronously (works regardless of existing loop)."""
    return asyncio.run(coro)


def _create_py(name: str, code: str, registry: ToolRegistry) -> str:
    """Shorthand: invoke create_python_tool handler directly."""
    return _run(create_python_tool._handler(name=name, code=code, registry=registry))


def _create_bash(name: str, description: str, script: str, registry: ToolRegistry) -> str:
    """Shorthand: invoke create_bash_tool handler directly."""
    return _run(create_bash_tool._handler(
        name=name, description=description, script=script, registry=registry,
    ))


# ── Python tool tests ───────────────────────────────────────────

class TestPythonToolCreation:

    def test_create_and_run_math_tool(self):
        """Create a math tool and verify it produces correct output."""
        registry = ToolRegistry()
        code = (
            'import math\n'
            'x = float(args.get("x", 0))\n'
            'result = math.factorial(int(x))\n'
            'print(json.dumps({"output": str(result)}))'
        )
        result = _create_py("factorial", code, registry)
        assert "Success" in result

        tool = registry.get("factorial")
        assert tool is not None

        exec_result = _run(tool.execute({"x": 5}))
        assert "120" in exec_result.output
        assert not exec_result.is_error

    def test_create_tool_with_error_handling(self):
        """Tool that raises returns error output, not crash."""
        registry = ToolRegistry()
        code = 'raise ValueError("intentional error")'
        msg = _create_py("error_tool", code, registry)
        assert "Success" in msg

        tool = registry.get("error_tool")
        exec_result = _run(tool.execute({}))
        # Subprocess exits non-zero -> handler returns "Error (...)" string
        assert "error" in exec_result.output.lower() or exec_result.is_error

    def test_reject_os_import(self):
        """Tool with os import is blocked at validation."""
        registry = ToolRegistry()
        result = _create_py("bad", "import os\nos.listdir('/')", registry)
        assert "Blocked" in result
        assert registry.get("bad") is None

    def test_reject_eval(self):
        """Tool with eval() is blocked at validation."""
        registry = ToolRegistry()
        result = _create_py("bad2", 'eval("1+1")', registry)
        assert "Blocked" in result

    def test_reject_exec(self):
        """Tool with exec() is blocked at validation."""
        registry = ToolRegistry()
        result = _create_py("bad3", 'exec("x=1")', registry)
        assert "Blocked" in result

    def test_reject_subprocess_import(self):
        """Tool with subprocess import is blocked at validation."""
        registry = ToolRegistry()
        result = _create_py("bad4", "import subprocess", registry)
        assert "Blocked" in result

    def test_multiple_tools_coexist(self):
        """Multiple dynamic tools can be registered and used."""
        registry = ToolRegistry()
        for i in range(3):
            code = f'print(json.dumps({{"output": "tool_{i}"}}))'
            result = _create_py(f"tool_{i}", code, registry)
            assert "Success" in result

        matching = [n for n in registry.list_tools() if n.startswith("tool_")]
        assert len(matching) == 3

    def test_tool_receives_args(self):
        """Verify that the args dict is correctly passed through stdin."""
        registry = ToolRegistry()
        code = (
            'name = args.get("name", "world")\n'
            'print(json.dumps({"output": f"hello {name}"}))'
        )
        _create_py("greet", code, registry)
        tool = registry.get("greet")

        exec_result = _run(tool.execute({"name": "sage"}))
        assert "hello sage" in exec_result.output
        assert not exec_result.is_error

    def test_no_registry_returns_error(self):
        """Calling without registry returns an error string."""
        result = _run(create_python_tool._handler(name="x", code="pass"))
        assert "Error" in result


# ── Bash tool tests ──────────────────────────────────────────────

_HAS_BASH = shutil.which("bash") is not None


@pytest.mark.skipif(not _HAS_BASH, reason="bash not available")
class TestBashToolCreation:

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Bash tool handler uses /bin/bash which is not available on Windows",
    )
    def test_create_and_run_echo(self):
        """Create a bash echo tool and verify output."""
        registry = ToolRegistry()
        result = _create_bash(
            "echo_test", "Echoes a message", 'echo "sandbox works"', registry,
        )
        assert "Success" in result

        tool = registry.get("echo_test")
        assert tool is not None

        exec_result = _run(tool.execute({}))
        assert "sandbox works" in exec_result.output

    def test_reject_rm_rf(self):
        """Bash tool blocks rm -rf."""
        registry = ToolRegistry()
        result = _create_bash("rm_tool", "Bad tool", "rm -rf /", registry)
        assert "Blocked" in result
        assert registry.get("rm_tool") is None

    def test_reject_mkfs(self):
        """Bash tool blocks mkfs."""
        registry = ToolRegistry()
        result = _create_bash("mkfs_tool", "Bad tool", "mkfs /dev/sda", registry)
        assert "Blocked" in result

    def test_reject_dd(self):
        """Bash tool blocks dd if=."""
        registry = ToolRegistry()
        result = _create_bash(
            "dd_tool", "Bad tool", "dd if=/dev/zero of=/dev/sda", registry,
        )
        assert "Blocked" in result

    def test_safe_script_allowed(self):
        """Safe bash scripts are registered normally."""
        registry = ToolRegistry()
        result = _create_bash(
            "date_tool", "Shows date", "date +%Y-%m-%d", registry,
        )
        assert "Success" in result
        assert registry.get("date_tool") is not None

    def test_no_registry_returns_error(self):
        """Calling without registry returns an error string."""
        result = _run(create_bash_tool._handler(
            name="x", description="x", script="echo hi",
        ))
        assert "Error" in result
