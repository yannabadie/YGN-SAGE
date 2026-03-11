"""Tests for sandbox_executor: AST validation + subprocess execution."""
from __future__ import annotations

import pytest

from sage.tools.sandbox_executor import (
    SandboxResult,
    execute_python_in_sandbox,
    validate_tool_code,
)


# ── Validation tests ─────────────────────────────────────────────


class TestValidateToolCode:
    def test_validate_clean_code(self):
        """import json, math should pass validation."""
        code = "import json\nimport math\nresult = json.dumps({'x': math.pi})\nprint(result)"
        errors = validate_tool_code(code)
        assert errors == []

    def test_validate_blocks_dangerous_imports(self):
        """subprocess, shutil, ctypes, from os import system should all be blocked."""
        for code in [
            "import subprocess",
            "import shutil",
            "import ctypes",
            "from os import system",
        ]:
            errors = validate_tool_code(code)
            assert len(errors) > 0, f"Expected block for: {code}"

    def test_validate_blocks_exec(self):
        """exec('x=1') should be blocked."""
        errors = validate_tool_code("exec('x=1')")
        assert len(errors) > 0
        assert any("exec" in e for e in errors)

    def test_validate_blocks_eval(self):
        """eval('1+1') should be blocked."""
        errors = validate_tool_code("eval('1+1')")
        assert len(errors) > 0
        assert any("eval" in e for e in errors)

    def test_validate_blocks_open(self):
        """open('file') should be blocked."""
        errors = validate_tool_code("open('file.txt')")
        assert len(errors) > 0
        assert any("open" in e for e in errors)

    def test_validate_blocks_dunder_import(self):
        """__import__('os') should be blocked."""
        errors = validate_tool_code("__import__('os')")
        assert len(errors) > 0

    def test_validate_syntax_error(self):
        """Malformed code returns SyntaxError."""
        errors = validate_tool_code("def f(\n")
        assert len(errors) == 1
        assert errors[0].startswith("SyntaxError")

    def test_validate_import_os(self):
        """import os should be blocked."""
        errors = validate_tool_code("import os")
        assert len(errors) > 0
        assert any("os" in e for e in errors)

    def test_validate_blocks_from_builtins(self):
        """from builtins import ... should be blocked."""
        errors = validate_tool_code("from builtins import open")
        assert len(errors) > 0

    def test_validate_blocks_breakpoint(self):
        """breakpoint() should be blocked."""
        errors = validate_tool_code("breakpoint()")
        assert len(errors) > 0

    def test_validate_blocks_compile(self):
        """compile() should be blocked."""
        errors = validate_tool_code("compile('x=1', '<string>', 'exec')")
        assert len(errors) > 0

    def test_validate_method_call_form(self):
        """os.system('cmd') — method call form should be blocked."""
        # Even if os isn't imported, the call name 'system' itself isn't blocked;
        # but 'exec' as a method call should be caught.
        errors = validate_tool_code("obj.exec()")
        assert len(errors) > 0


# ── Bypass regression tests (Audit3 F-01) ────────────────────────


class TestSandboxBypassVectors:
    """Regression tests for Audit3 F-01 bypass vectors."""

    def test_getattr_blocked(self):
        errs = validate_tool_code("getattr(object, '__subclasses__')")
        assert any("getattr" in e for e in errs)

    def test_setattr_blocked(self):
        errs = validate_tool_code("setattr(obj, 'x', 1)")
        assert any("setattr" in e for e in errs)

    def test_delattr_blocked(self):
        errs = validate_tool_code("delattr(obj, 'x')")
        assert any("delattr" in e for e in errs)

    def test_globals_blocked(self):
        errs = validate_tool_code("globals()")
        assert any("globals" in e for e in errs)

    def test_locals_blocked(self):
        errs = validate_tool_code("locals()")
        assert any("locals" in e for e in errs)

    def test_vars_blocked(self):
        errs = validate_tool_code("vars()")
        assert any("vars" in e for e in errs)

    def test_dir_blocked(self):
        errs = validate_tool_code("dir()")
        assert any("dir" in e for e in errs)

    def test_chr_blocked(self):
        errs = validate_tool_code("chr(101)+chr(118)+chr(97)+chr(108)")
        assert any("chr" in e for e in errs)

    def test_type_blocked(self):
        errs = validate_tool_code("type(compile)")
        assert any("type" in e for e in errs)

    def test_hasattr_blocked(self):
        errs = validate_tool_code("hasattr(obj, 'x')")
        assert any("hasattr" in e for e in errs)

    def test_dunder_class_attribute(self):
        errs = validate_tool_code("x = ().__class__")
        assert any("__class__" in e for e in errs)

    def test_dunder_mro_attribute(self):
        errs = validate_tool_code("x = ().__class__.__mro__")
        assert any("__mro__" in e for e in errs)

    def test_dunder_subclasses(self):
        errs = validate_tool_code(
            "[x for x in ().__class__.__mro__[-1].__subclasses__()]"
        )
        assert len(errs) > 0

    def test_dunder_globals_attr(self):
        errs = validate_tool_code("s.__init__.__globals__")
        assert any("__globals__" in e for e in errs)

    def test_dunder_builtins_attr(self):
        errs = validate_tool_code("s.__builtins__")
        assert any("__builtins__" in e for e in errs)

    def test_dunder_dict_attr(self):
        errs = validate_tool_code("obj.__dict__")
        assert any("__dict__" in e for e in errs)

    def test_full_exploit_chain(self):
        """Full exploit from Audit3 lines 72-78."""
        code = (
            "subs = ().__class__.__mro__[-1].__subclasses__()\n"
            "for s in subs:\n"
            "    if 'warning' in str(s).lower():\n"
            "        import_func = s.__init__.__globals__.get('__builtins__', {})\n"
            "        break\n"
        )
        errs = validate_tool_code(code)
        assert len(errs) >= 3  # __class__, __mro__, __subclasses__, __init__, __globals__


# ── Execution tests ──────────────────────────────────────────────


class TestExecutePythonInSandbox:
    @pytest.mark.asyncio
    async def test_execute_simple_expression(self):
        """print('hello world') should appear in stdout."""
        result = await execute_python_in_sandbox('print("hello world")', {})
        assert isinstance(result, SandboxResult)
        assert "hello world" in result.stdout
        assert result.exit_code == 0
        assert result.timed_out is False

    @pytest.mark.asyncio
    async def test_execute_with_args(self):
        """Receives JSON args via stdin and processes them."""
        code = (
            "name = args.get('name', 'unknown')\n"
            "print(f'Hello, {name}!')\n"
        )
        result = await execute_python_in_sandbox(code, {"name": "SAGE"})
        assert result.exit_code == 0
        assert "Hello, SAGE!" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """time.sleep(999) gets killed with short timeout."""
        code = "import time\ntime.sleep(999)\n"
        result = await execute_python_in_sandbox(code, {}, timeout=2)
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self):
        """Bad syntax returns non-zero exit code and SyntaxError in stderr."""
        result = await execute_python_in_sandbox("def f(\n", {})
        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_import_restriction(self):
        """validate_tool_code catches import os before execution."""
        errors = validate_tool_code("import os\nos.getcwd()")
        assert len(errors) > 0
        assert any("os" in e for e in errors)

    @pytest.mark.asyncio
    async def test_execute_returns_stderr(self):
        """Runtime error appears in stderr."""
        result = await execute_python_in_sandbox("raise ValueError('boom')", {})
        assert result.exit_code != 0
        assert "ValueError" in result.stderr
        assert "boom" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_result_dataclass(self):
        """SandboxResult fields are correct types."""
        result = await execute_python_in_sandbox("print(42)", {})
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.timed_out, bool)
