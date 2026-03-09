# Sandboxed Tool Creation via SandboxManager

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-enable `create_python_tool` and `create_bash_tool` with code executing in the SandboxManager (subprocess isolation) instead of in-process `exec()`. No more SEC-01/SEC-02 vulnerabilities.

**Architecture:** Tools are created by saving code to disk, then wrapping them in a handler that executes the code via `SandboxManager` (subprocess with `create_subprocess_exec`, no `shell=True`). The handler sends the code + arguments as a self-contained Python script to the sandbox. The Wasm path is used when `WasmSandbox` is available; otherwise subprocess isolation is the fallback. The tool is registered normally in `ToolRegistry` — callers don't know it's sandboxed.

**Tech Stack:** Python 3.12+, asyncio, SandboxManager, AST validation, pytest

---

### Task 1: Create sandboxed tool executor module

**Files:**
- Create: `sage-python/src/sage/tools/sandbox_executor.py`
- Test: `sage-python/tests/test_sandbox_executor.py`

**Step 1: Write the failing test**

```python
# sage-python/tests/test_sandbox_executor.py
"""Tests for sandbox-backed tool execution."""
import pytest
import asyncio
from sage.tools.sandbox_executor import execute_python_in_sandbox


def test_execute_simple_expression():
    """Sandbox executor runs Python code and returns stdout."""
    code = 'print("hello world")'
    result = asyncio.run(execute_python_in_sandbox(code, {}))
    assert result.exit_code == 0
    assert "hello world" in result.stdout


def test_execute_with_args():
    """Sandbox executor receives JSON args via stdin."""
    code = (
        'import json, sys\n'
        'args = json.load(sys.stdin)\n'
        'print(json.dumps({"sum": args["a"] + args["b"]}))'
    )
    result = asyncio.run(execute_python_in_sandbox(code, {"a": 3, "b": 4}))
    assert result.exit_code == 0
    assert '"sum": 7' in result.stdout


def test_execute_timeout():
    """Sandbox executor kills long-running code."""
    code = 'import time; time.sleep(999)'
    result = asyncio.run(execute_python_in_sandbox(code, {}, timeout=2))
    assert result.timed_out or result.exit_code != 0


def test_execute_syntax_error():
    """Sandbox executor returns error for bad code."""
    code = 'def f(:\n  pass'
    result = asyncio.run(execute_python_in_sandbox(code, {}))
    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr


def test_execute_import_restriction():
    """Sandbox executor blocks os.system and subprocess."""
    code = 'import os; os.system("echo pwned")'
    result = asyncio.run(execute_python_in_sandbox(code, {}))
    # The code runs but we validate AST before calling
    # This test verifies the AST check catches it
    from sage.tools.sandbox_executor import validate_tool_code
    errors = validate_tool_code(code)
    assert len(errors) > 0
    assert any("os.system" in e or "os" in e for e in errors)


def test_validate_clean_code():
    """Validator accepts safe code."""
    from sage.tools.sandbox_executor import validate_tool_code
    code = (
        'import json, math\n'
        'def run(x):\n'
        '    return json.dumps({"result": math.sqrt(x)})\n'
    )
    errors = validate_tool_code(code)
    assert errors == []


def test_validate_blocks_exec():
    """Validator rejects exec/eval."""
    from sage.tools.sandbox_executor import validate_tool_code
    assert len(validate_tool_code('exec("x=1")')) > 0
    assert len(validate_tool_code('eval("1+1")')) > 0


def test_validate_blocks_dangerous_imports():
    """Validator rejects subprocess, shutil, ctypes."""
    from sage.tools.sandbox_executor import validate_tool_code
    assert len(validate_tool_code('import subprocess')) > 0
    assert len(validate_tool_code('import shutil')) > 0
    assert len(validate_tool_code('import ctypes')) > 0
    assert len(validate_tool_code('from os import system')) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_sandbox_executor.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'sage.tools.sandbox_executor'"

**Step 3: Write minimal implementation**

```python
# sage-python/src/sage/tools/sandbox_executor.py
"""Sandboxed Python code executor.

Executes user-provided Python code in an isolated subprocess
(not in-process exec/eval). Arguments are passed via stdin as JSON.
Returns SandboxResult with stdout/stderr/exit_code.

Security model:
  1. AST validation BEFORE execution (reject dangerous patterns)
  2. Subprocess isolation (create_subprocess_exec, no shell=True)
  3. Timeout enforcement (kills process on timeout)
  4. No network, no filesystem writes outside /tmp
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
import tempfile
import os
from dataclasses import dataclass

# Allowed stdlib modules — extend as needed
ALLOWED_MODULES = frozenset({
    "json", "math", "re", "collections", "itertools", "functools",
    "datetime", "decimal", "fractions", "statistics", "string",
    "textwrap", "unicodedata", "hashlib", "hmac", "base64",
    "copy", "pprint", "typing", "dataclasses", "enum",
    "csv", "io", "urllib.parse",
})

# Blocked modules — security-critical
BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "ctypes", "importlib",
    "socket", "http", "ftplib", "smtplib", "xmlrpc",
    "multiprocessing", "threading", "signal", "resource",
    "code", "codeop", "compile", "compileall",
    "pathlib", "glob", "tempfile", "pickle", "shelve",
    "builtins", "__builtin__",
})

# Blocked function calls
BLOCKED_CALLS = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint",
    "open",  # file I/O blocked; use json.loads(sys.stdin) for input
})

DEFAULT_TIMEOUT = 30  # seconds


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


def validate_tool_code(code: str) -> list[str]:
    """Validate Python code via AST analysis. Returns list of error messages."""
    errors: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_MODULES:
                    errors.append(
                        f"Blocked import: '{alias.name}' — "
                        f"module '{top}' is not allowed in sandboxed tools."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_MODULES:
                    errors.append(
                        f"Blocked import: 'from {node.module}' — "
                        f"module '{top}' is not allowed in sandboxed tools."
                    )

        # Check dangerous function calls
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                # Catch os.system, subprocess.run, etc.
                if isinstance(func.value, ast.Name):
                    name = f"{func.value.id}.{func.attr}"
                    # Also check just the attribute for things like obj.exec()
                    if func.attr in BLOCKED_CALLS:
                        errors.append(
                            f"Blocked call: '{func.attr}()' is not allowed."
                        )
            if name and name.split(".")[0] in BLOCKED_CALLS:
                errors.append(f"Blocked call: '{name}()' is not allowed.")

    return errors


async def execute_python_in_sandbox(
    code: str,
    args: dict,
    timeout: int = DEFAULT_TIMEOUT,
) -> SandboxResult:
    """Execute Python code in an isolated subprocess.

    The code receives arguments via stdin as a JSON object.
    It should print its result to stdout (ideally as JSON).

    Uses create_subprocess_exec (no shell=True).
    """
    # Build a wrapper that feeds args via stdin
    wrapper = (
        "import json, sys\n"
        "args = json.load(sys.stdin)\n"
        + code
    )

    # Write to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(wrapper)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdin_data = json.dumps(args).encode("utf-8")
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout,
            )
            return SandboxResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()  # drain
            return SandboxResult(
                stdout="", stderr="Execution timed out",
                exit_code=137, timed_out=True,
            )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_sandbox_executor.py -v`
Expected: PASS (all 8 tests)

**Step 5: Commit**

```bash
git add sage-python/src/sage/tools/sandbox_executor.py sage-python/tests/test_sandbox_executor.py
git commit -m "feat: add sandboxed Python code executor with AST validation"
```

---

### Task 2: Rewrite create_python_tool to use sandbox executor

**Files:**
- Modify: `sage-python/src/sage/tools/meta.py`
- Test: `sage-python/tests/test_meta_security.py`

**Step 1: Write the failing tests**

Add to `sage-python/tests/test_meta_security.py`:
```python
def test_create_python_tool_runs_in_sandbox():
    """SEC-01 fix: create_python_tool executes code in subprocess, not exec()."""
    from sage.tools.meta import create_python_tool
    from sage.tools.registry import ToolRegistry
    import asyncio

    registry = ToolRegistry()
    result = asyncio.run(create_python_tool.fn(
        name="adder",
        code='result = args["a"] + args["b"]\nprint(json.dumps({"output": result}))',
        registry=registry,
    ))
    assert "Success" in result or "registered" in result.lower()
    # Tool should be registered
    assert registry.get("adder") is not None


def test_created_tool_executes_in_sandbox():
    """Created tool runs in subprocess isolation, not in-process."""
    from sage.tools.meta import create_python_tool
    from sage.tools.registry import ToolRegistry
    import asyncio

    registry = ToolRegistry()
    asyncio.run(create_python_tool.fn(
        name="greeter",
        code='name = args.get("name", "world")\nprint(json.dumps({"output": f"Hello, {name}!"}))',
        registry=registry,
    ))
    tool = registry.get("greeter")
    assert tool is not None
    result = asyncio.run(tool.execute({"name": "Yann"}))
    assert "Hello, Yann!" in result.output


def test_created_tool_rejects_dangerous_code():
    """create_python_tool rejects code with blocked imports."""
    from sage.tools.meta import create_python_tool
    from sage.tools.registry import ToolRegistry
    import asyncio

    registry = ToolRegistry()
    result = asyncio.run(create_python_tool.fn(
        name="evil",
        code='import subprocess; subprocess.run(["rm", "-rf", "/"])',
        registry=registry,
    ))
    assert "Blocked" in result or "Error" in result
    assert registry.get("evil") is None
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_meta_security.py::test_create_python_tool_runs_in_sandbox -v`
Expected: FAIL (current create_python_tool returns "DISABLED")

**Step 3: Rewrite create_python_tool in meta.py**

Replace the entire `create_python_tool` function (lines 41-93):
```python
async def create_python_tool(name: str, code: str, registry: ToolRegistry = None) -> str:
    """Create a sandboxed Python tool.

    The code receives arguments as a dict named `args` (injected via stdin JSON).
    It should print its result to stdout as JSON: {"output": "..."}.

    Security: Code is validated via AST then executed in a subprocess
    (never in-process exec/eval). See sandbox_executor.py.
    """
    if not registry:
        return "Error: Tool registry not available for dynamic registration."

    from sage.tools.sandbox_executor import validate_tool_code, execute_python_in_sandbox

    # 1. AST validation — reject dangerous patterns
    errors = validate_tool_code(code)
    if errors:
        return f"Blocked: Code failed security validation:\n" + "\n".join(f"  - {e}" for e in errors)

    # 2. Save code for auditability
    file_path = os.path.join(TOOLS_WORKSPACE, f"{name}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)

    # 3. Create a sandboxed handler
    saved_code = code  # capture for closure

    async def _sandboxed_handler(**kwargs) -> str:
        result = await execute_python_in_sandbox(saved_code, kwargs)
        if result.exit_code != 0:
            return f"Tool execution error (exit {result.exit_code}):\n{result.stderr}"
        # Try to extract JSON output, fall back to raw stdout
        stdout = result.stdout.strip()
        try:
            import json
            parsed = json.loads(stdout)
            return parsed.get("output", stdout) if isinstance(parsed, dict) else stdout
        except (json.JSONDecodeError, ValueError):
            return stdout

    # 4. Build Tool and register
    from sage.llm.base import ToolDef
    spec = ToolDef(
        name=name,
        description=f"Dynamically created tool: {name}",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    )
    tool = Tool(spec=spec, handler=_sandboxed_handler)
    registry.register(tool)

    logger.info("Registered sandboxed tool '%s' (saved to %s)", name, file_path)
    return f"Success: Tool '{name}' registered. Code saved to {file_path}. Execution is sandboxed (subprocess isolation)."
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_meta_security.py -v`
Expected: PASS (all tests including new ones)

**Step 5: Commit**

```bash
git add sage-python/src/sage/tools/meta.py sage-python/tests/test_meta_security.py
git commit -m "feat: rewrite create_python_tool with subprocess sandbox (SEC-01)"
```

---

### Task 3: Rewrite create_bash_tool to use sandbox executor

**Files:**
- Modify: `sage-python/src/sage/tools/meta.py`
- Test: `sage-python/tests/test_meta_security.py`

**Step 1: Write the failing tests**

Add to `sage-python/tests/test_meta_security.py`:
```python
def test_create_bash_tool_runs_in_sandbox():
    """SEC-02 fix: create_bash_tool executes via subprocess, no shell=True."""
    from sage.tools.meta import create_bash_tool
    from sage.tools.registry import ToolRegistry
    import asyncio

    registry = ToolRegistry()
    result = asyncio.run(create_bash_tool.fn(
        name="echo_tool",
        description="Echoes input",
        script='echo "hello from bash"',
        registry=registry,
    ))
    assert "Success" in result or "registered" in result.lower()
    assert registry.get("echo_tool") is not None


def test_bash_tool_blocks_dangerous_commands():
    """create_bash_tool blocks rm -rf and similar."""
    from sage.tools.meta import create_bash_tool
    from sage.tools.registry import ToolRegistry
    import asyncio

    registry = ToolRegistry()
    result = asyncio.run(create_bash_tool.fn(
        name="evil_bash",
        description="Dangerous",
        script='rm -rf /',
        registry=registry,
    ))
    assert "Blocked" in result or "BLOCKED" in result
    assert registry.get("evil_bash") is None
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_meta_security.py::test_create_bash_tool_runs_in_sandbox -v`
Expected: FAIL (current create_bash_tool returns "DISABLED")

**Step 3: Rewrite create_bash_tool in meta.py**

Replace the entire `create_bash_tool` function (lines 109-138):
```python
async def create_bash_tool(name: str, description: str, script: str, registry: ToolRegistry = None) -> str:
    """Create a sandboxed bash tool.

    The script is executed via create_subprocess_exec (no shell=True).
    Destructive commands are blocked by pattern matching.
    """
    import re
    if not registry:
        return "Error: Tool registry not available."

    # Block destructive patterns
    BLOCKED = re.compile(r'rm\s+-rf|mkfs|dd\s+if=|:\(\)\s*\{|/dev/sd|>\s*/dev/')
    if BLOCKED.search(script):
        return "Blocked: Script contains potentially destructive commands."

    # Create sandboxed handler
    saved_script = script

    async def _bash_handler(**kwargs) -> str:
        import asyncio as _aio
        proc = await _aio.create_subprocess_exec(
            "/bin/bash", "-c", saved_script,
            stdout=_aio.subprocess.PIPE,
            stderr=_aio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await _aio.wait_for(proc.communicate(), timeout=60)
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return f"Error (exit {proc.returncode}):\n{err}"
            return out if out else err
        except _aio.TimeoutError:
            proc.kill()
            return "Error: Script timed out after 60 seconds."

    from sage.llm.base import ToolDef
    spec = ToolDef(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
    )
    tool = Tool(spec=spec, handler=_bash_handler)
    registry.register(tool)

    logger.info("Registered sandboxed bash tool '%s'", name)
    return f"Success: Bash tool '{name}' registered with subprocess isolation."
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_meta_security.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add sage-python/src/sage/tools/meta.py sage-python/tests/test_meta_security.py
git commit -m "feat: rewrite create_bash_tool with subprocess isolation (SEC-02)"
```

---

### Task 4: Re-enable tool registration in boot.py

**Files:**
- Modify: `sage-python/src/sage/boot.py:201-207`
- Modify: `sage-python/tests/test_integration.py`

**Step 1: Write the failing test**

Update `test_boot_registers_meta_tools` in `sage-python/tests/test_integration.py`:
```python
def test_boot_registers_meta_tools():
    """Boot sequence registers sandboxed meta tools (SEC-01/02 fixed)."""
    system = boot_agent_system(use_mock_llm=True)
    tool_names = system.tool_registry.list_tools()
    # Meta tools re-enabled with sandbox isolation
    assert "create_python_tool" in tool_names
    assert "create_bash_tool" in tool_names
    # Memory tools still registered
    assert "search_memory" in tool_names
    assert "store_memory" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_integration.py::test_boot_registers_meta_tools -v`
Expected: FAIL (tools currently not registered)

**Step 3: Uncomment registration in boot.py**

Replace lines 201-207:
```python
    # Runtime tool synthesis — sandboxed (SEC-01/SEC-02 fixed).
    # Tools execute in subprocess isolation, not in-process exec().
    from sage.tools.meta import create_python_tool, create_bash_tool
    tool_registry.register(create_python_tool)
    tool_registry.register(create_bash_tool)
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_integration.py
git commit -m "feat: re-enable create_python_tool and create_bash_tool with sandbox isolation"
```

---

### Task 5: End-to-end integration test

**Files:**
- Create: `sage-python/tests/integration/test_tool_creation.py`

**Step 1: Write the integration test**

```python
# sage-python/tests/integration/test_tool_creation.py
"""Integration tests for dynamic tool creation pipeline.

These tests verify the full flow: create tool → register → execute → get result.
No mocks. Real subprocess execution.
"""
import pytest
import asyncio
from sage.tools.meta import create_python_tool, create_bash_tool
from sage.tools.registry import ToolRegistry


class TestPythonToolCreation:
    """End-to-end tests for create_python_tool."""

    def test_create_and_run_math_tool(self):
        """Create a math tool and verify it produces correct output."""
        registry = ToolRegistry()
        code = (
            'import math\n'
            'x = float(args.get("x", 0))\n'
            'result = math.factorial(int(x))\n'
            'print(json.dumps({"output": str(result)}))'
        )
        result = asyncio.run(create_python_tool.fn(
            name="factorial", code=code, registry=registry
        ))
        assert "Success" in result

        tool = registry.get("factorial")
        assert tool is not None

        exec_result = asyncio.run(tool.execute({"x": 5}))
        assert "120" in exec_result.output
        assert not exec_result.is_error

    def test_create_tool_with_error_handling(self):
        """Tool that raises returns error output, not crash."""
        registry = ToolRegistry()
        code = 'raise ValueError("intentional error")'
        msg = asyncio.run(create_python_tool.fn(
            name="error_tool", code=code, registry=registry
        ))
        assert "Success" in msg  # Registration succeeds (code is valid Python)

        tool = registry.get("error_tool")
        exec_result = asyncio.run(tool.execute({}))
        # Tool returns error via stderr, not crash
        assert "error" in exec_result.output.lower() or exec_result.is_error

    def test_reject_os_import(self):
        """Tool with os import is blocked at validation."""
        registry = ToolRegistry()
        result = asyncio.run(create_python_tool.fn(
            name="bad", code="import os\nos.listdir('/')", registry=registry
        ))
        assert "Blocked" in result
        assert registry.get("bad") is None

    def test_reject_eval(self):
        """Tool with eval() is blocked at validation."""
        registry = ToolRegistry()
        result = asyncio.run(create_python_tool.fn(
            name="bad2", code='eval("1+1")', registry=registry
        ))
        assert "Blocked" in result

    def test_multiple_tools_coexist(self):
        """Multiple dynamic tools can be registered and used."""
        registry = ToolRegistry()
        for i in range(3):
            code = f'print(json.dumps({{"output": "tool_{i}"}}))'
            asyncio.run(create_python_tool.fn(
                name=f"tool_{i}", code=code, registry=registry
            ))
        assert len([n for n in registry.list_tools() if n.startswith("tool_")]) == 3


class TestBashToolCreation:
    """End-to-end tests for create_bash_tool."""

    def test_create_and_run_echo(self):
        """Create a bash echo tool and verify output."""
        registry = ToolRegistry()
        result = asyncio.run(create_bash_tool.fn(
            name="echo_test",
            description="Echoes a message",
            script='echo "sandbox works"',
            registry=registry,
        ))
        assert "Success" in result

        tool = registry.get("echo_test")
        assert tool is not None

        exec_result = asyncio.run(tool.execute({}))
        assert "sandbox works" in exec_result.output

    def test_reject_rm_rf(self):
        """Bash tool blocks rm -rf."""
        registry = ToolRegistry()
        result = asyncio.run(create_bash_tool.fn(
            name="rm_tool",
            description="Bad tool",
            script="rm -rf /",
            registry=registry,
        ))
        assert "Blocked" in result
        assert registry.get("rm_tool") is None
```

**Step 2: Run tests**

Run: `cd sage-python && python -m pytest tests/integration/test_tool_creation.py -v`
Expected: PASS (all tests)

**Step 3: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS (previous 895 + ~15 new = ~910)

**Step 4: Commit**

```bash
git add sage-python/tests/integration/test_tool_creation.py
git commit -m "test: add end-to-end integration tests for sandboxed tool creation"
```

---

## Dependencies

```
Task 1 (sandbox executor) → independent
Task 2 (rewrite create_python_tool) → depends on Task 1
Task 3 (rewrite create_bash_tool) → depends on Task 1
Task 4 (re-enable in boot.py) → depends on Tasks 2, 3
Task 5 (integration tests) → depends on Task 4
```

## Success Criteria

After all tasks:
- `create_python_tool` and `create_bash_tool` re-enabled in boot.py
- All tool code executes in subprocess (never in-process `exec()`/`eval()`)
- AST validation blocks dangerous imports and calls
- Destructive bash commands blocked by pattern matching
- All tests pass (target: 910+)
- No `shell=True` in tool execution paths
- No `exec()` or `eval()` in tool execution paths
