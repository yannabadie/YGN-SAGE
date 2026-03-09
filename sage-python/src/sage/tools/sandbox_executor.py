"""Sandboxed Python code executor with AST-based validation.

Provides two main functions:
- validate_tool_code(code) -> list[str]: static AST analysis for dangerous patterns
- execute_python_in_sandbox(code, args, timeout) -> SandboxResult: subprocess execution
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


# ── Blocked sets ─────────────────────────────────────────────────

BLOCKED_IMPORTS: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "shutil", "ctypes", "importlib",
    "socket", "http", "ftplib", "smtplib", "xmlrpc",
    "multiprocessing", "threading", "signal", "resource",
    "code", "codeop", "pathlib", "glob", "tempfile",
    "pickle", "shelve", "builtins", "__builtin__",
})

BLOCKED_CALLS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint", "open",
})


# ── Data types ───────────────────────────────────────────────────

@dataclass
class SandboxResult:
    """Result of a sandboxed Python execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


# ── AST validation ───────────────────────────────────────────────

def validate_tool_code(code: str) -> list[str]:
    """Validate Python code via AST analysis.

    Returns a list of error messages. An empty list means the code is clean.
    Checks for:
    - Blocked module imports (import X, from X import Y)
    - Blocked function/method calls (exec, eval, open, etc.)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]

    errors: list[str] = []

    for node in ast.walk(tree):
        # ── import X / import X.Y ────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_IMPORTS:
                    errors.append(
                        f"Blocked import: '{alias.name}' "
                        f"(line {node.lineno})"
                    )

        # ── from X import Y / from X.Z import Y ─────────────
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_IMPORTS:
                    errors.append(
                        f"Blocked import: 'from {node.module} import ...' "
                        f"(line {node.lineno})"
                    )

        # ── function / method calls ──────────────────────────
        elif isinstance(node, ast.Call):
            name = _extract_call_name(node)
            if name and name in BLOCKED_CALLS:
                errors.append(
                    f"Blocked call: '{name}()' "
                    f"(line {node.lineno})"
                )

    return errors


def _extract_call_name(node: ast.Call) -> str | None:
    """Extract the function/method name from a Call node.

    Handles:
    - func()        -> Name.id = "func"
    - obj.method()  -> Attribute.attr = "method"
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ── Subprocess execution ────────────────────────────────────────

_WRAPPER_TEMPLATE = """\
import json, sys
args = json.load(sys.stdin)
{user_code}
"""


async def execute_python_in_sandbox(
    code: str,
    args: dict[str, Any],
    timeout: int = 30,
) -> SandboxResult:
    """Execute Python code in an isolated subprocess.

    The user code receives a pre-populated `args` dict (deserialized from
    JSON fed via stdin). stdout/stderr are captured and returned in a
    SandboxResult dataclass.

    Args:
        code: Python source code to execute.
        args: Dictionary passed to the script via stdin as JSON.
        timeout: Maximum execution time in seconds (default 30).

    Returns:
        SandboxResult with stdout, stderr, exit_code, and timed_out flag.
    """
    # Build the wrapper script
    wrapper = _WRAPPER_TEMPLATE.format(user_code=code)

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    )
    script_path = tmp.name
    try:
        tmp.write(wrapper)
        tmp.close()

        # Serialize args
        stdin_data = json.dumps(args).encode("utf-8")

        # Launch subprocess
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout,
            )
            return SandboxResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
                timed_out=False,
            )
        except asyncio.TimeoutError:
            # Kill the process on timeout
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # Drain any partial output
            try:
                stdout_bytes, stderr_bytes = await proc.communicate()
            except Exception:
                stdout_bytes, stderr_bytes = b"", b""
            return SandboxResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else "",
                stderr=stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
                exit_code=proc.returncode or -1,
                timed_out=True,
            )
    finally:
        # Clean up temp file
        try:
            import os as _os  # local import — only for cleanup, not exposed
            _os.unlink(script_path)
        except OSError:
            pass
