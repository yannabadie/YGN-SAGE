"""Meta-tools for dynamic tool synthesis.

Enables agents to write, register, and manage their own tools dynamically.
Uses subprocess-based sandboxed execution with AST validation to prevent
arbitrary code execution in the host process.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import logging
import shlex
from typing import Callable

from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry
from sage.tools.sandbox_executor import validate_tool_code, execute_python_in_sandbox
from sage.llm.base import ToolDef

logger = logging.getLogger(__name__)

TOOLS_WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tools")
os.makedirs(TOOLS_WORKSPACE, exist_ok=True)

_READ_ONLY_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("git", "diff"),
    ("git", "log"),
    ("git", "show"),
    ("git", "status"),
    ("cat",),
    ("date",),
    ("df",),
    ("du",),
    ("echo",),
    ("file",),
    ("find",),
    ("grep",),
    ("head",),
    ("ls",),
    ("printf",),
    ("pwd",),
    ("rg",),
    ("sort",),
    ("stat",),
    ("tail",),
    ("tree",),
    ("uniq",),
    ("wc",),
    ("whoami",),
)

_EXACT_SHELL_OPERATOR_TOKENS = {
    ";", ";;", "&", "&&", "|", "||", ">", ">>", "<", "<<", "<<<",
}
_CONTROL_CHARS = tuple(chr(i) for i in range(32) if chr(i) not in ("\t",))


def _format_allowed_commands() -> str:
    return ", ".join(" ".join(parts) for parts in _READ_ONLY_COMMANDS)


def _reject_control_chars(argv: list[str]) -> str | None:
    for arg in argv:
        if "\x00" in arg or any(ch in arg for ch in _CONTROL_CHARS):
            return "Blocked: Control characters are not allowed in bash tool commands."
    return None


def _reject_shell_operators(argv: list[str]) -> str | None:
    for arg in argv:
        if arg in _EXACT_SHELL_OPERATOR_TOKENS:
            return (
                "Blocked: Shell operators are not supported. "
                "Pass a single command with arguments only."
            )
    return None


def _validate_find_args(args: list[str]) -> str | None:
    blocked = {
        "-delete", "-exec", "-execdir", "-ok", "-okdir",
        "-fprint", "-fprint0", "-fprintf", "-fls",
    }
    for arg in args:
        if arg in blocked:
            return f"Blocked: find argument '{arg}' is not allowed."
    return None


def _validate_rg_args(args: list[str]) -> str | None:
    for arg in args:
        if arg == "--pre" or arg.startswith("--pre="):
            return "Blocked: rg --pre is not allowed in bash tools."
        if arg == "--pre-glob" or arg.startswith("--pre-glob="):
            return "Blocked: rg --pre-glob is not allowed in bash tools."
    return None


def _validate_git_args(args: list[str]) -> str | None:
    for arg in args:
        if arg == "--output" or arg.startswith("--output="):
            return "Blocked: git --output is not allowed in bash tools."
        if arg == "--ext-diff":
            return "Blocked: git --ext-diff is not allowed in bash tools."
        if arg == "--textconv":
            return "Blocked: git --textconv is not allowed in bash tools."
    return None


def _validator_for(prefix: tuple[str, ...]) -> Callable[[list[str]], str | None]:
    if prefix == ("find",):
        return _validate_find_args
    if prefix == ("rg",):
        return _validate_rg_args
    if prefix[0] == "git":
        return _validate_git_args
    return lambda _args: None


def _parse_bash_tool_command(script: str) -> tuple[list[str] | None, str | None]:
    script_stripped = script.strip()
    if not script_stripped:
        return None, "Blocked: Empty script."
    if "\n" in script_stripped or "\r" in script_stripped:
        return None, "Blocked: Multi-line bash tool scripts are not allowed."

    try:
        argv = shlex.split(script_stripped, posix=True)
    except ValueError as exc:
        return None, f"Blocked: Invalid shell quoting: {exc}"

    if not argv:
        return None, "Blocked: Empty script."

    control_error = _reject_control_chars(argv)
    if control_error:
        return None, control_error

    operator_error = _reject_shell_operators(argv)
    if operator_error:
        return None, operator_error

    matched_prefix: tuple[str, ...] | None = None
    for prefix in sorted(_READ_ONLY_COMMANDS, key=len, reverse=True):
        if argv[: len(prefix)] == list(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        return (
            None,
            "Blocked: Command not in allowlist. "
            f"Permitted commands: {_format_allowed_commands()}",
        )

    validator = _validator_for(matched_prefix)
    arg_error = validator(argv[len(matched_prefix):])
    if arg_error:
        return None, arg_error

    return argv, None


def _build_isolated_exec_wrapper(argv: list[str]) -> str:
    argv_literal = json.dumps(argv)
    return (
        "import json\n"
        "import subprocess\n"
        "import sys\n"
        f"argv = json.loads({argv_literal!r})\n"
        "proc = subprocess.run(argv, capture_output=True, text=True, shell=False)\n"
        "sys.stdout.write(proc.stdout)\n"
        "sys.stderr.write(proc.stderr)\n"
        "raise SystemExit(proc.returncode)\n"
    )

@Tool.define(
    name="create_python_tool",
    description="Dynamically writes and registers a new Python tool. The code is saved persistently to disk and formally validated before registration.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the new tool"},
            "code": {"type": "string", "description": "The complete Python code for the tool, including imports and @Tool.define"}
        },
        "required": ["name", "code"]
    }
)
async def create_python_tool(name: str, code: str, registry: ToolRegistry = None) -> str:
    if not registry:
        return "Error: Tool registry not available for dynamic registration."

    # Try Rust path first (tree-sitter validation + subprocess)
    try:
        from sage_core import ToolExecutor
        _executor = ToolExecutor()

        # Validate via tree-sitter
        validation = _executor.validate(code)
        if not validation.valid:
            return "Blocked: " + "; ".join(validation.errors)

        # Create sandboxed handler using Rust executor.
        # §5 flip (2026-04-22): now uses validate_and_execute — which
        # runs the code in the embedded RustPython wasm sandbox by
        # default. No opt-in required: the code was already tree-
        # sitter-validated above, and the Wasm layer enforces the
        # deny-by-default filesystem / network / env / subprocess
        # contract even if validation missed something.
        saved_code = code
        async def _rust_handler(**kwargs):
            try:
                result = _executor.validate_and_execute(saved_code, json.dumps(kwargs))
            except ValueError as e:
                # Re-validation inside validate_and_execute rejected
                # the code (shouldn't happen — we validated above —
                # but surface cleanly if it does).
                return f"Error (validation): {e}"
            if result.exit_code != 0:
                return f"Error (exit {result.exit_code}): {result.stderr.strip()}"
            stdout = result.stdout.strip()
            try:
                parsed = json.loads(stdout)
                if isinstance(parsed, dict) and "output" in parsed:
                    return str(parsed["output"])
            except (json.JSONDecodeError, TypeError):
                pass
            return stdout

        handler = _rust_handler
        logger.info("Using Rust ToolExecutor for tool '%s'", name)

    except (ImportError, AttributeError):
        # Fallback: Python sandbox_executor
        errors = validate_tool_code(code)
        if errors:
            return "Blocked: " + "; ".join(errors)

        saved_code = code
        async def _python_handler(**kwargs):
            result = await execute_python_in_sandbox(saved_code, kwargs)
            if result.exit_code != 0:
                return f"Error (exit {result.exit_code}): {result.stderr.strip()}"
            stdout = result.stdout.strip()
            try:
                parsed = json.loads(stdout)
                if isinstance(parsed, dict) and "output" in parsed:
                    return str(parsed["output"])
            except (json.JSONDecodeError, TypeError):
                pass
            return stdout

        handler = _python_handler
        logger.info("Using Python sandbox for tool '%s' (Rust ToolExecutor not available)", name)

    # Save code for auditability
    file_path = os.path.join(TOOLS_WORKSPACE, f"{name}.py")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
    except OSError as e:
        return f"Error: Could not save tool code: {e}"

    # Register tool
    tool_spec = ToolDef(
        name=name,
        description=f"Dynamically created tool '{name}' (sandboxed).",
        parameters={
            "type": "object",
            "properties": {"_": {"type": "string", "description": "Unused. Pass empty string."}},
        },
    )
    new_tool = Tool(spec=tool_spec, handler=handler)
    registry.register(new_tool)

    logger.info("Registered sandboxed tool '%s' (saved to %s)", name, file_path)
    return f"Success: Tool '{name}' has been created, validated, saved to {file_path}, and registered (sandboxed)."


@Tool.define(
    name="create_bash_tool",
    description="Creates a persistent tool that wraps a specific bash command or script. Executed via secure subprocess.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the new tool"},
            "description": {"type": "string", "description": "What the tool does"},
            "script": {"type": "string", "description": "The bash script to execute"}
        },
        "required": ["name", "description", "script"]
    }
)
async def create_bash_tool(name: str, description: str, script: str, registry: ToolRegistry = None) -> str:
    # 1. Registry required
    if not registry:
        return "Error: Tool registry not available for dynamic registration."

    # 2. Security gate: parse to argv and validate against an explicit read-only allowlist.
    argv, validation_error = _parse_bash_tool_command(script)
    if validation_error:
        return validation_error
    assert argv is not None

    # 3. Build sandbox-isolated handler closure
    saved_argv = list(argv)

    async def _bash_handler(**kwargs):
        try:
            # Use Rust sandbox executor if available, subprocess fallback with strict limits
            from sage.sandbox.isolated_executor import execute_isolated
            wrapper = _build_isolated_exec_wrapper(saved_argv)
            stdout, stderr, exit_code = execute_isolated(wrapper, timeout=30)
            if exit_code == 0:
                return stdout.strip()
            else:
                return f"Error (exit {exit_code}): {stderr.strip()}"
        except ImportError:
            # Fallback: direct argv execution with no shell.
            try:
                proc = await asyncio.create_subprocess_exec(
                    *saved_argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                if proc.returncode == 0:
                    return stdout_text
                return f"Error (exit {proc.returncode}): {stderr_text}"
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.communicate()
                return "Error: Script execution timed out after 30 seconds."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # 4. Register tool
    tool_spec = ToolDef(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"_": {"type": "string", "description": "Unused. Pass empty string."}},
        },
    )
    new_tool = Tool(spec=tool_spec, handler=_bash_handler)
    registry.register(new_tool)

    logger.info("Registered bash tool '%s' (subprocess-isolated)", name)
    return f"Success: Bash tool '{name}' has been created and registered (subprocess-isolated)."
