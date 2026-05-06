"""Meta-tools for dynamic tool synthesis.

Dynamic Python tools use sage_core.ToolExecutor by default: Rust tree-sitter
validation plus validate_and_execute() in the Wasm sandbox. The legacy Python
subprocess fallback is disabled unless SAGE_UNSAFE_PY_SUBPROCESS=1 is set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from typing import Any, Awaitable, Callable

from sage.llm.base import ToolDef
from sage.policy import ToolCapability
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry
from sage.tools.runtime_safety import (
    UNSAFE_PY_SUBPROCESS_ENV,
    load_tool_executor_or_raise,
    unsafe_py_subprocess_enabled,
)
from sage.tools.sandbox_executor import execute_python_in_sandbox, validate_tool_code

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
    capability=ToolCapability.DANGEROUS,
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
async def create_python_tool(name: str, code: str, registry: ToolRegistry | None = None) -> str:
    if not registry:
        return "Error: Tool registry not available for dynamic registration."

    try:
        ToolExecutor = load_tool_executor_or_raise()
    except ImportError:
        if not unsafe_py_subprocess_enabled():
            raise

        logger.warning(
            "%s=1: using timeout-only Python subprocess fallback for tool '%s'. "
            "This is unsafe and must not be used in production.",
            UNSAFE_PY_SUBPROCESS_ENV,
            name,
        )
        return await _create_python_tool_with_python_subprocess(name, code, registry)

    try:
        return await _create_python_tool_with_rust_executor(
            name,
            code,
            registry,
            ToolExecutor,
        )
    except Exception as exc:
        raise RuntimeError(
            "Rust ToolExecutor failed while validating/registering a dynamic "
            "Python tool; refusing to downgrade to Python subprocess fallback."
        ) from exc


async def _create_python_tool_with_rust_executor(
    name: str,
    code: str,
    registry: ToolRegistry,
    tool_executor_cls: type[Any],
) -> str:
    executor = tool_executor_cls()

    validation = executor.validate(code)
    if not validation.valid:
        return "Blocked: " + "; ".join(validation.errors)

    saved_code = code

    async def _rust_handler(**kwargs: Any) -> str:
        try:
            result = executor.validate_and_execute(saved_code, json.dumps(kwargs))
        except ValueError as e:
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

    return _register_generated_tool(name, code, registry, _rust_handler)


async def _create_python_tool_with_python_subprocess(
    name: str,
    code: str,
    registry: ToolRegistry,
) -> str:
    errors = validate_tool_code(code)
    if errors:
        return "Blocked: " + "; ".join(errors)

    saved_code = code

    async def _python_handler(**kwargs: Any) -> str:
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

    return _register_generated_tool(name, code, registry, _python_handler)


def _register_generated_tool(
    name: str,
    code: str,
    registry: ToolRegistry,
    handler: Callable[..., Awaitable[str]],
) -> str:
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
    # Phase 1.5d (cgpro VERIFY 2026-05-06): generated Python tools
    # carry capability=PURE because the wrapper executes code inside
    # the sandboxed validator (sage_core.ToolExecutor + RustPython
    # WASI / subprocess sandbox per ADR-013 §5). The ToolForge prompt
    # already forbids `os`, `sys`, `subprocess`, `socket`, network/
    # filesystem modules; the validator AST-rejects any survival of
    # those imports. The meta-tool `create_python_tool` is itself
    # DANGEROUS (it writes code AND registers a tool dynamically),
    # but the resulting tool can be PURE because its execution is
    # confined and emits no host I/O.
    new_tool = Tool(
        spec=tool_spec,
        handler=handler,
        capability=ToolCapability.PURE,
    )
    registry.register(new_tool)

    logger.info("Registered sandboxed tool '%s' (saved to %s)", name, file_path)
    return (
        f"Success: Tool '{name}' has been created, validated, saved to "
        f"{file_path}, and registered (sandboxed)."
    )


@Tool.define(
    capability=ToolCapability.DANGEROUS,
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
async def create_bash_tool(name: str, description: str, script: str, registry: ToolRegistry | None = None) -> str:
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
    # Phase 1.5d: generated bash tools carry capability=DANGEROUS.
    # Even with subprocess isolation, the spawned shell can read/write
    # the local filesystem and run any binary the operator's PATH
    # exposes. Per cgpro DESIGN classification rule "single label =
    # max-safe summary, multi-effect classifies as dangerous", a
    # generated bash tool gets the strictest tier. The meta-tool
    # `create_bash_tool` itself is also DANGEROUS for the same reason
    # as `create_python_tool` (it writes + registers).
    new_tool = Tool(
        spec=tool_spec,
        handler=_bash_handler,
        capability=ToolCapability.DANGEROUS,
    )
    registry.register(new_tool)

    logger.info("Registered bash tool '%s' (subprocess-isolated)", name)
    return f"Success: Bash tool '{name}' has been created and registered (subprocess-isolated)."
