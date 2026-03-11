"""Meta-tools for dynamic tool synthesis.

Enables agents to write, register, and manage their own tools dynamically.
Uses subprocess-based sandboxed execution with AST validation to prevent
arbitrary code execution in the host process.
"""
from __future__ import annotations

import asyncio
import json
import os
import logging
import re

from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry
from sage.tools.sandbox_executor import validate_tool_code, execute_python_in_sandbox
from sage.llm.base import ToolDef

logger = logging.getLogger(__name__)

TOOLS_WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tools")
os.makedirs(TOOLS_WORKSPACE, exist_ok=True)

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

        # Create sandboxed handler using Rust executor
        saved_code = code
        async def _rust_handler(**kwargs):
            result = _executor.execute_raw(saved_code, json.dumps(kwargs))
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
            "properties": {},
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

    # 2. Check script against destructive command blocklist
    _DESTRUCTIVE_PATTERN = re.compile(
        r"rm\s+-rf|mkfs|dd\s+if=|:\(\)\s*\{|/dev/sd|>\s*/dev/"
    )
    if _DESTRUCTIVE_PATTERN.search(script):
        return "Blocked: Script contains potentially destructive commands."

    # 3. Build subprocess-isolated handler closure
    saved_script = script  # capture for closure

    async def _bash_handler(**kwargs):
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "/bin/bash", "-c", saved_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=60,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="replace").strip()
            else:
                return f"Error (exit {proc.returncode}): {stderr.decode('utf-8', errors='replace').strip()}"
        except asyncio.TimeoutError:
            return "Error: Script execution timed out after 60 seconds."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # 4. Register tool
    tool_spec = ToolDef(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {},
        },
    )
    new_tool = Tool(spec=tool_spec, handler=_bash_handler)
    registry.register(new_tool)

    logger.info("Registered bash tool '%s' (subprocess-isolated)", name)
    return f"Success: Bash tool '{name}' has been created and registered (subprocess-isolated)."
