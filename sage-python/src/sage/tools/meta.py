"""Meta-tools for dynamic tool synthesis.

Enables agents to write, register, and manage their own tools dynamically.
Uses subprocess-based sandboxed execution with AST validation to prevent
arbitrary code execution in the host process.
"""
from __future__ import annotations

import json
import os
import logging
import warnings

from sage.tools.base import Tool, ToolResult
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
    # 1. Registry required
    if not registry:
        return "Error: Tool registry not available for dynamic registration."

    # 2. AST validation — reject dangerous patterns before execution
    errors = validate_tool_code(code)
    if errors:
        return "Blocked: " + "; ".join(errors)

    # 3. Save code to disk for auditability
    file_path = os.path.join(TOOLS_WORKSPACE, f"{name}.py")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
    except OSError as e:
        return f"Error: Could not save tool code: {e}"

    # 4. Build sandboxed handler closure
    saved_code = code  # capture for closure

    async def _sandboxed_handler(**kwargs):
        result = await execute_python_in_sandbox(saved_code, kwargs)
        if result.exit_code != 0:
            return f"Error (exit {result.exit_code}): {result.stderr.strip()}"
        # Try to parse structured output (JSON with "output" key)
        stdout = result.stdout.strip()
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict) and "output" in parsed:
                return str(parsed["output"])
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back to raw stdout
        return stdout

    # 5. Register tool
    tool_spec = ToolDef(
        name=name,
        description=f"Dynamically created tool '{name}' (sandboxed).",
        parameters={
            "type": "object",
            "properties": {},
        },
    )
    new_tool = Tool(spec=tool_spec, handler=_sandboxed_handler)
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
    warnings.warn(
        "create_bash_tool is disabled due to security vulnerabilities (SEC-02). "
        "Use statically defined tools or Wasm sandbox instead.",
        DeprecationWarning, stacklevel=2,
    )
    return "DISABLED: create_bash_tool is not available due to security constraints."

    if not registry:
         return "Error: Tool registry not available."

    # Generate the python wrapper for the bash script
    code = f"""
from sage.tools.base import Tool
import subprocess

@Tool.define(
    name="{name}",
    description="{description}",
    parameters={{"type": "object", "properties": {{}}}}
)
async def {name}() -> str:
    try:
        # Executed with isolation considerations
        result = subprocess.run({repr(script)}, shell=True, capture_output=True, text=True, timeout=60)
        return f"STDOUT:\\n{{result.stdout}}\\nSTDERR:\\n{{result.stderr}}"
    except Exception as e:
        return f"Execution error: {{e}}"
"""
    return await create_python_tool(name, code, registry)
