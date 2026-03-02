"""Built-in tools for YGN-SAGE agents."""
from __future__ import annotations

import asyncio
import subprocess
from sage.tools.base import Tool


bash_tool = Tool.define(
    name="bash",
    description="Execute a bash command and return its output.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        },
        "required": ["command"],
    },
)(lambda command, timeout=120: _run_bash(command, timeout))


async def _run_bash(command: str, timeout: int = 120) -> str:
    """Execute a bash command asynchronously."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: Command timed out after {timeout}s"

    output = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace")
        output += f"\nSTDERR:\n{err}\nExit code: {proc.returncode}"
    return output
