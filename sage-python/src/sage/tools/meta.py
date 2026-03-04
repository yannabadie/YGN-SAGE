"""Meta-tools for dynamic tool synthesis (OpenSage).

Enables agents to write, register, and manage their own tools dynamically.
"""
from __future__ import annotations

import os
import ast
import json
import logging
from typing import Any, Dict

from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# SOTA OpenSage: Tools to write tools
@Tool.define(
    name="create_python_tool",
    description="Dynamically writes and registers a new Python tool for the agent to use. The code must define an async function and use the @Tool.define decorator. The tool will be immediately available.",
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
        
    try:
        # Validate syntax
        ast.parse(code)
        
        # We need a safe namespace to execute the dynamic code
        namespace = {
            "Tool": Tool,
            "os": os,
            "json": json,
            "Any": Any,
            "Dict": Dict
        }
        
        # Execute the code to define the tool in the namespace
        exec(code, namespace)
        
        # Find the defined tool in the namespace
        new_tool = None
        for obj_name, obj in namespace.items():
            if isinstance(obj, Tool) and obj.name == name:
                new_tool = obj
                break
                
        if new_tool:
            registry.register(new_tool)
            return f"Success: Tool '{name}' has been compiled and registered. You can now use it."
        else:
            return f"Error: Could not find a Tool named '{name}' in the provided code. Did you use @Tool.define?"
            
    except Exception as e:
        return f"Error compiling dynamic tool: {str(e)}"


@Tool.define(
    name="create_bash_tool",
    description="Creates a reusable tool that wraps a specific bash command or script.",
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
    if not registry:
         return "Error: Tool registry not available."
         
    # Generate the python wrapper
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
        result = subprocess.run({repr(script)}, shell=True, capture_output=True, text=True, timeout=60)
        return f"STDOUT:\\n{{result.stdout}}\\nSTDERR:\\n{{result.stderr}}"
    except Exception as e:
        return f"Execution error: {{e}}"
"""
    return await create_python_tool(name, code, registry)
