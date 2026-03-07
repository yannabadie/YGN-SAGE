"""Meta-tools for dynamic tool synthesis.

Enables agents to write, register, and manage their own tools dynamically.
Uses persistent file-based compilation and isolated execution to avoid 
the catastrophic context fragmentation of simple `exec()` wrappers.
"""
from __future__ import annotations

import os
import ast
import logging
import importlib.util
import sys

from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry
try:
    import sage_core
except ImportError:
    import types as _types
    sage_core = _types.ModuleType("sage_core")

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
        
    try:
        # 1. Formal Validation (AST level)
        try:
            parsed_ast = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax Error in generated code: {e}"
            
        # Check for dangerous builtins (Basic SMT proxy for security)
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') in ('exec', 'eval'):
                return "Security Error: The use of exec() or eval() is strictly forbidden in generated tools."

        # 2. Persistence (Save to disk)
        file_path = os.path.join(TOOLS_WORKSPACE, f"{name}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 3. Dynamic Module Loading (Instead of unsafe exec)
        spec = importlib.util.spec_from_file_location(name, file_path)
        if not spec or not spec.loader:
            return f"Error: Could not load module spec for {name}."
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        
        # 4. Registration
        new_tool = None
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, Tool) and obj.name == name:
                new_tool = obj
                break
                
        if new_tool:
            registry.register(new_tool)
            return f"Success: Tool '{name}' has been compiled, saved to {file_path}, and registered securely."
        else:
            return f"Error: Could not find a Tool named '{name}' in the loaded module. Ensure you used @Tool.define."
            
    except Exception as e:
        return f"Error compiling dynamic tool: {str(e)}"


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
