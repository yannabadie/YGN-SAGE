"""CPython WASI tool runner — executes Python code inside Wasm sandbox.

Built with componentize-py to create a WASI component from CPython.
The component implements the tool-env WIT world (sage:plugin/tool-env).

Security: WASI capabilities are deny-by-default in the host (wasmtime-wasi).
The guest cannot access filesystem, network, env vars, or subprocess
unless the host explicitly grants those capabilities.

Build (Linux CI or Windows with componentize-py):
    componentize-py -d ../../interface.wit -w tool-env componentize app \\
        -o ../cpython-runner.wasm
"""

import json
import math

from wit_world import WitWorld, ToolInput, ToolOutput


class WitWorld(WitWorld):
    """Implements the tool-env WIT world for CPython WASI execution."""

    def run(self, input: ToolInput) -> ToolOutput:
        """Execute Python code from the input args in a restricted namespace.

        The code can use 'args' (parsed JSON), 'json', and 'math' modules.
        Result is read from 'result' or 'output' variable in the namespace.
        """
        try:
            raw_args = json.loads(input.args) if input.args else {}

            # Extract code: either from __code__ key or the name field
            code = raw_args.get("__code__", input.name)

            # Create a restricted namespace — only safe builtins
            namespace = {
                "args": raw_args,
                "json": json,
                "math": math,
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "isinstance": isinstance,
                "type": type,
                "hasattr": hasattr,
                "getattr": getattr,
            }

            exec(code, namespace)

            # Capture result from namespace
            result = namespace.get("result", namespace.get("output", ""))

            return ToolOutput(
                stdout=str(result),
                stderr="",
                exit_code=0,
                result_json=json.dumps({"output": str(result)}),
            )

        except Exception as e:
            return ToolOutput(
                stdout="",
                stderr=f"{type(e).__name__}: {e}",
                exit_code=1,
                result_json="",
            )
