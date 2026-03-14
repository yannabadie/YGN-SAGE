"""Custom tools: give your agent new capabilities."""

import asyncio
from sage import create, Tool


@Tool.define(
    name="calculate",
    description="Evaluate a math expression",
    parameters={"expression": {"type": "string", "description": "Math expression like '2+3*4'"}},
)
async def calculate(expression: str) -> str:
    """Safely evaluate math expressions."""
    import ast
    try:
        tree = ast.parse(expression, mode="eval")
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def main():
    system = await create(tools=[calculate])
    result = await system.run("What is 17 * 23 + 42?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
