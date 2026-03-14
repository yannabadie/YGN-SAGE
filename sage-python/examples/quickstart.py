"""Minimal SAGE quickstart — 10 lines to your first agent."""

import asyncio
from sage import create


async def main():
    system = await create()  # Auto-discovers LLM (Codex > Gemini)
    result = await system.run("Write a Python function that checks if a number is prime")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
