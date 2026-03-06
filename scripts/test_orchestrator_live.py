"""Live test: ModelRegistry auto-discovery + CognitiveOrchestrator with real APIs."""
import sys
import os
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sage-python" / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from sage.providers.registry import ModelRegistry
from sage.orchestrator import CognitiveOrchestrator
from sage.events.bus import EventBus


async def main():
    # 1. Boot registry with auto-discovery
    print("=== Model Registry: Auto-Discovery ===")
    registry = ModelRegistry()
    await registry.refresh()

    available = registry.list_available()
    print(f"Discovered {len(available)} available models:")
    for m in available:
        cost = f"${m.cost_input:.3f}/${m.cost_output:.3f}"
        print(f"  {m.id:40s} {m.provider:10s} code={m.code_score:.2f} reason={m.reasoning_score:.2f} {cost}")

    print()

    # 2. Test model selection
    print("=== Model Selection Tests ===")
    tests = [
        ({"code": 1.0, "max_cost_per_1m": 2.0}, "Best cheap coder"),
        ({"reasoning": 1.0}, "Best reasoner (any cost)"),
        ({"code": 0.5, "reasoning": 0.5, "max_cost_per_1m": 1.0}, "Balanced cheap"),
        ({"code": 1.0}, "Best coder (any cost)"),
    ]
    for needs, label in tests:
        m = registry.select(needs)
        if m:
            print(f"  {label:30s} -> {m.id} ({m.provider}) ${m.cost_input:.3f}/${m.cost_output:.3f}")
        else:
            print(f"  {label:30s} -> NO MODEL")

    print()

    # 3. Test CognitiveOrchestrator with real API calls
    print("=== CognitiveOrchestrator: Live Tests ===")
    bus = EventBus()
    orch = CognitiveOrchestrator(registry=registry, event_bus=bus)

    tasks = [
        ("What is 2+2?", "S1 trivial"),
        ("Write a Python function to check if a string is a palindrome.", "S2 code"),
    ]

    for task, label in tasks:
        print(f'\n--- {label}: "{task[:60]}" ---')
        t0 = time.perf_counter()
        try:
            result = await orch.run(task)
            elapsed = time.perf_counter() - t0
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Response: {result[:300]}")
            events = bus.query(last_n=10)
            for e in events:
                if e.type == "ORCHESTRATOR":
                    print(f"  Model: {e.model} | System: S{e.system} | Latency: {e.latency_ms:.0f}ms")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
