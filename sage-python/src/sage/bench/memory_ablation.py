"""Memory ablation: measure value of each memory tier.

4 configurations:
  1. no_memory — all memory disabled (WorkingMemory mock, no episodic/semantic)
  2. tier0_only — Rust WorkingMemory + S-MMU, no episodic/semantic/ExoCortex
  3. tier01 — tier0 + episodic, no semantic/ExoCortex
  4. full — all 4 tiers active

Runs same task set under each config, measures pass rate + quality.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class MemoryAblationResult:
    config: str
    tasks_run: int = 0
    tasks_passed: int = 0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.tasks_passed / self.tasks_run if self.tasks_run > 0 else 0.0


ABLATION_TASKS = [
    "Write a Python function to compute the nth Fibonacci number",
    "Implement a linked list with append and reverse methods",
    "Write a function that finds all prime numbers up to N using Sieve of Eratosthenes",
    "Create a class for a simple calculator with add, subtract, multiply, divide",
    "Write a function to check if a binary tree is balanced",
    "Implement merge sort in Python",
    "Write a function that converts Roman numerals to integers",
    "Create a Python decorator that caches function results",
    "Write a function to find the longest common subsequence of two strings",
    "Implement a simple regex matcher supporting . and * characters",
]


async def run_memory_ablation(
    boot_fn,
    tasks: list[str] | None = None,
    verbose: bool = False,
) -> list[MemoryAblationResult]:
    """Run memory ablation across 4 configurations.

    Parameters
    ----------
    boot_fn:
        Callable that returns an AgentSystem. Should accept keyword args
        for memory configuration overrides.
    tasks:
        Override task list. Defaults to ABLATION_TASKS.
    """
    task_list = tasks or ABLATION_TASKS
    configs = [
        ("no_memory", {"disable_memory": True}),
        ("tier0_only", {"disable_episodic": True, "disable_semantic": True, "disable_exocortex": True}),
        ("tier01", {"disable_semantic": True, "disable_exocortex": True}),
        ("full", {}),
    ]

    results = []
    for config_name, overrides in configs:
        if verbose:
            print(f"\n--- Config: {config_name} ---")
        result = MemoryAblationResult(config=config_name)

        try:
            system = await boot_fn(**overrides)
        except Exception as exc:
            result.errors.append(f"Boot failed: {exc}")
            results.append(result)
            continue

        qualities = []
        latencies = []

        for i, task in enumerate(task_list):
            result.tasks_run += 1
            start = time.perf_counter()
            try:
                response = await system.run(task)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)

                # Simple quality: non-empty + has code
                quality = 0.0
                if response and len(response.strip()) > 10:
                    quality = 0.5
                    if "def " in response or "class " in response:
                        quality = 1.0
                    result.tasks_passed += 1
                qualities.append(quality)

                if verbose:
                    status = "PASS" if quality >= 0.5 else "FAIL"
                    print(f"  [{i+1}/{len(task_list)}] {status} ({latency:.0f}ms): {task[:50]}")
            except Exception as exc:
                result.errors.append(f"Task {i+1}: {exc}")
                if verbose:
                    print(f"  [{i+1}/{len(task_list)}] ERROR: {exc}")

        result.avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        result.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        results.append(result)

    return results
