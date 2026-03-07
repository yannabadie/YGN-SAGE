"""Routing SELF-CONSISTENCY benchmark (NOT accuracy).

Measures whether MetacognitiveController's heuristic agrees with
hand-labeled tasks. Labels were calibrated against the heuristic,
so 100% agreement is expected and proves nothing about downstream
task quality.

To measure real routing accuracy, compare task outcomes (pass/fail)
across different routing decisions. See docs/plans/ for the
evidence-first routing benchmark design.

Heuristic routing summary (default thresholds):
  S1: complexity <= 0.35 AND uncertainty <= 0.3 AND tool_required is False
  S3: complexity > 0.7 OR uncertainty > 0.6
  S2: everything else

Complexity: base 0.3
  +0.3 if any(debug, fix, error, crash)
  +0.2 if any(optimize, evolve, design, architect)
  +0.1 if len(task) > 500

Uncertainty: base 0.2
  +0.2 if "?" in task
  +0.2 if any(maybe, possibly, explore, investigate)

tool_required: any(file, search, run, execute, compile, test, deploy)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

if TYPE_CHECKING:
    from sage.strategy.metacognition import MetacognitiveController

# fmt: off
LABELED_TASKS: list[dict] = [
    # ------------------------------------------------------------------
    # 10 x S1: trivial, no tool keywords, no boosters, no "?", short
    # Expected heuristic: complexity=0.3, uncertainty=0.2, tool=False -> S1
    # ------------------------------------------------------------------
    {"task": "What is 2+2",                                     "expected": 1},
    {"task": "What is the capital of France",                    "expected": 1},
    {"task": "Translate hello to Spanish",                       "expected": 1},
    {"task": "Name three primary colors",                        "expected": 1},
    {"task": "Convert 100 Celsius to Fahrenheit",                "expected": 1},
    {"task": "Spell the word onomatopoeia",                      "expected": 1},
    {"task": "What day comes after Monday",                      "expected": 1},
    {"task": "Sum of angles in a triangle",                      "expected": 1},
    {"task": "Name the largest planet in our solar system",       "expected": 1},
    {"task": "How many continents are there",                    "expected": 1},

    # ------------------------------------------------------------------
    # 10 x S2: moderate complexity or tool_required, NOT S3
    # Needs complexity in (0.35, 0.7] or tool_required but c <= 0.7 and u <= 0.6
    # ------------------------------------------------------------------
    # "debug" -> c=0.6, no "?" -> u=0.2, no tool kw -> S2
    {"task": "Debug this Python function that calculates fibonacci",                "expected": 2},
    # "fix" -> c=0.6, u=0.2 -> S2
    {"task": "Fix the off-by-one in this binary sort implementation",               "expected": 2},
    # "design" -> c=0.5, u=0.2 -> S2
    {"task": "Design a REST API for user authentication",                           "expected": 2},
    # "error" -> c=0.6, u=0.2 -> S2
    {"task": "Explain why this code throws a null pointer error",                   "expected": 2},
    # "crash" -> c=0.6, u=0.2 -> S2
    {"task": "Identify the root cause of this crash in the parser",                 "expected": 2},
    # "optimize" -> c=0.5, u=0.2 -> S2
    {"task": "Optimize this SQL query for better performance",                      "expected": 2},
    # tool_required via "test" -> c=0.3, u=0.2, tool=True -> S2
    {"task": "Write a unit test for the add function",                              "expected": 2},
    # "architect" -> c=0.5, u=0.2 -> S2
    {"task": "Architect a caching layer for this web service",                      "expected": 2},
    # tool_required via "deploy" -> c=0.3, u=0.2, tool=True -> S2
    {"task": "Deploy the container image to staging",                               "expected": 2},
    # tool_required via "compile" -> c=0.3, u=0.2, tool=True -> S2
    {"task": "Compile the Rust binary and verify the output",                       "expected": 2},

    # ------------------------------------------------------------------
    # 10 x S3: complexity > 0.7 (requires BOTH keyword groups)
    # "debug"/"fix"/"error"/"crash" (+0.3) AND "optimize"/"evolve"/"design"/"architect" (+0.2)
    # -> c = 0.3+0.3+0.2 = 0.8 -> S3
    # ------------------------------------------------------------------
    # fix + design -> c=0.8 -> S3
    {"task": "Fix the memory leak then design a garbage collection strategy",                       "expected": 3},
    # debug + optimize -> c=0.8 -> S3
    {"task": "Debug the race condition and optimize the lock-free queue",                            "expected": 3},
    # error + architect -> c=0.8 -> S3
    {"task": "Resolve the segfault error and architect a safe memory allocator",                     "expected": 3},
    # crash + evolve -> c=0.8 -> S3
    {"task": "Analyze why the system crash occurs and evolve the recovery mechanism",                "expected": 3},
    # debug + design -> c=0.8 -> S3
    {"task": "Debug the consensus protocol and design a Byzantine fault tolerance layer",            "expected": 3},
    # fix + optimize -> c=0.8 -> S3
    {"task": "Fix the numerical instability and optimize the gradient descent convergence",          "expected": 3},
    # error + design -> c=0.8 -> S3
    {"task": "Diagnose the serialization error and design a schema migration strategy",             "expected": 3},
    # crash + architect -> c=0.8 -> S3
    {"task": "Investigate the kernel crash and architect a fault-tolerant microkernel",              "expected": 3},
    # fix + evolve -> c=0.8 -> S3
    {"task": "Fix the deadlock in the scheduler and evolve the priority inversion handling",         "expected": 3},
    # debug + architect -> c=0.8 -> S3
    {"task": "Debug the distributed transaction failure and architect an idempotent retry layer",    "expected": 3},
]
# fmt: on


class RoutingAccuracyBench:
    """Benchmark that measures how accurately the metacognitive router
    classifies tasks into S1 / S2 / S3 against labeled ground truth.
    """

    def __init__(self, metacognition: MetacognitiveController) -> None:
        self.metacognition = metacognition
        self.manifest: BenchmarkManifest | None = None

    async def run(self) -> BenchReport:
        """Evaluate all labeled tasks and return a :class:`BenchReport`."""
        results: list[TaskResult] = []
        self.manifest = BenchmarkManifest(
            benchmark="routing_accuracy", model="heuristic"
        )

        for idx, item in enumerate(LABELED_TASKS):
            task_text: str = item["task"]
            expected_system: int = item["expected"]

            t0 = time.perf_counter()
            profile = self.metacognition.assess_complexity(task_text)
            decision = self.metacognition.route(profile)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            passed = decision.system == expected_system
            error = "" if passed else (
                f"expected S{expected_system}, got S{decision.system}"
            )
            results.append(
                TaskResult(
                    task_id=f"routing_{idx:03d}",
                    passed=passed,
                    system_used=decision.system,
                    latency_ms=elapsed_ms,
                    error=error,
                )
            )
            self.manifest.add(TaskTrace(
                task_id=f"routing_{idx:03d}",
                passed=passed,
                latency_ms=round(elapsed_ms, 3),
                cost_usd=0.0,
                model="heuristic",
                routing=f"S{decision.system}",
                error=error,
            ))

        return BenchReport.from_results("routing_accuracy", results)
