"""Non-circular routing ground truth benchmark.

Labels created by human expert (not reverse-engineered from heuristic).
Measures how well the router assigns tasks to the correct cognitive system.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

GT_PATH = Path(__file__).parent.parent.parent.parent / "config" / "routing_ground_truth.json"


@dataclass
class RoutingGTResult:
    total: int = 0
    correct: int = 0
    per_system: dict[int, dict[str, int]] = field(default_factory=dict)
    misroutes: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def run_routing_gt(router, gt_path: Path | None = None, verbose: bool = False) -> RoutingGTResult:
    """Run routing ground truth benchmark.

    Parameters
    ----------
    router:
        ComplexityRouter (assess_complexity + route), AdaptiveRouter,
        or Rust SystemRouter (route(task, budget) → RoutingDecision).
    gt_path:
        Path to ground truth JSON. Defaults to config/routing_ground_truth.json.
    """
    path = gt_path or GT_PATH
    with open(path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    tasks = gt_data["tasks"]
    result = RoutingGTResult(total=len(tasks))

    start = time.perf_counter()
    for task_entry in tasks:
        task_text = task_entry["task"]
        expected = task_entry["expected_system"]

        try:
            if hasattr(router, "assess_complexity") and hasattr(router, "route"):
                # ComplexityRouter: assess_complexity(task) → CognitiveProfile,
                # then route(profile) → RoutingDecision with .system
                profile = router.assess_complexity(task_text)
                decision = router.route(profile)
                actual = int(decision.system)
            elif hasattr(router, "route"):
                # Rust SystemRouter: route(task, budget) → RoutingDecision
                decision = router.route(task_text, 10.0)
                actual = int(decision.system)
            else:
                _log.warning("Router has no known interface, skipping")
                result.total -= 1
                continue
        except Exception as exc:
            _log.warning("Router failed on task %d: %s", task_entry["id"], exc)
            actual = -1

        # Track per-system stats
        for sys in [1, 2, 3]:
            if sys not in result.per_system:
                result.per_system[sys] = {"total": 0, "correct": 0}

        result.per_system[expected]["total"] += 1
        if actual == expected:
            result.correct += 1
            result.per_system[expected]["correct"] += 1
        else:
            result.misroutes.append({
                "id": task_entry["id"],
                "task": task_text[:80],
                "expected": expected,
                "actual": actual,
                "domain": task_entry.get("domain", ""),
            })

        if verbose:
            status = "OK" if actual == expected else f"MISS (got S{actual})"
            print(f"  [{task_entry['id']:2d}] S{expected} {status}: {task_text[:60]}")

    result.elapsed_ms = (time.perf_counter() - start) * 1000
    return result
