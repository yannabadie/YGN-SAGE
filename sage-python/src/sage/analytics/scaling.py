"""Empirical scaling law -- when does topology beat model selection?"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RunRecord:
    """A single run observation for scaling analysis."""

    task_type: str
    model_id: str
    topology_type: str
    quality_score: float
    cost_usd: float
    latency_ms: float


class ScalingAnalyzer:
    """Collects run data and derives empirical scaling insights.

    The core question: does changing **topology** or changing **model**
    produce more variance in quality?  If topology variance dominates,
    the system should invest in topology search; otherwise, pick a
    stronger model.
    """

    def __init__(self) -> None:
        self._records: list[RunRecord] = []

    def add(self, record: RunRecord) -> None:
        """Append a run record."""
        self._records.append(record)

    def analyze(self) -> dict[str, Any]:
        """Derive when topology change > model change in quality impact.

        Returns ``{"status": "insufficient_data", ...}`` when fewer than
        10 records have been collected.  Otherwise returns model vs
        topology variance, a dominance boolean, and a recommendation.
        """
        if len(self._records) < 10:
            return {"status": "insufficient_data", "records": len(self._records)}

        # Group quality scores by model and by topology
        by_model: dict[str, list[float]] = {}
        by_topology: dict[str, list[float]] = {}
        for r in self._records:
            by_model.setdefault(r.model_id, []).append(r.quality_score)
            by_topology.setdefault(r.topology_type, []).append(r.quality_score)

        model_variance = self._variance_across_groups(by_model)
        topology_variance = self._variance_across_groups(by_topology)

        return {
            "status": "analyzed",
            "records": len(self._records),
            "model_variance": round(model_variance, 4),
            "topology_variance": round(topology_variance, 4),
            "topology_dominates": topology_variance > model_variance,
            "recommendation": (
                "Optimize TOPOLOGY (structure matters more than model choice)"
                if topology_variance > model_variance
                else "Optimize MODEL (model quality matters more than structure)"
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _variance_across_groups(self, groups: dict[str, list[float]]) -> float:
        """Variance of group means (mean-of-means approach).

        With fewer than 2 groups there is no between-group variance.
        """
        if len(groups) < 2:
            return 0.0
        means = [sum(v) / len(v) for v in groups.values() if v]
        if not means:
            return 0.0
        overall_mean = sum(means) / len(means)
        return sum((m - overall_mean) ** 2 for m in means) / len(means)
