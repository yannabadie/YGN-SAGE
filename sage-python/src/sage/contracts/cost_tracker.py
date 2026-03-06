"""CostTracker — cumulative cost accounting for DAG execution.

Tracks per-node and total spend against an optional budget cap.
When budget_usd is 0 (or negative), tracking is unlimited (never over budget).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostTracker:
    """Track cumulative cost across DAG node executions.

    Parameters
    ----------
    budget_usd:
        Total budget cap.  0 means unlimited (no cap).
    """

    budget_usd: float = 0.0
    _spent: dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, node_id: str, cost_usd: float) -> None:
        """Record cost for a node (additive if called multiple times)."""
        self._spent[node_id] = self._spent.get(node_id, 0.0) + cost_usd

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total_spent(self) -> float:
        return sum(self._spent.values())

    @property
    def remaining(self) -> float:
        if self.budget_usd <= 0:
            return float("inf")
        return max(self.budget_usd - self.total_spent, 0.0)

    @property
    def is_over_budget(self) -> bool:
        if self.budget_usd <= 0:
            return False
        return self.total_spent > self.budget_usd

    def cost_for(self, node_id: str) -> float:
        return self._spent.get(node_id, 0.0)

    def stats(self) -> dict[str, Any]:
        return {
            "total_spent": self.total_spent,
            "budget": self.budget_usd,
            "remaining": self.remaining if self.budget_usd > 0 else None,
            "per_node": dict(self._spent),
        }
