"""CostTracker — cumulative cost accounting for DAG execution.

Tracks per-node and total spend against an optional budget cap.
When budget_usd is 0 (or negative), tracking is unlimited (never over budget).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BudgetUpdateResult:
    """Outcome of a ``tighten_remaining_budget`` call.

    ``accepted`` is True when the new cap was applied. ``reason`` is a
    short slug suitable for a ``failure.payload.error_type`` field on
    rejection (``budget_loosen_rejected`` / ``budget_invalid_value``)
    or a ``budget.payload.kind`` slug on acceptance (``budget_tightened``).
    The four numeric fields reflect the tracker's state AFTER the call —
    on rejection they're identical to the pre-call snapshot.
    """

    accepted: bool
    reason: str
    budget_usd: float
    remaining: float
    total_spent: float


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
    _spend_seq: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, node_id: str, cost_usd: float) -> None:
        """Record cost for a node (additive if called multiple times).

        Negative costs are clamped to zero to prevent corrupted Pareto
        frontier analysis from provider billing glitches.
        """
        cost_usd = max(0.0, cost_usd)
        self._spent[node_id] = self._spent.get(node_id, 0.0) + cost_usd

    def record_spend(self, cost_usd: float) -> None:
        """Record an unkeyed spend event."""
        node_id = f"spend-{self._spend_seq}"
        self._spend_seq += 1
        self.record(node_id, cost_usd)

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
        # Use small epsilon to avoid floating-point edge cases
        return self.total_spent > self.budget_usd + 1e-9

    def cost_for(self, node_id: str) -> float:
        return self._spent.get(node_id, 0.0)

    def stats(self) -> dict[str, Any]:
        return {
            "total_spent": self.total_spent,
            "budget": self.budget_usd,
            "remaining": self.remaining if self.budget_usd > 0 else None,
            "per_node": dict(self._spent),
        }

    # ------------------------------------------------------------------
    # Mid-run mutation (CLI ``set_budget`` v0 — tightening only)
    # ------------------------------------------------------------------

    def tighten_remaining_budget(
        self, new_remaining_usd: float
    ) -> BudgetUpdateResult:
        """Tighten the remaining budget cap. NEVER loosens.

        ``new_remaining_usd`` is the new REMAINING-budget cap (NOT the
        absolute total cap). Per ``docs/contracts/SAGE_CLI_PROTOCOL.md``
        invariant 7 (timeout/budget enforcement, attacker-can't-loosen):

          - ``<= 0``, NaN, or ±inf values are REJECTED
            (``budget_invalid_value``). Zero is rejected because
            ``budget_usd <= 0`` is the unlimited sentinel — accepting
            zero from an unlimited tracker would silently keep the
            tracker unlimited.
          - When the tracker is currently unlimited (``budget_usd <= 0``),
            any positive finite value is accepted as a tightening
            from infinite remaining.
          - When the tracker has a finite cap, the new value MUST be
            ``<= self.remaining``. Otherwise REJECTED
            (``budget_loosen_rejected``).
          - On accept, ``budget_usd`` is set to ``total_spent + new_remaining_usd``
            so the cap is anchored at the current spend (the new
            ``remaining`` exactly equals ``new_remaining_usd``).
        """
        # Per cgpro Stage B VERIFY round-2 (2026-05-07): reject ``<= 0.0``
        # because ``budget_usd <= 0`` is the unlimited sentinel; accepting
        # zero from an unlimited tracker would silently keep the tracker
        # unlimited, violating the tighten-only invariant rather than
        # enforcing it. ``new_remaining_usd`` MUST be a positive finite
        # number.
        if not math.isfinite(new_remaining_usd) or new_remaining_usd <= 0.0:
            return BudgetUpdateResult(
                accepted=False,
                reason="budget_invalid_value",
                budget_usd=self.budget_usd,
                remaining=self.remaining,
                total_spent=self.total_spent,
            )
        # Loosen check — only meaningful when the tracker has a finite cap.
        # When unlimited (budget_usd <= 0), any finite value is a tightening
        # from infinity.
        if self.budget_usd > 0 and new_remaining_usd > self.remaining + 1e-9:
            return BudgetUpdateResult(
                accepted=False,
                reason="budget_loosen_rejected",
                budget_usd=self.budget_usd,
                remaining=self.remaining,
                total_spent=self.total_spent,
            )
        self.budget_usd = self.total_spent + new_remaining_usd
        return BudgetUpdateResult(
            accepted=True,
            reason="budget_tightened",
            budget_usd=self.budget_usd,
            remaining=self.remaining,
            total_spent=self.total_spent,
        )
