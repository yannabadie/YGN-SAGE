"""Baseline comparison framework for routing evidence gates.

Provides 4 baseline strategies to compare against the Rust cognitive router:
1. AlwaysBestModel: route every task to the most expensive model
2. CheapestUnderSLA: cheapest model meeting a quality threshold
3. FixedAVR: always use AVR topology with fixed reviewer model
4. RoutingOnOff: A/B paired comparison (router ON vs router OFF)

Evidence gate: Rust router must match or beat "always_best_model" on cost-quality
Pareto front before any Python routing code can be deleted.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sage.bench.routing_downstream import DownstreamEvaluator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BaselineDecision:
    """Decision returned by a baseline strategy."""

    model_id: str
    topology: str  # e.g., "sequential", "avr"
    expected_cost_usd: float
    strategy_name: str


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaselineStrategy(ABC):
    """Abstract base for baseline routing strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def select(self, task_features: dict) -> BaselineDecision:
        """Select a model/topology for a given task.

        Args:
            task_features: Dictionary of task characteristics (ignored by
                most baselines, but available for extensibility).

        Returns:
            A BaselineDecision with model, topology, and expected cost.
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class AlwaysBestModel(BaselineStrategy):
    """Route every task to the most expensive (highest quality) model.

    This is the upper-bound quality baseline.  If the router cannot match
    this quality at lower cost, the router adds no value.

    Args:
        models: List of ``(model_id, cost_per_call)`` tuples.
    """

    def __init__(self, models: list[tuple[str, float]]) -> None:
        if not models:
            raise ValueError("models list must not be empty")
        # Sort descending by cost; most expensive first.
        self._models = sorted(models, key=lambda m: m[1], reverse=True)
        self._best_id, self._best_cost = self._models[0]

    @property
    def name(self) -> str:
        return "always_best_model"

    def select(self, task_features: dict) -> BaselineDecision:
        return BaselineDecision(
            model_id=self._best_id,
            topology="sequential",
            expected_cost_usd=self._best_cost,
            strategy_name=self.name,
        )


class CheapestUnderSLA(BaselineStrategy):
    """Pick the cheapest model whose estimated quality >= threshold.

    If no model meets the SLA, falls back to the cheapest overall.

    Args:
        models: List of ``(model_id, cost_per_call, quality_estimate)`` tuples.
        min_quality: Minimum acceptable quality score (0.0-1.0).
    """

    def __init__(
        self,
        models: list[tuple[str, float, float]],
        min_quality: float = 0.8,
    ) -> None:
        if not models:
            raise ValueError("models list must not be empty")
        self._models = models
        self._min_quality = min_quality

    @property
    def name(self) -> str:
        return "cheapest_under_sla"

    def select(self, task_features: dict) -> BaselineDecision:
        # Filter models that meet the quality SLA.
        qualifying = [
            (mid, cost, qual)
            for mid, cost, qual in self._models
            if qual >= self._min_quality
        ]
        if qualifying:
            # Cheapest among qualifying.
            chosen_id, chosen_cost, _ = min(qualifying, key=lambda m: m[1])
        else:
            # Fallback: cheapest overall.
            chosen_id, chosen_cost, _ = min(self._models, key=lambda m: m[1])
            log.warning(
                "No model meets quality SLA %.2f; falling back to cheapest: %s",
                self._min_quality,
                chosen_id,
            )
        return BaselineDecision(
            model_id=chosen_id,
            topology="sequential",
            expected_cost_usd=chosen_cost,
            strategy_name=self.name,
        )


class FixedAVR(BaselineStrategy):
    """Always use Act-Verify-Refine topology with a fixed reviewer.

    Cost is modeled as 2x the base cost (one generate + one review pass).

    Args:
        reviewer_model_id: The model used for the review pass.
        base_cost_usd: Cost of a single LLM call for the reviewer.
    """

    def __init__(
        self,
        reviewer_model_id: str,
        base_cost_usd: float = 0.01,
    ) -> None:
        self._reviewer = reviewer_model_id
        self._base_cost = base_cost_usd

    @property
    def name(self) -> str:
        return "fixed_avr"

    def select(self, task_features: dict) -> BaselineDecision:
        return BaselineDecision(
            model_id=self._reviewer,
            topology="avr",
            expected_cost_usd=self._base_cost * 2,
            strategy_name=self.name,
        )


# ---------------------------------------------------------------------------
# Baseline comparison orchestrator
# ---------------------------------------------------------------------------

@dataclass
class _StrategyRecord:
    """Internal: accumulated outcomes for one strategy."""

    evaluator: DownstreamEvaluator = field(default_factory=DownstreamEvaluator)
    total_cost_usd: float = 0.0
    total_quality: float = 0.0
    count: int = 0


class BaselineComparison:
    """Orchestrates multiple baseline strategies and the router for comparison.

    Usage::

        comp = BaselineComparison(["always_best_model", "cheapest_under_sla"])
        comp.record("always_best_model", quality=0.95, cost_usd=0.10)
        comp.record("cheapest_under_sla", quality=0.80, cost_usd=0.01)
        comp.record("router", quality=0.92, cost_usd=0.03)
        table = comp.compare()

    Args:
        strategy_names: List of strategy names to track (plus implicit "router").
    """

    ROUTER_KEY = "router"

    def __init__(self, strategy_names: list[str] | None = None) -> None:
        self._records: dict[str, _StrategyRecord] = {}
        # Always include a slot for the router.
        self._records[self.ROUTER_KEY] = _StrategyRecord()
        for sn in strategy_names or []:
            self._records[sn] = _StrategyRecord()

    def record(
        self,
        strategy_name: str,
        quality: float,
        cost_usd: float = 0.0,
        tier: int = 2,
        latency_ms: float = 0.0,
        routing_ms: float = 0.0,
        escalated: bool = False,
    ) -> None:
        """Record a single task outcome for a strategy.

        Args:
            strategy_name: Which strategy produced this outcome.
            quality: Quality score (0.0-1.0).
            cost_usd: Cost of the LLM call(s).
            tier: Routing tier used (default 2).
            latency_ms: End-to-end latency.
            routing_ms: Routing decision latency.
            escalated: Whether task was escalated/re-routed.
        """
        if strategy_name not in self._records:
            self._records[strategy_name] = _StrategyRecord()
        rec = self._records[strategy_name]
        rec.evaluator.record(
            tier=tier,
            quality=quality,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            routing_ms=routing_ms,
            escalated=escalated,
        )
        rec.total_cost_usd += cost_usd
        rec.total_quality += quality
        rec.count += 1

    def compare(self) -> dict[str, dict]:
        """Return a comparison table across all recorded strategies.

        Returns:
            Dict mapping strategy name to
            ``{"avg_quality", "avg_cost", "count", "pareto_dominates_router"}``.
        """
        table: dict[str, dict] = {}
        router_avg_q, router_avg_c = self._avg("router")

        for sn, rec in self._records.items():
            avg_q = rec.total_quality / rec.count if rec.count else 0.0
            avg_c = rec.total_cost_usd / rec.count if rec.count else 0.0
            # A strategy Pareto-dominates the router if it has
            # >= quality AND <= cost, with at least one strict inequality.
            dominates = False
            if sn != self.ROUTER_KEY and rec.count > 0 and router_avg_q is not None:
                dominates = (
                    avg_q >= router_avg_q
                    and avg_c <= router_avg_c
                    and (avg_q > router_avg_q or avg_c < router_avg_c)
                )
            table[sn] = {
                "avg_quality": round(avg_q, 4),
                "avg_cost": round(avg_c, 6),
                "count": rec.count,
                "pareto_dominates_router": dominates,
            }
        return table

    def router_beats_best_model(self) -> bool:
        """Evidence gate: does the router Pareto-dominate 'always_best_model'?

        Returns True when:
        - Router quality >= best_model quality, AND
        - Router cost < best_model cost.

        Both strategies must have at least one recorded outcome.
        """
        best_key = "always_best_model"
        if best_key not in self._records:
            log.warning("No '%s' baseline recorded; evidence gate fails.", best_key)
            return False
        r_q, r_c = self._avg(self.ROUTER_KEY)
        b_q, b_c = self._avg(best_key)
        if r_q is None or b_q is None:
            return False
        return r_q >= b_q and r_c < b_c

    # -- helpers --

    def _avg(self, key: str) -> tuple[float | None, float]:
        """Return (avg_quality, avg_cost) for a strategy, or (None, 0) if empty."""
        rec = self._records.get(key)
        if rec is None or rec.count == 0:
            return None, 0.0
        return rec.total_quality / rec.count, rec.total_cost_usd / rec.count


# ---------------------------------------------------------------------------
# A/B paired comparison: router ON vs OFF
# ---------------------------------------------------------------------------

@dataclass
class _PairOutcome:
    """Single paired observation."""

    task_id: str
    on_quality: float
    on_cost: float
    off_quality: float
    off_cost: float


class RoutingOnOff:
    """A/B paired comparison of routing ON vs routing OFF.

    Each task is executed twice (once with routing, once without), and the
    outcomes are compared pairwise.

    Usage::

        ab = RoutingOnOff()
        ab.record_pair("task_1", on_quality=0.9, on_cost=0.03,
                        off_quality=0.8, off_cost=0.10)
        print(ab.summary())
    """

    def __init__(self) -> None:
        self._pairs: list[_PairOutcome] = []

    def record_pair(
        self,
        task_id: str,
        on_quality: float,
        on_cost: float,
        off_quality: float,
        off_cost: float,
    ) -> None:
        """Record a paired (ON, OFF) observation for one task.

        Args:
            task_id: Unique identifier for this task.
            on_quality: Quality with routing enabled.
            on_cost: Cost with routing enabled.
            off_quality: Quality with routing disabled.
            off_cost: Cost with routing disabled.
        """
        self._pairs.append(
            _PairOutcome(
                task_id=task_id,
                on_quality=on_quality,
                on_cost=on_cost,
                off_quality=off_quality,
                off_cost=off_cost,
            )
        )

    def summary(self) -> dict:
        """Return aggregated A/B comparison summary.

        Returns:
            Dict with keys: pairs, on_wins, off_wins, ties,
            on_avg_quality, off_avg_quality, on_avg_cost, off_avg_cost.
        """
        n = len(self._pairs)
        if n == 0:
            return {
                "pairs": 0,
                "on_wins": 0,
                "off_wins": 0,
                "ties": 0,
                "on_avg_quality": 0.0,
                "off_avg_quality": 0.0,
                "on_avg_cost": 0.0,
                "off_avg_cost": 0.0,
            }

        on_wins = 0
        off_wins = 0
        ties = 0
        for p in self._pairs:
            if p.on_quality > p.off_quality:
                on_wins += 1
            elif p.off_quality > p.on_quality:
                off_wins += 1
            else:
                ties += 1

        return {
            "pairs": n,
            "on_wins": on_wins,
            "off_wins": off_wins,
            "ties": ties,
            "on_avg_quality": round(
                sum(p.on_quality for p in self._pairs) / n, 4
            ),
            "off_avg_quality": round(
                sum(p.off_quality for p in self._pairs) / n, 4
            ),
            "on_avg_cost": round(
                sum(p.on_cost for p in self._pairs) / n, 6
            ),
            "off_avg_cost": round(
                sum(p.off_cost for p in self._pairs) / n, 6
            ),
        }
