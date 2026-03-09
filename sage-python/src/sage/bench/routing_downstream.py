"""Downstream quality evaluation for routing.

Metrics:
  1. Tier Precision: success rate per routed tier
  2. Escalation Rate: % of tasks re-routed (<20% target)
  3. Routing Latency: P50/P95/P99 of routing decision time
  4. Avg Quality: mean quality across all tasks
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class TierMetrics:
    """Per-tier downstream metrics."""

    tier: int
    total: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    @property
    def precision(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.total if self.total else 0.0


@dataclass
class DownstreamResult:
    """Aggregated downstream evaluation result."""

    total_tasks: int = 0
    escalations: int = 0
    tier_metrics: dict[int, TierMetrics] = field(default_factory=dict)
    routing_latencies_ms: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.total_tasks if self.total_tasks else 0.0

    @property
    def avg_quality(self) -> float:
        return (
            sum(self.quality_scores) / len(self.quality_scores)
            if self.quality_scores
            else 0.0
        )

    @property
    def routing_p50_ms(self) -> float:
        if not self.routing_latencies_ms:
            return 0.0
        s = sorted(self.routing_latencies_ms)
        return s[len(s) // 2]

    @property
    def routing_p99_ms(self) -> float:
        if not self.routing_latencies_ms:
            return 0.0
        s = sorted(self.routing_latencies_ms)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "escalation_rate": round(self.escalation_rate, 3),
            "avg_quality": round(self.avg_quality, 3),
            "routing_p50_ms": round(self.routing_p50_ms, 2),
            "routing_p99_ms": round(self.routing_p99_ms, 2),
            "tier_precision": {
                t: round(m.precision, 3) for t, m in self.tier_metrics.items()
            },
            "tier_avg_cost": {
                t: round(m.avg_cost_usd, 5) for t, m in self.tier_metrics.items()
            },
        }


class DownstreamEvaluator:
    """Collects routing outcomes and computes downstream quality metrics."""

    def __init__(self) -> None:
        self._result = DownstreamResult()

    def record(
        self,
        tier: int,
        quality: float,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        routing_ms: float = 0.0,
        escalated: bool = False,
    ) -> None:
        """Record a single task routing outcome.

        Args:
            tier: The tier the task was routed to (1, 2, or 3).
            quality: Quality score of the result (0.0-1.0).
            latency_ms: End-to-end latency of task execution in ms.
            cost_usd: Cost of the LLM call in USD.
            routing_ms: Time spent on the routing decision in ms.
            escalated: Whether the task was re-routed / escalated.
        """
        self._result.total_tasks += 1
        self._result.quality_scores.append(quality)
        self._result.routing_latencies_ms.append(routing_ms)
        if escalated:
            self._result.escalations += 1

        if tier not in self._result.tier_metrics:
            self._result.tier_metrics[tier] = TierMetrics(tier=tier)
        tm = self._result.tier_metrics[tier]
        tm.total += 1
        if quality >= 0.5:
            tm.successes += 1
        tm.total_latency_ms += latency_ms
        tm.total_cost_usd += cost_usd

    def result(self) -> DownstreamResult:
        """Return the accumulated downstream result."""
        return self._result
