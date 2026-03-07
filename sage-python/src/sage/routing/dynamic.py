"""DynamicRouter — round-level routing inside verified capability envelope.

Selects the optimal provider for each TaskNode based on:
1. Capability requirements (hard constraint — filter)
2. Cost sensitivity vs quality (soft scoring)
3. Budget constraints (hard cap)
4. Historical performance feedback (adaptive penalty/bonus)

Inspired by DyTopo (2602.06039): round-level routing decisions
constrained by a verified envelope (CapabilityMatrix).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from sage.contracts.task_node import TaskNode
from sage.providers.capabilities import CapabilityMatrix

log = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    provider: str
    score: float
    reason: str


class DynamicRouter:
    """Selects provider per TaskNode with capability constraints + feedback.

    Parameters
    ----------
    capability_matrix:
        Registry of provider capabilities (hard filter).
    provider_costs:
        Provider name -> relative cost per 1M tokens (normalized).
    provider_quality:
        Provider name -> quality score 0.0-1.0 (from benchmarks/profiling).
    """

    def __init__(
        self,
        capability_matrix: CapabilityMatrix,
        provider_costs: dict[str, float] | None = None,
        provider_quality: dict[str, float] | None = None,
    ) -> None:
        self.capability_matrix = capability_matrix
        self.provider_costs = provider_costs or {}
        self.provider_quality = provider_quality or {}
        # Feedback tracking: provider -> list of (success, latency_ms)
        self._feedback: dict[str, list[tuple[bool, float]]] = defaultdict(list)

    def route(
        self,
        node: TaskNode,
        cost_sensitivity: float = 0.5,
    ) -> RoutingDecision:
        """Select the best provider for this TaskNode.

        Parameters
        ----------
        node:
            The task node to route.
        cost_sensitivity:
            0.0 = pure quality, 1.0 = pure cost optimization.
        """
        # 1. Hard filter: capability requirements
        candidates = self._filter_by_capabilities(node)
        if not candidates:
            caps = node.capabilities_required
            raise ValueError(
                f"No provider supports required capabilities: {caps}"
            )

        # 2. Hard filter: budget constraint
        if node.budget.max_cost_usd > 0:
            candidates = [
                p for p in candidates
                if self.provider_costs.get(p, 0) <= node.budget.max_cost_usd * 100
            ] or candidates  # Fallback to all if budget filter too strict

        # 3. Score each candidate
        scored: list[tuple[str, float]] = []
        for provider in candidates:
            score = self._score_provider(provider, cost_sensitivity)
            scored.append((provider, score))

        if not scored:
            caps = node.capabilities_required
            raise ValueError(
                f"No provider scored for task (capabilities: {caps})"
            )

        # 4. Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        best_provider, best_score = scored[0]

        reason = (
            f"Selected {best_provider} (score={best_score:.3f}, "
            f"cost_sensitivity={cost_sensitivity:.1f})"
        )
        log.debug(reason)

        return RoutingDecision(
            provider=best_provider,
            score=best_score,
            reason=reason,
        )

    def report_outcome(
        self,
        provider: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Report execution outcome for adaptive routing."""
        self._feedback[provider].append((success, latency_ms))

    def _filter_by_capabilities(self, node: TaskNode) -> list[str]:
        """Return providers that satisfy all required capabilities."""
        if not node.capabilities_required:
            # No specific requirements — all providers are candidates
            return list(self.provider_costs.keys()) or list(
                self.provider_quality.keys()
            )

        # Build requirements dict for CapabilityMatrix
        requirements = {cap: True for cap in node.capabilities_required}
        try:
            return self.capability_matrix.require(**requirements)
        except ValueError:
            return []

    def _score_provider(
        self,
        provider: str,
        cost_sensitivity: float,
    ) -> float:
        """Score a provider based on quality, cost, and feedback."""
        quality = self.provider_quality.get(provider, 0.5)
        cost = self.provider_costs.get(provider, 1.0)

        # Normalize cost to 0-1 scale (lower cost = higher score)
        max_cost = max(self.provider_costs.values()) if self.provider_costs else 1.0
        cost_score = 1.0 - (cost / max_cost) if max_cost > 0 else 0.5

        # Blend quality and cost
        base_score = (1 - cost_sensitivity) * quality + cost_sensitivity * cost_score

        # Apply feedback penalty/bonus
        feedback_modifier = self._feedback_modifier(provider)

        return base_score + feedback_modifier

    def _feedback_modifier(self, provider: str) -> float:
        """Calculate feedback-based score modifier.

        Each failure penalizes by -0.05 (up to -0.3).
        Each success gives +0.01 (up to +0.1).
        """
        history = self._feedback.get(provider, [])
        if not history:
            return 0.0

        # Use recent history (last 20 entries)
        recent = history[-20:]
        successes = sum(1 for s, _ in recent if s)
        failures = sum(1 for s, _ in recent if not s)

        penalty = min(failures * 0.05, 0.3)
        bonus = min(successes * 0.01, 0.1)

        return bonus - penalty
