"""Drift monitor -- detects behavioral degradation in agents.

Analyzes a window of AgentEvent objects and computes a composite drift score
from three signals:

  - **Latency trend** (40% weight): ratio of second-half to first-half mean
    latencies.  A 3x increase maps to 1.0 (via ``(ratio - 1) / 2``).
  - **Error rate** (40% weight): proportion of events carrying an ``error``
    key in their ``meta`` dict.
  - **Cost trend** (20% weight): same half-comparison as latency, but more
    tolerant (ratio of 6x maps to 1.0, via ``(ratio - 1) / 5``).

The composite score is ``max(weighted_average, max_signal * 0.85)`` so that
a single catastrophic signal (e.g. 100% error rate) is never masked by the
other two being healthy.

The monitor recommends one of three actions based on drift severity:

  - ``CONTINUE`` -- drift < 0.4, system is healthy.
  - ``SWITCH_MODEL`` -- 0.4 <= drift <= 0.7, try a different LLM tier.
  - ``RESET_AGENT`` -- drift > 0.7, full agent reset recommended.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Result of a drift analysis."""

    drift_score: float  # 0.0 (stable) to 1.0 (severe drift)
    action: str  # CONTINUE | SWITCH_MODEL | RESET_AGENT
    details: dict[str, float] | None = None


class DriftMonitor:
    """Analyzes event patterns to detect agent behavioral drift.

    Usage::

        from sage.monitoring.drift import DriftMonitor
        from sage.agent_loop import AgentEvent

        monitor = DriftMonitor()
        report = monitor.analyze(recent_events)
        if report.action != "CONTINUE":
            log.warning("Drift detected: %s", report)
    """

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(self, events: list[Any]) -> DriftReport:
        """Return a *DriftReport* summarising behavioral drift.

        Parameters
        ----------
        events:
            A list of :class:`~sage.agent_loop.AgentEvent` (or compatible)
            objects.  At least 2 events are needed for a meaningful score;
            otherwise drift is reported as 0.0.
        """
        if not events or len(events) < 2:
            return DriftReport(drift_score=0.0, action="CONTINUE")

        latency_drift = self._latency_trend(events)
        error_rate = self._error_rate(events)
        cost_drift = self._cost_trend(events)

        weighted = latency_drift * 0.4 + error_rate * 0.4 + cost_drift * 0.2
        # Ensure a single catastrophic signal is never masked:
        max_signal = max(latency_drift, error_rate, cost_drift)
        drift = max(weighted, max_signal * 0.85)
        drift = min(1.0, max(0.0, drift))

        if drift > 0.7:
            action = "RESET_AGENT"
        elif drift > 0.4:
            action = "SWITCH_MODEL"
        else:
            action = "CONTINUE"

        report = DriftReport(
            drift_score=round(drift, 3),
            action=action,
            details={
                "latency": round(latency_drift, 4),
                "errors": round(error_rate, 4),
                "cost": round(cost_drift, 4),
            },
        )
        log.debug("DriftMonitor: score=%.3f action=%s", report.drift_score, action)
        return report

    # ------------------------------------------------------------------ #
    # Private signal extractors                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _latency_trend(events: list[Any]) -> float:
        """Compare mean latency of first half vs second half.

        A 3x increase maps to 1.0 (via ``(ratio - 1) / 2``).
        """
        latencies = [e.latency_ms for e in events if e.latency_ms and e.latency_ms > 0]
        if len(latencies) < 3:
            return 0.0
        mid = len(latencies) // 2
        first_half = sum(latencies[:mid]) / max(mid, 1)
        second_half = sum(latencies[mid:]) / max(len(latencies) - mid, 1)
        if first_half <= 0:
            return 0.0
        ratio = second_half / first_half
        return min(1.0, max(0.0, (ratio - 1.0) / 2.0))

    @staticmethod
    def _error_rate(events: list[Any]) -> float:
        """Proportion of events that carry an ``error`` key in ``meta``."""
        errors = sum(1 for e in events if e.meta.get("error"))
        return min(1.0, errors / max(len(events), 1))

    @staticmethod
    def _cost_trend(events: list[Any]) -> float:
        """Compare mean cost of first half vs second half.

        A 6x increase maps to 1.0 (via ``(ratio - 1) / 5``).
        """
        costs = [e.cost_usd for e in events if e.cost_usd and e.cost_usd > 0]
        if len(costs) < 3:
            return 0.0
        mid = len(costs) // 2
        first_half = sum(costs[:mid]) / max(mid, 1)
        second_half = sum(costs[mid:]) / max(len(costs) - mid, 1)
        if first_half <= 0:
            return 0.0
        ratio = second_half / first_half
        return min(1.0, max(0.0, (ratio - 1.0) / 5.0))
