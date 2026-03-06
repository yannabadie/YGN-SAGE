"""Tests for DriftMonitor — behavioral degradation detection."""
from __future__ import annotations

import time

import pytest

from sage.agent_loop import AgentEvent
from sage.monitoring.drift import DriftMonitor, DriftReport


def _evt(
    type_: str = "THINK",
    step: int = 1,
    latency: float = 100.0,
    cost: float = 0.001,
    error: bool = False,
) -> AgentEvent:
    return AgentEvent(
        type=type_,
        step=step,
        timestamp=time.time(),
        latency_ms=latency,
        cost_usd=cost,
        meta={"error": "fail"} if error else {},
    )


# --- Stable events: low drift ------------------------------------------

def test_no_drift_on_stable_events():
    dm = DriftMonitor()
    events = [_evt(step=i, latency=100 + i) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score < 0.3
    assert report.action == "CONTINUE"


# --- Escalating latency: high drift ------------------------------------

def test_high_drift_on_escalating_latency():
    dm = DriftMonitor()
    events = [_evt(step=i, latency=100 * (i + 1)) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.5


# --- Repeated errors: high drift ---------------------------------------

def test_drift_on_repeated_errors():
    dm = DriftMonitor()
    events = [_evt(step=i, error=True) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.7
    assert report.action in ("SWITCH_MODEL", "RESET_AGENT")


# --- Escalating cost: medium drift -------------------------------------

def test_drift_on_escalating_cost():
    dm = DriftMonitor()
    events = [_evt(step=i, cost=0.001 * (2 ** i)) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.4


# --- Empty events: zero drift ------------------------------------------

def test_empty_events():
    dm = DriftMonitor()
    report = dm.analyze([])
    assert report.drift_score == 0.0
    assert report.action == "CONTINUE"


# --- DriftReport dataclass check ---------------------------------------

def test_drift_report_has_details():
    dm = DriftMonitor()
    events = [_evt(step=i) for i in range(10)]
    report = dm.analyze(events)
    assert isinstance(report, DriftReport)
    assert report.details is not None
    assert "latency" in report.details
    assert "errors" in report.details
    assert "cost" in report.details


# --- Action thresholds --------------------------------------------------

def test_continue_action_below_0_4():
    dm = DriftMonitor()
    # Slight increase, no errors -> low drift -> CONTINUE
    events = [_evt(step=i, latency=100 + i * 5) for i in range(10)]
    report = dm.analyze(events)
    assert report.action == "CONTINUE"


def test_reset_agent_action_above_0_7():
    dm = DriftMonitor()
    # All errors (0.4 * 1.0 = 0.4 from errors alone) + escalating latency
    events = [_evt(step=i, latency=100 * (i + 1), error=True) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.7
    assert report.action == "RESET_AGENT"
