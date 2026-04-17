"""Tests for scripts/decide_next_phase.py gate logic.

Keeps the thresholds in the doc and the code in sync.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from decide_next_phase import decide, GATE_A_THRESHOLD, GATE_B_THRESHOLD


def _ablation(tmp_path: Path, configs: dict[str, float]) -> dict:
    """Build a fake ablation JSON with the given pass rates."""
    results = []
    for name, rate in configs.items():
        report_path = tmp_path / f"report_{name}.json"
        report_path.write_text(json.dumps({"pass_rate": rate}))
        results.append({"config": name, "report_path": str(report_path)})
    return {"results": results}


def test_gate_a_when_full_above_threshold(tmp_path):
    data = _ablation(tmp_path, {"full": 0.40, "bare": 0.10})
    gate, reason, code = decide(data)
    assert gate == "A"
    assert code == 0
    assert "40.0%" in reason


def test_gate_b_when_full_below_low_threshold(tmp_path):
    data = _ablation(tmp_path, {"full": 0.15, "bare": 0.10})
    gate, reason, code = decide(data)
    assert gate == "B"
    assert code == 1
    assert "architecture not the bottleneck" in reason


def test_gate_c_when_in_mid_band(tmp_path):
    data = _ablation(
        tmp_path,
        {
            "full": 0.28,
            "no_toolforge": 0.18,
            "no_sage_recurse": 0.25,
            "bare": 0.10,
        },
    )
    gate, reason, code = decide(data)
    assert gate == "C"
    assert code == 2
    # Biggest drop from no_toolforge (0.10 pp).
    assert "no_toolforge" in reason


def test_missing_full_returns_error(tmp_path):
    data = _ablation(tmp_path, {"bare": 0.10})
    gate, reason, code = decide(data)
    assert gate == "ERROR"
    assert code == 3


def test_empty_results_returns_error():
    gate, _, code = decide({"results": []})
    assert gate == "ERROR"
    assert code == 3


def test_gate_c_works_without_bare(tmp_path):
    """If bare is missing, still give a verdict based on `full` alone."""
    data = _ablation(tmp_path, {"full": 0.28})
    gate, _, code = decide(data)
    assert gate == "C"
    assert code == 2


def test_threshold_constants_match_doc():
    """Guard against silent drift between code and ROADMAP_SPRINT6_DECISION.md."""
    assert GATE_A_THRESHOLD == 0.35
    assert GATE_B_THRESHOLD == 0.20
