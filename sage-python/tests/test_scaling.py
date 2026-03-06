"""Tests for the empirical scaling law analyzer (sage.analytics.scaling)."""

import pytest

from sage.analytics.scaling import RunRecord, ScalingAnalyzer


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

def test_run_record_creation():
    r = RunRecord(
        task_type="coding",
        model_id="gemini-3-flash",
        topology_type="sequential",
        quality_score=0.85,
        cost_usd=0.003,
        latency_ms=1200.0,
    )
    assert r.task_type == "coding"
    assert r.model_id == "gemini-3-flash"
    assert r.topology_type == "sequential"
    assert r.quality_score == 0.85
    assert r.cost_usd == 0.003
    assert r.latency_ms == 1200.0


def test_run_record_equality():
    r1 = RunRecord("a", "m1", "seq", 0.5, 0.01, 100.0)
    r2 = RunRecord("a", "m1", "seq", 0.5, 0.01, 100.0)
    assert r1 == r2


# ---------------------------------------------------------------------------
# ScalingAnalyzer — add + basic state
# ---------------------------------------------------------------------------

def test_analyzer_starts_empty():
    sa = ScalingAnalyzer()
    assert len(sa._records) == 0


def test_analyzer_add_records():
    sa = ScalingAnalyzer()
    sa.add(RunRecord("coding", "m1", "seq", 0.8, 0.01, 100.0))
    sa.add(RunRecord("coding", "m2", "par", 0.9, 0.02, 200.0))
    assert len(sa._records) == 2


# ---------------------------------------------------------------------------
# ScalingAnalyzer — insufficient data
# ---------------------------------------------------------------------------

def test_analyze_insufficient_data_zero():
    sa = ScalingAnalyzer()
    result = sa.analyze()
    assert result["status"] == "insufficient_data"
    assert result["records"] == 0


def test_analyze_insufficient_data_nine():
    sa = ScalingAnalyzer()
    for i in range(9):
        sa.add(RunRecord("coding", f"m{i}", "seq", 0.5, 0.01, 100.0))
    result = sa.analyze()
    assert result["status"] == "insufficient_data"
    assert result["records"] == 9


def test_analyze_sufficient_at_ten():
    sa = ScalingAnalyzer()
    for i in range(10):
        sa.add(RunRecord("coding", f"m{i % 2}", f"topo{i % 3}", float(i) / 10, 0.01, 100.0))
    result = sa.analyze()
    assert result["status"] == "analyzed"
    assert result["records"] == 10


# ---------------------------------------------------------------------------
# ScalingAnalyzer — variance computation
# ---------------------------------------------------------------------------

def test_variance_across_groups_basic():
    sa = ScalingAnalyzer()
    groups = {"a": [1.0, 1.0], "b": [3.0, 3.0]}
    # means: a=1.0, b=3.0, overall_mean=2.0
    # variance = ((1-2)^2 + (3-2)^2) / 2 = 1.0
    assert sa._variance_across_groups(groups) == 1.0


def test_variance_single_group():
    sa = ScalingAnalyzer()
    groups = {"a": [1.0, 2.0]}
    assert sa._variance_across_groups(groups) == 0.0


def test_variance_identical_groups():
    sa = ScalingAnalyzer()
    groups = {"a": [5.0, 5.0], "b": [5.0, 5.0]}
    assert sa._variance_across_groups(groups) == 0.0


# ---------------------------------------------------------------------------
# ScalingAnalyzer — topology dominates
# ---------------------------------------------------------------------------

def test_topology_dominates_true():
    """When topology variation causes more quality spread than model variation."""
    sa = ScalingAnalyzer()
    # Model m1 and m2 have similar scores across topologies
    # But topology "seq" and "par" have very different scores
    for _ in range(5):
        sa.add(RunRecord("coding", "m1", "seq", 0.9, 0.01, 100.0))
        sa.add(RunRecord("coding", "m2", "seq", 0.85, 0.01, 100.0))
    for _ in range(5):
        sa.add(RunRecord("coding", "m1", "par", 0.3, 0.01, 100.0))
        sa.add(RunRecord("coding", "m2", "par", 0.25, 0.01, 100.0))
    result = sa.analyze()
    assert result["status"] == "analyzed"
    assert result["topology_dominates"] is True
    assert "TOPOLOGY" in result["recommendation"]


def test_model_dominates():
    """When model variation causes more quality spread than topology variation."""
    sa = ScalingAnalyzer()
    # Model "strong" always scores high, "weak" always scores low
    # regardless of topology
    for _ in range(5):
        sa.add(RunRecord("coding", "strong", "seq", 0.95, 0.01, 100.0))
        sa.add(RunRecord("coding", "strong", "par", 0.90, 0.01, 100.0))
    for _ in range(5):
        sa.add(RunRecord("coding", "weak", "seq", 0.2, 0.01, 100.0))
        sa.add(RunRecord("coding", "weak", "par", 0.15, 0.01, 100.0))
    result = sa.analyze()
    assert result["status"] == "analyzed"
    assert result["topology_dominates"] is False
    assert "MODEL" in result["recommendation"]


# ---------------------------------------------------------------------------
# ScalingAnalyzer — edge cases
# ---------------------------------------------------------------------------

def test_analyze_all_same_scores():
    sa = ScalingAnalyzer()
    for i in range(10):
        sa.add(RunRecord("coding", f"m{i % 2}", f"t{i % 2}", 0.5, 0.01, 100.0))
    result = sa.analyze()
    assert result["status"] == "analyzed"
    assert result["model_variance"] == 0.0
    assert result["topology_variance"] == 0.0
    assert result["topology_dominates"] is False


def test_analyze_result_keys():
    sa = ScalingAnalyzer()
    for i in range(12):
        sa.add(RunRecord("coding", f"m{i % 3}", f"t{i % 2}", float(i) / 12, 0.01, 100.0))
    result = sa.analyze()
    expected_keys = {"status", "records", "model_variance", "topology_variance",
                     "topology_dominates", "recommendation"}
    assert set(result.keys()) == expected_keys


def test_analyze_variances_are_rounded():
    sa = ScalingAnalyzer()
    for i in range(10):
        sa.add(RunRecord("coding", f"m{i % 2}", f"t{i % 3}", float(i) / 10, 0.01, 100.0))
    result = sa.analyze()
    # Check that variances are rounded to 4 decimal places
    mv_str = str(result["model_variance"])
    tv_str = str(result["topology_variance"])
    if "." in mv_str:
        assert len(mv_str.split(".")[1]) <= 4
    if "." in tv_str:
        assert len(tv_str.split(".")[1]) <= 4
