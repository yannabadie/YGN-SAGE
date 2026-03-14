"""Tests for ablation statistical analysis."""
import pytest
from sage.bench.ablation import compute_ablation_stats


def test_compute_ablation_stats_structure():
    """Stats output has required fields."""
    results = {
        "full": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        "baseline": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    }
    stats = compute_ablation_stats(results)
    assert "pairwise" in stats
    pair = stats["pairwise"]["full_vs_baseline"]
    assert "mcnemar_p" in pair
    assert "cohens_d" in pair
    assert "bootstrap_ci_95" in pair
    assert len(pair["bootstrap_ci_95"]) == 2


def test_compute_ablation_stats_identical():
    """Identical results produce p=1.0 and d=0."""
    results = {
        "a": [1, 1, 1, 1, 1],
        "b": [1, 1, 1, 1, 1],
    }
    stats = compute_ablation_stats(results)
    pair = stats["pairwise"]["a_vs_b"]
    assert pair["mcnemar_p"] >= 0.99
    assert abs(pair["cohens_d"]) < 0.01
