"""Tests for baseline comparison framework (Task 9: evidence gates)."""
from __future__ import annotations

import pytest

from sage.bench.baseline_comparison import (
    AlwaysBestModel,
    BaselineComparison,
    BaselineDecision,
    CheapestUnderSLA,
    FixedAVR,
    RoutingOnOff,
)


# ---------------------------------------------------------------------------
# AlwaysBestModel
# ---------------------------------------------------------------------------

class TestAlwaysBestModel:
    """AlwaysBestModel always routes to the most expensive model."""

    def test_picks_most_expensive(self) -> None:
        models = [
            ("gemini-flash", 0.001),
            ("gpt-5.3-codex", 0.10),
            ("gemini-pro", 0.05),
        ]
        strategy = AlwaysBestModel(models)
        decision = strategy.select({})
        assert decision.model_id == "gpt-5.3-codex"
        assert decision.expected_cost_usd == 0.10

    def test_topology_is_sequential(self) -> None:
        strategy = AlwaysBestModel([("model-a", 0.05)])
        decision = strategy.select({"task": "hello"})
        assert decision.topology == "sequential"

    def test_strategy_name(self) -> None:
        strategy = AlwaysBestModel([("model-a", 0.05)])
        assert strategy.name == "always_best_model"
        decision = strategy.select({})
        assert decision.strategy_name == "always_best_model"

    def test_single_model(self) -> None:
        strategy = AlwaysBestModel([("only-model", 0.02)])
        decision = strategy.select({})
        assert decision.model_id == "only-model"
        assert decision.expected_cost_usd == 0.02

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            AlwaysBestModel([])

    def test_ignores_task_features(self) -> None:
        """Decision is the same regardless of task features."""
        strategy = AlwaysBestModel([("cheap", 0.001), ("expensive", 0.50)])
        d1 = strategy.select({"complexity": "low"})
        d2 = strategy.select({"complexity": "high", "code": True})
        assert d1.model_id == d2.model_id == "expensive"


# ---------------------------------------------------------------------------
# CheapestUnderSLA
# ---------------------------------------------------------------------------

class TestCheapestUnderSLA:
    """CheapestUnderSLA picks cheapest model meeting quality threshold."""

    def test_picks_cheapest_above_threshold(self) -> None:
        models = [
            ("expensive-good", 0.10, 0.95),
            ("cheap-good", 0.01, 0.85),
            ("cheap-bad", 0.005, 0.50),
        ]
        strategy = CheapestUnderSLA(models, min_quality=0.80)
        decision = strategy.select({})
        assert decision.model_id == "cheap-good"
        assert decision.expected_cost_usd == 0.01

    def test_falls_back_to_cheapest_if_none_qualify(self) -> None:
        models = [
            ("model-a", 0.10, 0.50),
            ("model-b", 0.05, 0.40),
            ("model-c", 0.02, 0.30),
        ]
        strategy = CheapestUnderSLA(models, min_quality=0.90)
        decision = strategy.select({})
        # None meet 0.90, so falls back to cheapest overall.
        assert decision.model_id == "model-c"
        assert decision.expected_cost_usd == 0.02

    def test_exact_threshold_qualifies(self) -> None:
        """A model at exactly min_quality should qualify."""
        models = [("borderline", 0.03, 0.80)]
        strategy = CheapestUnderSLA(models, min_quality=0.80)
        decision = strategy.select({})
        assert decision.model_id == "borderline"

    def test_strategy_name(self) -> None:
        strategy = CheapestUnderSLA([("m", 0.01, 0.9)])
        assert strategy.name == "cheapest_under_sla"

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CheapestUnderSLA([])

    def test_topology_is_sequential(self) -> None:
        strategy = CheapestUnderSLA([("m", 0.01, 0.9)])
        decision = strategy.select({})
        assert decision.topology == "sequential"


# ---------------------------------------------------------------------------
# FixedAVR
# ---------------------------------------------------------------------------

class TestFixedAVR:
    """FixedAVR always returns AVR topology with fixed reviewer."""

    def test_topology_is_avr(self) -> None:
        strategy = FixedAVR("gemini-pro", base_cost_usd=0.05)
        decision = strategy.select({})
        assert decision.topology == "avr"

    def test_reviewer_model(self) -> None:
        strategy = FixedAVR("gpt-5.3-codex")
        decision = strategy.select({})
        assert decision.model_id == "gpt-5.3-codex"

    def test_cost_is_double_base(self) -> None:
        strategy = FixedAVR("model-x", base_cost_usd=0.04)
        decision = strategy.select({})
        assert decision.expected_cost_usd == pytest.approx(0.08)

    def test_default_base_cost(self) -> None:
        strategy = FixedAVR("model-y")
        decision = strategy.select({})
        assert decision.expected_cost_usd == pytest.approx(0.02)

    def test_strategy_name(self) -> None:
        strategy = FixedAVR("model-z")
        assert strategy.name == "fixed_avr"

    def test_ignores_task_features(self) -> None:
        strategy = FixedAVR("reviewer")
        d1 = strategy.select({"complexity": "high"})
        d2 = strategy.select({})
        assert d1.model_id == d2.model_id
        assert d1.topology == d2.topology


# ---------------------------------------------------------------------------
# BaselineComparison
# ---------------------------------------------------------------------------

class TestBaselineComparison:
    """BaselineComparison orchestrates and compares multiple strategies."""

    def test_compare_returns_all_strategies(self) -> None:
        comp = BaselineComparison(["always_best_model", "cheapest_under_sla"])
        comp.record("always_best_model", quality=0.95, cost_usd=0.10)
        comp.record("cheapest_under_sla", quality=0.80, cost_usd=0.01)
        comp.record("router", quality=0.90, cost_usd=0.03)

        table = comp.compare()
        assert "always_best_model" in table
        assert "cheapest_under_sla" in table
        assert "router" in table

    def test_compare_correct_averages(self) -> None:
        comp = BaselineComparison(["always_best_model"])
        # Record 2 outcomes for each.
        comp.record("always_best_model", quality=0.90, cost_usd=0.10)
        comp.record("always_best_model", quality=1.00, cost_usd=0.12)
        comp.record("router", quality=0.92, cost_usd=0.02)
        comp.record("router", quality=0.88, cost_usd=0.04)

        table = comp.compare()
        assert table["always_best_model"]["avg_quality"] == pytest.approx(0.95, abs=1e-4)
        assert table["always_best_model"]["avg_cost"] == pytest.approx(0.11, abs=1e-6)
        assert table["always_best_model"]["count"] == 2
        assert table["router"]["avg_quality"] == pytest.approx(0.90, abs=1e-4)
        assert table["router"]["avg_cost"] == pytest.approx(0.03, abs=1e-6)
        assert table["router"]["count"] == 2

    def test_pareto_dominates_when_better_on_both(self) -> None:
        """A baseline dominates the router if quality >= and cost <=, one strict."""
        comp = BaselineComparison(["always_best_model"])
        # Baseline: higher quality, lower cost.
        comp.record("always_best_model", quality=0.98, cost_usd=0.01)
        comp.record("router", quality=0.80, cost_usd=0.05)

        table = comp.compare()
        assert table["always_best_model"]["pareto_dominates_router"] is True

    def test_no_pareto_when_router_is_better(self) -> None:
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.80, cost_usd=0.10)
        comp.record("router", quality=0.90, cost_usd=0.03)

        table = comp.compare()
        assert table["always_best_model"]["pareto_dominates_router"] is False

    def test_no_pareto_when_equal(self) -> None:
        """Equal on both axes is not strict Pareto dominance."""
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.90, cost_usd=0.05)
        comp.record("router", quality=0.90, cost_usd=0.05)

        table = comp.compare()
        assert table["always_best_model"]["pareto_dominates_router"] is False

    def test_router_beats_best_model_true(self) -> None:
        """Evidence gate passes: router quality >= best, router cost < best."""
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.90, cost_usd=0.10)
        comp.record("router", quality=0.92, cost_usd=0.03)

        assert comp.router_beats_best_model() is True

    def test_router_beats_best_model_equal_quality_lower_cost(self) -> None:
        """Evidence gate passes: equal quality, strictly lower cost."""
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.90, cost_usd=0.10)
        comp.record("router", quality=0.90, cost_usd=0.03)

        assert comp.router_beats_best_model() is True

    def test_router_beats_best_model_false_worse_quality(self) -> None:
        """Evidence gate fails: router quality < best_model quality."""
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.95, cost_usd=0.10)
        comp.record("router", quality=0.85, cost_usd=0.03)

        assert comp.router_beats_best_model() is False

    def test_router_beats_best_model_false_higher_cost(self) -> None:
        """Evidence gate fails: router cost >= best_model cost."""
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.90, cost_usd=0.05)
        comp.record("router", quality=0.92, cost_usd=0.05)

        assert comp.router_beats_best_model() is False

    def test_router_beats_best_model_no_baseline(self) -> None:
        """Evidence gate fails if no 'always_best_model' recorded."""
        comp = BaselineComparison([])
        comp.record("router", quality=0.95, cost_usd=0.01)
        assert comp.router_beats_best_model() is False

    def test_router_beats_best_model_no_router_records(self) -> None:
        """Evidence gate fails if router has no recorded outcomes."""
        comp = BaselineComparison(["always_best_model"])
        comp.record("always_best_model", quality=0.90, cost_usd=0.10)
        assert comp.router_beats_best_model() is False

    def test_empty_comparison(self) -> None:
        """No records at all: compare returns zero counts, no dominance."""
        comp = BaselineComparison([])
        table = comp.compare()
        assert table["router"]["count"] == 0
        assert table["router"]["avg_quality"] == 0.0

    def test_auto_creates_strategy_on_record(self) -> None:
        """Recording for an unknown strategy name creates it automatically."""
        comp = BaselineComparison([])
        comp.record("new_strategy", quality=0.80, cost_usd=0.02)
        table = comp.compare()
        assert "new_strategy" in table
        assert table["new_strategy"]["count"] == 1


# ---------------------------------------------------------------------------
# RoutingOnOff
# ---------------------------------------------------------------------------

class TestRoutingOnOff:
    """RoutingOnOff A/B paired comparison."""

    def test_empty_summary(self) -> None:
        ab = RoutingOnOff()
        s = ab.summary()
        assert s["pairs"] == 0
        assert s["on_wins"] == 0
        assert s["off_wins"] == 0
        assert s["ties"] == 0
        assert s["on_avg_quality"] == 0.0
        assert s["off_avg_quality"] == 0.0
        assert s["on_avg_cost"] == 0.0
        assert s["off_avg_cost"] == 0.0

    def test_single_pair_on_wins(self) -> None:
        ab = RoutingOnOff()
        ab.record_pair("t1", on_quality=0.9, on_cost=0.03,
                        off_quality=0.7, off_cost=0.10)
        s = ab.summary()
        assert s["pairs"] == 1
        assert s["on_wins"] == 1
        assert s["off_wins"] == 0
        assert s["ties"] == 0

    def test_single_pair_off_wins(self) -> None:
        ab = RoutingOnOff()
        ab.record_pair("t1", on_quality=0.6, on_cost=0.03,
                        off_quality=0.9, off_cost=0.10)
        s = ab.summary()
        assert s["on_wins"] == 0
        assert s["off_wins"] == 1

    def test_tie(self) -> None:
        ab = RoutingOnOff()
        ab.record_pair("t1", on_quality=0.8, on_cost=0.05,
                        off_quality=0.8, off_cost=0.10)
        s = ab.summary()
        assert s["ties"] == 1

    def test_multiple_pairs_correct_counts(self) -> None:
        ab = RoutingOnOff()
        # on wins
        ab.record_pair("t1", on_quality=0.9, on_cost=0.03,
                        off_quality=0.7, off_cost=0.10)
        # off wins
        ab.record_pair("t2", on_quality=0.5, on_cost=0.03,
                        off_quality=0.8, off_cost=0.10)
        # tie
        ab.record_pair("t3", on_quality=0.8, on_cost=0.05,
                        off_quality=0.8, off_cost=0.10)
        # on wins
        ab.record_pair("t4", on_quality=0.95, on_cost=0.04,
                        off_quality=0.85, off_cost=0.08)

        s = ab.summary()
        assert s["pairs"] == 4
        assert s["on_wins"] == 2
        assert s["off_wins"] == 1
        assert s["ties"] == 1

    def test_average_quality_and_cost(self) -> None:
        ab = RoutingOnOff()
        ab.record_pair("t1", on_quality=0.8, on_cost=0.02,
                        off_quality=0.6, off_cost=0.10)
        ab.record_pair("t2", on_quality=0.9, on_cost=0.04,
                        off_quality=0.8, off_cost=0.12)

        s = ab.summary()
        assert s["on_avg_quality"] == pytest.approx(0.85, abs=1e-4)
        assert s["off_avg_quality"] == pytest.approx(0.70, abs=1e-4)
        assert s["on_avg_cost"] == pytest.approx(0.03, abs=1e-6)
        assert s["off_avg_cost"] == pytest.approx(0.11, abs=1e-6)


# ---------------------------------------------------------------------------
# BaselineDecision dataclass
# ---------------------------------------------------------------------------

class TestBaselineDecision:
    """Basic dataclass tests for BaselineDecision."""

    def test_fields(self) -> None:
        d = BaselineDecision(
            model_id="gemini-pro",
            topology="sequential",
            expected_cost_usd=0.05,
            strategy_name="test",
        )
        assert d.model_id == "gemini-pro"
        assert d.topology == "sequential"
        assert d.expected_cost_usd == 0.05
        assert d.strategy_name == "test"
