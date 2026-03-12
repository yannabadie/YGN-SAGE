"""Basic tests for sage-router standalone package."""
import warnings

import pytest

from sage_router import ComplexityRouter, CognitiveProfile, RoutingDecision, QualityEstimator
from sage_router.structural_features import StructuralFeatures
from sage_router.adaptive_router import AdaptiveRouter, AdaptiveRoutingResult


# ---------------------------------------------------------------------------
# ComplexityRouter
# ---------------------------------------------------------------------------

def test_complexity_router_heuristic():
    router = ComplexityRouter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = router.assess_complexity("hello world")
    assert isinstance(profile, CognitiveProfile)
    assert 0.0 <= profile.complexity <= 1.0


def test_complexity_router_complex_task():
    router = ComplexityRouter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = router.assess_complexity(
            "implement a distributed concurrent algorithm with deadlock detection"
        )
    assert profile.complexity > 0.3


def test_routing_decision():
    router = ComplexityRouter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = router.assess_complexity("simple greeting")
    decision = router.route(profile)
    assert isinstance(decision, RoutingDecision)
    assert decision.system in (1, 2, 3)


def test_s1_routing():
    """Low-complexity, low-uncertainty, no-tool task should go to S1."""
    router = ComplexityRouter()
    profile = CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 1
    assert decision.llm_tier == "fast"
    assert decision.use_z3 is False


def test_s3_routing():
    """High complexity should go to S3."""
    router = ComplexityRouter()
    profile = CognitiveProfile(complexity=0.9, uncertainty=0.1, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 3
    assert decision.use_z3 is True


def test_s2_routing():
    """Mid-range task should go to S2."""
    router = ComplexityRouter()
    profile = CognitiveProfile(complexity=0.55, uncertainty=0.4, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 2


def test_cgrs_brake():
    """CGRS: should_brake() returns True when window filled with low entropy."""
    router = ComplexityRouter(brake_window=3, brake_entropy_threshold=0.15)
    assert not router.should_brake()
    router.record_output_entropy(0.05)
    router.record_output_entropy(0.08)
    assert not router.should_brake()  # only 2 entries
    router.record_output_entropy(0.10)
    assert router.should_brake()


def test_cgrs_no_brake_high_entropy():
    """CGRS: should_brake() returns False when any entropy is high."""
    router = ComplexityRouter(brake_window=3, brake_entropy_threshold=0.15)
    router.record_output_entropy(0.05)
    router.record_output_entropy(0.50)  # high entropy breaks the streak
    router.record_output_entropy(0.08)
    assert not router.should_brake()


# ---------------------------------------------------------------------------
# QualityEstimator
# ---------------------------------------------------------------------------

def test_quality_estimator_basic():
    score = QualityEstimator.estimate(
        task="write a function",
        result="def foo(): return 42",
    )
    assert 0.0 < score <= 1.0


def test_quality_estimator_empty():
    score = QualityEstimator.estimate(task="anything", result="")
    assert score == 0.0


def test_quality_estimator_whitespace_only():
    score = QualityEstimator.estimate(task="anything", result="   \n  ")
    assert score == 0.0


def test_quality_estimator_with_code():
    """Code task + code result should score higher than no code."""
    score_with_code = QualityEstimator.estimate(
        task="implement a function to sort a list",
        result="def sort_list(lst):\n    return sorted(lst)\n",
    )
    score_no_code = QualityEstimator.estimate(
        task="implement a function to sort a list",
        result="here is the answer",
    )
    assert score_with_code > score_no_code


def test_quality_estimator_error_penalty():
    """Error-flagged results should score lower."""
    score_ok = QualityEstimator.estimate(
        task="explain sorting", result="sorting works by comparison", had_errors=False
    )
    score_err = QualityEstimator.estimate(
        task="explain sorting", result="sorting works by comparison", had_errors=True
    )
    assert score_ok >= score_err


def test_quality_estimator_avr_bonus():
    """Fast AVR convergence (≤2 iterations) should add bonus."""
    score_avr = QualityEstimator.estimate(
        task="write code", result="def f(): pass", avr_iterations=1
    )
    score_no_avr = QualityEstimator.estimate(
        task="write code", result="def f(): pass", avr_iterations=0
    )
    assert score_avr > score_no_avr


# ---------------------------------------------------------------------------
# StructuralFeatures
# ---------------------------------------------------------------------------

def test_structural_features_simple():
    f = StructuralFeatures.extract("hello world")
    assert f.word_count == 2
    assert not f.has_code_block
    assert not f.has_question_mark
    assert f.keyword_complexity >= 0.2  # base


def test_structural_features_algorithm():
    f = StructuralFeatures.extract("implement a concurrent distributed algorithm")
    assert f.keyword_complexity > 0.4  # algo keyword bonus


def test_structural_features_uncertainty():
    f = StructuralFeatures.extract("maybe explore the flaky intermittent behavior")
    assert f.keyword_uncertainty > 0.0


def test_structural_features_tool():
    f = StructuralFeatures.extract("search for files and execute the tests")
    assert f.tool_required is True


def test_structural_features_code_block():
    f = StructuralFeatures.extract("here is some code:\n```python\nprint('hi')\n```")
    assert f.has_code_block is True
    assert f.keyword_complexity > 0.2


# ---------------------------------------------------------------------------
# AdaptiveRouter
# ---------------------------------------------------------------------------

def test_adaptive_router_no_knn():
    """AdaptiveRouter without kNN should fall through to structural features."""
    router = AdaptiveRouter()
    profile = router.assess_complexity("hello world")
    assert isinstance(profile, CognitiveProfile)
    assert 0.0 <= profile.complexity <= 1.0


def test_adaptive_router_route_adaptive():
    """route_adaptive() should return AdaptiveRoutingResult with method info."""
    router = AdaptiveRouter()
    result = router.route_adaptive("implement a sorting algorithm")
    assert isinstance(result, AdaptiveRoutingResult)
    assert result.method in ("knn", "structural", "heuristic", "entropy_s2")
    assert isinstance(result.decision, RoutingDecision)
    assert result.decision.system in (1, 2, 3)


def test_adaptive_router_has_rust_false():
    """Standalone package always reports has_rust=False."""
    router = AdaptiveRouter()
    assert router.has_rust is False


def test_adaptive_router_duck_type():
    """AdaptiveRouter is duck-type compatible with ComplexityRouter."""
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False)
    decision = router.route(profile)
    assert isinstance(decision, RoutingDecision)
    assert decision.system == 1
