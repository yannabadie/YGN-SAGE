"""Tests for sage.strategy.adaptive_router -- AdaptiveRouter with duck-type ComplexityRouter compat."""
import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest

from sage.strategy.adaptive_router import AdaptiveRouter, AdaptiveRoutingResult
from sage.strategy.metacognition import CognitiveProfile, RoutingDecision


# -- Duck-type compatibility (same interface as ComplexityRouter) -------------


def test_has_route_method():
    router = AdaptiveRouter()
    assert callable(getattr(router, "route", None))


def test_has_assess_complexity_method():
    router = AdaptiveRouter()
    assert callable(getattr(router, "assess_complexity", None))


def test_has_assess_complexity_async_method():
    router = AdaptiveRouter()
    assert callable(getattr(router, "assess_complexity_async", None))


def test_has_record_output_entropy_method():
    router = AdaptiveRouter()
    assert callable(getattr(router, "record_output_entropy", None))


def test_has_should_brake_method():
    router = AdaptiveRouter()
    assert callable(getattr(router, "should_brake", None))


# -- route() accepts CognitiveProfile, returns RoutingDecision ---------------


def test_route_returns_routing_decision():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.2, uncertainty=0.1, tool_required=False)
    decision = router.route(profile)
    assert isinstance(decision, RoutingDecision)


def test_route_simple_to_s1():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 1
    assert decision.llm_tier == "fast"
    assert decision.validation_level == 1


def test_route_moderate_to_s2():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.5, uncertainty=0.4, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 2
    assert decision.llm_tier == "mutator"
    assert decision.validation_level == 2


def test_route_complex_to_s3():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True)
    decision = router.route(profile)
    assert decision.system == 3
    assert decision.validation_level == 3
    assert decision.use_z3


def test_route_high_complexity_codex_tier():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.9, uncertainty=0.5, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 3
    assert decision.llm_tier == "codex"


def test_route_s2_reasoner_for_higher_complexity():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.65, uncertainty=0.55, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 2
    assert decision.llm_tier == "reasoner"


def test_route_tool_required_prevents_s1():
    router = AdaptiveRouter()
    profile = CognitiveProfile(complexity=0.3, uncertainty=0.2, tool_required=True)
    decision = router.route(profile)
    assert decision.system == 2  # tool_required forces S2


# -- assess_complexity (heuristic) ------------------------------------------


def test_assess_complexity_simple_task():
    router = AdaptiveRouter()
    profile = router.assess_complexity("What is the capital of France?")
    assert isinstance(profile, CognitiveProfile)
    assert profile.complexity < 0.5
    assert not profile.tool_required


def test_assess_complexity_code_task():
    router = AdaptiveRouter()
    profile = router.assess_complexity(
        "Write a Python function to parse JSON and run the tests"
    )
    assert profile.complexity >= 0.35
    assert profile.tool_required  # "run" + "test" trigger tool detection


def test_assess_complexity_complex_task():
    router = AdaptiveRouter()
    profile = router.assess_complexity(
        "Debug and fix the crash in the authentication system, "
        "then run the test suite"
    )
    assert profile.complexity > 0.5
    assert profile.tool_required


def test_assess_complexity_reasoning_field():
    router = AdaptiveRouter()
    profile = router.assess_complexity("Hello")
    assert profile.reasoning == "heuristic"


# -- assess_complexity_async -------------------------------------------------


@pytest.mark.asyncio
async def test_assess_complexity_async_returns_profile():
    router = AdaptiveRouter()
    profile = await router.assess_complexity_async("What is 2+2?")
    assert isinstance(profile, CognitiveProfile)
    assert profile.complexity < 0.5


# -- Self-braking (CGRS) ----------------------------------------------------


def test_should_brake_detects_convergence():
    router = AdaptiveRouter()
    router.record_output_entropy(0.1)
    router.record_output_entropy(0.08)
    router.record_output_entropy(0.05)
    assert router.should_brake()


def test_should_brake_allows_divergence():
    router = AdaptiveRouter()
    router.record_output_entropy(0.9)
    router.record_output_entropy(0.85)
    assert not router.should_brake()


def test_should_brake_needs_full_window():
    router = AdaptiveRouter(brake_window=5)
    router.record_output_entropy(0.01)
    router.record_output_entropy(0.01)
    # Only 2 samples, window is 5
    assert not router.should_brake()


# -- AdaptiveRoutingResult ---------------------------------------------------


def test_adaptive_routing_result_has_all_fields():
    result = AdaptiveRoutingResult(
        decision=RoutingDecision(
            system=1, llm_tier="fast", max_tokens=2048, use_z3=False
        ),
        profile=CognitiveProfile(
            complexity=0.2, uncertainty=0.1, tool_required=False
        ),
        stage=0,
        confidence=0.9,
        method="heuristic",
    )
    assert result.stage == 0
    assert result.confidence == 0.9
    assert result.method == "heuristic"
    assert result.decision.system == 1
    assert result.profile.complexity == 0.2


# -- route_adaptive (extended API) ------------------------------------------


def test_route_adaptive_simple_task():
    router = AdaptiveRouter()
    result = router.route_adaptive("What is the capital of France?")
    assert isinstance(result, AdaptiveRoutingResult)
    assert result.decision.system == 1
    assert result.method == "heuristic"
    assert 0.0 <= result.confidence <= 1.0


def test_route_adaptive_code_task_routes_s2():
    router = AdaptiveRouter()
    result = router.route_adaptive(
        "Write a Python function to parse JSON, then run the test suite"
    )
    assert result.decision.system >= 2


def test_route_adaptive_complex_task_routes_s3():
    router = AdaptiveRouter()
    padding = "using advanced techniques " * 20
    result = router.route_adaptive(
        f"Implement a distributed consensus algorithm with lock-free data structures {padding}"
    )
    assert result.decision.system == 3


@pytest.mark.asyncio
async def test_route_adaptive_async_works():
    router = AdaptiveRouter()
    result = await router.route_adaptive_async("What is 2+2?")
    assert isinstance(result, AdaptiveRoutingResult)
    assert result.decision.system == 1


# -- Confidence in valid range -----------------------------------------------


def test_confidence_in_valid_range():
    router = AdaptiveRouter()
    tasks = [
        "Hello world",
        "Write a function to sort a list",
        "Implement a compiler for a subset of Haskell with formal verification",
        "Debug the intermittent flaky race condition in the concurrent queue",
    ]
    for task in tasks:
        result = router.route_adaptive(task)
        assert (
            0.0 <= result.confidence <= 1.0
        ), f"Confidence {result.confidence} out of range for: {task}"


# -- S1 quality: trivial tasks should route correctly -----------------------


def test_trivial_tasks_route_s1():
    router = AdaptiveRouter()
    trivial = [
        "What is 2+2?",
        "Hello world",
        "What color is the sky?",
        "Say hi",
        "How are you?",
    ]
    for task in trivial:
        result = router.route_adaptive(task)
        assert (
            result.decision.system == 1
        ), f"Trivial task should route S1: '{task}' got S{result.decision.system}"


# -- Over-routing: simple tasks should NOT route to S3 -----------------------


def test_simple_tasks_not_routed_to_s3():
    router = AdaptiveRouter()
    simple = [
        "What is the capital of Japan?",
        "Explain what a variable is",
        "Hello there",
        "What time is it?",
        "Define recursion",
    ]
    for task in simple:
        result = router.route_adaptive(task)
        assert (
            result.decision.system < 3
        ), f"Simple task over-routed to S3: '{task}'"


# -- record_feedback does not crash ------------------------------------------


def test_record_feedback_no_crash():
    router = AdaptiveRouter()
    # Should not raise even without Rust backend
    router.record_feedback(
        task="test", routed_tier=2, actual_quality=0.9, latency_ms=100, cost_usd=0.01
    )


# -- Properties -------------------------------------------------------------


def test_has_rust_property():
    router = AdaptiveRouter()
    # Without Rust build, this should be False
    assert isinstance(router.has_rust, bool)


def test_has_classifier_property():
    router = AdaptiveRouter()
    assert isinstance(router.has_classifier, bool)
    # Without Rust, classifier is never available
    if not router.has_rust:
        assert not router.has_classifier


# -- Constructor parameters --------------------------------------------------


def test_custom_thresholds():
    router = AdaptiveRouter(
        s1_complexity_ceil=0.60,
        s3_complexity_floor=0.80,
    )
    # With wider S1 ceiling, more tasks route to S1
    profile = CognitiveProfile(complexity=0.55, uncertainty=0.2, tool_required=False)
    decision = router.route(profile)
    assert decision.system == 1  # Would be S2 with default 0.50 ceil


def test_entropy_probe_disabled_by_default():
    router = AdaptiveRouter()
    assert not router._enable_entropy


def test_entropy_probe_can_be_enabled():
    router = AdaptiveRouter(enable_entropy_probe=True)
    assert router._enable_entropy
