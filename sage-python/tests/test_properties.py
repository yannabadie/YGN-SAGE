"""Property-based tests using Hypothesis.

These tests verify that key components never crash and always produce
well-typed outputs across arbitrary inputs.
"""
import pytest
from hypothesis import given, strategies as st, settings


@given(task=st.text(min_size=0, max_size=1000))
@settings(max_examples=200)
def test_complexity_router_never_crashes(task):
    """ComplexityRouter.assess_complexity + route must never raise on any text input."""
    from sage.strategy.metacognition import ComplexityRouter

    router = ComplexityRouter()
    profile = router.assess_complexity(task)
    result = router.route(profile)
    assert result.system in (1, 2, 3)


@given(
    complexity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    uncertainty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    tool_required=st.booleans(),
)
@settings(max_examples=200)
def test_complexity_router_route_is_total(complexity, uncertainty, tool_required):
    """route() must return a valid system for every CognitiveProfile combination."""
    from sage.strategy.metacognition import ComplexityRouter, CognitiveProfile

    router = ComplexityRouter()
    profile = CognitiveProfile(
        complexity=complexity,
        uncertainty=uncertainty,
        tool_required=tool_required,
    )
    result = router.route(profile)
    assert result.system in (1, 2, 3)
    assert isinstance(result.llm_tier, str)
    assert result.max_tokens > 0


@given(code=st.text(min_size=0, max_size=500))
@settings(max_examples=200)
def test_sandbox_validator_never_crashes(code):
    """ToolExecutor.validate() must return a bool for any code string."""
    try:
        from sage_core import ToolExecutor
    except ImportError:
        pytest.skip("sage_core not available")

    executor = ToolExecutor()
    result = executor.validate(code)
    assert isinstance(result, bool)


@given(
    task=st.text(min_size=0, max_size=500),
    result=st.text(min_size=0, max_size=500),
    latency_ms=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    had_errors=st.booleans(),
    avr_iterations=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=200)
def test_quality_estimator_score_in_range(task, result, latency_ms, had_errors, avr_iterations):
    """QualityEstimator.estimate() must always return a score in [0.0, 1.0]."""
    from sage.quality_estimator import QualityEstimator

    score = QualityEstimator.estimate(
        task=task,
        result=result,
        latency_ms=latency_ms,
        had_errors=had_errors,
        avr_iterations=avr_iterations,
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@given(
    key=st.text(min_size=1, max_size=100),
    content=st.text(min_size=0, max_size=1000),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_episodic_memory_store_and_search(key, content):
    """EpisodicMemory.store() then search() must never crash and search must return a list."""
    from sage.memory.episodic import EpisodicMemory

    mem = EpisodicMemory()  # in-memory mode (no db_path)
    await mem.store(key=key, content=content)
    results = await mem.search(query=content[:50] if content else "x", top_k=5)
    assert isinstance(results, list)
    for item in results:
        assert "key" in item
        assert "content" in item
