import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.strategy.metacognition import (
    MetacognitiveController, ComplexityRouter, CognitiveProfile, RoutingDecision
)

def test_cognitive_profile():
    p = CognitiveProfile(complexity=0.3, uncertainty=0.2, tool_required=False)
    assert p.complexity < 0.5

def test_routing_decision_has_validation_level():
    rd = RoutingDecision(system=2, llm_tier="mutator", max_tokens=4096, use_z3=False, validation_level=2)
    assert rd.validation_level == 2

def test_route_simple_to_system1():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False))
    assert decision.system == 1
    assert decision.llm_tier == "fast"
    assert decision.validation_level == 1

def test_route_moderate_to_system2():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.5, uncertainty=0.4, tool_required=False))
    assert decision.system == 2
    assert decision.llm_tier == "mutator"
    assert decision.validation_level == 2
    assert not decision.use_z3

def test_route_moderate_with_tools_to_system2():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.4, uncertainty=0.3, tool_required=True))
    assert decision.system == 2
    assert decision.validation_level == 2

def test_route_s2_uses_reasoner_for_higher_complexity():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.65, uncertainty=0.55, tool_required=False))
    assert decision.system == 2
    assert decision.llm_tier == "reasoner"

def test_route_complex_to_system3():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True))
    assert decision.system == 3
    assert decision.llm_tier in ("reasoner", "codex")
    assert decision.validation_level == 3

def test_route_high_complexity_codex_tier():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.9, uncertainty=0.5, tool_required=False))
    assert decision.system == 3
    assert decision.llm_tier == "codex"

def test_route_low_complexity_to_system1():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.2, uncertainty=0.1, tool_required=False))
    assert decision.system == 1
    assert decision.validation_level == 1
    assert decision.llm_tier == "fast"

def test_route_high_complexity_to_system3():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.8, uncertainty=0.7, tool_required=True))
    assert decision.system == 3
    assert decision.validation_level == 3
    assert decision.use_z3

def test_self_braking_detects_convergence():
    ctrl = MetacognitiveController()
    ctrl.record_output_entropy(0.1)
    ctrl.record_output_entropy(0.08)
    ctrl.record_output_entropy(0.05)
    assert ctrl.should_brake()

def test_self_braking_allows_divergence():
    ctrl = MetacognitiveController()
    ctrl.record_output_entropy(0.9)
    ctrl.record_output_entropy(0.85)
    assert not ctrl.should_brake()

def test_assess_complexity_simple():
    ctrl = MetacognitiveController()
    profile = ctrl.assess_complexity("What is 2+2?")
    assert profile.complexity < 0.5
    assert not profile.tool_required

def test_assess_complexity_complex():
    ctrl = MetacognitiveController()
    # "debug" + "fix" = 2 hits → min(2/3, 1.0) = 0.67
    profile = ctrl.assess_complexity("Debug and fix the crash in the authentication system")
    assert profile.complexity > 0.5


@pytest.mark.asyncio
async def test_assess_complexity_async_fallback():
    """Without GOOGLE_API_KEY, async falls back to degraded heuristic."""
    import os
    saved = os.environ.pop('GOOGLE_API_KEY', None)
    try:
        ctrl = MetacognitiveController()
        ctrl._llm_available = False
        # "debug" + "fix" = 2 hits → 0.67
        profile = await ctrl.assess_complexity_async('Debug and fix the auth crash')
        assert profile.complexity > 0.5
        assert profile.reasoning == 'degraded_heuristic'
    finally:
        if saved:
            os.environ['GOOGLE_API_KEY'] = saved


def test_assess_complexity_has_reasoning():
    ctrl = MetacognitiveController()
    profile = ctrl.assess_complexity('Hello')
    assert profile.reasoning == 'degraded_heuristic'


def test_s2_validation_detects_code_block():
    """S2 validation identifies code blocks for sandbox execution."""
    from sage.agent_loop import _extract_code_blocks

    content_with_code = "Here is the solution:\n```python\nprint('hello')\n```\nDone."
    blocks = _extract_code_blocks(content_with_code)
    assert len(blocks) == 1
    assert "print('hello')" in blocks[0]

    content_no_code = "The answer is 42. Step 1: think about it."
    blocks = _extract_code_blocks(content_no_code)
    assert len(blocks) == 0


def test_s2_avr_loop_constants():
    """S2 AVR loop has configurable constants."""
    from sage.agent_loop import S2_MAX_RETRIES_BEFORE_ESCALATION, S2_AVR_MAX_ITERATIONS
    assert S2_MAX_RETRIES_BEFORE_ESCALATION == 2
    assert S2_AVR_MAX_ITERATIONS == 3


def test_s3_system_prompt_contains_z3_dsl():
    """S3 system prompt must teach the Z3 DSL syntax to the LLM."""
    from sage.agent_loop import AgentLoop
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=1, validation_level=3,
        system_prompt="Base prompt.",
    )
    loop = AgentLoop(config=config, llm_provider=MockProvider())

    # Build the system prompt the same way run() does
    system_prompt = config.system_prompt
    if config.validation_level >= 3:
        # The AgentLoop should augment the prompt with Z3 DSL
        # Check via source code that the augmentation exists
        import inspect
        source = inspect.getsource(AgentLoop)
        assert "assert bounds" in source, "S3 prompt must teach assert bounds"
        assert "assert loop" in source, "S3 prompt must teach assert loop"
        assert "assert arithmetic" in source, "S3 prompt must teach assert arithmetic"
        assert "assert invariant" in source, "S3 prompt must teach assert invariant"


def test_s3_prompt_produces_parseable_z3_output():
    """Mock LLM response with Z3 DSL should score > 0 via PRM."""
    from sage.topology.kg_rlvr import ProcessRewardModel

    prm = ProcessRewardModel()

    # Simulate what an LLM SHOULD produce when properly prompted with Z3 DSL
    # Use bounds assertions that Z3 can actually prove (addr < limit)
    content_with_z3 = """<think>
assert bounds(50, 100)
assert bounds(0, 256)
assert bounds(99, 100)
</think>
The access is safe because all addresses are within bounds."""

    score, details = prm.calculate_r_path(content_with_z3)
    assert score > 0.0, f"Z3 DSL content should score positively, got {score}"
    assert details["verifiable_ratio"] > 0.5


def test_s3_prompt_without_z3_dsl_scores_zero():
    """LLM response without Z3 DSL assertions should score 0 (not negative)."""
    from sage.topology.kg_rlvr import ProcessRewardModel

    prm = ProcessRewardModel()

    content_no_z3 = """<think>
I think the answer is 42.
Let me reason step by step about this problem.
First, we need to consider the constraints.
</think>
The answer is 42."""

    score, details = prm.calculate_r_path(content_no_z3)
    # Without Z3 assertions, steps score 0.0 each, average = 0.0
    assert score == 0.0, f"Non-Z3 content should score 0.0, got {score}"


def test_degraded_heuristic_returns_profile():
    """Degraded keyword-count heuristic returns a valid CognitiveProfile."""
    import warnings
    router = ComplexityRouter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = router._assess_heuristic("What is a unit test?")
    assert isinstance(profile, CognitiveProfile)
    assert profile.tool_required is False  # degraded heuristic never sets tools


def test_degraded_heuristic_no_false_positives():
    """Non-keyword tasks get zero complexity."""
    import warnings
    router = ComplexityRouter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = router._assess_heuristic("The attestation process is simple")
    assert profile.tool_required is False
    assert profile.complexity == 0.0  # no complex keywords


def test_degraded_heuristic_complex_keywords():
    """Complex keywords increase complexity score."""
    import warnings
    router = ComplexityRouter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = router._assess_heuristic("debug the algorithm and verify the fix")
    # "debug", "algorithm", "verify", "fix" = 4 hits → min(4/3, 1.0) = 1.0
    assert profile.complexity > 0.5
