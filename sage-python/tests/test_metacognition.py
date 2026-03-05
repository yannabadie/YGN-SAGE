import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.strategy.metacognition import (
    MetacognitiveController, CognitiveProfile, RoutingDecision
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
    profile = ctrl.assess_complexity("Debug and fix the crash in the authentication system, then run the test suite")
    assert profile.complexity > 0.5
    assert profile.tool_required


@pytest.mark.asyncio
async def test_assess_complexity_async_fallback():
    """Without GOOGLE_API_KEY, async falls back to heuristic."""
    import os
    saved = os.environ.pop('GOOGLE_API_KEY', None)
    try:
        ctrl = MetacognitiveController()
        ctrl._llm_available = False
        profile = await ctrl.assess_complexity_async('Debug the crash in auth')
        assert profile.complexity > 0.5
        assert profile.reasoning == 'heuristic'
    finally:
        if saved:
            os.environ['GOOGLE_API_KEY'] = saved


def test_assess_complexity_has_reasoning():
    ctrl = MetacognitiveController()
    profile = ctrl.assess_complexity('Hello')
    assert profile.reasoning == 'heuristic'


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


def test_s2_escalation_threshold():
    """S2->S3 escalation constant is defined."""
    from sage.agent_loop import S2_MAX_RETRIES_BEFORE_ESCALATION
    assert S2_MAX_RETRIES_BEFORE_ESCALATION == 2
