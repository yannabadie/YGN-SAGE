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

def test_route_simple_to_system1():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False))
    assert decision.system == 1
    assert decision.llm_tier == "fast"

def test_route_complex_to_system3():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True))
    assert decision.system == 3
    assert decision.llm_tier in ("reasoner", "codex")

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
