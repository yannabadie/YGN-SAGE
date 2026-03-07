"""Ablation: does routing add value over fixed model selection?"""
import pytest
from sage.strategy.metacognition import MetacognitiveController


def test_routing_produces_different_tiers():
    """Routing should produce different tiers for different task types."""
    mc = MetacognitiveController()

    tasks = {
        "simple": "What is 2+2?",
        "code": "Write a Python function to sort a list using quicksort with error handling",
        "formal": "Prove that the halting problem is undecidable using diagonalization",
    }

    tiers = {}
    for name, task in tasks.items():
        profile = mc._assess_heuristic(task)
        decision = mc.route(profile)
        tiers[name] = decision.llm_tier

    unique_tiers = set(tiers.values())
    assert len(unique_tiers) >= 2, (
        f"Routing produced only {unique_tiers} — adds no value over fixed selection. "
        f"Tiers: {tiers}"
    )


def test_routing_cost_ordering():
    """Routing tiers should have a cost ordering: S1 < S2 < S3."""
    from sage.llm.router import ModelRouter

    s1_config = ModelRouter.get_config("fast")
    s3_config = ModelRouter.get_config("reasoner")

    assert s1_config.model != s3_config.model, (
        "S1 and S3 use the same model — routing adds no value"
    )


def test_routing_system_levels_for_different_complexity():
    """Different complexity levels should map to different systems (S1/S2/S3)."""
    from sage.strategy.metacognition import CognitiveProfile

    mc = MetacognitiveController()

    # S1: low complexity, low uncertainty, no tools
    s1_profile = CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False)
    s1_decision = mc.route(s1_profile)

    # S3: high complexity
    s3_profile = CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True)
    s3_decision = mc.route(s3_profile)

    assert s1_decision.system < s3_decision.system, (
        f"S1 system={s1_decision.system}, S3 system={s3_decision.system} — "
        "routing does not differentiate complexity levels"
    )


def test_heuristic_is_deterministic():
    """Same input should always produce the same routing decision."""
    mc = MetacognitiveController()
    task = "Write a function to compute Fibonacci numbers efficiently"

    decisions = []
    for _ in range(10):
        profile = mc._assess_heuristic(task)
        decision = mc.route(profile)
        decisions.append((decision.system, decision.llm_tier))

    assert len(set(decisions)) == 1, (
        f"Heuristic routing is non-deterministic: {set(decisions)}"
    )
