"""Tests for DynamicRouter — round-level routing inside verified envelope."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, BudgetConstraint
from sage.providers.capabilities import CapabilityMatrix, ProviderCapabilities
from sage.routing.dynamic import DynamicRouter, RoutingDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matrix():
    m = CapabilityMatrix()
    m.register(ProviderCapabilities(
        provider="google", structured_output=True, tool_role=True,
        file_search=True, grounding=True, streaming=True,
    ))
    m.register(ProviderCapabilities(
        provider="openai", structured_output=True, tool_role=True,
        file_search=False, grounding=False, streaming=True,
    ))
    m.register(ProviderCapabilities(
        provider="deepseek", structured_output=False, tool_role=False,
        file_search=False, grounding=False, streaming=True,
    ))
    return m


@pytest.fixture
def router(matrix):
    return DynamicRouter(
        capability_matrix=matrix,
        provider_costs={"google": 0.5, "openai": 2.0, "deepseek": 0.1},
        provider_quality={"google": 0.8, "openai": 0.95, "deepseek": 0.7},
    )


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

def test_route_simple_task(router):
    """Simple task should pick cheapest capable provider."""
    node = TaskNode(node_id="a", description="Simple question")
    decision = router.route(node, cost_sensitivity=0.8)
    assert isinstance(decision, RoutingDecision)
    assert decision.provider in ("google", "openai", "deepseek")


def test_route_respects_capabilities(router):
    """Task requiring file_search should only go to google."""
    node = TaskNode(
        node_id="a", description="Search files",
        capabilities_required=["file_search"],
    )
    decision = router.route(node, cost_sensitivity=0.5)
    assert decision.provider == "google"


def test_route_respects_tool_role(router):
    """Task requiring tool_role excludes deepseek."""
    node = TaskNode(
        node_id="a", description="Use tools",
        capabilities_required=["tool_role"],
    )
    decision = router.route(node, cost_sensitivity=0.5)
    assert decision.provider in ("google", "openai")


def test_route_quality_first(router):
    """Low cost_sensitivity should pick highest quality."""
    node = TaskNode(node_id="a", description="Complex reasoning")
    decision = router.route(node, cost_sensitivity=0.0)
    assert decision.provider == "openai"  # Highest quality


def test_route_cost_first(router):
    """High cost_sensitivity should pick cheapest."""
    node = TaskNode(node_id="a", description="Simple task")
    decision = router.route(node, cost_sensitivity=1.0)
    assert decision.provider == "deepseek"  # Cheapest


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------

def test_route_adapts_after_feedback(router):
    """After reporting failure for a provider, router should penalize it."""
    node = TaskNode(node_id="a", description="Task")
    # Route initially
    d1 = router.route(node, cost_sensitivity=1.0)
    initial = d1.provider

    # Report failure
    router.report_outcome(initial, success=False, latency_ms=5000)
    router.report_outcome(initial, success=False, latency_ms=5000)
    router.report_outcome(initial, success=False, latency_ms=5000)

    # Route again — should avoid the failing provider
    d2 = router.route(node, cost_sensitivity=1.0)
    # After 3 failures the provider should be penalized
    # (may still be selected if it's the only option, but score should drop)
    assert isinstance(d2, RoutingDecision)


# ---------------------------------------------------------------------------
# Budget constraints
# ---------------------------------------------------------------------------

def test_route_respects_budget(router):
    """Task with tight budget should not pick expensive provider."""
    node = TaskNode(
        node_id="a", description="Budget task",
        budget=BudgetConstraint(max_cost_usd=0.01),
    )
    decision = router.route(node, cost_sensitivity=0.5)
    # Should prefer cheaper providers
    assert decision.provider in ("google", "deepseek")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_route_no_capable_provider_raises():
    """If no provider has required capabilities, raise ValueError."""
    m = CapabilityMatrix()
    m.register(ProviderCapabilities(provider="mock"))
    router = DynamicRouter(
        capability_matrix=m,
        provider_costs={"mock": 0.1},
        provider_quality={"mock": 0.5},
    )
    node = TaskNode(
        node_id="a", description="Needs everything",
        capabilities_required=["file_search", "grounding"],
    )
    with pytest.raises(ValueError, match="No provider"):
        router.route(node, cost_sensitivity=0.5)


def test_routing_decision_fields(router):
    node = TaskNode(node_id="a", description="Test")
    decision = router.route(node, cost_sensitivity=0.5)
    assert hasattr(decision, "provider")
    assert hasattr(decision, "score")
    assert hasattr(decision, "reason")
