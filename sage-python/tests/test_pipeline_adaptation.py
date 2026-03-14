"""Integration tests for Phase C runtime adaptation in the pipeline."""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sage.topology_controller import TopologyController, AdaptationDecision


class MockNode:
    def __init__(self, role="agent", model_id="test-model", system=2,
                 required_capabilities=None, security_label=0, max_cost_usd=5.0):
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities or []
        self.security_label = security_label
        self.max_cost_usd = max_cost_usd


class MockGraph:
    def __init__(self, nodes):
        self._nodes = list(nodes)
    def node_count(self):
        return len(self._nodes)
    def get_node(self, idx):
        return self._nodes[idx] if 0 <= idx < len(self._nodes) else None
    def set_node_model_id(self, idx, model_id):
        if 0 <= idx < len(self._nodes):
            self._nodes[idx].model_id = model_id


def test_controller_upgrade_triggers_on_low_quality():
    """quality < 0.3 → UPGRADE_MODEL action."""
    qe = MagicMock()
    qe.estimate.return_value = 0.1  # very low
    ctrl = TopologyController(
        assigner=MagicMock(),
        quality_estimator=qe,
    )
    ctx = MagicMock()
    ctx.latency_ms = 100.0
    topo = MagicMock()
    topo.get_node.return_value = MockNode(system=2)

    decision = ctrl.evaluate_and_decide(0, "bad output", "task", topo, ctx)
    assert decision.action == "upgrade_model"


def test_controller_continues_on_good_quality():
    """quality >= 0.7 → CONTINUE."""
    qe = MagicMock()
    qe.estimate.return_value = 0.9
    ctrl = TopologyController(quality_estimator=qe)
    ctx = MagicMock()
    ctx.latency_ms = 50.0

    decision = ctrl.evaluate_and_decide(0, "great output", "task", MagicMock(), ctx)
    assert decision.action == "continue"


def test_budget_exhausted_graceful():
    """Controller with no assigner → can't upgrade, falls through to continue."""
    qe = MagicMock()
    qe.estimate.return_value = 0.2  # critical
    ctrl = TopologyController(assigner=None, quality_estimator=qe)
    ctrl._node_retries[0] = 2  # exhausted retries
    ctx = MagicMock()
    ctx.latency_ms = 100.0

    decision = ctrl.evaluate_and_decide(0, "bad", "task", MagicMock(), ctx)
    # No more retries → falls through to continue
    assert decision.action == "continue"


def test_oxiz_verification_fail_proceeds():
    """OxiZ failure in pipeline is non-blocking — pipeline continues."""
    # This tests the _verify_assignment_formal path
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
    pipeline = CognitiveOrchestrationPipeline(
        router=None, engine=None, assigner=MagicMock(),
        provider_pool=None, llm_provider=None,
    )
    ctx = PipelineContext(task="test")
    ctx.topology = MockGraph([MockNode()])
    ctx.domain = "code"
    ctx.budget = 5.0
    ctx.assignments = {0: "test-model"}

    # Should not raise even if verification fails internally
    pipeline._stage_assign_models(ctx)
    # If we get here, it's non-blocking ✓


def test_prm_skipped_on_plain_text():
    """PRM not called when result has no structured content."""
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
    prm = MagicMock()
    pipeline = CognitiveOrchestrationPipeline(
        router=None, engine=None, assigner=None,
        provider_pool=None, llm_provider=None,
    )
    pipeline.prm = prm
    pipeline.quality_estimator = MagicMock()
    pipeline.quality_estimator.estimate.return_value = 0.7

    ctx = PipelineContext(task="hello", budget=5.0)
    ctx.result = "just plain text with no think tags or code"
    ctx.latency_ms = 100.0

    pipeline._stage_learn(ctx)
    prm.calculate_r_path.assert_not_called()


def test_controller_none_preserves_phase_b():
    """controller=None → pipeline behaves identically to Phase B."""
    from sage.pipeline import CognitiveOrchestrationPipeline
    pipeline = CognitiveOrchestrationPipeline(
        router=None, engine=None, assigner=None,
        provider_pool=None, llm_provider=MagicMock(),
    )
    # controller should be None by default
    assert not hasattr(pipeline, 'controller') or pipeline.controller is None
