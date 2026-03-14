"""Tests for OxiZ formal verification wired into pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


# ── Helper mocks ─────────────────────────────────────────────────────────────


class _MockTopologyNode:
    """Mock topology node with model_id and capabilities."""

    def __init__(self, model_id: str = "gemini-2.5-flash", capabilities: list[str] | None = None) -> None:
        self.model_id = model_id
        self.capabilities = capabilities or []


class _MockTopology:
    """Mock TopologyGraph-like object with nodes that have capabilities."""

    def __init__(self, nodes: list[_MockTopologyNode] | None = None) -> None:
        self._nodes = nodes or [_MockTopologyNode()]

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> _MockTopologyNode | None:
        return self._nodes[idx] if idx < len(self._nodes) else None


class _MockAssigner:
    def assign_models(self, topology: object, domain: str, budget: float) -> int:
        return topology.node_count() if hasattr(topology, "node_count") else 0  # type: ignore[union-attr]


class _MockLLMResponse:
    content: str = "test response"


class _MockLLMProvider:
    async def generate(self, messages: object, config: object = None) -> _MockLLMResponse:
        return _MockLLMResponse()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_verify_assignment_called_after_assign() -> None:
    """_verify_assignment_formal is called after assigner completes in _stage_assign_models."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    ctx = PipelineContext(
        task="Write a function to sort a list",
        topology=_MockTopology(),
        domain="code",
        budget=5.0,
    )

    called: list[bool] = []

    original = pipeline._verify_assignment_formal

    def _spy(c: PipelineContext) -> None:
        called.append(True)
        original(c)

    pipeline._verify_assignment_formal = _spy  # type: ignore[method-assign]
    pipeline._stage_assign_models(ctx)

    assert called, "_verify_assignment_formal was not called after model assignment"


def test_verify_assignment_fail_non_blocking() -> None:
    """If OxiZ verification fails, _stage_assign_models still succeeds (non-blocking)."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    ctx = PipelineContext(
        task="test task",
        topology=_MockTopology(),
        domain="code",
        budget=5.0,
    )

    # Patch _verify_assignment_formal to raise an exception
    def _always_raise(c: PipelineContext) -> None:
        raise RuntimeError("Simulated OxiZ failure")

    pipeline._verify_assignment_formal = _always_raise  # type: ignore[method-assign]

    # Must not raise — verification is non-blocking
    result_ctx = pipeline._stage_assign_models(ctx)
    assert result_ctx is ctx  # context returned unchanged


def test_verify_assignment_skipped_without_oxiz() -> None:
    """Without sage_core SMT or z3-solver, _verify_assignment_formal skips gracefully."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    ctx = PipelineContext(
        task="test task",
        topology=_MockTopology(),
        domain="code",
        budget=5.0,
    )

    # Patch verify_provider_assignment to raise ImportError (no SMT backend)
    with patch(
        "sage.pipeline.verify_provider_assignment",
        side_effect=ImportError("No SMT backend: install sage_core[smt] or z3-solver"),
    ):
        # Should not raise — import errors are caught gracefully
        result_ctx = pipeline._stage_assign_models(ctx)
        assert result_ctx is ctx


def test_verify_assignment_skipped_no_topology() -> None:
    """When topology is None, Stage 3 returns early without calling verify."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    ctx = PipelineContext(task="test task", topology=None, domain="code", budget=5.0)

    called: list[bool] = []

    def _spy(c: PipelineContext) -> None:
        called.append(True)

    pipeline._verify_assignment_formal = _spy  # type: ignore[method-assign]
    pipeline._stage_assign_models(ctx)

    # No topology → early return → verify not called
    assert not called, "_verify_assignment_formal should not be called when topology is None"


def test_verify_assignment_formal_builds_adapter() -> None:
    """_verify_assignment_formal builds a proper adapter from topology nodes."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    # Topology with nodes that have specific capabilities
    nodes = [
        _MockTopologyNode(model_id="gemini-2.5-flash", capabilities=["tool_role"]),
        _MockTopologyNode(model_id="gpt-5.3-codex", capabilities=["code_execution"]),
    ]
    ctx = PipelineContext(
        task="test task",
        topology=_MockTopology(nodes=nodes),
        domain="code",
        budget=5.0,
    )

    captured_args: list[tuple] = []

    def _mock_verify(dag: object, providers: object) -> MagicMock:
        captured_args.append((dag, providers))
        result = MagicMock()
        result.satisfied = True
        result.counterexample = None
        return result

    with patch("sage.pipeline.verify_provider_assignment", side_effect=_mock_verify):
        pipeline._verify_assignment_formal(ctx)

    assert len(captured_args) == 1
    dag_adapter, provider_specs = captured_args[0]

    # Verify adapter has node_ids
    assert hasattr(dag_adapter, "node_ids"), "dag adapter must have node_ids"
    node_ids = list(dag_adapter.node_ids)
    assert len(node_ids) == 2

    # Verify adapter get_node returns objects with capabilities_required
    node0 = dag_adapter.get_node(node_ids[0])
    assert hasattr(node0, "capabilities_required")
    assert "tool_role" in node0.capabilities_required

    # Verify providers built correctly
    assert len(provider_specs) == 2
    model_ids = {p.name for p in provider_specs}
    assert "gemini-2.5-flash" in model_ids
    assert "gpt-5.3-codex" in model_ids


def test_verify_assignment_formal_logs_warning_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    """When verification fails (UNSAT), a warning is logged."""
    import logging

    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    nodes = [_MockTopologyNode(model_id="model-a", capabilities=["quantum_reasoning"])]
    ctx = PipelineContext(
        task="test task",
        topology=_MockTopology(nodes=nodes),
        domain="code",
        budget=5.0,
    )

    def _mock_verify_unsat(dag: object, providers: object) -> MagicMock:
        result = MagicMock()
        result.satisfied = False
        result.counterexample = "node '0' (model-a: missing {'quantum_reasoning'})"
        return result

    with patch("sage.pipeline.verify_provider_assignment", side_effect=_mock_verify_unsat):
        with caplog.at_level(logging.WARNING, logger="sage.pipeline"):
            pipeline._verify_assignment_formal(ctx)

    # A warning must have been logged
    assert any("provider" in msg.lower() or "assignment" in msg.lower() or "formal" in msg.lower()
               for msg in caplog.messages), f"Expected warning about provider assignment, got: {caplog.messages}"
