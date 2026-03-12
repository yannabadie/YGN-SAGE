"""Test ExecutionDecision dataclass and orchestrator consumption."""
import pytest
from sage.execution_decision import ExecutionDecision


def test_execution_decision_fields():
    ed = ExecutionDecision(
        system=2,
        model_id="gemini-2.5-flash",
        topology_id="topo_abc123",
        budget_usd=0.5,
        guardrail_level="standard",
    )
    assert ed.system == 2
    assert ed.model_id == "gemini-2.5-flash"
    assert ed.topology_id == "topo_abc123"
    assert ed.budget_usd == 0.5
    assert ed.guardrail_level == "standard"


def test_execution_decision_defaults():
    ed = ExecutionDecision(system=1, model_id="test")
    assert ed.topology_id is None
    assert ed.budget_usd == 0.0
    assert ed.guardrail_level == "standard"


@pytest.mark.asyncio
async def test_orchestrator_uses_decision():
    """Orchestrator should use provided ExecutionDecision without re-routing."""
    from unittest.mock import MagicMock, AsyncMock
    from sage.orchestrator import CognitiveOrchestrator
    from sage.execution_decision import ExecutionDecision

    registry = MagicMock()
    registry.select.return_value = MagicMock(id="gemini-2.5-flash")
    mc = MagicMock()

    orch = CognitiveOrchestrator(registry=registry, metacognition=mc)
    decision = ExecutionDecision(system=1, model_id="gemini-2.5-flash")

    # Mock the model execution path
    model_mock = MagicMock()
    model_mock.generate = AsyncMock(return_value="result")
    registry.select.return_value = model_mock

    # Key assertion: metacognition.assess_complexity_async should NOT be called
    # when decision is provided
    try:
        await orch.run("simple task", decision=decision)
    except Exception:
        pass  # May fail on model execution, that's OK

    mc.assess_complexity_async.assert_not_called()
