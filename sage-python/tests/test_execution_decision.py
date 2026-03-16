"""Test ExecutionDecision dataclass."""
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
