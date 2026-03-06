"""Tests for DynamicAgentFactory — self-programming agent creation."""
import pytest
from sage.agents.factory import DynamicAgentFactory, AgentBlueprint
from sage.providers.registry import ModelProfile


def _mock_profile():
    return ModelProfile(id="test-model", provider="test", family="test", available=True,
                       code_score=0.8, reasoning_score=0.7, cost_input=1.0, cost_output=5.0)


def test_blueprint_creation():
    bp = AgentBlueprint(
        name="sql-validator",
        role="Validate SQL queries for safety",
        needs_code=True,
        needs_reasoning=False,
    )
    assert bp.name == "sql-validator"
    assert bp.needs_code


def test_factory_creates_agent():
    factory = DynamicAgentFactory()
    bp = AgentBlueprint(name="test", role="Test agent")
    profile = _mock_profile()
    agent = factory.create(bp, profile)
    assert agent.name == "test"
    assert agent.model.id == "test-model"


def test_factory_generates_system_prompt():
    factory = DynamicAgentFactory()
    bp = AgentBlueprint(name="analyzer", role="Analyze code for bugs", needs_code=True)
    prompt = factory._build_prompt(bp)
    assert "Analyze code for bugs" in prompt
    assert "code" in prompt.lower()


@pytest.mark.asyncio
async def test_factory_parse_blueprints_from_text():
    factory = DynamicAgentFactory()
    text = """
    1. [CODE] Write the sorting function
    2. [REASON] Prove it terminates correctly
    3. [GENERAL] Write unit tests
    """
    blueprints = factory.parse_blueprints(text)
    assert len(blueprints) == 3
    assert blueprints[0].needs_code
    assert blueprints[1].needs_reasoning
