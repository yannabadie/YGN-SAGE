"""Test A2A agent wrapper."""
import pytest

a2a = pytest.importorskip("a2a", reason="a2a-sdk not installed")


def test_build_agent_card():
    from sage.protocols.a2a_server import build_agent_card

    card = build_agent_card(name="test-sage", url="http://localhost:8002")
    assert card.name == "test-sage"
    assert len(card.skills) >= 1  # At least the "general" skill


def test_agent_card_has_three_skills():
    from sage.protocols.a2a_server import build_agent_card

    card = build_agent_card()
    skill_ids = [s.id for s in card.skills]
    assert "general" in skill_ids
    assert "code" in skill_ids
    assert "research" in skill_ids


def test_sage_agent_executor_without_loop():
    """SageAgentExecutor is constructable without an agent loop."""
    from sage.protocols.a2a_server import SageAgentExecutor

    executor = SageAgentExecutor(agent_loop=None)
    assert executor._agent_loop is None
