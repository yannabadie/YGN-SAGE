"""Conformance tests for MCP and A2A protocol servers."""
import pytest
from unittest.mock import MagicMock, AsyncMock

try:
    from sage.protocols.mcp_server import create_mcp_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

try:
    from sage.protocols.a2a_server import build_agent_card, SageAgentExecutor
    HAS_A2A = True
except ImportError:
    HAS_A2A = False


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MCP, reason="MCP deps not installed")
def test_mcp_server_creates():
    """MCP server can be created with mock components."""
    tool_registry = MagicMock()
    tool_registry._tools = {}
    agent_loop = MagicMock()
    event_bus = MagicMock()
    event_bus.query.return_value = []
    server = create_mcp_server(tool_registry, agent_loop, event_bus)
    assert server is not None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_A2A, reason="A2A deps not installed")
def test_a2a_agent_card():
    """A2A agent card has required skills."""
    card = build_agent_card("test", "http://localhost:8002", "Test agent")
    assert len(card.skills) == 3
    skill_ids = [s.id for s in card.skills]
    assert "general" in skill_ids
    assert "code" in skill_ids
    assert "research" in skill_ids
