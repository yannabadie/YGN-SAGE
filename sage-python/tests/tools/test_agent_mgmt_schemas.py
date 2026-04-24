"""A14 closure: ``output_schema`` attached + validated on agent_mgmt tools.

These tests confirm that the 2 JSON-returning tools in `agent_mgmt.py`
— ``call_agent`` and ``list_active_agents`` — actually round-trip
through Pydantic validation end-to-end (not just "schema attached, no
tool uses it"). This moves AUDIT3 #17 from ⚠️ library-only to ❌
on two concrete entry points.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.tools.agent_mgmt import (
    AgentInfo,
    CallAgentResult,
    ListActiveAgentsResult,
    call_agent,
    list_active_agents,
)


# -------- schema attachment (library-only sanity, unchanged from A14) --------


def test_call_agent_has_output_schema_attached() -> None:
    assert call_agent.output_schema is CallAgentResult


def test_list_active_agents_has_output_schema_attached() -> None:
    assert list_active_agents.output_schema is ListActiveAgentsResult


# -------- end-to-end: handler output round-trips through schema --------


@pytest.mark.asyncio
async def test_call_agent_output_validates_against_schema() -> None:
    """Invoke call_agent with agent_pool injection; validate JSON output.

    `Tool.execute()` only forwards declared parameters from the spec;
    contextual injections (``agent_pool``, ``parent_agent``) are bound
    at the registry layer at runtime. This test exercises the handler
    directly and then wraps the output in a ToolResult to round-trip
    through `validate_output`, matching the real runtime path.
    """
    sub_agent = MagicMock()
    sub_agent.run = AsyncMock(return_value="Sub-agent completed task.")
    agent_pool = {"worker": sub_agent}

    output = await call_agent._handler(
        agent_name="worker",
        task_message="do something",
        agent_pool=agent_pool,
    )

    from sage.tools.base import ToolResult
    result = ToolResult(output=output)

    # Round-trip: raw JSON string -> validated Pydantic instance
    parsed = result.validate_output(CallAgentResult)
    assert isinstance(parsed, CallAgentResult)
    assert parsed.agent == "worker"
    assert parsed.status == "success"
    assert parsed.result == "Sub-agent completed task."
    assert result.validated is parsed


@pytest.mark.asyncio
async def test_call_agent_handler_with_pool_injection() -> None:
    """The decorator strips `agent_pool` kwarg; call the handler directly."""
    sub_agent = MagicMock()
    sub_agent.run = AsyncMock(return_value="done")
    output = await call_agent._handler(
        agent_name="w",
        task_message="t",
        agent_pool={"w": sub_agent},
    )
    # Direct handler returns the JSON string; schema validates it
    from sage.tools.base import ToolResult
    result = ToolResult(output=output)
    parsed = result.validate_output(CallAgentResult)
    assert parsed.agent == "w"
    assert parsed.status == "success"


@pytest.mark.asyncio
async def test_list_active_agents_output_validates_against_schema() -> None:
    """list_active_agents returns JSON; confirm schema round-trip."""
    class FakeConfig:
        tools = ["read_file", "write_file"]

    class FakeAgent:
        def __init__(self):
            self.config = FakeConfig()
            self.step_count = 3

    agent_pool = {"alpha": FakeAgent(), "beta": FakeAgent()}

    output = await list_active_agents._handler(agent_pool=agent_pool)

    from sage.tools.base import ToolResult
    result = ToolResult(output=output)
    parsed = result.validate_output(ListActiveAgentsResult)
    assert isinstance(parsed, ListActiveAgentsResult)
    assert len(parsed.active_agents) == 2
    names = {a.name for a in parsed.active_agents}
    assert names == {"alpha", "beta"}
    for a in parsed.active_agents:
        assert isinstance(a, AgentInfo)
        assert a.steps_taken == 3


@pytest.mark.asyncio
async def test_call_agent_error_path_returns_string_not_json() -> None:
    """Error paths return plain strings; schema validation returns None.

    This documents the 2c opt-in policy: the schema models the SUCCESS
    contract only. Error paths are free-form (backwards compat).
    """
    # agent_pool=None → error path, returns "Error: Agent pool not available."
    output = await call_agent._handler(
        agent_name="x",
        task_message="y",
        agent_pool=None,
    )
    from sage.tools.base import ToolResult
    result = ToolResult(output=output)
    # Non-JSON string → validate_output returns None (warn-silent)
    assert result.validate_output(CallAgentResult) is None


def test_strict_env_raises_on_non_json_error_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAGE_TOOLRESULT_VALIDATE=1 promotes validate_output error to raise."""
    monkeypatch.setenv("SAGE_TOOLRESULT_VALIDATE", "1")
    from sage.tools.base import ToolResult
    result = ToolResult(output="Error: Agent pool not available.")
    with pytest.raises(ValueError, match="not valid JSON"):
        result.validate_output(CallAgentResult)
