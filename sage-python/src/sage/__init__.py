"""YGN-SAGE: Self-Adaptive Generation Engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

from sage.agent import Agent, AgentConfig
from sage.llm import LLMConfig
from sage.tools import Tool, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from sage.boot import AgentSystem


async def create(
    *,
    mock: bool = False,
    tier: str = "auto",
    name: str = "sage-agent",
    tools: list[Tool] | None = None,
) -> AgentSystem:
    """Create a ready-to-use AgentSystem.

    Args:
        mock: If True, use a mock LLM (for testing).
        tier: Model tier to use ("auto" picks the best available).
        name: Name for the root agent.
        tools: Optional list of extra tools to register.

    Returns:
        A fully booted AgentSystem instance.
    """
    from sage.boot import boot_agent_system  # noqa: E402
    from sage.events.bus import EventBus  # noqa: E402

    system = boot_agent_system(
        use_mock_llm=mock,
        llm_tier=tier,
        agent_name=name,
        event_bus=EventBus(),
    )

    if tools:
        for tool in tools:
            system.tool_registry.register(tool)

    return system


__all__ = [
    "Agent",
    "AgentConfig",
    "LLMConfig",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "create",
]
