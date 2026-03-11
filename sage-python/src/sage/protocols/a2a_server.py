"""A2A (Agent-to-Agent) protocol server for YGN-SAGE.

Exposes YGN-SAGE as an A2A-compatible agent, enabling delegation from
Google ADK, LangGraph, CrewAI, or any A2A client.

Requires a2a-sdk >= 1.0 (pinned to v1.0, breaking changes from v0.3 confirmed).

Usage:
    from sage.protocols.a2a_server import create_a2a_app
    app = create_a2a_app(agent_loop, tool_registry, event_bus)
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""
from __future__ import annotations

import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor  # type: ignore[import-untyped]
from a2a.server.apps import A2AStarletteApplication  # type: ignore[import-untyped]
from a2a.server.request_handlers import DefaultRequestHandler  # type: ignore[import-untyped]
from a2a.server.tasks import InMemoryTaskStore  # type: ignore[import-untyped]
from a2a.types import AgentCapabilities, AgentCard, AgentSkill  # type: ignore[import-untyped]
from a2a.utils.helpers import new_agent_text_message  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)


class SageAgentExecutor(AgentExecutor):
    """Wraps YGN-SAGE AgentLoop as an A2A AgentExecutor."""

    def __init__(self, agent_loop: Any | None = None):
        self._agent_loop = agent_loop

    async def execute(self, context: Any, event_queue: Any) -> None:
        """Execute a task via the SAGE cognitive pipeline."""
        # Extract task text from A2A message
        task_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    task_text += part.text

        if not task_text:
            await event_queue.enqueue_event(
                new_agent_text_message("Error: empty task")
            )
            return

        if self._agent_loop is None:
            await event_queue.enqueue_event(
                new_agent_text_message("Error: AgentLoop not configured")
            )
            return

        try:
            result = await self._agent_loop.run(task_text)
            text = result if isinstance(result, str) else str(result)
            await event_queue.enqueue_event(new_agent_text_message(text))
        except Exception as exc:
            _log.error("A2A execution error: %s", exc)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {exc}")
            )

    async def cancel(self, context: Any, event_queue: Any) -> None:
        """Cancel not yet supported."""
        raise NotImplementedError("Task cancellation not yet supported")


def build_agent_card(
    name: str = "YGN-SAGE",
    url: str = "http://localhost:8002",
    description: str | None = None,
) -> AgentCard:
    """Build an A2A AgentCard describing SAGE capabilities."""
    return AgentCard(
        name=name,
        description=description or (
            "YGN-SAGE: Self-Adaptive Generation Engine with cognitive routing "
            "(S1/S2/S3), formal verification, evolutionary topology search, "
            "4-tier memory, and 7-provider model selection."
        ),
        url=url,
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="general",
                name="General Task Execution",
                description=(
                    "Execute any task through the cognitive pipeline "
                    "with automatic S1/S2/S3 routing."
                ),
                tags=["general", "coding", "reasoning", "math"],
                examples=["Write a Python function", "Prove sqrt(2) is irrational"],
            ),
            AgentSkill(
                id="code",
                name="Code Generation & Analysis",
                description="Generate, review, and fix code with formal verification.",
                tags=["code", "python", "rust", "review"],
                examples=["Implement a binary search tree", "Fix this bug"],
            ),
            AgentSkill(
                id="research",
                name="Knowledge Retrieval",
                description=(
                    "Search ExoCortex research store (500+ papers) "
                    "and answer questions."
                ),
                tags=["research", "papers", "knowledge"],
                examples=["What is MAP-Elites?", "Summarize PSRO"],
            ),
        ],
    )


def create_a2a_app(
    agent_loop: Any | None = None,
    tool_registry: Any | None = None,  # noqa: ARG001
    event_bus: Any | None = None,  # noqa: ARG001
    name: str = "YGN-SAGE",
    url: str = "http://localhost:8002",
) -> A2AStarletteApplication:
    """Create an A2A Starlette application wrapping SAGE.

    Parameters
    ----------
    agent_loop:
        AgentLoop instance for task execution.
    tool_registry:
        ToolRegistry instance (reserved for future skill auto-discovery).
    event_bus:
        EventBus instance (reserved for future event streaming).
    name:
        Agent name for the AgentCard.
    url:
        Public URL where this agent is reachable.
    """
    agent_card = build_agent_card(name=name, url=url)
    executor = SageAgentExecutor(agent_loop=agent_loop)
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    _log.info("A2A server created: %s at %s", name, url)
    return app
