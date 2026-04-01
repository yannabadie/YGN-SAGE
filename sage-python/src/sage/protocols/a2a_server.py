"""A2A (Agent-to-Agent) protocol server for YGN-SAGE.

Exposes YGN-SAGE as an A2A-compatible agent with streaming per-node events,
task lifecycle management, and cancellation support.

Requires a2a-sdk (a2a-python). Pinned to latest API (April 2026).

Usage:
    from sage.protocols.a2a_server import create_a2a_app
    app = create_a2a_app(agent_loop, pipeline)
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""
from __future__ import annotations

import logging
from typing import Any

from a2a.server.agent import AgentExecutor  # type: ignore[import-untyped]
from a2a.server.request import RequestContext  # type: ignore[import-untyped]
from a2a.server.events import EventQueue  # type: ignore[import-untyped]
from a2a.server.tasks import InMemoryTaskStore  # type: ignore[import-untyped]
from a2a.server.handlers import DefaultRequestHandler  # type: ignore[import-untyped]
from a2a.server.fastapi import A2AFastAPIApplication  # type: ignore[import-untyped]
from a2a.agent import create_agent_card  # type: ignore[import-untyped]
from a2a.messages import new_task, new_agent_text_message  # type: ignore[import-untyped]
from a2a.tasks import TaskUpdater  # type: ignore[import-untyped]
from a2a.types import (  # type: ignore[import-untyped]
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Part,
    TextPart,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus,
)

try:
    from a2a.server.task_state import TaskState  # type: ignore[import-untyped]
except ImportError:
    # Fallback enum if task_state module not available
    class TaskState:  # type: ignore[no-redef]
        working = "working"
        completed = "completed"
        failed = "failed"
        canceled = "canceled"
        input_required = "input-required"

_log = logging.getLogger(__name__)


class SageAgentExecutor(AgentExecutor):
    """Wraps YGN-SAGE pipeline as an A2A AgentExecutor with streaming.

    Streams per-node topology events via TaskArtifactUpdateEvent,
    reports task lifecycle via TaskStatusUpdateEvent.
    """

    def __init__(
        self,
        agent_loop: Any | None = None,
        pipeline: Any | None = None,
    ):
        self._agent_loop = agent_loop
        self._pipeline = pipeline
        self._cancelled: set[str] = set()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a task via the SAGE cognitive pipeline with streaming."""
        # Extract task text from A2A message
        task_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "root") and hasattr(part.root, "text"):
                    task_text += part.root.text
                elif hasattr(part, "text"):
                    task_text += part.text

        task = context.task or new_task(request=context.message)
        updater = TaskUpdater(event_queue, task)

        if not task_text:
            await updater.failed(
                new_agent_text_message("Error: empty task", context.context_id, task.id)
            )
            return

        if self._agent_loop is None:
            await updater.failed(
                new_agent_text_message(
                    "Error: AgentLoop not configured", context.context_id, task.id
                )
            )
            return

        # Signal working state
        await event_queue.put(
            TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        try:
            # Try streaming path via pipeline's TopologyRunner
            if self._pipeline and hasattr(self._pipeline, "run_stream"):
                final_output = ""
                async for event in self._pipeline.run_stream(task_text):
                    # Check cancellation
                    if task.id in self._cancelled:
                        self._cancelled.discard(task.id)
                        await updater.cancel()
                        return

                    if event.get("type") == "node_start":
                        # Stream progress update
                        artifact = Artifact(
                            artifact_id="progress",
                            name="Topology Progress",
                            parts=[
                                Part(
                                    root=TextPart(
                                        text=f"[{event['role']}] starting..."
                                    )
                                )
                            ],
                        )
                        await event_queue.put(
                            TaskArtifactUpdateEvent(
                                task_id=task.id,
                                context_id=task.context_id,
                                artifact=artifact,
                                append=True,
                                last_chunk=False,
                            )
                        )

                    elif event.get("type") == "node_done":
                        final_output = event.get("output", "")
                        artifact = Artifact(
                            artifact_id=f"node-{event.get('node_idx', 0)}",
                            name=event.get("role", "agent"),
                            parts=[
                                Part(root=TextPart(text=final_output[:2000]))
                            ],
                        )
                        await event_queue.put(
                            TaskArtifactUpdateEvent(
                                task_id=task.id,
                                context_id=task.context_id,
                                artifact=artifact,
                                append=False,
                                last_chunk=False,
                            )
                        )

                    elif event.get("type") == "topology_done":
                        final_output = event.get("final_output", final_output)

                # Complete with final result
                await updater.complete(
                    new_agent_text_message(final_output, context.context_id, task.id)
                )
            else:
                # Fallback: synchronous full run
                result = await self._agent_loop.run(task_text)
                text = result if isinstance(result, str) else str(result)
                await updater.complete(
                    new_agent_text_message(text, context.context_id, task.id)
                )
        except Exception as exc:
            _log.error("A2A execution error: %s", exc)
            await updater.failed(
                new_agent_text_message(
                    f"Error: {exc}", context.context_id, task.id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        if context.task:
            self._cancelled.add(context.task.id)
            updater = TaskUpdater(event_queue, context.task)
            await updater.cancel()
        else:
            _log.warning("A2A cancel requested but no task context")


def build_agent_card(
    name: str = "YGN-SAGE",
    url: str = "http://localhost:8002",
    description: str | None = None,
) -> AgentCard:
    """Build an A2A AgentCard describing SAGE capabilities."""
    return AgentCard(
        name=name,
        description=description
        or (
            "YGN-SAGE: Self-Adaptive Generation Engine with cognitive routing "
            "(S1/S2/S3), 11 topology templates, formal verification, evolutionary "
            "topology search, 4-tier memory, 7-provider model selection, and "
            "per-node streaming."
        ),
        url=url,
        version="0.2.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="general",
                name="General Task Execution",
                description=(
                    "Execute any task through the cognitive pipeline "
                    "with automatic S1/S2/S3 routing and multi-agent topology."
                ),
                tags=["general", "coding", "reasoning", "math"],
                examples=[
                    "Write a Python function",
                    "Prove sqrt(2) is irrational",
                ],
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
    pipeline: Any | None = None,
    tool_registry: Any | None = None,  # noqa: ARG001
    event_bus: Any | None = None,  # noqa: ARG001
    name: str = "YGN-SAGE",
    url: str = "http://localhost:8002",
) -> Any:
    """Create an A2A FastAPI application wrapping SAGE.

    Parameters
    ----------
    agent_loop:
        AgentLoop instance for fallback task execution.
    pipeline:
        CognitiveOrchestrationPipeline instance for streaming execution.
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
    executor = SageAgentExecutor(agent_loop=agent_loop, pipeline=pipeline)
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    app_builder = A2AFastAPIApplication(agent_card, handler)
    app = app_builder.build()
    _log.info("A2A server created: %s at %s (streaming=%s)", name, url, True)
    return app
