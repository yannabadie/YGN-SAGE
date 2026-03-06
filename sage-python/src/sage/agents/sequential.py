"""SequentialAgent -- chain agents in series."""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class Runnable(Protocol):
    """Minimal protocol: anything with a *name* and an async *run*."""

    name: str

    async def run(self, task: str) -> str: ...


class SequentialAgent:
    """Run a list of agents sequentially, piping each output to the next.

    Parameters
    ----------
    name:
        Human-readable name for this composition.
    agents:
        Ordered list of runnable agents.  Each must expose ``name: str``
        and ``async def run(self, task: str) -> str``.
    shared_state:
        Optional mutable dict shared across all agents in the sequence.
    """

    def __init__(
        self,
        name: str,
        agents: list[Any],
        shared_state: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.agents = agents
        self.shared_state: dict[str, Any] = shared_state if shared_state is not None else {}

    async def run(self, task: str) -> str:
        """Execute agents in order, chaining outputs.

        The initial *task* is fed to the first agent; each subsequent agent
        receives the output of its predecessor.  Returns the final output.
        """
        current_input = task
        for agent in self.agents:
            log.debug("[%s] running agent %s", self.name, agent.name)
            current_input = await agent.run(current_input)
        return current_input
