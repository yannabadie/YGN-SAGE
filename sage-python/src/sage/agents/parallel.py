"""ParallelAgent -- fan-out via asyncio.gather, aggregate results."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


def _default_aggregator(results: dict[str, str]) -> str:
    """Join results as ``[agent_name]: result`` blocks separated by blank lines."""
    return "\n\n".join(f"[{name}]: {output}" for name, output in results.items())


class ParallelAgent:
    """Run multiple agents concurrently and aggregate their results.

    Parameters
    ----------
    name:
        Human-readable name for this composition.
    agents:
        List of runnable agents.  Each must expose ``name: str``
        and ``async def run(self, task: str) -> str``.
    aggregator:
        Optional callable ``(dict[str, str]) -> str`` to combine per-agent
        results into a single string.  Receives a dict mapping agent name to
        its output.  Defaults to a newline-separated ``[name]: output`` join.
    """

    def __init__(
        self,
        name: str,
        agents: list[Any],
        aggregator: Callable[[dict[str, str]], str] | None = None,
    ) -> None:
        self.name = name
        self.agents = agents
        self._aggregator = aggregator or _default_aggregator

    async def run(self, task: str) -> str:
        """Fan-out *task* to all agents concurrently and aggregate results."""

        async def _run_one(agent: Any) -> tuple[str, str]:
            log.debug("[%s] launching agent %s", self.name, agent.name)
            output = await agent.run(task)
            return agent.name, output

        pairs = await asyncio.gather(*[_run_one(a) for a in self.agents])
        # Preserve original agent ordering in the results dict
        results: dict[str, str] = {name: output for name, output in pairs}
        return self._aggregator(results)
