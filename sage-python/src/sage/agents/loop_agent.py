"""LoopAgent -- iterative refinement with exit condition."""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


class LoopAgent:
    """Run a single agent in a loop, feeding its output back as input.

    Parameters
    ----------
    name:
        Human-readable name for this composition.
    agent:
        Runnable agent with ``name: str`` and ``async def run(self, task: str) -> str``.
    max_iterations:
        Hard upper bound on the number of iterations.
    exit_condition:
        Optional callable ``(str) -> bool``.  If it returns ``True`` for an
        iteration's output, the loop terminates early.  When *None*, the loop
        always runs for *max_iterations*.
    """

    def __init__(
        self,
        name: str,
        agent: Any,
        max_iterations: int = 10,
        exit_condition: Callable[[str], bool] | None = None,
    ) -> None:
        self.name = name
        self.agent = agent
        self.max_iterations = max_iterations
        self._exit_condition = exit_condition

    async def run(self, task: str) -> str:
        """Execute the agent repeatedly, returning the last output."""
        current_input = task
        result = current_input  # fallback if max_iterations == 0

        for i in range(1, self.max_iterations + 1):
            log.debug("[%s] iteration %d/%d", self.name, i, self.max_iterations)
            result = await self.agent.run(current_input)

            if self._exit_condition is not None and self._exit_condition(result):
                log.debug("[%s] exit condition met at iteration %d", self.name, i)
                break

            current_input = result

        return result
