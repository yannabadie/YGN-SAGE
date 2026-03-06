"""Handoff -- transfer control between agents."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class HandoffResult:
    """Outcome of a handoff execution."""

    output: str
    target_name: str


class Handoff:
    """Transfer a task to a target agent, optionally transforming the input.

    Parameters
    ----------
    target:
        Runnable agent with ``name: str`` and ``async def run(self, task: str) -> str``.
    description:
        Human-readable description of what this handoff does (useful for
        routing decisions and observability).
    input_filter:
        Optional callable ``(str) -> str`` to transform the input before
        sending it to the target agent.
    on_handoff:
        Optional callback ``(target_name: str, task: str) -> None`` invoked
        immediately before the target agent runs.  Useful for logging,
        metrics, or side-effects.
    """

    def __init__(
        self,
        target: Any,
        description: str,
        input_filter: Callable[[str], str] | None = None,
        on_handoff: Callable[[str, str], None] | None = None,
    ) -> None:
        self.target = target
        self.description = description
        self._input_filter = input_filter
        self._on_handoff = on_handoff

    async def execute(self, task: str) -> HandoffResult:
        """Hand off *task* to the target agent and return a :class:`HandoffResult`.

        1. Apply ``input_filter`` (if provided).
        2. Fire ``on_handoff`` callback (if provided).
        3. Run the target agent.
        4. Wrap and return the result.
        """
        effective_task = self._input_filter(task) if self._input_filter else task

        if self._on_handoff is not None:
            self._on_handoff(self.target.name, effective_task)

        log.debug(
            "[Handoff] %s -> %s: %s",
            self.description,
            self.target.name,
            effective_task[:80],
        )

        output = await self.target.run(effective_task)
        return HandoffResult(output=output, target_name=self.target.name)
