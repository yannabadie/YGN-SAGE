"""GapDetector — detect missing tool capabilities during execution.

Listens for unknown tool calls and creates CreationTickets for ToolForge.
Research basis: UCT (arXiv 2602.01983) — "creation ticket" triggered when
needed tool doesn't exist. Bounded queue prevents runaway ticket creation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CreationTicket:
    """Request to create a new tool to fill a capability gap."""

    task: str                    # Original task that triggered the gap
    gap_description: str         # What capability is missing
    required_interface: str      # Expected input/output schema (from tool_call args)
    context: str                 # Predecessor node outputs (truncated)
    created_at: int              # Task counter value at creation time
    attempts: int = 0            # Build loop attempts so far (max 3)
    tool_name_hint: str = ""     # Suggested tool name from the LLM's tool_call


class GapDetector:
    """Detect tool capability gaps and queue creation tickets.

    Bounded: max ``MAX_PENDING`` tickets. Tickets older than
    ``TICKET_TTL`` tasks are automatically expired.
    """

    MAX_PENDING = 5
    TICKET_TTL = 100  # tasks

    def __init__(self) -> None:
        self._pending: list[CreationTicket] = []
        self._task_count: int = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def on_unknown_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | Any,
        task: str,
        context: str = "",
    ) -> CreationTicket | None:
        """Called when a tool_call references an unknown tool.

        Returns a CreationTicket if one was created, or None if the queue
        is full or the tool name is empty.
        """
        if not tool_name or len(self._pending) >= self.MAX_PENDING:
            return None

        # Deduplicate: don't create ticket for a tool we already have a ticket for
        existing_names = {t.tool_name_hint for t in self._pending}
        if tool_name in existing_names:
            return None

        args_str = str(tool_args)[:300] if tool_args else ""
        ticket = CreationTicket(
            task=task[:500],
            gap_description=f"Tool '{tool_name}' not found in registry",
            required_interface=args_str,
            context=context[:500],
            created_at=self._task_count,
            tool_name_hint=tool_name,
        )
        self._pending.append(ticket)
        return ticket

    def tick(self) -> None:
        """Advance task counter and expire old tickets."""
        self._task_count += 1
        self._pending = [
            t for t in self._pending
            if self._task_count - t.created_at < self.TICKET_TTL
        ]

    def pop_tickets(self) -> list[CreationTicket]:
        """Return and clear all pending tickets."""
        tickets = list(self._pending)
        self._pending.clear()
        return tickets

    def clear(self) -> None:
        """Clear all pending tickets."""
        self._pending.clear()
