"""Dynamic sub-agent pool for multi-agent orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SubAgentSpec:
    """Specification for creating a sub-agent."""
    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    llm_tier: str = "fast"  # fast, mutator, reasoner, codex
    max_steps: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentPool:
    """Thread-safe pool for managing dynamically created sub-agents.

    OpenSAGE-style: parent agent can create, run, and ensemble sub-agents.
    """

    def __init__(self):
        self._specs: dict[str, SubAgentSpec] = {}
        self._results: dict[str, str] = {}
        self._running: set[str] = set()

    def register(self, spec: SubAgentSpec) -> None:
        """Register a new sub-agent specification."""
        self._specs[spec.name] = spec
        log.info(f"Registered sub-agent: {spec.name} (role={spec.role})")

    def deregister(self, name: str) -> None:
        """Remove a sub-agent from the pool."""
        self._specs.pop(name, None)
        self._results.pop(name, None)
        self._running.discard(name)

    def get_spec(self, name: str) -> SubAgentSpec | None:
        return self._specs.get(name)

    def list_agents(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "role": s.role,
                "llm_tier": s.llm_tier,
                "tools": s.tools,
                "running": s.name in self._running,
                "has_result": s.name in self._results,
            }
            for s in self._specs.values()
        ]

    def mark_running(self, name: str) -> None:
        self._running.add(name)

    def store_result(self, name: str, result: str) -> None:
        self._results[name] = result
        self._running.discard(name)

    def collect_results(self) -> dict[str, str]:
        """Collect all completed results (ensemble pattern)."""
        return dict(self._results)

    def clear_results(self) -> None:
        self._results.clear()
