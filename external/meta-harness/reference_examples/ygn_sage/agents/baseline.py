"""Baseline candidate: the default SAGE system, no modifications.

Establishes the floor score every other candidate must beat. If the
proposer can't improve over this, Meta-Harness search returns empty.
"""
from __future__ import annotations

from typing import Any

from ..sage_candidate import SageCandidate


class Baseline(SageCandidate):
    name = "baseline"
    hypothesis = "SAGE default topology + routing + memory, no changes"
    axis = "baseline"

    def build_system(self, hints: dict[str, Any] | None = None) -> Any:
        from sage.boot import boot_agent_system
        # Default SAGE boot: loads cards.toml, wires provider pool,
        # initializes TopologyEngine, SystemRouter, ModelAssigner, etc.
        # Returns an AgentSystem on which `await system.run(task)` can
        # be called.
        return boot_agent_system()


CANDIDATE = Baseline()
