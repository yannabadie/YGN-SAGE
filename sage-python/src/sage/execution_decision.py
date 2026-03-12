"""ExecutionDecision — single authoritative routing decision.

Produced by AgentSystem.run(), consumed by CognitiveOrchestrator.run().
Eliminates split-brain routing where both systems make independent decisions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionDecision:
    """Authoritative routing decision for a task."""

    system: int  # 1, 2, or 3
    model_id: str
    topology_id: str | None = None
    budget_usd: float = 0.0
    guardrail_level: str = "standard"
