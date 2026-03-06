"""Multi-agent composition patterns for YGN-SAGE.

Provides four composition primitives:
- SequentialAgent: chain agents in series (output -> next input)
- ParallelAgent: fan-out via asyncio.gather, aggregate results
- LoopAgent: iterative refinement with exit condition
- Handoff / HandoffResult: transfer control between agents
"""
from __future__ import annotations

from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.agents.loop_agent import LoopAgent
from sage.agents.handoff import Handoff, HandoffResult

__all__ = [
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "Handoff",
    "HandoffResult",
]
