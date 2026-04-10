"""Per-node AgentLoop factory for topology execution.

Phase 2 of unified entry point: each topology node gets an independent
AgentLoop with role-filtered tools and per-node validation.

Hazards addressed:
- H6: Verifier nodes run with validation_level=0 (no recursive AVR/Z3)
- H8: Each call creates a fresh instance (no shared mutable state)
"""
from __future__ import annotations

from typing import Any

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.llm.base import LLMConfig, LLMProvider
from sage.tools.registry import ToolRegistry

# Tool sets per role (H6: prevent recursive validation on verifiers)
_VERIFIER_TOOLS = ["execute_bash", "stm_read", "stm_write", "ltm_recall"]
_FORMATTER_TOOLS = ["stm_read", "stm_write", "ltm_recall"]

# Roles that get restricted validation (H6)
_NO_VALIDATION_ROLES = {"verifier", "output_formatter", "formatter", "aggregator", "critic"}


def create_node_agent_loop(
    node_role: str,
    node_name: str,
    llm_provider: LLMProvider,
    llm_config: LLMConfig,
    tool_registry: ToolRegistry,
    system_prompt: str,
    system_level: int,
    on_event: Any = None,
) -> AgentLoop:
    """Create an independent AgentLoop for a topology node.

    Each call returns a FRESH instance with its own WorkingMemory,
    CircuitBreakers, and DriftMonitor (H8: no shared mutable state).

    Tool filtering (H6):
    - actor/coder/planner: all tools (config.tools = None)
    - verifier: execute_bash + memory (can run tests, no code gen)
    - output_formatter/aggregator: memory only (no code execution)

    Validation (H6):
    - actor/coder: full validation from system_level
    - verifier/formatter/aggregator: validation_level=0 (no AVR/Z3)
    """
    role_lower = node_role.lower()

    # Tool filtering
    tools: list[str] | None = None  # all tools for actors
    if any(r in role_lower for r in ("verif",)):
        tools = _VERIFIER_TOOLS
    elif any(r in role_lower for r in ("format", "output", "aggregat")):
        tools = _FORMATTER_TOOLS

    # Validation level (H6: no validation on verifiers to prevent recursion)
    if any(r in role_lower for r in _NO_VALIDATION_ROLES):
        validation = 0
    elif system_level >= 3:
        validation = 3
    elif system_level >= 2:
        validation = 2
    else:
        validation = 1

    config = AgentConfig(
        name=node_name,
        llm=llm_config,
        system_prompt=system_prompt,
        max_steps=5,  # topology nodes: 1-3 steps typical, 5 max to prevent timeouts
        validation_level=validation,
        tools=tools,
    )

    loop = AgentLoop(
        config=config,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        on_event=on_event,
    )

    # H1/H4 carryover: pipeline already handled routing and topology
    loop._skip_routing = True
    loop._current_topology = None

    return loop
