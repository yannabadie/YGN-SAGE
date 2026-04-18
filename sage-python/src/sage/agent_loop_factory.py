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


def _is_high_rigour_domain(task_domain: str) -> bool:
    """True for domains where Z3 PRM <think>-block reasoning is meaningful.

    Mirrors the Rust `is_high_rigour_domain` in sage-core/src/routing/
    model_assigner.rs (F7 floor — same predicate, same intent: separate
    "this task wants formal proofs" from "this task wants code". Don't
    drift between Python and Rust — both should answer the same question.

    For non-rigour domains (code/general), validation_level >= 3 still
    enables AVR (level 2) but skips PRM/CEGAR — Z3 assertions on a Python
    bug-fix patch don't carry signal, and the `<think>` block requirement
    triggers a retry+CEGAR thrashing loop on every model that doesn't
    natively emit them (observed 2026-04-17 SWE-bench smoke: 17 CEGAR
    failures, 6 RESET_AGENT, 6 SWITCH_MODEL → 0/3 patches generated).
    """
    d = task_domain.lower()
    return "math" in d or "formal" in d


def create_node_agent_loop(
    node_role: str,
    node_name: str,
    llm_provider: LLMProvider,
    llm_config: LLMConfig,
    tool_registry: ToolRegistry,
    system_prompt: str,
    system_level: int,
    task_domain: str = "",
    on_event: Any = None,
    on_drift: Any = None,
) -> AgentLoop:
    """Create an independent AgentLoop for a topology node.

    Each call returns a FRESH instance with its own WorkingMemory,
    CircuitBreakers, and DriftMonitor (H8: no shared mutable state).

    Tool filtering (H6):
    - actor/coder/planner: all tools (config.tools = None)
    - verifier: execute_bash + memory (can run tests, no code gen)
    - output_formatter/aggregator: memory only (no code execution)

    Validation (H6 + 2026-04-17 PM domain gate):
    - verifier/formatter/aggregator: validation_level=0 (no AVR/Z3)
    - S3 task + math/formal domain: validation=3 (PRM Z3 is meaningful)
    - S3 task + other domains:      validation=2 (AVR only — PRM thrash on code)
    - S2 task: validation=2
    - S1 task: validation=1

    `task_domain` defaults to "" (treated as non-rigour). The default
    keeps existing direct callers (without F7 wiring) working — they get
    AVR-only on S3, which is the safer choice for an unknown domain.
    """
    role_lower = node_role.lower()

    # Tool filtering
    tools: list[str] | None = None  # all tools for actors
    if any(r in role_lower for r in ("verif",)):
        tools = _VERIFIER_TOOLS
    elif any(r in role_lower for r in ("format", "output", "aggregat")):
        tools = _FORMATTER_TOOLS

    # Validation level. Sink/verifier roles always get 0 (H6: no recursive
    # validation). S3 tasks get PRM (level 3) ONLY when the domain wants
    # formal reasoning; other S3 tasks fall back to AVR (level 2) so the
    # Z3 <think>-block requirement doesn't trigger CEGAR thrashing on
    # code/general domains where `<think>` blocks aren't part of the
    # model's normal output (2026-04-17 SWE-bench smoke regression).
    if any(r in role_lower for r in _NO_VALIDATION_ROLES):
        validation = 0
    elif system_level >= 3:
        validation = 3 if _is_high_rigour_domain(task_domain) else 2
    elif system_level >= 2:
        validation = 2
    else:
        validation = 1

    # Per-node step budget: scales with system level so complex S3 tasks
    # (SWE-bench, agentic debugging) get enough room to explore + patch.
    # The old flat cap of 5 produced empty results on SWE-bench Lite smoke
    # (2026-04-17), because each node burned its budget on execute_bash
    # exploration before ever emitting a final answer — learn.py's
    # "Agent finished at step N" fallback then masked the failure
    # downstream. See docs/benchmarks/2026-04-17-swebench-smoke-debug.md.
    if system_level >= 3:
        node_max_steps = 20
    elif system_level >= 2:
        node_max_steps = 10
    else:
        node_max_steps = 5

    # D8 soft-cap (2026-04-18 audit, revised after first smoke): break
    # out of consecutive-tool-turn thrash with enough headroom for
    # legitimate multi-call work. First revision used max_steps//2
    # which killed a 5-task SWE-Lite run (0/5 real vs 3/5 baseline —
    # coder bailed at 5/10, synthesizer at 2/5, before either could
    # emit a patch). SWE-bench coders need 8-15 tool turns (grep +
    # read_file + run_tests + edit) before final content. The cap
    # has to be "almost the full budget" — catch the pathological
    # 20-for-20 thrash, not normal exploration.
    #
    # Heuristic: leave the final 3 steps available for content
    # emission. Small S1 budgets (max=5) disable D8 entirely (cap=0)
    # since a 3-step thrash window would still break legitimate
    # execute_bash → read_file → final-answer chains.
    if node_max_steps <= 5:
        node_stall_cap = 0  # disabled for S1
    else:
        node_stall_cap = max(0, node_max_steps - 3)

    config = AgentConfig(
        name=node_name,
        llm=llm_config,
        system_prompt=system_prompt,
        max_steps=node_max_steps,
        validation_level=validation,
        tools=tools,
        stall_after_tool_steps=node_stall_cap,
    )

    loop = AgentLoop(
        config=config,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        on_event=on_event,
    )
    # D6 audit fix (2026-04-18): wire the drift callback so SWITCH_MODEL/
    # RESET_AGENT drift classifications from monitoring/drift.py get
    # forwarded to ProviderPool.record_failure via the runner's wiring.
    if on_drift is not None:
        loop._on_drift = on_drift

    # H1/H4 carryover: pipeline already handled routing and topology
    loop._skip_routing = True
    loop._current_topology = None

    return loop
