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
# F5 audit fix attempt (2026-04-18): stripped execute_bash from planner,
# hoping to force plain-text emission on fast-tier S1 model. Smoke v5
# result: sentinels went from 1 (v3/v4) back to 3 — WORSE signal.
# Hypothesis: without execute_bash, the planner emits text BUT the plan
# is structurally weaker (no repo grounding) and downstream coder can't
# execute it, cascading new sentinels. Reverted. Log retained for
# posterity at docs/benchmarks/2026-04-18-swebench-smoke-v5-f5-structural-reverted.log.
# _PLANNER_TOOLS = ["stm_read", "stm_write", "ltm_recall", "search_memory"]  # F5 reverted

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
    write_gate: Any = None,
    task_text: str = "",
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
    elif any(r in role_lower for r in ("format", "output", "aggregat", "synth")):
        # F10 audit fix (2026-04-19 docs/audits/2026-04-18-astropy-14995-*):
        # synthesizer was missing from the formatter filter — it matched
        # neither "format" nor "output" nor "aggregat". That let it
        # receive the full actor toolset including execute_bash. In v9,
        # 4/4 sentinels came from synthesizer nodes burning their 5-step
        # S1 budget on tool calls instead of emitting the SINK_NODE_PROMPT
        # "output ONLY the final answer". Restricting to memory-only tools
        # forces text emission.
        tools = _FORMATTER_TOOLS
    # F5 attempted planner tool-stripping here — reverted, see the
    # _PLANNER_TOOLS comment block above for the result. F10 targets
    # sinks instead (they SHOULD have no tools by prompt; now enforced
    # structurally).

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

    # D8 soft-cap (2026-04-18 audit) — revised four times on empirical
    # smoke results:
    #   Rev 1 (max_steps//2): 0/5 real, coder bailed at 5/10.
    #   Rev 2 (max_steps-3):  1/5 real, astropy-6938 regressed.
    #   Rev 3 (S2: max-1, S3: max-3) — F1: S2 coder routinely uses full
    #     10-step budget; only 1-step headroom catches 20-for-20 thrash.
    #   Rev 4 (S3: max-1) — F12 (2026-04-19): after F8 put coder on S3
    #     (max=20), v11 showed 3/5 EMPTY tasks stalled at exactly 17/20
    #     on legitimate grep+read+edit cycles that would have completed
    #     at step 18-19. Match S2's logic: 1-step headroom is enough for
    #     the pathological case, preserves 2 more exploration steps for
    #     the typical case.
    if node_max_steps <= 5:
        node_stall_cap = 0  # S1 — budget too tight for any window
    else:
        node_stall_cap = node_max_steps - 1  # S2: 9, S3: 19

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

    # G-series audit fix (2026-04-19): share the pipeline's write gate across
    # nodes and derive the source tier from the assigned model id via cards.toml.
    # When write_gate is None (e.g. legacy direct callers), memory writes stay
    # ungated — same behavior as before.
    loop.write_gate = write_gate
    loop.gate_current_task = task_text
    if llm_config and getattr(llm_config, "model", None):
        from sage.memory.write_gate import infer_source_tier
        loop.gate_source_tier = infer_source_tier(llm_config.model)

    # H1/H4 carryover: pipeline already handled routing and topology
    loop._skip_routing = True
    loop._current_topology = None

    return loop
