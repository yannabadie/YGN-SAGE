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
    episodic_memory: Any = None,
    semantic_memory: Any = None,
    memory_agent: Any = None,
    causal_memory: Any = None,
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

    # T2 phase 0/1 (cgpro 2026-04-29 cycle-7 post-flip): pass the 4
    # memory collaborators through to per-node agent loops so the write
    # gate can target real backends instead of always hitting
    # ``memory_backend_unwired``. Backends are optional — None preserves
    # legacy "ungated, no-op" behavior. boot creates these in
    # boot_memory.py, pipeline carries them on self, runner forwards
    # via the partial(create_node_agent_loop, ...) factory.
    if episodic_memory is not None:
        loop.episodic_memory = episodic_memory
    if semantic_memory is not None:
        loop.semantic_memory = semantic_memory
    if memory_agent is not None:
        loop.memory_agent = memory_agent
    if causal_memory is not None:
        loop.causal_memory = causal_memory

    # H1/H4 carryover: pipeline already handled routing and topology
    loop._skip_routing = True
    loop._current_topology = None

    return loop


def create_bypass_agent_loop(
    *,
    singleton: AgentLoop,
    llm_provider: LLMProvider,
    llm_config: LLMConfig,
    system_level: int,
    write_gate: Any | None = None,
    task_text: str = "",
    on_drift: Any = None,
    run_frame_builder: Any | None = None,
    runtime_node_run_id: str | None = None,
) -> AgentLoop:
    """Per-run AgentLoop for the single-agent bypass path (P6-A Phase A).

    Replaces the ~150-line snapshot/restore mutation block at
    ``pipeline.py:_stage_execute`` (lines 2300-2465) with a fresh
    AgentLoop instance carrying all per-ctx state from the start.
    See ``docs/adr/ADR-016-agent-loop-bypass-factory.md`` for the
    full design + migration plan.

    Cycle-11 Phase A: this function is callable but not yet wired
    into ``pipeline.py``. Phase B (cycle-12, behind cgpro DESIGN
    review) swaps the bypass mutation block for a single call to
    this factory, then removes the P6-B lock + ContextVar reentry
    guard. All 25 cycle-11 P9 phase 1 tests must pass byte-identically
    against the post-swap pipeline.

    Field ownership matrix
    ======================
    Per-run (factory args, never copied from singleton):
      - ``llm_provider`` / ``config.llm``
      - ``write_gate`` / ``gate_current_task`` / ``gate_source_tier``
      - ``_on_drift``
      - ``_run_frame_builder`` / ``_runtime_node_run_id``

    Derived from ``system_level`` (parametric, matches the existing
    bypass mutation block at pipeline.py:2356-2386):
      - ``config.validation_level`` (1/2/3 + sandbox-aware on S2)
      - ``config.max_steps`` ({1: 5, 2: 10, 3: 20})
      - ``config.stall_after_tool_steps`` (max-1 if max>5 else 0)

    Per-run constants for bypass path:
      - ``_skip_routing = True`` (H1: routing was Stage 0)
      - ``_current_topology = None`` (H4: bypass = no topology)

    Shared with singleton (copied by reference):
      - ``_tools`` (ToolRegistry)
      - ``sandbox_manager`` / ``exocortex`` / ``guardrail_pipeline``
      - ``episodic_memory`` / ``semantic_memory`` / ``memory_agent``
      - ``causal_memory`` / ``consolidator``
      - ``tool_executor`` / ``topology_engine``
      - ``agent_pool`` / ``metacognition`` / ``topology_population``
      - Ablation flags (``_skip_memory`` / ``_skip_avr`` /
        ``_skip_guardrails`` / ``_auto_evolve``) — set at boot,
        same across all runs in a process.

    Per-instance fresh (via ``AgentLoop.__init__`` defaults):
      - ``working_memory`` (new WorkingMemory(agent_id=config.name))
      - ``prm`` (new ProcessRewardModel)
      - All stats fields zeroed
        (``step_count``, ``total_inference_time``, ``total_cost_usd``,
        ``tool_call_count``, ``tool_turn_count``, ``executed_commands``)

    Args:
        singleton: The boot AgentLoop singleton (source of injected
            deps: sandbox, memory backends, tool_registry, etc.).
        llm_provider: Per-run LLM provider (bandit-selected or
            default routing).
        llm_config: Per-run LLM config matching the provider.
        system_level: 1/2/3 — drives max_steps + validation_level.
        write_gate: Per-task write gate (built by pipeline). None
            preserves "ungated, no-op" behavior.
        task_text: Task text for write_gate dedup. "" if no gate.
        on_drift: Drift callback (forwards to ProviderPool.record_failure).
        run_frame_builder: Per-run RunFrame builder.
        runtime_node_run_id: Per-run/per-node run id.

    Returns:
        Fresh AgentLoop ready for ``await loop.run(task)``. The
        instance is independent of the singleton: mutations to the
        returned loop do NOT affect the singleton or any concurrent
        bypass loop.
    """
    # config.validation_level scaling matches pipeline.py:2356-2361.
    # On S2 the singleton checks ``self._agent_loop.sandbox_manager``
    # — we mirror that here using the singleton's sandbox_manager
    # (since it's the field we copy from).
    if system_level >= 3:
        validation_level = 3
    elif system_level >= 2 and singleton.sandbox_manager is not None:
        validation_level = 2
    else:
        validation_level = 1

    # max_steps scaling matches pipeline.py:2372 (singleton mutation)
    # AND agent_loop_factory.py:136-141 (per-node factory). They are
    # the same formula by design.
    max_steps = {1: 5, 2: 10, 3: 20}.get(system_level, 10)

    # D8 stall cap mirrors agent_loop_factory.py:155-158.
    stall_after_tool_steps = max_steps - 1 if max_steps > 5 else 0

    # Build a config that copies stable fields from the singleton's
    # config (name, system_prompt, tools list, sandbox/exhaustion
    # flags) and overrides the per-run fields.
    base_config = singleton.config
    config = AgentConfig(
        name=base_config.name,
        llm=llm_config,
        system_prompt=base_config.system_prompt,
        max_steps=max_steps,
        tools=base_config.tools,  # tool name list, shared
        use_docker_sandbox=base_config.use_docker_sandbox,
        snapshot_to_restore=base_config.snapshot_to_restore,
        validation_level=validation_level,
        raise_on_exhaustion=base_config.raise_on_exhaustion,
        stall_after_tool_steps=stall_after_tool_steps,
        # Cycle-12 P6-A Phase B prep (cgpro DESIGN 2026-05-05 trap Q7):
        # `dangerous_tools` is a per-process flag (toggled by
        # `SAGE_DANGEROUS_TOOLS=1` for SWE-bench / advanced research
        # paths). The singleton inherits it from boot; the factory must
        # propagate it so post-swap bypass runs preserve the same
        # tool-registration semantics. Default-False would silently
        # block `execute_bash` on bench paths that need it.
        dangerous_tools=base_config.dangerous_tools,
    )

    loop = AgentLoop(
        config=config,
        llm_provider=llm_provider,
        tool_registry=singleton._tools,  # shared
        memory_compressor=singleton.memory_compressor,
        on_event=singleton._on_event,
    )

    # Shared injected deps — copy by reference from singleton.
    loop.sandbox_manager = singleton.sandbox_manager
    loop.exocortex = singleton.exocortex
    loop.guardrail_pipeline = singleton.guardrail_pipeline
    loop.episodic_memory = singleton.episodic_memory
    loop.semantic_memory = singleton.semantic_memory
    loop.memory_agent = singleton.memory_agent
    loop.causal_memory = singleton.causal_memory
    loop.consolidator = singleton.consolidator
    loop.tool_executor = singleton.tool_executor
    loop.topology_engine = singleton.topology_engine
    loop.agent_pool = singleton.agent_pool
    loop.metacognition = singleton.metacognition
    loop.topology_population = singleton.topology_population
    # Cycle-12 P6-A Phase B prep (cgpro DESIGN 2026-05-05 trap Q7):
    # `toolforge` is wired by boot onto the singleton AgentLoop; pre-swap
    # the bypass path was using the singleton directly so this attribute
    # was implicitly available. Post-swap the per-run instance MUST
    # carry it too — `AgentLoop._execute_tool_call` reads
    # `getattr(self, "toolforge", None)` and falls back to "unknown
    # tool" when absent, which silently breaks autonomous-synthesis
    # paths.
    loop.toolforge = getattr(singleton, "toolforge", None)
    # Same logic for `evolution_memory`: cycle-7 wiring on the singleton
    # exposes `loop.evolution_memory` to the per-turn evolution gate.
    # Without this propagation the bypass path would silently degrade
    # to "no evolution feedback" on every bypass run.
    loop.evolution_memory = getattr(singleton, "evolution_memory", None)

    # Ablation flags — set at boot via AblationConfig.apply, shared
    # across all runs in a process (NOT per-task ablation).
    loop._skip_memory = singleton._skip_memory
    loop._skip_avr = singleton._skip_avr
    loop._skip_guardrails = singleton._skip_guardrails
    loop._auto_evolve = singleton._auto_evolve

    # Per-run state for the bypass path (H1: routing already done in
    # Stage 0; H4: bypass means no topology was selected).
    loop._skip_routing = True
    loop._current_topology = None

    # Per-run write gate + drift callback. These are the cycle-7+
    # G-series + H6 wiring carried from the singleton mutation block.
    loop.write_gate = write_gate
    loop.gate_current_task = task_text
    if llm_config and getattr(llm_config, "model", None):
        from sage.memory.write_gate import infer_source_tier
        loop.gate_source_tier = infer_source_tier(llm_config.model)
    if on_drift is not None:
        loop._on_drift = on_drift

    # RunFrame correlation IDs — per-run, set by pipeline.run() before
    # this factory is called.
    loop._run_frame_builder = run_frame_builder
    loop._runtime_node_run_id = runtime_node_run_id

    return loop
