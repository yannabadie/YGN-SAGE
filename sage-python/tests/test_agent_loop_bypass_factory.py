"""P6-A Phase A: characterization tests for create_bypass_agent_loop.

Locks the field-set + ownership matrix that ADR-016 defines for the
per-run AgentLoop bypass factory. Phase A (this commit) adds the
factory function alongside ``create_node_agent_loop()`` but does NOT
yet wire it into ``pipeline.py:_stage_execute``. Phase B (cycle-12,
behind cgpro DESIGN review) swaps the bypass mutation block for a
single ``create_bypass_agent_loop()`` call.

These tests prove the factory's output state is correct so cycle-12
phase B can swap with confidence — the byte-identical 25 cycle-11
P9 phase 1 tests + these factory tests + cgpro DESIGN review form
the cycle-12 acceptance gate.

Reference
=========
- ``docs/adr/ADR-016-agent-loop-bypass-factory.md`` (full design)
- ``sage-python/src/sage/pipeline.py:_stage_execute`` lines 2300-2465
  (the snapshot/restore mutation block being replaced)
- ``sage-python/src/sage/agent_loop_factory.py:create_node_agent_loop``
  (sibling factory, same pattern)
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.agent_loop_factory import create_bypass_agent_loop
from sage.llm.base import LLMConfig


def _build_singleton() -> AgentLoop:
    """Build a minimal singleton AgentLoop with all injected deps populated.

    The factory must copy these by reference. We use sentinel
    MagicMock instances so the post-factory tests can assert
    ``loop.X is singleton.X`` for shared fields.
    """
    config = AgentConfig(
        name="boot-singleton",
        llm=LLMConfig(provider="default", model="default-model"),
        system_prompt="You are a helpful assistant.",
        max_steps=20,           # singleton boot value (overridden by factory)
        tools=None,             # all tools available
        validation_level=1,
        stall_after_tool_steps=0,
    )
    singleton = AgentLoop(
        config=config,
        llm_provider=MagicMock(name="default-llm"),
        tool_registry=MagicMock(name="tool-registry"),
        on_event=MagicMock(name="on-event"),
    )
    # Inject shared deps as MagicMock sentinels.
    singleton.sandbox_manager = MagicMock(name="sandbox-manager")
    singleton.exocortex = MagicMock(name="exocortex")
    singleton.guardrail_pipeline = MagicMock(name="guardrail-pipeline")
    singleton.episodic_memory = MagicMock(name="episodic-memory")
    singleton.semantic_memory = MagicMock(name="semantic-memory")
    singleton.memory_agent = MagicMock(name="memory-agent")
    singleton.causal_memory = MagicMock(name="causal-memory")
    singleton.consolidator = MagicMock(name="consolidator")
    singleton.tool_executor = MagicMock(name="tool-executor")
    singleton.topology_engine = MagicMock(name="topology-engine")
    singleton.agent_pool = MagicMock(name="agent-pool")
    singleton.metacognition = MagicMock(name="metacognition")
    singleton.topology_population = MagicMock(name="topology-population")
    # Ablation flags (set by AblationConfig.apply at boot).
    singleton._skip_memory = False
    singleton._skip_avr = False
    singleton._skip_guardrails = False
    singleton._auto_evolve = True
    return singleton


# ─────────────────────────────────────────────────────────────────
# Test 1: per-run state is fresh (no singleton mutation)
# ─────────────────────────────────────────────────────────────────


def test_factory_returns_fresh_instance_distinct_from_singleton() -> None:
    """The factory MUST return a NEW AgentLoop, not the singleton itself.

    This is the load-bearing assertion for the entire P6-A design.
    If the factory returned the singleton (e.g. mutation-in-place
    that we forgot to refactor), the structural-isolation guarantee
    would be void and we'd be back at P6-B (lock + ContextVar).
    """
    singleton = _build_singleton()
    bypass_loop = create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=MagicMock(name="bandit-llm"),
        llm_config=LLMConfig(provider="bandit", model="bandit-model"),
        system_level=2,
    )

    assert bypass_loop is not singleton, (
        "create_bypass_agent_loop returned the singleton itself. The "
        "P6-A factory MUST return a NEW AgentLoop instance per call "
        "— this is the structural-isolation guarantee. See ADR-016."
    )
    assert bypass_loop.config is not singleton.config, (
        "Factory returned an instance that shares config with the "
        "singleton. config must be a fresh AgentConfig — otherwise "
        "max_steps/validation_level overrides would mutate the "
        "singleton's config field."
    )


# ─────────────────────────────────────────────────────────────────
# Test 2: shared injected deps are copied by reference
# ─────────────────────────────────────────────────────────────────


def test_factory_shares_injected_deps_with_singleton() -> None:
    """All boot-injected deps MUST be ``is``-identical to the singleton's.

    Per ADR-016 §"Field ownership matrix", the heavy injected
    backends (sandbox_manager, memory backends, tool_executor, etc.)
    are SHARED — the factory copies them by reference, not by value.
    Building fresh sandboxes/memory per run would defeat the
    factory's "cheap to construct" property.
    """
    singleton = _build_singleton()
    bypass_loop = create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=MagicMock(),
        llm_config=LLMConfig(provider="bandit", model="bandit-model"),
        system_level=2,
    )

    shared_attrs = [
        "sandbox_manager",
        "exocortex",
        "guardrail_pipeline",
        "episodic_memory",
        "semantic_memory",
        "memory_agent",
        "causal_memory",
        "consolidator",
        "tool_executor",
        "topology_engine",
        "agent_pool",
        "metacognition",
        "topology_population",
    ]
    for attr in shared_attrs:
        assert getattr(bypass_loop, attr) is getattr(singleton, attr), (
            f"Factory did not share {attr!r} with the singleton. "
            f"This dep is heavy / boot-injected and must be reference-"
            f"copied, not duplicated. See ADR-016 ownership matrix."
        )

    # tool_registry is also shared (passed via singleton._tools).
    assert bypass_loop._tools is singleton._tools, (
        "Factory built a fresh ToolRegistry instead of sharing the "
        "singleton's. Per-run tool registries would lose any tools "
        "added at boot."
    )


# ─────────────────────────────────────────────────────────────────
# Test 3: per-run state is correctly populated from args
# ─────────────────────────────────────────────────────────────────


def test_factory_populates_per_run_state_from_args() -> None:
    """LLM/write_gate/drift/run_frame fields come from factory args, not singleton."""
    singleton = _build_singleton()
    bandit_llm = MagicMock(name="bandit-llm")
    write_gate_sentinel = MagicMock(name="write-gate")
    on_drift_sentinel = MagicMock(name="on-drift")
    run_frame_builder_sentinel = MagicMock(name="run-frame-builder")

    bypass_loop = create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=bandit_llm,
        llm_config=LLMConfig(provider="bandit", model="bandit-model"),
        system_level=2,
        write_gate=write_gate_sentinel,
        task_text="def add(a, b):\n    return a + b",
        on_drift=on_drift_sentinel,
        run_frame_builder=run_frame_builder_sentinel,
        runtime_node_run_id="01TESTNODE000000000000001",
    )

    # Per-run LLM (not from singleton).
    assert bypass_loop._llm is bandit_llm
    assert bypass_loop._llm is not singleton._llm
    assert bypass_loop.config.llm.model == "bandit-model"

    # Per-run write gate + task.
    assert bypass_loop.write_gate is write_gate_sentinel
    assert bypass_loop.gate_current_task == "def add(a, b):\n    return a + b"

    # Per-run drift callback.
    assert bypass_loop._on_drift is on_drift_sentinel

    # Per-run RunFrame correlation IDs.
    assert bypass_loop._run_frame_builder is run_frame_builder_sentinel
    assert bypass_loop._runtime_node_run_id == "01TESTNODE000000000000001"


# ─────────────────────────────────────────────────────────────────
# Test 4: system_level scaling matches pipeline.py mutation block
# ─────────────────────────────────────────────────────────────────


def test_factory_system_level_scaling_matches_existing_mutation() -> None:
    """max_steps + stall_after_tool_steps + validation_level scaling per system_level.

    The same formulas live at:
      - ``pipeline.py:2356-2386`` (current bypass mutation block)
      - ``agent_loop_factory.py:create_node_agent_loop`` (per-node factory)

    Phase B will replace the pipeline.py mutation with a factory call.
    The scaling MUST match exactly so the byte-identical golden test
    (cycle-11 P9 phase 1 #1) continues to pass.

    Per system_level:
      S1: max_steps=5,  stall=0,            validation=1
      S2: max_steps=10, stall=9 (max-1),    validation=2 (sandbox-aware)
      S3: max_steps=20, stall=19 (max-1),   validation=3
    """
    singleton = _build_singleton()
    cases = [
        (1, 5, 0, 1),
        (2, 10, 9, 2),
        (3, 20, 19, 3),
    ]
    for system_level, expected_max, expected_stall, expected_validation in cases:
        loop = create_bypass_agent_loop(
            singleton=singleton,
            llm_provider=MagicMock(),
            llm_config=LLMConfig(provider="x", model="x-model"),
            system_level=system_level,
        )
        assert loop.config.max_steps == expected_max, (
            f"system_level={system_level}: max_steps={loop.config.max_steps}, "
            f"expected {expected_max}. Formula must match "
            f"pipeline.py:2372 + agent_loop_factory.py:136-141."
        )
        assert loop.config.stall_after_tool_steps == expected_stall, (
            f"system_level={system_level}: stall_after_tool_steps="
            f"{loop.config.stall_after_tool_steps}, expected {expected_stall}. "
            f"Formula: (max_steps - 1) if max_steps > 5 else 0."
        )
        assert loop.config.validation_level == expected_validation, (
            f"system_level={system_level}: validation_level="
            f"{loop.config.validation_level}, expected {expected_validation}. "
            f"S2 expects sandbox-aware (singleton has sandbox_manager set)."
        )


# ─────────────────────────────────────────────────────────────────
# Test 5: per-run constants for the bypass path
# ─────────────────────────────────────────────────────────────────


def test_factory_sets_skip_routing_and_clears_topology() -> None:
    """``_skip_routing=True`` + ``_current_topology=None`` are bypass-path constants.

    H1 carryover (2026-04-19 audit): pipeline already routed in
    Stage 0; AgentLoop should not re-route inside.
    H4 carryover (2026-04-20 audit): bypass means topology selection
    decided "no topology"; AgentLoop should not see a stale value.

    These are NOT factory args — the factory always sets them.
    Cycle-12 phase B's swap removes the equivalent two lines from
    the pipeline.py mutation block; the factory takes ownership.
    """
    singleton = _build_singleton()
    # Pre-poison the singleton with non-default values to prove the
    # factory doesn't inherit them.
    singleton._skip_routing = False
    singleton._current_topology = MagicMock(name="stale-topology")

    bypass_loop = create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=MagicMock(),
        llm_config=LLMConfig(provider="x", model="x-model"),
        system_level=1,
    )

    assert bypass_loop._skip_routing is True, (
        "Factory did not set _skip_routing=True. Bypass path requires "
        "this — pipeline already handled routing in Stage 0."
    )
    assert bypass_loop._current_topology is None, (
        "Factory did not clear _current_topology. Bypass = no "
        "topology selected; a stale topology would mislead AgentLoop "
        "internals."
    )


# ─────────────────────────────────────────────────────────────────
# Test 6: ablation flags inherited from singleton
# ─────────────────────────────────────────────────────────────────


def test_factory_inherits_ablation_flags_from_singleton() -> None:
    """``_skip_memory`` / ``_skip_avr`` / ``_skip_guardrails`` / ``_auto_evolve`` come from singleton.

    These are set at boot via ``AblationConfig.apply()`` and are
    process-wide, NOT per-run. The factory must copy them so bypass
    runs honor the same ablation profile as multi-agent / boot-singleton
    runs.
    """
    # Singleton with ablation: skip_memory=True, skip_avr=True
    singleton = _build_singleton()
    singleton._skip_memory = True
    singleton._skip_avr = True
    singleton._skip_guardrails = False
    singleton._auto_evolve = False

    bypass_loop = create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=MagicMock(),
        llm_config=LLMConfig(provider="x", model="x-model"),
        system_level=1,
    )

    assert bypass_loop._skip_memory is True
    assert bypass_loop._skip_avr is True
    assert bypass_loop._skip_guardrails is False
    assert bypass_loop._auto_evolve is False


# ─────────────────────────────────────────────────────────────────
# Test 7: factory does NOT mutate the singleton
# ─────────────────────────────────────────────────────────────────


def test_factory_does_not_mutate_singleton() -> None:
    """The whole point of P6-A: zero mutation of singleton state.

    Snapshot every singleton field BEFORE the factory call;
    snapshot AGAIN after; assert byte-identical. This is the
    structural-isolation property the cycle-11 P6-B lock papered
    over but didn't eliminate.
    """
    singleton = _build_singleton()
    # Capture mutable / mutation-prone fields before.
    before = {
        "config": singleton.config,
        "_llm": singleton._llm,
        "_tools": singleton._tools,
        "_skip_routing": singleton._skip_routing,
        "_current_topology": singleton._current_topology,
        "_run_frame_builder": getattr(singleton, "_run_frame_builder", None),
        "_runtime_node_run_id": getattr(singleton, "_runtime_node_run_id", None),
        "write_gate": getattr(singleton, "write_gate", None),
        "gate_current_task": getattr(singleton, "gate_current_task", None),
        "_on_drift": getattr(singleton, "_on_drift", None),
        "max_steps": singleton.config.max_steps,
        "validation_level": singleton.config.validation_level,
        "stall_after_tool_steps": singleton.config.stall_after_tool_steps,
        "config_llm": singleton.config.llm,
    }

    create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=MagicMock(),
        llm_config=LLMConfig(provider="bandit", model="bandit-model"),
        system_level=3,
        write_gate=MagicMock(),
        task_text="task",
        on_drift=MagicMock(),
        run_frame_builder=MagicMock(),
        runtime_node_run_id="01TEST",
    )

    after = {
        "config": singleton.config,
        "_llm": singleton._llm,
        "_tools": singleton._tools,
        "_skip_routing": singleton._skip_routing,
        "_current_topology": singleton._current_topology,
        "_run_frame_builder": getattr(singleton, "_run_frame_builder", None),
        "_runtime_node_run_id": getattr(singleton, "_runtime_node_run_id", None),
        "write_gate": getattr(singleton, "write_gate", None),
        "gate_current_task": getattr(singleton, "gate_current_task", None),
        "_on_drift": getattr(singleton, "_on_drift", None),
        "max_steps": singleton.config.max_steps,
        "validation_level": singleton.config.validation_level,
        "stall_after_tool_steps": singleton.config.stall_after_tool_steps,
        "config_llm": singleton.config.llm,
    }

    for key in before:
        assert before[key] is after[key] or before[key] == after[key], (
            f"Singleton field {key!r} was mutated by the factory: "
            f"before={before[key]!r}, after={after[key]!r}. The "
            f"P6-A structural-isolation guarantee is BROKEN — the "
            f"P6-B lock would still be needed."
        )
