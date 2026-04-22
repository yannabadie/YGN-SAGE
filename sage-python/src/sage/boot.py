"""Boot sequence: initialize the full YGN-SAGE agent stack."""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger("sage.boot")

# Use OS certificate store (Windows/macOS) instead of certifi.
# Fixes SSL errors behind corporate proxies (e.g., AD Groupe *.adgroupe.com).
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # truststore not installed — uses certifi (may fail behind proxy)

# Load .env if present (for GOOGLE_API_KEY etc.)
try:
    from dotenv import load_dotenv
    # Walk up to find .env (works from sage-python/ or repo root)
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

from sage.agent import AgentConfig  # noqa: E402
from sage.agent_loop import AgentLoop  # noqa: E402
from sage.input.types import TaskInput  # noqa: E402
from sage.agent_pool import AgentPool  # noqa: E402
from sage.llm.base import LLMConfig  # noqa: E402
from sage.strategy.adaptive_router import AdaptiveRouter  # noqa: E402
from sage.topology.evo_topology import TopologyEvolver, TopologyPopulation  # noqa: E402
from sage.tools.registry import ToolRegistry  # noqa: E402
from sage.events.bus import EventBus  # noqa: E402
from sage.constants import (  # noqa: E402
    DEFAULT_BUDGET_USD,
    COST_GUARDRAIL_MAX_USD,
    OUTPUT_GUARDRAIL_MIN_LENGTH,
    MAX_AGENT_STEPS,
)

# Sub-modules (extracted for readability)
from sage.boot_providers import init_llm_provider  # noqa: E402
from sage.boot_tools import (  # noqa: E402
    init_tools, check_sandbox_availability, init_metacognition,
)
from sage.boot_memory import init_memory  # noqa: E402
from sage.boot_topology import init_topology  # noqa: E402
from sage.boot_pipeline import init_pipeline  # noqa: E402

# Re-export for backward compat (test_boot_sandbox_warning imports this)
_check_sandbox_availability = check_sandbox_availability


@dataclass
class AgentSystem:
    """The complete YGN-SAGE agent system."""
    agent_loop: AgentLoop
    agent_pool: AgentPool
    metacognition: AdaptiveRouter
    topology_evolver: TopologyEvolver
    topology_population: TopologyPopulation
    memory_agent: Any
    tool_registry: ToolRegistry
    event_bus: EventBus
    registry: Any = None
    capability_matrix: Any = None
    rust_router: Any = None
    # Phase 6: Rust TopologyEngine (6-path generate, MAP-Elites, bandit)
    topology_engine: Any = None
    # Phase 6: Standalone ContextualBandit for model selection
    bandit: Any = None
    # Rust ModelRegistry (None if sage_core not compiled or cards.toml not found)
    _rust_registry: Any = None
    # CognitiveOrchestrationPipeline (5-stage: classify->decompose->topology->assign->execute)
    pipeline: Any = None
    # Execution path tracking (Issue A audit fix): set after each run()
    _last_execution_path: str = ""
    # C4 observability: the TaskInput the bench passed (if any). `None`
    # when run() was called with a raw string (legacy path).
    _last_task_input: Any = None

    @property
    def model_info(self) -> dict[str, str]:
        """Return resolved model metadata for benchmark artifacts."""
        info: dict[str, str] = {"model": "unknown", "provider": "", "tier": ""}
        loop = self.agent_loop
        if hasattr(loop, "_llm") and loop._llm:
            info["model"] = getattr(loop._llm, "model_id", "unknown")
            info["provider"] = type(loop._llm).__name__
        if hasattr(self, "metacognition") and self.metacognition:
            info["tier"] = getattr(self.metacognition, "_current_tier", "")
        return info

    async def run(
        self,
        task: "str | TaskInput",
        *,
        system_hint: int | None = None,
    ) -> str:
        """Run a task through the agent system.

        Mock mode: direct AgentLoop (tested exception, preserves 2001 tests).
        Non-mock: CognitiveOrchestrationPipeline (5-stage).
        Fallback: direct AgentLoop if pipeline not initialized.

        Args:
            task: The user's task. Accepts either a raw string (legacy
                path, still the canonical API for chat interfaces and
                anything that hasn't gone through a normalizer) OR a
                `TaskInput` produced by `sage.input.normalize_*`
                (universal input adapter, C4 of the
                2026-04-21-universal-input-adapter-design plan). When a
                TaskInput is passed, the source tag selects the
                appropriate renderer (`swebench` →
                `render_swebench_prompt`, `bcb` → `render_bcb_prompt`,
                `chat` / any other → the raw `prompt` field). The
                rendered string then flows through the same pipeline as
                a legacy string task — `perceive()` still sees a
                `str` argument. A deeper refactor teaching
                `perceive()` to consume `TaskInput` directly is
                deferred to a follow-up (C4b).
            system_hint: Optional override for Stage 0 routing (1, 2, or 3).
                Passed through to ``pipeline.run(system_hint=...)`` when the
                pipeline is active. Benchmark adapters that already know the
                task class (e.g. SWE-bench tasks are always S3) use this to
                skip router misclassification.
        """
        # C4 (2026-04-22): TaskInput dispatch at entry point.
        # Each bench used to call `render_X_prompt(normalize_X(task))`
        # itself; now it just passes `normalize_X(task)` here and we
        # dispatch. Zero behavioral change — byte-identical output for
        # every source, because the renderers are the same functions
        # the benches used to call directly.
        if isinstance(task, TaskInput):
            self._last_task_input = task
            if task.source == "swebench":
                from sage.input.swebench import render_swebench_prompt
                task = render_swebench_prompt(task)
            elif task.source == "bcb":
                from sage.input.bcb import render_bcb_prompt
                task = render_bcb_prompt(task)
            else:
                # `chat` and any future source: the raw NL prompt IS the
                # task text. Benches that want richer rendering ship
                # their own renderer + source tag.
                task = task.prompt
        else:
            self._last_task_input = None

        _budget = self._guardrail_budget if hasattr(self, '_guardrail_budget') else DEFAULT_BUDGET_USD

        # Mock bypass: tested exception (H9).
        # Mock goes direct to agent_loop so phase events (PERCEIVE/THINK/ACT/LEARN)
        # and guardrail wiring are exercised — important for 2001 tests.
        if self.agent_loop.config.llm.provider == "mock":
            self._last_execution_path = "mock"
            self.agent_loop._current_topology = None
            result = await self.agent_loop.run(task)
            return result

        # Pipeline is THE execution path.
        # Pipeline Stage 4 now calls agent_loop.run() for bypass (Phase 1),
        # giving every task tools + S2/S3 validation + guardrails + memory.
        if self.pipeline:
            result = await self.pipeline.run(
                task, budget_usd=_budget, system_hint=system_hint,
            )
            self._last_execution_path = "pipeline"
            await self._persist_memory()
            return result

        # Fallback: pipeline not initialized (missing deps at boot).
        # Direct agent_loop.run() — still gets tools + validation.
        _log.warning("Pipeline not available — using direct agent_loop")
        self._last_execution_path = "direct"
        self.agent_loop._current_topology = None
        result = await self.agent_loop.run(task)
        await self._persist_memory()
        return result

    def _record_topology_outcome(self, task: str, result: str, topology_result: Any, bandit_decision: Any = None, run_start: float = 0.0) -> None:
        """Record outcome into topology engine's learning loop (S-MMU + MAP-Elites)."""
        if not self.topology_engine or topology_result is None:
            return
        try:
            # Estimate quality from result — returns float or None (abstain)
            from sage.quality_estimator import QualityEstimator  # noqa: E402
            _qe = QualityEstimator()
            quality = _qe.estimate(
                task, result, latency_ms=(time.perf_counter() - run_start) * 1000,
            )
            cost = self.agent_loop.total_cost_usd
            latency_ms = (time.perf_counter() - run_start) * 1000

            # When quality is unknown (None), use 0.5 as neutral recording.
            # The bandit needs observations to learn, even imprecise ones.
            # Only truly empty results (quality=0.0) are definitively bad.
            if quality is None:
                quality = 0.5
                _log.info("Topology outcome: quality=None -> 0.5 (neutral recording)")

            # Extract keywords from task
            keywords = list(set(
                w.lower() for w in re.findall(r'\b\w{4,}\b', task)
            ))[:10]

            topology_id = topology_result.topology.id
            # Compute real embedding for S-MMU record
            outcome_embedding = None
            try:
                from sage.memory.embedder import Embedder
                _emb = Embedder()
                if _emb.is_semantic:
                    outcome_embedding = _emb.embed(task[:500])
            except (ImportError, RuntimeError):
                pass
            self.topology_engine.record_outcome(
                topology_id,
                task[:200],
                keywords,
                outcome_embedding,
                quality,
                cost,
                latency_ms,
            )
            _log.info(
                "Topology outcome recorded: id=%s, quality=%.2f, cost=%.4f, latency=%.0fms",
                topology_id, quality, cost, latency_ms,
            )

            # Update bandit posteriors
            if self.bandit and bandit_decision is not None:
                try:
                    self.bandit.record(
                        bandit_decision.decision_id, quality, cost, latency_ms,
                    )
                except (ImportError, RuntimeError) as e2:
                    _log.warning("Bandit outcome recording failed (%s)", e2)

            # Feed telemetry back to SystemRouter
            if self.rust_router and hasattr(self.rust_router, 'record_outcome'):
                try:
                    _decision = getattr(self, '_last_decision', None)
                    self.rust_router.record_outcome(
                        getattr(_decision, 'decision_id', ''),
                        quality, cost, latency_ms,
                    )
                except (ImportError, RuntimeError) as e3:
                    _log.debug("Router telemetry recording failed (%s)", e3)
        except (ImportError, RuntimeError, ValueError) as e:
            _log.warning("Topology outcome recording failed (%s)", e)

    async def _persist_memory(self) -> None:
        """Persist semantic and causal memory after a run."""
        if hasattr(self.agent_loop, "semantic_memory") and self.agent_loop.semantic_memory:
            try:
                self.agent_loop.semantic_memory.save()
            except (IOError, OSError):
                _log.warning("Failed to persist semantic memory", exc_info=True)
        if hasattr(self.agent_loop, "causal_memory") and self.agent_loop.causal_memory:
            try:
                self.agent_loop.causal_memory.save()
            except (IOError, OSError):
                _log.warning("Failed to persist causal memory", exc_info=True)


def boot_agent_system(
    use_mock_llm: bool = False,
    llm_tier: str = "auto",
    agent_name: str = "sage-main",
    event_bus: EventBus | None = None,
) -> AgentSystem:
    """Initialize the complete agent stack.

    Args:
        llm_tier: Model tier to use. "auto" (default) picks the best
                  available provider: Codex CLI if installed, else Google
                  Gemini if GOOGLE_API_KEY is set, else raises.
    """
    # 1. LLM provider
    provider, llm_config = init_llm_provider(use_mock_llm, llm_tier)

    # 2. Tools, sandbox, kNN router
    tool_registry, sandbox_manager, knn_router = init_tools(event_bus, provider, use_mock_llm)

    # 3. Metacognition (AdaptiveRouter + kNN)
    metacognition = init_metacognition(provider, use_mock_llm, knn_router)

    # 4. Topology engine, bandit, shadow router
    topo = init_topology(rust_registry=None, metacognition=metacognition)
    rust_router = topo["rust_router"]
    rust_registry = topo["rust_registry"]
    py_model_registry = topo["py_model_registry"]
    rust_topology_engine = topo["topology_engine"]
    rust_bandit = topo["bandit"]

    # 5. Evolution + memory agent
    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    agent_pool = AgentPool()

    # Agent config
    config = AgentConfig(
        name=agent_name,
        llm=llm_config,
        system_prompt=(
            "You are YGN-SAGE, a precise AI assistant. "
            "Think step-by-step. Be concise. Answer the user task directly."
        ),
        max_steps=MAX_AGENT_STEPS,
        validation_level=1,  # Default S1 — routing promotes to S2 only for code tasks
    )

    # Event bus (central nervous system)
    event_bus = event_bus or EventBus()

    # 6. Memory tiers
    # We need the agent loop first to wire memory into it, but we need memory_compressor
    # for the loop constructor. So we create a temporary compressor for the loop.
    from sage.memory.compressor import MemoryCompressor
    from sage.memory.embedder import Embedder
    from sage.constants import MEMORY_COMPRESSION_THRESHOLD, MEMORY_KEEP_RECENT
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=MEMORY_COMPRESSION_THRESHOLD,
        keep_recent=MEMORY_KEEP_RECENT,
    )
    memory_compressor.embedder = Embedder()

    # Agent loop
    loop = AgentLoop(
        config=config,
        llm_provider=provider,
        tool_registry=tool_registry,
        memory_compressor=memory_compressor,
        on_event=event_bus.emit,
    )
    loop.agent_pool = agent_pool
    loop.metacognition = metacognition
    loop.topology_population = topology_population
    loop.sandbox_manager = sandbox_manager

    # Wire memory tiers into loop
    mem = init_memory(event_bus, provider, use_mock_llm, loop)
    memory_agent = mem["memory_agent"]
    episodic_memory = mem["episodic_memory"]
    consolidator = mem["consolidator"]
    causal_memory = mem["causal_memory"]

    # ToolExecutor for S2 AVR code validation (Rust tree-sitter + subprocess)
    try:
        from sage_core import ToolExecutor as RustToolExecutor
        tool_executor = RustToolExecutor()
        _log.info("ToolExecutor (Rust): tree-sitter validator + subprocess executor")
    except ImportError:
        tool_executor = None
        _log.info("ToolExecutor (Rust) not available — S2 AVR uses Python sandbox")
    loop.tool_executor = tool_executor
    loop.topology_engine = rust_topology_engine

    # Enable online evolution when Rust TopologyEngine is available
    if rust_topology_engine is not None:
        loop._auto_evolve = True
        _log.info("Online evolution enabled (Rust TopologyEngine available)")

    # ToolForge: autonomous tool synthesis (UCT arXiv 2602.01983)
    # When agent calls a tool that doesn't exist, GapDetector creates a ticket,
    # ToolForge synthesizes code+tests via LLM, validates (AST+sandbox), registers.
    # Skip when SAGE_ABLATION_NO_TOOLFORGE=1 (Sprint 5 ablation config).
    if os.environ.get("SAGE_ABLATION_NO_TOOLFORGE") == "1":
        _log.info("ToolForge disabled by SAGE_ABLATION_NO_TOOLFORGE=1 (ablation)")
    else:
        try:
            from sage.tools.forge import ToolForge
            _toolforge = ToolForge(
                registry=tool_registry,
                llm_provider=provider,
                llm_config=llm_config,
                event_bus=event_bus,
            )
            loop.toolforge = _toolforge
            _log.info("ToolForge wired (GapDetector + BuildLoop for autonomous tool creation)")
        except Exception as exc:
            _log.debug("ToolForge not available: %s", exc)

    # CORAL Phase 1: persistent evolution memory (arXiv 2604.01658)
    # Lazy init: EvolutionMemory.initialize() is called on first async use,
    # not here (boot is synchronous, can't safely run async init).
    try:
        from sage.evolution.memory import EvolutionMemory
        _evo_mem = EvolutionMemory()
        # Wire into evolution engine if available
        if hasattr(loop, '_evolution_engine') and loop._evolution_engine:
            loop._evolution_engine._evolution_memory = _evo_mem
        # Wire into LLM mutator for skill injection
        if hasattr(loop, '_mutator') and hasattr(loop._mutator, 'evolution_memory'):
            loop._mutator.evolution_memory = _evo_mem
        loop.evolution_memory = _evo_mem
        _log.info("EvolutionMemory wired (CORAL persistent skills at %s)", _evo_mem._db_path)
    except Exception as exc:
        _log.debug("EvolutionMemory not available: %s", exc)

    # AgeMem: 8 memory tools (3 STM + 4 LTM + 1 Causal)
    from sage.tools.memory_tools import create_memory_tools
    for tool in create_memory_tools(loop.working_memory, episodic_memory, memory_compressor, causal_memory=causal_memory):
        tool_registry.register(tool)

    # Core code tools: bash execution for file reading, git, tests
    # These are the foundation for SWE-bench, code review, and self-programming.
    from sage.tools.base import Tool
    from sage.llm.base import ToolDef

    async def _execute_bash_handler(command: str, timeout: int = 30, **_kwargs) -> str:
        """Execute a bash command in a subprocess. Returns stdout+stderr.

        Uses bash (git bash on Windows) for Unix compatibility.
        Inherits current working directory (set by SWE-bench to repo root).
        """
        import asyncio, subprocess, shutil
        # Use bash for Unix commands (git bash on Windows, /bin/bash on Linux)
        bash = shutil.which("bash") or shutil.which("sh")
        if bash:
            cmd_args = [bash, "-c", command]
        else:
            cmd_args = command  # Fallback to shell=True
        try:
            if isinstance(cmd_args, list):
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    ),
                    timeout=timeout,
                )
            else:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        cmd_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    ),
                    timeout=timeout,
                )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = (stdout or b"").decode("utf-8", errors="replace")
            if stderr:
                output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")
            return output[:10000]  # Cap output to 10K chars
        except asyncio.TimeoutError:
            return f"[TIMEOUT after {timeout}s]"
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"

    bash_tool = Tool(
        spec=ToolDef(
            name="execute_bash",
            description="Execute a bash/shell command. Use for: reading files (cat, head), searching code (grep, find), running tests (pytest, python), git operations (git diff, git log), and any system command. Returns stdout+stderr (max 10K chars).",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Max seconds (default 30)", "default": 30},
                },
                "required": ["command"],
            },
        ),
        handler=_execute_bash_handler,
    )
    tool_registry.register(bash_tool)
    _log.info("Core tools: execute_bash registered (file read, git, tests, system commands)")

    # ExoCortex tools (research-paper search — not library docs)
    from sage.tools.exocortex_tools import create_exocortex_tools
    for tool in create_exocortex_tools(mem["exocortex"]):
        tool_registry.register(tool)

    # Context7 library-docs tool (C2c 2026-04-21) — bridges the gap that
    # search_exocortex is research-paper-scoped; this is the right tool
    # for django/astropy/requests/etc. API-contract questions.
    from sage.tools.context7_tools import create_context7_tools
    for tool in create_context7_tools():
        tool_registry.register(tool)
        _log.info("Core tools: %s registered (Context7 library docs)", tool.spec.name)

    # Guardrails
    from sage.guardrails.base import GuardrailPipeline
    from sage.guardrails.builtin import CostGuardrail, OutputGuardrail
    loop.guardrail_pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=COST_GUARDRAIL_MAX_USD),
        OutputGuardrail(min_length=OUTPUT_GUARDRAIL_MIN_LENGTH),
    ])

    # Sandbox availability check — warn loudly if neither Wasm nor Docker present
    check_sandbox_availability()

    # 7. Pipeline, controller, quality, ToolForge
    from sage.providers.registry import ModelRegistry
    registry = ModelRegistry()

    pipe = init_pipeline(
        router=metacognition,
        engine=rust_topology_engine,
        provider=provider,
        llm_config=llm_config,
        bandit=rust_bandit,
        rust_registry=rust_registry,
        py_model_registry=py_model_registry,
        registry=registry,
        event_bus=event_bus,
        use_mock_llm=use_mock_llm,
        consolidator=consolidator,
        working_memory=loop.working_memory,
        episodic_memory=episodic_memory,
        tool_registry=tool_registry,
        memory_compressor=memory_compressor,
        rust_router=rust_router,
        agent_loop=loop,
    )

    # Log capability surface at boot (Issue E audit fix)
    from sage.memory.working import get_memory_backend
    _log.info("Memory backend: %s", get_memory_backend())
    _log.info(
        "Capabilities: pipeline=%s, quality=%s, bandit=%s, engine=%s",
        pipe["pipeline"] is not None,
        pipe["quality_estimator"] is not None,
        rust_bandit is not None,
        rust_topology_engine is not None,
    )

    system = AgentSystem(
        agent_loop=loop,
        agent_pool=agent_pool,
        metacognition=metacognition,
        topology_evolver=topology_evolver,
        topology_population=topology_population,
        memory_agent=memory_agent,
        tool_registry=tool_registry,
        event_bus=event_bus,
        registry=pipe["registry"],
        capability_matrix=pipe["capability_matrix"],
        rust_router=rust_router,
        topology_engine=rust_topology_engine,
        bandit=rust_bandit,
        _rust_registry=rust_registry or py_model_registry,
        pipeline=pipe["pipeline"],
    )

    # sage_recurse: recursive self-invocation for test-time scaling
    # (The Conductor, arXiv 2512.04388, ICLR 2026). Bound here so the tool
    # can call system.run() without creating an import-level cycle.
    # Skip when SAGE_ABLATION_NO_RECURSE=1 (Sprint 5 ablation config).
    if os.environ.get("SAGE_ABLATION_NO_RECURSE") == "1":
        _log.info("sage_recurse disabled by SAGE_ABLATION_NO_RECURSE=1 (ablation)")
    else:
        try:
            from sage.tools.sage_recurse import build_sage_recurse_tool
            controller = getattr(system.pipeline, "controller", None)
            tool_registry.register(
                build_sage_recurse_tool(system.run, controller=controller)
            )
            _log.info(
                "Core tools: sage_recurse registered (max depth 3, budget-gated=%s)",
                controller is not None,
            )
        except (ImportError, RuntimeError) as exc:
            _log.debug("sage_recurse not available: %s", exc)

    return system
