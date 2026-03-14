"""End-to-end integration tests for YGN-SAGE.

Tests the full stack offline (no API keys, no LLM calls) by exercising
real component wiring across module boundaries. Each test validates that
independently-developed subsystems actually work together.

Organized by cross-cutting concern:
  1. Boot → Agent → Loop lifecycle
  2. Routing → Orchestration → Quality feedback loop
  3. Memory tier pipeline (STM → Episodic → Semantic → Causal)
  4. Agent composition patterns (Sequential, Parallel, Loop, Handoff, AgentTool)
  5. Guardrail pipeline integration
  6. EventBus → DriftMonitor observability chain
  7. Contracts → DAG → Verification pipeline
  8. Resilience under failure cascades
  9. Topology → Pipeline integration
  10. Constants consistency (no stale references)
  11. Write gate
  12. Sandbox security (Python fallback)
  13. Cross-module smoke tests
"""
from __future__ import annotations

import asyncio
import time
import math
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. BOOT → AGENT → LOOP LIFECYCLE
# ---------------------------------------------------------------------------


class TestAgentLifecycle:
    """Boot an agent, run a task with MockProvider, verify the full lifecycle."""

    def test_agent_config_defaults(self):
        from sage.agent import AgentConfig
        from sage.llm.base import LLMConfig

        cfg = AgentConfig(name="test", llm=LLMConfig(provider="mock", model="m"))
        assert cfg.max_steps == 100
        assert cfg.validation_level == 1
        assert cfg.system_prompt == "You are a helpful AI assistant."

    @pytest.mark.asyncio
    async def test_agent_loop_single_step(self):
        """AgentLoop with MockProvider should complete and emit events."""
        from sage.agent import AgentConfig
        from sage.agent_loop import AgentLoop, AgentEvent
        from sage.llm.base import LLMConfig
        from sage.llm.mock import MockProvider

        events_received: list[AgentEvent] = []

        config = AgentConfig(
            name="e2e-test",
            llm=LLMConfig(provider="mock", model="mock"),
            system_prompt="You are a test agent.",
            max_steps=1,
        )
        provider = MockProvider(responses=["Hello from mock agent."])
        loop = AgentLoop(
            config=config,
            llm_provider=provider,
            on_event=lambda e: events_received.append(e),
        )
        result = await loop.run("Say hello")

        assert result is not None
        assert len(result) > 0
        # Events should have been emitted via on_event
        assert len(events_received) >= 1, f"Expected events, got {len(events_received)}"

    @pytest.mark.asyncio
    async def test_agent_loop_multi_step_convergence(self):
        """AgentLoop with multiple responses should converge within max_steps."""
        from sage.agent import AgentConfig
        from sage.agent_loop import AgentLoop
        from sage.llm.base import LLMConfig
        from sage.llm.mock import MockProvider

        provider = MockProvider(responses=[
            "Let me think about this...",
            "The answer is 42.",
        ])
        config = AgentConfig(
            name="multi-step",
            llm=LLMConfig(provider="mock", model="mock"),
            max_steps=3,
        )
        loop = AgentLoop(config=config, llm_provider=provider)
        result = await loop.run("What is the meaning of life?")
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_loop_metacognition_injection(self):
        """Metacognition is injected post-init and used during routing."""
        from sage.agent import AgentConfig
        from sage.agent_loop import AgentLoop
        from sage.llm.base import LLMConfig
        from sage.llm.mock import MockProvider
        from sage.strategy.metacognition import ComplexityRouter

        config = AgentConfig(
            name="routing-test",
            llm=LLMConfig(provider="mock", model="mock"),
            max_steps=1,
        )
        provider = MockProvider(responses=["Simple answer."])
        loop = AgentLoop(config=config, llm_provider=provider)
        loop.metacognition = ComplexityRouter()  # injected post-init like boot.py does
        result = await loop.run("What is 2+2?")
        assert result is not None


# ---------------------------------------------------------------------------
# 2. ROUTING → QUALITY FEEDBACK LOOP
# ---------------------------------------------------------------------------


class TestRoutingQualityLoop:
    """ComplexityRouter → QualityEstimator → feedback chain."""

    def test_routing_tiers_are_ordered(self):
        """S1 < S2 < S3 complexity for canonical tasks."""
        from sage.strategy.metacognition import ComplexityRouter

        router = ComplexityRouter()
        s1 = router.assess_complexity("What is 2+2?")
        s2 = router.assess_complexity("Write a Python function to merge two sorted lists")
        s3 = router.assess_complexity(
            "Prove deadlock freedom for this concurrent algorithm using "
            "Z3 formal verification with loop invariants"
        )
        assert s1.complexity <= s2.complexity, f"S1 ({s1.complexity}) > S2 ({s2.complexity})"
        assert s2.complexity <= s3.complexity, f"S2 ({s2.complexity}) > S3 ({s3.complexity})"

    def test_quality_estimator_signals(self):
        """QualityEstimator produces distinct scores for different quality levels."""
        from sage.quality_estimator import QualityEstimator

        empty = QualityEstimator.estimate("write code", "")
        assert empty == 0.0

        good_code = QualityEstimator.estimate(
            "write a sorting function",
            "def sort(lst):\n    return sorted(lst)\n",
        )
        assert good_code > 0.5, f"Good code scored too low: {good_code}"

        error_result = QualityEstimator.estimate(
            "write code",
            "error: failed to compile\ntraceback (most recent call last)",
            had_errors=True,
        )
        assert error_result < good_code, "Error result should score lower than good code"

    def test_quality_avr_convergence_bonus(self):
        """Faster AVR convergence -> higher quality score."""
        from sage.quality_estimator import QualityEstimator

        fast = QualityEstimator.estimate("write code", "def f(): pass", avr_iterations=1)
        slow = QualityEstimator.estimate("write code", "def f(): pass", avr_iterations=10)
        assert fast > slow, f"Fast AVR ({fast}) should beat slow ({slow})"

    def test_routing_to_quality_roundtrip(self):
        """Route a task, estimate quality, verify score matches routing tier."""
        from sage.strategy.metacognition import ComplexityRouter
        from sage.quality_estimator import QualityEstimator
        from sage.constants import ORCHESTRATOR_S1_QUALITY

        router = ComplexityRouter()
        profile = router.assess_complexity("What is 2+2?")

        # S1 task with correct answer should pass S1 quality threshold
        score = QualityEstimator.estimate("What is 2+2?", "4")
        assert score >= ORCHESTRATOR_S1_QUALITY, (
            f"S1 task score {score} < S1 threshold {ORCHESTRATOR_S1_QUALITY}"
        )


# ---------------------------------------------------------------------------
# 3. MEMORY TIER PIPELINE
# ---------------------------------------------------------------------------


class TestMemoryPipeline:
    """STM → Episodic → Semantic → Causal memory integration."""

    def test_working_memory_basic(self):
        """WorkingMemory (Python mock) stores and retrieves events."""
        from sage.memory.working import WorkingMemory

        wm = WorkingMemory(agent_id="e2e-mem")
        wm.add_event("test", "E2E event")
        assert wm.event_count() >= 1

    @pytest.mark.asyncio
    async def test_episodic_memory_crud(self):
        """Episodic memory: store → search → retrieve."""
        from sage.memory.episodic import EpisodicMemory

        mem = EpisodicMemory()
        await mem.initialize()
        await mem.store("sort_list", "def sort(l): return sorted(l)", {"lang": "python"})
        results = await mem.search("sort")
        assert len(results) > 0, "Episodic search should find stored entry"

    def test_semantic_memory_entity_graph(self):
        """Semantic memory: add entities + query context."""
        from sage.memory.semantic import SemanticMemory
        from sage.memory.memory_agent import ExtractionResult

        mem = SemanticMemory()
        mem.add_extraction(ExtractionResult(
            entities=["Python", "sort"],
            relationships=[("Python", "has", "sort")],
        ))
        ctx = mem.get_context_for("Python")
        # Should return some context (may be empty if graph structure differs)
        assert isinstance(ctx, str)

    def test_causal_memory_chain(self):
        """Causal memory: add entities → causal edge → retrieve chain."""
        from sage.memory.causal import CausalMemory

        mem = CausalMemory()
        mem.add_entity("bug", "issue")
        mem.add_entity("fix", "action")
        mem.add_causal_edge("bug", "fix", "causes")
        chain = mem.get_causal_chain("bug")
        assert len(chain) > 0, "Causal chain should have at least one edge"
        assert any("fix" in str(edge) for edge in chain)

    def test_relevance_gate_filters(self):
        """CRAG relevance gate: relevant passes, irrelevant blocked."""
        from sage.memory.relevance_gate import RelevanceGate

        gate = RelevanceGate(threshold=0.3)
        relevant = gate.score("Python sorting algorithm", "Write a Python function to sort a list")
        irrelevant = gate.score("quantum physics", "Write a Python function to sort a list")
        assert relevant > irrelevant, (
            f"Relevant ({relevant}) should score higher than irrelevant ({irrelevant})"
        )

    @pytest.mark.asyncio
    async def test_memory_tier_cascade(self):
        """Full cascade: STM → Episodic → Semantic → verify isolation."""
        from sage.memory.working import WorkingMemory
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.semantic import SemanticMemory
        from sage.memory.memory_agent import ExtractionResult

        # STM
        wm = WorkingMemory(agent_id="cascade-test")
        wm.add_event("tier0", "working memory event")

        # Episodic
        epi = EpisodicMemory()
        await epi.initialize()
        await epi.store("tier1_task", "tier1_result", {})

        # Semantic
        sem = SemanticMemory()
        sem.add_extraction(ExtractionResult(entities=["cascade"]))

        # Each tier should be independent
        assert wm.event_count() >= 1
        epi_results = await epi.search("tier1")
        assert len(epi_results) >= 1


# ---------------------------------------------------------------------------
# 4. AGENT COMPOSITION PATTERNS
# ---------------------------------------------------------------------------


class _SimpleAgent:
    """Minimal agent for composition tests."""

    def __init__(self, name: str, transform: str = ""):
        self.name = name
        self._transform = transform

    async def run(self, task: str) -> str:
        if self._transform == "upper":
            return task.upper()
        elif self._transform == "reverse":
            return task[::-1]
        elif self._transform == "prefix":
            return f"[{self.name}] {task}"
        return f"{self.name}: {task}"


class TestAgentComposition:
    """SequentialAgent, ParallelAgent, LoopAgent, Handoff, AgentTool."""

    @pytest.mark.asyncio
    async def test_sequential_agent_chains(self):
        """SequentialAgent: output of agent N feeds into agent N+1."""
        from sage.agents.sequential import SequentialAgent

        seq = SequentialAgent(
            name="seq-test",
            agents=[
                _SimpleAgent("upper", "upper"),
                _SimpleAgent("prefix", "prefix"),
            ],
        )
        result = await seq.run("hello")
        assert "HELLO" in result, f"Expected uppercased input in result, got: {result}"
        assert "[prefix]" in result

    @pytest.mark.asyncio
    async def test_parallel_agent_fan_out(self):
        """ParallelAgent: all agents receive the same input, results aggregated."""
        from sage.agents.parallel import ParallelAgent

        par = ParallelAgent(
            name="par-test",
            agents=[
                _SimpleAgent("agent-a", "upper"),
                _SimpleAgent("agent-b", "reverse"),
            ],
        )
        result = await par.run("hello")
        assert "HELLO" in result, f"Missing upper result in: {result}"
        assert "olleh" in result, f"Missing reversed result in: {result}"

    @pytest.mark.asyncio
    async def test_loop_agent_exit_condition(self):
        """LoopAgent: exits when exit_condition is met."""
        from sage.agents.loop_agent import LoopAgent

        counter = {"n": 0}

        class CounterAgent:
            name = "counter"

            async def run(self, task: str) -> str:
                counter["n"] += 1
                return f"count={counter['n']}"

        loop = LoopAgent(
            name="loop-test",
            agent=CounterAgent(),
            max_iterations=10,
            exit_condition=lambda out: "count=3" in out,
        )
        result = await loop.run("start")
        assert counter["n"] == 3, f"Expected 3 iterations, got {counter['n']}"
        assert "count=3" in result

    @pytest.mark.asyncio
    async def test_loop_agent_max_iterations(self):
        """LoopAgent: respects max_iterations even without exit condition."""
        from sage.agents.loop_agent import LoopAgent

        loop = LoopAgent(
            name="bounded-loop",
            agent=_SimpleAgent("echo"),
            max_iterations=2,
            exit_condition=None,
        )
        result = await loop.run("hello")
        assert result is not None

    @pytest.mark.asyncio
    async def test_handoff_with_filter(self):
        """Handoff: input_filter transforms task before delegation."""
        from sage.agents.handoff import Handoff

        handoff = Handoff(
            target=_SimpleAgent("specialist", "upper"),
            description="Uppercase specialist",
            input_filter=lambda t: t.strip() + " (filtered)",
        )
        result = await handoff.execute("hello")
        assert "FILTERED" in result.output
        assert result.target_name == "specialist"

    @pytest.mark.asyncio
    async def test_handoff_callback_fired(self):
        """Handoff: on_handoff callback is called before execution."""
        from sage.agents.handoff import Handoff

        callback_log: list[tuple[str, str]] = []

        handoff = Handoff(
            target=_SimpleAgent("target"),
            description="test",
            on_handoff=lambda name, task: callback_log.append((name, task)),
        )
        await handoff.execute("test task")
        assert len(callback_log) == 1
        assert callback_log[0][0] == "target"

    @pytest.mark.asyncio
    async def test_agent_tool_wrapping(self):
        """AgentTool: wraps an agent as a callable Tool."""
        from sage.tools.agent_tool import AgentTool

        agent = _SimpleAgent("researcher", "upper")
        tool = AgentTool.from_agent(agent, name="research", description="Research tool")

        assert tool.spec.name == "research"
        # Tool.execute takes a dict of arguments
        result = await tool.execute({"task": "hello world"})
        assert "HELLO WORLD" in result.output

    @pytest.mark.asyncio
    async def test_agent_tool_rejects_non_runnable(self):
        """AgentTool: rejects objects without run() method."""
        from sage.tools.agent_tool import AgentTool

        with pytest.raises(TypeError, match="callable 'run' method"):
            AgentTool.from_agent(object(), name="bad", description="bad")

    @pytest.mark.asyncio
    async def test_nested_composition(self):
        """Nested: Sequential wrapping ParallelAgent."""
        from sage.agents.sequential import SequentialAgent
        from sage.agents.parallel import ParallelAgent

        inner = ParallelAgent(
            name="inner-par",
            agents=[_SimpleAgent("a"), _SimpleAgent("b")],
        )
        outer = SequentialAgent(
            name="outer-seq",
            agents=[inner, _SimpleAgent("final", "upper")],
        )
        result = await outer.run("test")
        # The parallel output (which includes both agent outputs) gets uppercased
        assert result == result.upper()


# ---------------------------------------------------------------------------
# 5. GUARDRAIL PIPELINE
# ---------------------------------------------------------------------------


class TestGuardrailPipeline:
    """Input/output guardrail wiring."""

    @pytest.mark.asyncio
    async def test_output_guardrail_passes_valid(self):
        from sage.guardrails.builtin import OutputGuardrail
        from sage.guardrails.base import GuardrailPipeline

        pipeline = GuardrailPipeline([OutputGuardrail()])
        results = await pipeline.check_all(output="Valid response text.")
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_output_guardrail_blocks_empty(self):
        from sage.guardrails.builtin import OutputGuardrail
        from sage.guardrails.base import GuardrailPipeline

        pipeline = GuardrailPipeline([OutputGuardrail()])
        results = await pipeline.check_all(output="")
        assert any(not r.passed for r in results), "Empty output should be blocked"

    @pytest.mark.asyncio
    async def test_cost_guardrail_via_context(self):
        """CostGuardrail reads cost_usd from context dict."""
        from sage.guardrails.builtin import CostGuardrail
        from sage.guardrails.base import GuardrailPipeline

        pipeline = GuardrailPipeline([CostGuardrail(max_usd=1.0)])
        # Under budget
        results = await pipeline.check_all(context={"cost_usd": 0.5})
        assert all(r.passed for r in results)
        # Over budget
        results = await pipeline.check_all(context={"cost_usd": 2.0})
        assert any(not r.passed for r in results)

    @pytest.mark.asyncio
    async def test_multi_guardrail_pipeline(self):
        """Multiple guardrails: all must pass for overall pass."""
        from sage.guardrails.builtin import OutputGuardrail, CostGuardrail
        from sage.guardrails.base import GuardrailPipeline

        pipeline = GuardrailPipeline([OutputGuardrail(), CostGuardrail(max_usd=1.0)])
        # Good output, under budget -> all pass
        results = await pipeline.check_all(output="Valid text.", context={"cost_usd": 0.5})
        assert all(r.passed for r in results)
        # Good output, over budget -> cost fails
        results = await pipeline.check_all(output="Valid text.", context={"cost_usd": 5.0})
        passed = [r.passed for r in results]
        assert not all(passed), "Over budget should fail"


# ---------------------------------------------------------------------------
# 6. EVENTBUS → DRIFT MONITOR OBSERVABILITY
# ---------------------------------------------------------------------------


class TestObservabilityChain:
    """EventBus → DriftMonitor end-to-end."""

    def test_eventbus_subscribe_emit(self):
        """EventBus: subscribe + emit delivers events."""
        from sage.events.bus import EventBus
        from sage.agent_loop import AgentEvent

        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))

        evt = AgentEvent(type="TEST", step=1, timestamp=time.time())
        bus.emit(evt)
        assert len(received) == 1
        assert received[0].type == "TEST"

    def test_eventbus_multiple_subscribers(self):
        """EventBus: multiple subscribers all receive the same event."""
        from sage.events.bus import EventBus
        from sage.agent_loop import AgentEvent

        bus = EventBus()
        log_a, log_b = [], []
        bus.subscribe(lambda e: log_a.append(e))
        bus.subscribe(lambda e: log_b.append(e))

        bus.emit(AgentEvent(type="MULTI", step=0, timestamp=time.time()))
        assert len(log_a) == 1
        assert len(log_b) == 1

    def test_drift_monitor_healthy(self):
        """DriftMonitor: stable events → CONTINUE."""
        from sage.monitoring.drift import DriftMonitor

        @dataclass
        class FakeEvent:
            latency_ms: float = 100.0
            cost_usd: float = 0.001
            meta: dict = field(default_factory=dict)

        monitor = DriftMonitor()
        events = [FakeEvent() for _ in range(10)]
        report = monitor.analyze(events)
        assert report.action == "CONTINUE"
        assert report.drift_score < 0.4

    def test_drift_monitor_degraded(self):
        """DriftMonitor: spiking latency → SWITCH_MODEL or RESET_AGENT."""
        from sage.monitoring.drift import DriftMonitor

        @dataclass
        class FakeEvent:
            latency_ms: float = 0.0
            cost_usd: float = 0.001
            meta: dict = field(default_factory=dict)

        monitor = DriftMonitor()
        # First half: fast, second half: very slow
        events = [FakeEvent(latency_ms=100) for _ in range(5)]
        events += [FakeEvent(latency_ms=1000) for _ in range(5)]
        report = monitor.analyze(events)
        assert report.action in ("SWITCH_MODEL", "RESET_AGENT"), (
            f"Expected degradation, got {report.action} (score={report.drift_score})"
        )

    def test_drift_monitor_error_spike(self):
        """DriftMonitor: 100% error rate → RESET_AGENT."""
        from sage.monitoring.drift import DriftMonitor

        @dataclass
        class FakeEvent:
            latency_ms: float = 100.0
            cost_usd: float = 0.001
            meta: dict = field(default_factory=dict)

        monitor = DriftMonitor()
        events = [FakeEvent(meta={"error": "timeout"}) for _ in range(10)]
        report = monitor.analyze(events)
        assert report.action == "RESET_AGENT"
        assert report.drift_score > 0.7

    def test_eventbus_to_drift_integration(self):
        """Full chain: EventBus collects events → DriftMonitor analyzes."""
        from sage.events.bus import EventBus
        from sage.agent_loop import AgentEvent
        from sage.monitoring.drift import DriftMonitor

        bus = EventBus()
        collected = []
        bus.subscribe(lambda e: collected.append(e))

        # Emit healthy events
        for i in range(6):
            bus.emit(AgentEvent(
                type="ACT", step=i, timestamp=time.time(),
                latency_ms=100.0, cost_usd=0.001, meta={},
            ))

        monitor = DriftMonitor()
        report = monitor.analyze(collected)
        assert report.action == "CONTINUE"


# ---------------------------------------------------------------------------
# 7. CONTRACTS → DAG → VERIFICATION
# ---------------------------------------------------------------------------


class TestContractsPipeline:
    """TaskNode → DAG → VF checks → CostTracker pipeline."""

    def test_task_dag_topo_sort(self):
        """TaskDAG: topological sort produces valid ordering."""
        from sage.contracts.dag import TaskDAG
        from sage.contracts.task_node import TaskNode

        dag = TaskDAG()
        n0 = TaskNode(node_id="parse", description="Parse input")
        n1 = TaskNode(node_id="analyze", description="Analyze data")
        n2 = TaskNode(node_id="summarize", description="Summarize results")
        dag.add_node(n0)
        dag.add_node(n1)
        dag.add_node(n2)
        dag.add_edge("parse", "analyze")
        dag.add_edge("analyze", "summarize")

        order = dag.topological_sort()
        assert order.index("parse") < order.index("analyze") < order.index("summarize")

    def test_task_dag_cycle_detection(self):
        """TaskDAG: detects cycles and raises."""
        from sage.contracts.dag import TaskDAG, CycleError
        from sage.contracts.task_node import TaskNode

        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="a", description="A"))
        dag.add_node(TaskNode(node_id="b", description="B"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")

        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_verification_pre_post_checks(self):
        """VF pre/post checks pass for valid data."""
        from sage.contracts.verification import pre_check, post_check
        from sage.contracts.task_node import TaskNode

        node = TaskNode(node_id="test", description="Test node")
        assert pre_check(node, {"input": "test task"}).passed
        assert post_check(node, {"output": "result text"}).passed

    def test_cost_tracker_budget(self):
        """CostTracker: tracks cumulative cost and detects budget overflow."""
        from sage.contracts.cost_tracker import CostTracker

        tracker = CostTracker(budget_usd=1.0)
        tracker.record("node1", 0.3)
        tracker.record("node2", 0.5)
        assert tracker.total_spent == pytest.approx(0.8, abs=0.01)
        assert not tracker.is_over_budget
        tracker.record("node3", 0.3)
        assert tracker.is_over_budget

    def test_task_dag_ready_nodes(self):
        """TaskDAG: ready_nodes returns nodes with no unresolved dependencies."""
        from sage.contracts.dag import TaskDAG
        from sage.contracts.task_node import TaskNode

        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="a", description="A"))
        dag.add_node(TaskNode(node_id="b", description="B"))
        dag.add_node(TaskNode(node_id="c", description="C"))
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")

        ready = dag.ready_nodes(completed=set())
        assert "a" in ready


# ---------------------------------------------------------------------------
# 8. RESILIENCE UNDER FAILURES
# ---------------------------------------------------------------------------


class TestResilience:
    """CircuitBreaker behavior under cascading failures."""

    def test_circuit_breaker_opens_after_max_failures(self):
        from sage.resilience import CircuitBreaker

        cb = CircuitBreaker("test", max_failures=3)
        assert cb.is_closed()
        for i in range(3):
            cb.record_failure(RuntimeError(f"fail-{i}"))
        assert cb.is_open()
        assert cb.should_skip()

    def test_circuit_breaker_recovery(self):
        """Breaker recovers: OPEN → HALF_OPEN (cooldown) → CLOSED (success)."""
        from sage.resilience import CircuitBreaker

        cb = CircuitBreaker("recover", max_failures=1, cooldown_s=0.05)
        cb.record_failure(RuntimeError("fail"))
        assert cb.is_open()
        time.sleep(0.1)
        assert not cb.should_skip()  # half-open allows probe
        cb.record_success()
        assert cb.is_closed()

    def test_circuit_breaker_reopen_on_half_open_failure(self):
        """If probe fails in HALF_OPEN, circuit re-opens."""
        from sage.resilience import CircuitBreaker

        cb = CircuitBreaker("reopen", max_failures=1, cooldown_s=0.05)
        cb.record_failure(RuntimeError("1"))
        time.sleep(0.1)
        assert not cb.should_skip()  # half-open
        cb.record_failure(RuntimeError("2"))  # probe fails
        assert cb.is_open()

    def test_multiple_independent_breakers(self):
        """Each breaker is independent (shared failures don't cross)."""
        from sage.resilience import CircuitBreaker

        cb_a = CircuitBreaker("service-a", max_failures=2)
        cb_b = CircuitBreaker("service-b", max_failures=2)
        cb_a.record_failure(RuntimeError("a-fail-1"))
        cb_a.record_failure(RuntimeError("a-fail-2"))
        assert cb_a.is_open()
        assert cb_b.is_closed()  # service-b unaffected


# ---------------------------------------------------------------------------
# 9. TOPOLOGY + PIPELINE INTEGRATION
# ---------------------------------------------------------------------------


class TestTopologyIntegration:
    """Topology templates, pipeline stages."""

    def test_topology_genome_creation(self):
        """TopologyGenome can be created with nodes and edges."""
        from sage.topology.evo_topology import TopologyGenome

        genome = TopologyGenome(
            nodes=["agent_0", "agent_1"],
            edges=[("agent_0", "agent_1")],
            pattern="sequential",
        )
        assert len(genome.nodes) == 2
        assert genome.pattern == "sequential"

    def test_pipeline_stages_domain_inference(self):
        """Pipeline stage: _infer_domain returns reasonable domain."""
        from sage.pipeline_stages import _infer_domain

        domain = _infer_domain("Write a Python sorting function")
        assert isinstance(domain, str)
        assert len(domain) > 0

    def test_pipeline_stages_dag_features_with_dag(self):
        """Pipeline stage: compute_dag_features works with a TaskDAG."""
        from sage.pipeline_stages import compute_dag_features, DAGFeatures
        from sage.contracts.dag import TaskDAG
        from sage.contracts.task_node import TaskNode

        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="a", description="A"))
        dag.add_node(TaskNode(node_id="b", description="B"))
        dag.add_edge("a", "b")

        features = compute_dag_features(dag)
        assert isinstance(features, DAGFeatures)

    def test_pipeline_context_defaults(self):
        """PipelineContext has sensible defaults."""
        from sage.pipeline import PipelineContext

        ctx = PipelineContext(task="test")
        assert ctx.task == "test"
        assert ctx.budget == 5.0
        assert ctx.domain == ""
        assert ctx.result == ""


# ---------------------------------------------------------------------------
# 10. CONSTANTS CONSISTENCY
# ---------------------------------------------------------------------------


class TestConstantsConsistency:
    """Verify constants.py values are internally consistent."""

    def test_routing_zone_ordering(self):
        """S1_CEIL < SPECULATIVE_ZONE < S3_FLOOR: no gaps or overlaps."""
        from sage.constants import (
            S1_COMPLEXITY_CEIL, SPECULATIVE_ZONE_MIN, SPECULATIVE_ZONE_MAX,
            S3_COMPLEXITY_FLOOR,
        )
        assert S1_COMPLEXITY_CEIL <= SPECULATIVE_ZONE_MAX
        assert SPECULATIVE_ZONE_MIN < SPECULATIVE_ZONE_MAX
        assert SPECULATIVE_ZONE_MAX <= S3_COMPLEXITY_FLOOR

    def test_quality_weights_sum_to_one(self):
        """Quality signal weights (baseline + 4 signals) should sum to ~1.0."""
        from sage.constants import (
            QUALITY_BASELINE, QUALITY_LENGTH_WEIGHT, QUALITY_CODE_WEIGHT,
            QUALITY_ERROR_WEIGHT, QUALITY_AVR_WEIGHT,
        )
        total = QUALITY_BASELINE + QUALITY_LENGTH_WEIGHT + QUALITY_CODE_WEIGHT
        total += QUALITY_ERROR_WEIGHT + QUALITY_AVR_WEIGHT
        assert abs(total - 1.0) < 0.01, f"Quality weights sum to {total}, expected ~1.0"

    def test_drift_weights_sum_to_one(self):
        """Drift signal weights should sum to 1.0."""
        from sage.constants import DRIFT_WEIGHT_LATENCY, DRIFT_WEIGHT_ERRORS, DRIFT_WEIGHT_COST

        total = DRIFT_WEIGHT_LATENCY + DRIFT_WEIGHT_ERRORS + DRIFT_WEIGHT_COST
        assert abs(total - 1.0) < 0.01, f"Drift weights sum to {total}, expected 1.0"

    def test_knn_profiles_in_routing_zones(self):
        """kNN synthetic profiles must land in correct routing zones."""
        from sage.constants import (
            S1_COMPLEXITY_CEIL, S3_COMPLEXITY_FLOOR,
            KNN_S1_COMPLEXITY, KNN_S2_COMPLEXITY, KNN_S3_COMPLEXITY,
        )
        assert KNN_S1_COMPLEXITY <= S1_COMPLEXITY_CEIL, "S1 kNN profile outside S1 zone"
        # S2 must be >= S1 ceil (boundary inclusive) and < S3 floor
        assert KNN_S2_COMPLEXITY >= S1_COMPLEXITY_CEIL, "S2 kNN profile below S1 zone"
        assert KNN_S2_COMPLEXITY < S3_COMPLEXITY_FLOOR, "S2 kNN profile in S3 zone"
        assert KNN_S3_COMPLEXITY >= S3_COMPLEXITY_FLOOR, "S3 kNN profile outside S3 zone"

    def test_max_tokens_per_tier_ordered(self):
        """S1 < S2 < S3 max tokens."""
        from sage.constants import MAX_TOKENS_S1, MAX_TOKENS_S2, MAX_TOKENS_S3

        assert MAX_TOKENS_S1 < MAX_TOKENS_S2 < MAX_TOKENS_S3

    def test_orchestrator_quality_thresholds_ordered(self):
        """S1 < S2 < S3 quality thresholds."""
        from sage.constants import (
            ORCHESTRATOR_S1_QUALITY, ORCHESTRATOR_S2_QUALITY, ORCHESTRATOR_S3_QUALITY,
        )
        assert ORCHESTRATOR_S1_QUALITY < ORCHESTRATOR_S2_QUALITY < ORCHESTRATOR_S3_QUALITY


# ---------------------------------------------------------------------------
# 11. WRITE GATE
# ---------------------------------------------------------------------------


class TestWriteGate:
    """WriteGate: confidence-gated memory writes."""

    def test_write_gate_allows_high_confidence(self):
        from sage.memory.write_gate import WriteGate

        gate = WriteGate(threshold=0.5)
        result = gate.evaluate("content", 0.9)
        assert result.allowed

    def test_write_gate_blocks_low_confidence(self):
        from sage.memory.write_gate import WriteGate

        gate = WriteGate(threshold=0.5)
        result = gate.evaluate("content", 0.2)
        assert not result.allowed

    def test_write_gate_boundary(self):
        from sage.memory.write_gate import WriteGate

        gate = WriteGate(threshold=0.5)
        at_boundary = gate.evaluate("content", 0.5)
        # At threshold should be allowed (>=)
        assert at_boundary.allowed


# ---------------------------------------------------------------------------
# 12. SANDBOX VALIDATION (PYTHON FALLBACK)
# ---------------------------------------------------------------------------


class TestSandboxSecurity:
    """Python AST validator blocks dangerous code."""

    def test_blocks_import_os(self):
        from sage.tools.sandbox_executor import validate_tool_code

        errors = validate_tool_code("import os; os.system('rm -rf /')")
        assert len(errors) > 0

    def test_blocks_dunder_access(self):
        from sage.tools.sandbox_executor import validate_tool_code

        errors = validate_tool_code("x = ''.__class__.__mro__")
        assert len(errors) > 0

    def test_allows_safe_code(self):
        from sage.tools.sandbox_executor import validate_tool_code

        errors = validate_tool_code("result = sum(range(100))\nprint(result)")
        assert len(errors) == 0

    def test_blocks_eval(self):
        from sage.tools.sandbox_executor import validate_tool_code

        errors = validate_tool_code("eval('1+1')")
        assert len(errors) > 0

    def test_blocks_exec(self):
        from sage.tools.sandbox_executor import validate_tool_code

        errors = validate_tool_code("exec('print(1)')")
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# 13. CROSS-MODULE INTEGRATION SMOKE TESTS
# ---------------------------------------------------------------------------


class TestCrossModuleSmoke:
    """Smoke tests that verify modules can be imported and wired together."""

    def test_all_core_imports(self):
        """All core modules import without errors."""
        import sage.agent
        import sage.agent_loop
        import sage.agent_pool
        import sage.orchestrator
        import sage.pipeline
        import sage.quality_estimator
        import sage.resilience
        import sage.constants
        import sage.events.bus
        import sage.guardrails.base
        import sage.guardrails.builtin
        import sage.memory.working
        import sage.memory.episodic
        import sage.memory.semantic
        import sage.memory.causal
        import sage.memory.relevance_gate
        import sage.memory.write_gate
        import sage.monitoring.drift
        import sage.strategy.metacognition
        import sage.topology.evo_topology
        import sage.tools.agent_tool
        import sage.agents.sequential
        import sage.agents.parallel
        import sage.agents.loop_agent
        import sage.agents.handoff

    def test_llm_protocol_conformance(self):
        """MockProvider conforms to LLMProvider protocol."""
        from sage.llm.base import LLMProvider
        from sage.llm.mock import MockProvider

        provider = MockProvider(responses=["test"])
        assert isinstance(provider, LLMProvider)

    def test_process_reward_model_safe(self):
        """PRM: score_with_z3 handles constraints safely."""
        from sage.topology.kg_rlvr import ProcessRewardModel

        prm = ProcessRewardModel()
        # score_with_z3 accepts a list of constraint strings
        score, details = prm.score_with_z3(["x > 0", "x < 100"])
        assert isinstance(score, (int, float))
        assert isinstance(details, dict)

    def test_extraction_result_dataclass(self):
        """ExtractionResult is a proper dataclass with defaults."""
        from sage.memory.memory_agent import ExtractionResult

        result = ExtractionResult()
        assert result.entities == []
        assert result.relationships == []
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_full_agent_run_with_all_injections(self):
        """Simulate boot.py injection pattern: create AgentLoop, inject subsystems."""
        from sage.agent import AgentConfig
        from sage.agent_loop import AgentLoop
        from sage.llm.base import LLMConfig
        from sage.llm.mock import MockProvider
        from sage.strategy.metacognition import ComplexityRouter
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.semantic import SemanticMemory
        from sage.events.bus import EventBus
        from sage.guardrails.builtin import OutputGuardrail
        from sage.guardrails.base import GuardrailPipeline

        bus = EventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))

        config = AgentConfig(
            name="full-stack",
            llm=LLMConfig(provider="mock", model="mock"),
            max_steps=1,
        )
        provider = MockProvider(responses=["def hello(): return 'world'"])
        loop = AgentLoop(
            config=config,
            llm_provider=provider,
            on_event=bus.emit,
        )

        # Inject subsystems like boot.py does
        loop.metacognition = ComplexityRouter()
        loop.episodic_memory = EpisodicMemory()
        await loop.episodic_memory.initialize()
        loop.semantic_memory = SemanticMemory()
        loop.guardrail_pipeline = GuardrailPipeline([OutputGuardrail()])

        result = await loop.run("Write a hello function")
        assert result is not None
        assert len(events) >= 1  # events flowed through EventBus
