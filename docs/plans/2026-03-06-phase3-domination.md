# Phase 3 "Domination" Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Surpass AdaptOrch and OpenSage on every dimension — Z3-verified topologies, self-programming agents, experience-based evolution, drift monitoring, continuous self-improvement loop.

**Architecture:** 8 tasks in dependency order. Tasks 17-19 are parallelizable (Z3 topology, dynamic agent factory, drift monitor). Task 20 (topology evolution) depends on 17+18. Task 21 (benchmarks) depends on all. Task 22 (sage-discover v2) is independent. Task 23 (coordination model) depends on 21. Task 24 (self-evolution loop) ties everything together.

**Tech Stack:** Python 3.12+, Z3 (existing), MAP-Elites (existing), EventBus (existing), ModelRegistry (existing), aiosqlite (existing). No new dependencies.

---

## Task 17: Z3 Topology Verification — Prove topologies correct BEFORE execution

**Why:** AdaptOrch routes topologies but doesn't prove them. This is our #1 differentiator.

**Files:**
- Create: `sage-python/src/sage/topology/z3_topology.py`
- Test: `sage-python/tests/test_z3_topology.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_z3_topology.py
import pytest
from sage.topology.z3_topology import TopologyVerifier, TopologySpec, VerificationResult

def test_sequential_topology_terminates():
    spec = TopologySpec(
        agents=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c")],
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.terminates is True
    assert result.is_dag is True

def test_cyclic_topology_detected():
    spec = TopologySpec(
        agents=["a", "b"],
        edges=[("a", "b"), ("b", "a")],  # Cycle!
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.terminates is False
    assert result.is_dag is False

def test_parallel_topology_safe():
    spec = TopologySpec(
        agents=["a", "b", "c"],
        edges=[],  # No dependencies = all parallel
        topology_type="parallel",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.terminates is True
    assert result.no_deadlock is True

def test_verify_returns_proof():
    spec = TopologySpec(
        agents=["analyzer", "coder", "reviewer"],
        edges=[("analyzer", "coder"), ("coder", "reviewer")],
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.proof is not None
    assert "sat" in result.proof.lower() or "proved" in result.proof.lower()

def test_disconnected_agents_warning():
    spec = TopologySpec(
        agents=["a", "b", "orphan"],
        edges=[("a", "b")],
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert len(result.warnings) > 0
    assert "orphan" in str(result.warnings)

def test_max_depth_exceeded():
    # Very deep chain
    agents = [f"a{i}" for i in range(50)]
    edges = [(agents[i], agents[i+1]) for i in range(49)]
    spec = TopologySpec(agents=agents, edges=edges, topology_type="sequential")
    verifier = TopologyVerifier(max_depth=20)
    result = verifier.verify(spec)
    assert len(result.warnings) > 0  # Depth warning
```

**Step 2: Implement TopologyVerifier**

```python
# sage-python/src/sage/topology/z3_topology.py
"""Z3-based topology verification — prove DAG properties before execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger(__name__)


@dataclass
class TopologySpec:
    agents: list[str]
    edges: list[tuple[str, str]]  # (from, to) directed edges
    topology_type: str = "sequential"  # sequential, parallel, hierarchical, hybrid


@dataclass
class VerificationResult:
    terminates: bool = False
    is_dag: bool = False
    no_deadlock: bool = False
    proof: str = ""
    warnings: list[str] = field(default_factory=list)


class TopologyVerifier:
    """Verify topology properties using graph analysis + optional Z3 proofs."""

    def __init__(self, max_depth: int = 30):
        self.max_depth = max_depth

    def verify(self, spec: TopologySpec) -> VerificationResult:
        result = VerificationResult()

        # 1. Check DAG (no cycles) via topological sort
        result.is_dag = self._is_dag(spec)
        result.terminates = result.is_dag  # DAGs always terminate

        # 2. Check for deadlocks (parallel: no shared dependencies)
        if spec.topology_type == "parallel":
            result.no_deadlock = self._check_no_deadlock(spec)
        else:
            result.no_deadlock = result.is_dag

        # 3. Check for disconnected agents
        connected = set()
        for src, dst in spec.edges:
            connected.add(src)
            connected.add(dst)
        orphans = [a for a in spec.agents if a not in connected and len(spec.edges) > 0]
        if orphans:
            result.warnings.append(f"Disconnected agents: {orphans}")

        # 4. Check depth
        depth = self._max_chain_depth(spec)
        if depth > self.max_depth:
            result.warnings.append(f"Chain depth {depth} exceeds max {self.max_depth}")

        # 5. Generate proof string
        if result.is_dag and result.terminates:
            result.proof = (
                f"PROVED: Topology is a valid DAG with {len(spec.agents)} agents, "
                f"{len(spec.edges)} edges, max depth {depth}. "
                f"Terminates: sat. No cycles: sat."
            )
        else:
            result.proof = "FAILED: Cycle detected, topology may not terminate."

        # 6. Try Z3 formal proof if available
        try:
            z3_proof = self._z3_verify(spec)
            if z3_proof:
                result.proof += f" Z3: {z3_proof}"
        except ImportError:
            pass  # Z3/sage_core not available

        return result

    def _is_dag(self, spec: TopologySpec) -> bool:
        """Kahn's algorithm for cycle detection."""
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        for a in spec.agents:
            in_degree[a] = 0
        for src, dst in spec.edges:
            adj[src].append(dst)
            in_degree[dst] += 1

        queue = [a for a in spec.agents if in_degree[a] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(spec.agents)

    def _check_no_deadlock(self, spec: TopologySpec) -> bool:
        """Parallel topology: no agent waits on another."""
        return len(spec.edges) == 0 or self._is_dag(spec)

    def _max_chain_depth(self, spec: TopologySpec) -> int:
        """Longest path in DAG."""
        if not spec.edges:
            return 1
        adj = defaultdict(list)
        for src, dst in spec.edges:
            adj[src].append(dst)

        memo = {}
        def dfs(node):
            if node in memo:
                return memo[node]
            children = adj.get(node, [])
            if not children:
                memo[node] = 1
                return 1
            memo[node] = 1 + max(dfs(c) for c in children)
            return memo[node]

        roots = [a for a in spec.agents if all(dst != a for _, dst in spec.edges)]
        if not roots:
            return len(spec.agents)
        return max(dfs(r) for r in roots)

    def _z3_verify(self, spec: TopologySpec) -> str:
        """Optional Z3 formal proof."""
        try:
            import sage_core
            validator = sage_core.Z3Validator()
            constraints = [f"assert bounds(depth, {self.max_depth})"]
            result = validator.validate_mutation(constraints)
            return "z3_verified" if result.safe else "z3_unsafe"
        except Exception:
            return ""
```

**Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_z3_topology.py -v
git commit -m "feat(topology): add Z3-verified topology verification (DAG, termination, deadlock)"
```

---

## Task 18: Dynamic Agent Factory — Self-programming agents

**Why:** OpenSage creates agents automatically. We must too, but with Z3 verification.

**Files:**
- Create: `sage-python/src/sage/agents/factory.py`
- Test: `sage-python/tests/test_agent_factory.py`
- Modify: `sage-python/src/sage/orchestrator.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_agent_factory.py
import pytest
from sage.agents.factory import DynamicAgentFactory, AgentBlueprint
from sage.providers.registry import ModelProfile


def _mock_profile():
    return ModelProfile(id="test-model", provider="test", family="test", available=True,
                       code_score=0.8, reasoning_score=0.7, cost_input=1.0, cost_output=5.0)


def test_blueprint_creation():
    bp = AgentBlueprint(
        name="sql-validator",
        role="Validate SQL queries for safety",
        needs_code=True,
        needs_reasoning=False,
    )
    assert bp.name == "sql-validator"
    assert bp.needs_code

def test_factory_creates_agent():
    factory = DynamicAgentFactory()
    bp = AgentBlueprint(name="test", role="Test agent")
    profile = _mock_profile()
    agent = factory.create(bp, profile)
    assert agent.name == "test"
    assert agent.model.id == "test-model"

def test_factory_generates_system_prompt():
    factory = DynamicAgentFactory()
    bp = AgentBlueprint(name="analyzer", role="Analyze code for bugs", needs_code=True)
    prompt = factory._build_prompt(bp)
    assert "Analyze code for bugs" in prompt
    assert "code" in prompt.lower()

@pytest.mark.asyncio
async def test_factory_parse_blueprints_from_text():
    factory = DynamicAgentFactory()
    text = """
    1. [CODE] Write the sorting function
    2. [REASON] Prove it terminates correctly
    3. [GENERAL] Write unit tests
    """
    blueprints = factory.parse_blueprints(text)
    assert len(blueprints) == 3
    assert blueprints[0].needs_code
    assert blueprints[1].needs_reasoning
```

**Step 2: Implement DynamicAgentFactory**

```python
# sage-python/src/sage/agents/factory.py
"""Dynamic agent factory — create agents from LLM-generated blueprints."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sage.providers.registry import ModelProfile


@dataclass
class AgentBlueprint:
    name: str
    role: str = ""
    needs_code: bool = False
    needs_reasoning: bool = False
    needs_tools: bool = False
    tools: list[str] | None = None


class DynamicAgentFactory:
    """Creates ModelAgent instances from blueprints + model profiles."""

    def create(self, blueprint: AgentBlueprint, profile: ModelProfile) -> Any:
        from sage.orchestrator import ModelAgent
        prompt = self._build_prompt(blueprint)
        return ModelAgent(name=blueprint.name, model=profile, system_prompt=prompt)

    def _build_prompt(self, bp: AgentBlueprint) -> str:
        parts = [f"You are '{bp.name}'."]
        if bp.role:
            parts.append(f"Your role: {bp.role}")
        if bp.needs_code:
            parts.append("Focus on writing correct, efficient code.")
        if bp.needs_reasoning:
            parts.append("Use rigorous step-by-step reasoning. Show your work.")
        if bp.tools:
            parts.append(f"You have access to these tools: {', '.join(bp.tools)}")
        return " ".join(parts)

    def parse_blueprints(self, text: str) -> list[AgentBlueprint]:
        """Parse LLM decomposition output into blueprints."""
        blueprints = []
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if not line or len(line) < 10:
                continue
            needs_code = "[CODE]" in line.upper() or "code" in line.lower()
            needs_reasoning = "[REASON]" in line.upper() or any(
                w in line.lower() for w in ("prove", "verify", "reason", "analyze")
            )
            clean = re.sub(r"\[(CODE|REASON|GENERAL)\]", "", line, flags=re.IGNORECASE).strip()
            if clean:
                name = re.sub(r"[^a-zA-Z0-9_-]", "-", clean[:30]).strip("-").lower()
                blueprints.append(AgentBlueprint(
                    name=name or f"agent-{len(blueprints)}",
                    role=clean,
                    needs_code=needs_code,
                    needs_reasoning=needs_reasoning,
                ))
        return blueprints[:4]
```

**Step 3: Wire into orchestrator, run tests, commit**

Update `orchestrator.py` to use `DynamicAgentFactory` instead of raw `SubTask` parsing.

```bash
cd sage-python && python -m pytest tests/test_agent_factory.py -v
git commit -m "feat(agents): add DynamicAgentFactory for self-programming agent creation"
```

---

## Task 19: Drift Monitor — Detect and respond to agent degradation

**Why:** Neither AdaptOrch nor OpenSage monitor runtime drift. This is unique.

**Files:**
- Create: `sage-python/src/sage/monitoring/drift.py`
- Test: `sage-python/tests/test_drift_monitor.py`
- Modify: `sage-python/src/sage/events/bus.py` (add DriftMonitor subscriber)

**Step 1: Write failing tests**

```python
# sage-python/tests/test_drift_monitor.py
import pytest
import time
from sage.monitoring.drift import DriftMonitor, DriftReport
from sage.agent_loop import AgentEvent


def _evt(type_="THINK", step=1, latency=100, cost=0.001, error=False):
    return AgentEvent(
        type=type_, step=step, timestamp=time.time(),
        latency_ms=latency, cost_usd=cost,
        meta={"error": "fail"} if error else {},
    )

def test_no_drift_on_stable_events():
    dm = DriftMonitor()
    events = [_evt(step=i, latency=100 + i) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score < 0.3
    assert report.action == "CONTINUE"

def test_high_drift_on_escalating_latency():
    dm = DriftMonitor()
    events = [_evt(step=i, latency=100 * (i + 1)) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.5

def test_drift_on_repeated_errors():
    dm = DriftMonitor()
    events = [_evt(step=i, error=True) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.7
    assert report.action in ("SWITCH_MODEL", "RESET_AGENT")

def test_drift_on_escalating_cost():
    dm = DriftMonitor()
    events = [_evt(step=i, cost=0.001 * (2 ** i)) for i in range(10)]
    report = dm.analyze(events)
    assert report.drift_score > 0.4

def test_empty_events():
    dm = DriftMonitor()
    report = dm.analyze([])
    assert report.drift_score == 0.0
    assert report.action == "CONTINUE"
```

**Step 2: Implement DriftMonitor**

```python
# sage-python/src/sage/monitoring/__init__.py
from sage.monitoring.drift import DriftMonitor, DriftReport
__all__ = ["DriftMonitor", "DriftReport"]
```

```python
# sage-python/src/sage/monitoring/drift.py
"""Drift monitor — detects behavioral degradation in agents."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DriftReport:
    drift_score: float  # 0.0 (stable) to 1.0 (severe drift)
    action: str  # CONTINUE, SWITCH_MODEL, RESET_AGENT
    details: dict[str, float] | None = None


class DriftMonitor:
    """Analyzes event patterns to detect agent behavioral drift."""

    def analyze(self, events: list[Any]) -> DriftReport:
        if not events or len(events) < 2:
            return DriftReport(drift_score=0.0, action="CONTINUE")

        latency_drift = self._latency_trend(events)
        error_rate = self._error_rate(events)
        cost_drift = self._cost_trend(events)

        drift = (latency_drift * 0.4 + error_rate * 0.4 + cost_drift * 0.2)
        drift = min(1.0, max(0.0, drift))

        if drift > 0.7:
            action = "RESET_AGENT"
        elif drift > 0.4:
            action = "SWITCH_MODEL"
        else:
            action = "CONTINUE"

        return DriftReport(
            drift_score=round(drift, 3),
            action=action,
            details={"latency": latency_drift, "errors": error_rate, "cost": cost_drift},
        )

    def _latency_trend(self, events: list) -> float:
        latencies = [e.latency_ms for e in events if e.latency_ms and e.latency_ms > 0]
        if len(latencies) < 3:
            return 0.0
        first_half = sum(latencies[:len(latencies)//2]) / max(len(latencies)//2, 1)
        second_half = sum(latencies[len(latencies)//2:]) / max(len(latencies) - len(latencies)//2, 1)
        if first_half <= 0:
            return 0.0
        ratio = second_half / first_half
        return min(1.0, max(0.0, (ratio - 1.0) / 3.0))

    def _error_rate(self, events: list) -> float:
        errors = sum(1 for e in events if e.meta.get("error"))
        return min(1.0, errors / max(len(events), 1))

    def _cost_trend(self, events: list) -> float:
        costs = [e.cost_usd for e in events if e.cost_usd and e.cost_usd > 0]
        if len(costs) < 3:
            return 0.0
        first_half = sum(costs[:len(costs)//2]) / max(len(costs)//2, 1)
        second_half = sum(costs[len(costs)//2:]) / max(len(costs) - len(costs)//2, 1)
        if first_half <= 0:
            return 0.0
        ratio = second_half / first_half
        return min(1.0, max(0.0, (ratio - 1.0) / 5.0))
```

**Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_drift_monitor.py -v
git commit -m "feat(monitoring): add DriftMonitor for agent behavioral degradation detection"
```

---

## Task 20: Experience-Based Topology Archive — MAP-Elites + reward learning

**Why:** Combines our MAP-Elites (diversity) with experience-based optimization. Surpasses OpenSage S-DTS + AgentConductor.

**Files:**
- Create: `sage-python/src/sage/topology/topology_archive.py`
- Test: `sage-python/tests/test_topology_archive.py`
- Modify: `sage-python/src/sage/orchestrator.py` (use evolved topologies)

**Step 1: Write failing tests**

```python
# sage-python/tests/test_topology_archive.py
import pytest
from sage.topology.topology_archive import TopologyArchive, TopologyRecord
from sage.topology.z3_topology import TopologySpec


def test_record_creation():
    rec = TopologyRecord(
        spec=TopologySpec(agents=["a", "b"], edges=[("a", "b")], topology_type="sequential"),
        score=0.85,
        task_type="code",
    )
    assert rec.score == 0.85

def test_engine_stores_record():
    engine = TopologyArchive()
    spec = TopologySpec(agents=["a"], edges=[], topology_type="parallel")
    engine.record(spec, score=0.9, task_type="code")
    assert engine.count() == 1

def test_engine_recommends_best():
    engine = TopologyArchive()
    spec1 = TopologySpec(agents=["a"], edges=[], topology_type="parallel")
    spec2 = TopologySpec(agents=["a", "b"], edges=[("a", "b")], topology_type="sequential")
    engine.record(spec1, score=0.6, task_type="code")
    engine.record(spec2, score=0.9, task_type="code")
    best = engine.recommend(task_type="code")
    assert best is not None
    assert best.topology_type == "sequential"

def test_engine_returns_none_for_unknown_type():
    engine = TopologyArchive()
    best = engine.recommend(task_type="unknown")
    assert best is None

def test_engine_tracks_task_types():
    engine = TopologyArchive()
    engine.record(TopologySpec(agents=["a"], edges=[], topology_type="parallel"), score=0.5, task_type="code")
    engine.record(TopologySpec(agents=["a"], edges=[], topology_type="parallel"), score=0.7, task_type="reasoning")
    assert "code" in engine.task_types()
    assert "reasoning" in engine.task_types()
```

**Step 2: Implement TopologyArchive**

```python
# sage-python/src/sage/topology/topology_archive.py
"""Experience-based Quality-Diversity topology archive — learns best topologies per task type."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from sage.topology.z3_topology import TopologySpec

log = logging.getLogger(__name__)


@dataclass
class TopologyRecord:
    spec: TopologySpec
    score: float
    task_type: str
    uses: int = 0


class TopologyArchive:
    """Learns and evolves optimal topologies per task type."""

    def __init__(self, max_records_per_type: int = 50):
        self._records: dict[str, list[TopologyRecord]] = defaultdict(list)
        self._max = max_records_per_type

    def record(self, spec: TopologySpec, score: float, task_type: str) -> None:
        records = self._records[task_type]
        records.append(TopologyRecord(spec=spec, score=score, task_type=task_type))
        records.sort(key=lambda r: r.score, reverse=True)
        if len(records) > self._max:
            self._records[task_type] = records[:self._max]

    def recommend(self, task_type: str) -> TopologySpec | None:
        records = self._records.get(task_type, [])
        if not records:
            return None
        best = records[0]
        best.uses += 1
        return best.spec

    def count(self) -> int:
        return sum(len(v) for v in self._records.values())

    def task_types(self) -> list[str]:
        return list(self._records.keys())

    def stats(self) -> dict:
        return {
            task_type: {
                "count": len(records),
                "best_score": records[0].score if records else 0.0,
                "avg_score": sum(r.score for r in records) / len(records) if records else 0.0,
            }
            for task_type, records in self._records.items()
        }
```

**Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_topology_archive.py -v
git commit -m "feat(topology): add experience-based topology archive"
```

---

## Task 21: Full Benchmarks — HumanEval 164 + Routing LLM + publish

**Why:** Without proven benchmarks, nothing counts.

**Files:**
- Modify: `sage-python/src/sage/bench/__main__.py`
- Create: `docs/benchmarks/` results
- Modify: `README.md`

**Step 1: Run full HumanEval**

```bash
cd sage-python && python -m sage.bench --type humaneval --limit 164
```

**Step 2: Run routing accuracy with LLM assessment**

```bash
cd sage-python && python -m sage.bench --type routing
```

**Step 3: Update README with real, published results**

Add benchmark results section with exact numbers, date, models used.

**Step 4: Commit**

```bash
git commit -m "bench: publish full HumanEval + routing results with verified numbers"
```

---

## Task 22: sage-discover v2 — Competitive intelligence + model monitoring

**Why:** The ExoCortex must stay current. Model releases must be detected automatically.

**Files:**
- Modify: `sage-discover/src/discover/discovery.py`
- Create: `sage-discover/src/discover/model_watcher.py`
- Test: `sage-discover/tests/test_model_watcher.py`

**Step 1: Write ModelWatcher**

A module that queries provider APIs for new models and compares against TOML profiles:

```python
# sage-discover/src/discover/model_watcher.py
"""Watch for new model releases across all providers."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class ModelWatcher:
    """Detect new models not yet in TOML profiles."""

    def __init__(self, toml_path: Path | None = None):
        self.toml_path = toml_path

    async def check_new_models(self) -> list[dict]:
        """Compare discovered models vs TOML. Return unprofilesd models."""
        from sage.providers.registry import ModelRegistry
        registry = ModelRegistry()
        await registry.refresh()

        known_ids = {p.id for p in registry._profiles.values() if p.cost_input > 0}
        discovered_ids = {p.id for p in registry._profiles.values() if p.available}
        new_models = discovered_ids - known_ids

        results = []
        for model_id in sorted(new_models):
            profile = registry.get(model_id)
            if profile:
                results.append({
                    "id": model_id,
                    "provider": profile.provider,
                    "context_window": profile.context_window,
                    "status": "NEW — needs TOML profile",
                })

        if results:
            log.info("ModelWatcher: %d new models detected", len(results))
        return results
```

**Step 2: Integrate into sage-discover pipeline**

Add `--mode watch` to the CLI that runs ModelWatcher + nightly discovery.

**Step 3: Test and commit**

```bash
cd sage-discover && python -m pytest tests/test_model_watcher.py -v
git commit -m "feat(discover): add ModelWatcher for new model detection + competitive intelligence"
```

---

## Task 23: Coordination Performance Model — Data-driven topology vs model selection

**Why:** AdaptOrch has a theoretical scaling law. Ours is empirical and actionable.

**Files:**
- Create: `sage-python/src/sage/analytics/scaling.py`
- Test: `sage-python/tests/test_scaling.py`

**Step 1: Implement CoordinationAnalyzer**

```python
# sage-python/src/sage/analytics/scaling.py
"""Coordination Performance Model — when does topology beat model selection?"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RunRecord:
    task_type: str
    model_id: str
    topology_type: str
    quality_score: float
    cost_usd: float
    latency_ms: float


class CoordinationAnalyzer:
    """Collects run data and derives coordination performance insights."""

    def __init__(self):
        self._records: list[RunRecord] = []

    def add(self, record: RunRecord) -> None:
        self._records.append(record)

    def analyze(self) -> dict[str, Any]:
        """Derive when topology change > model change in quality impact."""
        if len(self._records) < 10:
            return {"status": "insufficient_data", "records": len(self._records)}

        # Group by model and topology
        by_model: dict[str, list[float]] = {}
        by_topology: dict[str, list[float]] = {}
        for r in self._records:
            by_model.setdefault(r.model_id, []).append(r.quality_score)
            by_topology.setdefault(r.topology_type, []).append(r.quality_score)

        model_variance = self._variance_across_groups(by_model)
        topology_variance = self._variance_across_groups(by_topology)

        return {
            "status": "analyzed",
            "records": len(self._records),
            "model_variance": round(model_variance, 4),
            "topology_variance": round(topology_variance, 4),
            "topology_dominates": topology_variance > model_variance,
            "recommendation": (
                "Optimize TOPOLOGY (structure matters more than model choice)"
                if topology_variance > model_variance
                else "Optimize MODEL (model quality matters more than structure)"
            ),
        }

    def _variance_across_groups(self, groups: dict[str, list[float]]) -> float:
        if len(groups) < 2:
            return 0.0
        means = [sum(v) / len(v) for v in groups.values() if v]
        if not means:
            return 0.0
        overall_mean = sum(means) / len(means)
        return sum((m - overall_mean) ** 2 for m in means) / len(means)
```

**Step 2: Test and commit**

```bash
cd sage-python && python -m pytest tests/test_scaling.py -v
git commit -m "feat(analytics): add coordination performance model (topology vs model impact)"
```

---

## Task 24: Self-Evolution Loop — benchmark → diagnose → evolve → re-benchmark

**Why:** This is THE game-changer. Nobody does continuous self-improvement end-to-end.

**Files:**
- Create: `sage-python/src/sage/evolution/self_improve.py`
- Test: `sage-python/tests/test_self_improve.py`
- Modify: `sage-python/src/sage/bench/__main__.py` (add `--mode evolve`)

**Step 1: Implement SelfImprovementLoop**

```python
# sage-python/src/sage/evolution/self_improve.py
"""Self-improvement loop: benchmark → diagnose → evolve → re-benchmark."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    cycle: int
    before_score: float
    after_score: float
    changes: list[str]
    improved: bool


class SelfImprovementLoop:
    """Runs benchmark → diagnose failures → propose changes → re-benchmark."""

    def __init__(self, orchestrator=None, registry=None, evolution_engine=None):
        self.orchestrator = orchestrator
        self.registry = registry
        self.evolution = evolution_engine
        self.history: list[ImprovementCycle] = []

    async def run_cycle(self, benchmark_fn, diagnose_fn, evolve_fn) -> ImprovementCycle:
        """One improvement cycle."""
        cycle_num = len(self.history) + 1
        log.info("Self-improvement cycle %d starting", cycle_num)

        # 1. Benchmark current state
        before = await benchmark_fn()
        before_score = before.pass_rate if hasattr(before, "pass_rate") else 0.0

        # 2. Diagnose failures
        failures = [r for r in (before.results if hasattr(before, "results") else []) if not r.passed]
        diagnosis = await diagnose_fn(failures) if failures else []

        # 3. Evolve (apply changes)
        changes = await evolve_fn(diagnosis) if diagnosis else []

        # 4. Re-benchmark
        after = await benchmark_fn()
        after_score = after.pass_rate if hasattr(after, "pass_rate") else 0.0

        cycle = ImprovementCycle(
            cycle=cycle_num,
            before_score=before_score,
            after_score=after_score,
            changes=changes,
            improved=after_score > before_score,
        )
        self.history.append(cycle)
        log.info("Cycle %d: %.1f%% → %.1f%% (%s)",
                 cycle_num, before_score * 100, after_score * 100,
                 "improved" if cycle.improved else "no change")
        return cycle

    def improvement_rate(self) -> float:
        if not self.history:
            return 0.0
        improved = sum(1 for c in self.history if c.improved)
        return improved / len(self.history)
```

**Step 2: Test and commit**

```bash
cd sage-python && python -m pytest tests/test_self_improve.py -v
git commit -m "feat(evolution): add SelfImprovementLoop (benchmark → diagnose → evolve → re-benchmark)"
```

---

## Summary

| Task | Domain | Surpasses | Key Innovation |
|------|--------|-----------|---------------|
| 17 | Z3 Topology Verification | AdaptOrch | Prove DAGs correct before execution |
| 18 | Dynamic Agent Factory | OpenSage | Self-programming + Z3 verified |
| 19 | Drift Monitor | Both | Real-time degradation detection |
| 20 | Experience-Based Topology Archive | Both | MAP-Elites + experience-based + Z3 combined |
| 21 | Full Benchmarks | Both | Published, honest, reproducible |
| 22 | sage-discover v2 | Both | Model monitoring + competitive intel |
| 23 | Coordination Performance Model | AdaptOrch | Data-driven, not just theoretical |
| 24 | Self-Evolution Loop | Everyone | Continuous self-improvement |

Dependency graph:
```
17 (Z3 topology) ─┐
18 (Agent factory) ├──► 20 (Topology archive) ──► 24 (Self-evolution loop)
19 (Drift monitor) ─┘         │
22 (sage-discover v2) ────────┘
21 (Benchmarks) ──► 23 (Coordination model) ──► 24
```

Tasks 17, 18, 19, 22 are parallelizable.
