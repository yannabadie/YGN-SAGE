from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from types import SimpleNamespace
from typing import Any

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_stages import DAGFeatures
from sage.pipeline_v2.select_topology import select_topology


class _Topology:
    def __init__(self, template_type: str = "sequential", nodes: int = 3) -> None:
        self.template_type = template_type
        self.id = f"{template_type}-test-id"
        self._nodes = nodes

    def node_count(self) -> int:
        return self._nodes

    def get_edges(self) -> list[tuple[int, int, int]]:
        return []


@dataclass
class _GenerateResult:
    topology: _Topology
    source: str = "engine_dynamic"
    confidence: float = 0.42
    candidates: list[Any] | None = None


@dataclass
class _Candidate:
    topology: _Topology
    source: str
    confidence: float


class _Engine:
    def __init__(self, result: _GenerateResult) -> None:
        self.result = result
        self.calls = 0

    def generate(
        self,
        task: str,
        task_embedding: Any,
        system: int,
        budget: float,
    ) -> _GenerateResult:
        del task, task_embedding, system, budget
        self.calls += 1
        return self.result


class _FailingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        task: str,
        task_embedding: Any,
        system: int,
        budget: float,
    ) -> _GenerateResult:
        del task, task_embedding, system, budget
        self.calls += 1
        raise RuntimeError("engine unavailable")


class _NoSemanticEmbedder:
    is_semantic = False


def _stub_embedder(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sage.memory.embedder",
        SimpleNamespace(Embedder=lambda: _NoSemanticEmbedder()),
    )


def _ctx() -> PipelineContext:
    return PipelineContext(
        task="solve a code task",
        domain="code",
        system=2,
        budget=10.0,
        dag_features=DAGFeatures(omega=1, delta=1, gamma=0.0),
    )


def test_skip_dag_template_routes_through_engine(monkeypatch, caplog) -> None:
    _stub_embedder(monkeypatch)
    monkeypatch.setenv("SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE", "1")
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    engine = _Engine(_GenerateResult(topology=_Topology("engine_graph", nodes=4)))
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=engine,
        assigner=None,
        provider_pool=None,
    )
    setattr(pipeline, "_build_topology_from_hint", lambda hint: _Topology(hint))

    result = select_topology(pipeline, _ctx())

    assert engine.calls == 1
    assert result.topology is engine.result.topology
    source_lines = [
        record.getMessage()
        for record in caplog.records
        if "topology.source" in record.getMessage()
    ]
    assert any("source=engine_dynamic" in line for line in source_lines)
    assert not any("source=dag_template" in line for line in source_lines)


def test_log_all_candidates_emits_per_candidate_lines(monkeypatch, caplog) -> None:
    _stub_embedder(monkeypatch)
    monkeypatch.setenv("SAGE_TOPOLOGY_FORCE_ENGINE", "1")
    monkeypatch.setenv("SAGE_TOPOLOGY_LOG_ALL_CANDIDATES", "1")
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    candidates = [
        _Candidate(_Topology("archive_graph", nodes=2), "archive_hit", 0.91),
        _Candidate(_Topology("mutation_graph", nodes=5), "mutation", 0.64),
    ]
    engine = _Engine(
        _GenerateResult(
            topology=candidates[0].topology,
            source=candidates[0].source,
            confidence=candidates[0].confidence,
            candidates=candidates,
        )
    )
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=engine,
        assigner=None,
        provider_pool=None,
    )

    select_topology(pipeline, _ctx())

    candidate_lines = [
        record.getMessage()
        for record in caplog.records
        if "topology.candidate" in record.getMessage()
    ]
    assert len(candidate_lines) == len(candidates)
    assert "path=1 source=archive_hit archive_hit=true" in candidate_lines[0]
    assert "path=2 source=mutation archive_hit=false" in candidate_lines[1]


def test_default_candidate_mode_evaluates_engine_over_dag_template(
    monkeypatch, caplog,
) -> None:
    _stub_embedder(monkeypatch)
    monkeypatch.delenv("SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE", raising=False)
    monkeypatch.delenv("SAGE_TOPOLOGY_FORCE_ENGINE", raising=False)
    monkeypatch.delenv("SAGE_TOPOLOGY_LOG_ALL_CANDIDATES", raising=False)
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    engine_topology = _Topology("engine_graph")
    engine = _Engine(_GenerateResult(topology=engine_topology))
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=engine,
        assigner=None,
        provider_pool=None,
    )
    setattr(pipeline, "_build_topology_from_hint", lambda hint: _Topology(hint))

    result = select_topology(pipeline, _ctx())

    assert engine.calls == 1
    assert result.topology is engine_topology
    lines = [record.getMessage() for record in caplog.records]
    assert any("topology.source source=dag_template" in line for line in lines)
    assert any("topology.source source=engine_dynamic" in line for line in lines)
    assert not any("topology.candidate" in line for line in lines)


def test_engine_failure_falls_back_to_dag_template(monkeypatch, caplog) -> None:
    _stub_embedder(monkeypatch)
    monkeypatch.delenv("SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE", raising=False)
    monkeypatch.delenv("SAGE_TOPOLOGY_FORCE_ENGINE", raising=False)
    monkeypatch.delenv("SAGE_TOPOLOGY_LOG_ALL_CANDIDATES", raising=False)
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    engine = _FailingEngine()
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=engine,
        assigner=None,
        provider_pool=None,
    )

    result = select_topology(pipeline, _ctx())

    assert engine.calls == 1
    assert result.topology is not None
    assert result.topology.template_type == "sequential"
    lines = [record.getMessage() for record in caplog.records]
    assert any("Stage 2 topology engine failed" in line for line in lines)
    assert any("topology.source source=dag_template" in line for line in lines)


def test_engine_no_allowed_path_falls_back_to_dag_template(monkeypatch, caplog) -> None:
    _stub_embedder(monkeypatch)
    monkeypatch.delenv("SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE", raising=False)
    monkeypatch.delenv("SAGE_TOPOLOGY_FORCE_ENGINE", raising=False)
    monkeypatch.delenv("SAGE_TOPOLOGY_LOG_ALL_CANDIDATES", raising=False)
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    no_allowed_topology = _Topology("no_allowed_graph")
    engine = _Engine(
        _GenerateResult(
            topology=no_allowed_topology,
            source="no_allowed_path",
            confidence=0.0,
        )
    )
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=engine,
        assigner=None,
        provider_pool=None,
    )
    setattr(pipeline, "_build_topology_from_hint", lambda hint: _Topology(hint))

    result = select_topology(pipeline, _ctx())

    assert engine.calls == 1
    assert result.topology is not no_allowed_topology
    assert result.topology is not None
    lines = [record.getMessage() for record in caplog.records]
    assert any("Stage 2 topology engine abstained" in line for line in lines)
    assert any("topology.source source=dag_template" in line for line in lines)
    assert not any("topology.source source=no_allowed_path" in line for line in lines)
