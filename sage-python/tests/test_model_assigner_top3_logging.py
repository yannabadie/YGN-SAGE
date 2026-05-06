from __future__ import annotations

import logging
import math

from sage.llm.model_assigner import ModelAssigner
from sage.llm.model_registry import ModelRegistry
from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_v2.assign_models import assign_models


_CATALOG_TOML = """
[[models]]
id = "fast"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.4
tool_use_score = 0.3
math_score = 0.3
formal_z3_strength = 0.1
cost_input_per_m = 0.05
cost_output_per_m = 0.1
latency_ttft_ms = 50.0
tokens_per_sec = 600.0
s1_affinity = 0.9
s2_affinity = 0.4
s3_affinity = 0.2
supports_tools = false
supports_json_mode = true
context_window = 32000

[models.domain_scores]
code = 0.60

[[models]]
id = "balanced"
provider = "test"
family = "test"
code_score = 0.8
reasoning_score = 0.75
tool_use_score = 0.85
math_score = 0.6
formal_z3_strength = 0.4
cost_input_per_m = 0.5
cost_output_per_m = 1.5
latency_ttft_ms = 300.0
tokens_per_sec = 200.0
s1_affinity = 0.5
s2_affinity = 0.8
s3_affinity = 0.5
supports_tools = true
supports_json_mode = true
context_window = 128000

[models.domain_scores]
code = 0.85

[[models]]
id = "expert"
provider = "test"
family = "test"
code_score = 0.9
reasoning_score = 0.95
tool_use_score = 0.8
math_score = 0.95
formal_z3_strength = 0.9
cost_input_per_m = 1.0
cost_output_per_m = 2.0
latency_ttft_ms = 800.0
tokens_per_sec = 50.0
s1_affinity = 0.1
s2_affinity = 0.9
s3_affinity = 0.95
supports_tools = true
supports_json_mode = true
context_window = 200000

[models.domain_scores]
code = 0.92
"""


class _Node:
    def __init__(self) -> None:
        self.role = "coder"
        self.system = 2
        self.required_capabilities: list[str] = []
        self.max_cost_usd = 10.0
        self.model_id = ""


class _Graph:
    def __init__(self) -> None:
        self.node = _Node()

    def node_count(self) -> int:
        return 1

    def get_node(self, idx: int) -> _Node | None:
        return self.node if idx == 0 else None

    def set_node_model_id(self, idx: int, model_id: str) -> None:
        if idx == 0:
            self.node.model_id = model_id


def _assigner() -> ModelAssigner:
    return ModelAssigner(ModelRegistry.from_toml_str(_CATALOG_TOML))


def test_assigner_log_top3_flag_emits_ranked_candidates(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("SAGE_ASSIGNER_LOG_TOP3", "1")
    caplog.set_level(logging.INFO, logger="sage.llm.model_assigner")

    count = _assigner().assign_models(_Graph(), task_domain="code", budget_usd=10.0)

    assert count == 1
    lines = [
        record.getMessage()
        for record in caplog.records
        if "model_assigner.candidates" in record.getMessage()
    ]
    assert len(lines) == 3
    assert "node_id=0 rank=1" in lines[0]
    assert "rank=2" in lines[1]
    assert "rank=3" in lines[2]
    assert all("source=python_fallback" in line for line in lines)
    assert all("reason_code=ok" in line for line in lines)
    assert all("affinity=" in line and "cost_norm=" in line for line in lines)
    assert not any("nan" in line.lower() for line in lines)


def test_assigner_top3_logging_default_off_is_silent(monkeypatch, caplog) -> None:
    monkeypatch.delenv("SAGE_ASSIGNER_LOG_TOP3", raising=False)
    caplog.set_level(logging.INFO, logger="sage.llm.model_assigner")

    count = _assigner().assign_models(_Graph(), task_domain="code", budget_usd=10.0)

    assert count == 1
    assert not [
        record
        for record in caplog.records
        if "model_assigner.candidates" in record.getMessage()
    ]


class _RustLikeAssigner:
    def assign_models(
        self,
        graph,
        task_domain,
        budget_usd,
        hints=None,
        task_system=None,
    ) -> int:
        del task_domain, budget_usd, hints, task_system
        graph.set_node_model_id(0, "rust-choice")
        return 1


def test_pipeline_logs_chosen_model_when_top3_not_derivable(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SAGE_ASSIGNER_LOG_TOP3", "1")
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=_RustLikeAssigner(),
        provider_pool=None,
    )
    ctx = PipelineContext(
        task="assign",
        domain="code",
        system=2,
        budget=10.0,
        topology=_Graph(),
    )

    assign_models(pipeline, ctx)

    lines = [
        record.getMessage()
        for record in caplog.records
        if "model_assigner.candidates" in record.getMessage()
    ]
    assert lines == [
        "model_assigner.candidates node_id=0 rank=1 model=rust-choice "
        "source=wrapper_fallback reason_code=non_finite_score "
        "score=0.000000 affinity=0.000000 domain=0.000000 cost_norm=0.000000 "
        "hint_bonus=0.000000 diversity_penalty=0.000000"
    ]


class _NanCard:
    id = "nan-card"
    supports_tools = True
    supports_json_mode = True

    def estimate_cost(self, _input_tokens: int, _output_tokens: int) -> float:
        return 0.1

    def domain_score(self, _task_domain: str) -> float:
        return math.nan


class _NanCatalog:
    def all_models(self) -> list[_NanCard]:
        return [_NanCard()]

    def calibrated_affinity(self, _model_id: str, _system: int) -> float:
        return math.nan


def test_assigner_top3_logging_guards_non_finite_scores(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("SAGE_ASSIGNER_LOG_TOP3", "1")
    caplog.set_level(logging.INFO, logger="sage.llm.model_assigner")

    count = ModelAssigner(_NanCatalog()).assign_models(
        _Graph(),
        task_domain="code",
        budget_usd=10.0,
    )

    assert count == 1
    lines = [
        record.getMessage()
        for record in caplog.records
        if "model_assigner.candidates" in record.getMessage()
    ]
    assert lines == [
        "model_assigner.candidates node_id=0 rank=1 model=nan-card "
        "source=python_fallback reason_code=non_finite_score "
        "score=0.000000 affinity=0.000000 domain=0.000000 cost_norm=1.000000 "
        "hint_bonus=0.000000 diversity_penalty=0.000000"
    ]
