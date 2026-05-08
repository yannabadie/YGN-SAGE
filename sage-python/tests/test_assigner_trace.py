"""P1-2B: Rust ModelAssigner top-3 trace — Python boundary tests."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sage_core = pytest.importorskip("sage_core")


def test_rust_assigner_last_assignment_trace_exists():
    """last_assignment_trace is exposed on Rust ModelAssigner."""
    _repo = Path(__file__).resolve().parents[2]
    registry = sage_core.ModelRegistry.from_toml_file(
        str(_repo / "sage-core" / "config" / "cards.toml"),
    )
    assigner = sage_core.ModelAssigner(registry)
    assert hasattr(assigner, "last_assignment_trace")


def test_rust_assigner_trace_empty_before_any_assignment():
    """Trace is empty before assign_models is called."""
    _repo = Path(__file__).resolve().parents[2]
    registry = sage_core.ModelRegistry.from_toml_file(
        str(_repo / "sage-core" / "config" / "cards.toml"),
    )
    assigner = sage_core.ModelAssigner(registry)
    trace = list(assigner.last_assignment_trace())
    assert trace == []


def test_log_model_assigner_trace_if_available_logs_real_scores(
    monkeypatch, caplog,
):
    """When Rust trace is available, log_model_assigner_trace_if_available
    emits log lines with real scores."""
    monkeypatch.setenv("SAGE_ASSIGNER_LOG_TOP3", "1")
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    trace_item = SimpleNamespace(
        node_idx=0, rank=1, model_id="smart",
        total_score=0.912345, affinity_score=0.95, domain_score=0.90,
        cost_norm=0.20, hint_bonus=0.15, diversity_penalty=0.08,
        filtered_reason="ok",
    )

    class _Assigner:
        def last_assignment_trace(self):
            return [trace_item]

    from sage.pipeline_v2.assign_models import (
        log_model_assigner_trace_if_available,
    )

    pipeline = SimpleNamespace(assigner=_Assigner())
    ctx = SimpleNamespace()

    assert log_model_assigner_trace_if_available(pipeline, ctx) is True

    messages = [r.getMessage() for r in caplog.records]
    assert any("model_assigner.candidates" in m for m in messages), messages
    assert any("trace_available=true" in m for m in messages), messages
    assert any("source=rust_pyo3_trace" in m for m in messages), messages
    assert any("score=0.912345" in m for m in messages), messages


def test_fallback_logs_trace_unavailable_when_no_rust_trace(
    monkeypatch, caplog,
):
    """When the assigner has no trace API, fallback emits trace_available=false."""
    monkeypatch.setenv("SAGE_ASSIGNER_LOG_TOP3", "1")
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    class _OpaqueAssigner:
        pass

    node = SimpleNamespace(model_id="chosen-model")
    topology = SimpleNamespace(
        node_count=lambda: 1,
        get_node=lambda idx: node,
    )

    from sage.pipeline_v2.assign_models import (
        log_model_assigner_chosen_fallback,
        log_model_assigner_trace_if_available,
    )

    pipeline = SimpleNamespace(assigner=_OpaqueAssigner())
    ctx = SimpleNamespace(topology=topology, assignments={0: "chosen-model"})

    assert log_model_assigner_trace_if_available(pipeline, ctx) is False
    log_model_assigner_chosen_fallback(pipeline, ctx)

    messages = [r.getMessage() for r in caplog.records]
    assert any("model_assigner.selected" in m for m in messages), messages
    assert any("trace_available=false" in m for m in messages), messages
    assert any("reason_code=rust_trace_not_exposed" in m for m in messages), messages
