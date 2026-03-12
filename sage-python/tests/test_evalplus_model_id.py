"""Test that benchmark reports capture model_id from router, not 'unknown'."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from sage.bench.runner import BenchReport, TaskResult


def test_bench_report_has_model_field():
    """BenchReport must have a 'model' field."""
    report = BenchReport.from_results("test", [])
    assert hasattr(report, "model")


def test_bench_report_model_propagated():
    """Model ID from results should propagate to report."""
    results = [TaskResult(task_id="t1", passed=True)]
    report = BenchReport.from_results("test", results, model_config={"model": "gemini-2.5-flash"})
    assert report.model == "gemini-2.5-flash"
