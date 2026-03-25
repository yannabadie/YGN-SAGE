"""Tests for execution path tracking.

Issue A audit fix: AgentSystem must track and log which execution path
was taken (pipeline, pipeline_fallback_legacy, or legacy).
"""
from sage.boot import AgentSystem


def test_last_execution_path_attribute_exists():
    """AgentSystem must have _last_execution_path attribute."""
    assert hasattr(AgentSystem, "_last_execution_path")
