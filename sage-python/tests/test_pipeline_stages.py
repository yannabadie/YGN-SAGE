"""Tests for pipeline_stages — domain inference and macro topology selection."""
from __future__ import annotations

import pytest
from sage.pipeline_stages import DAGFeatures, _infer_domain, select_macro_topology


# ---------------------------------------------------------------------------
# Stage 0: domain inference
# ---------------------------------------------------------------------------


def test_infer_domain_code():
    """'write a Python function' should map to 'code'."""
    result = _infer_domain("write a Python function that sorts a list")
    assert result == "code"


def test_infer_domain_math():
    """'solve the integral' should map to 'math'."""
    result = _infer_domain("solve the integral of x squared from 0 to 1")
    assert result == "math"


def test_infer_domain_reasoning():
    """'analyze the pros and cons' should map to 'reasoning'."""
    result = _infer_domain("analyze the pros and cons of microservices vs monoliths")
    assert result == "reasoning"


def test_infer_domain_default():
    """'hello' with no domain keywords should return 'general'."""
    result = _infer_domain("hello")
    assert result == "general"


def test_infer_domain_formal():
    """Text with formal verification keywords should return 'formal'."""
    result = _infer_domain("verify the invariant holds with smt solver")
    assert result == "formal"


def test_infer_domain_creative():
    """Text with creative writing keywords should return 'creative'."""
    result = _infer_domain("write a story about a robot who learns to paint")
    # 'write' matches creative; 'story' confirms it
    assert result == "creative"


def test_infer_domain_empty_string():
    """Empty string should return 'general'."""
    result = _infer_domain("")
    assert result == "general"


def test_infer_domain_case_insensitive():
    """Domain matching should be case-insensitive."""
    result = _infer_domain("IMPLEMENT an ALGORITHM in PYTHON")
    assert result == "code"


# ---------------------------------------------------------------------------
# Stage 2: macro topology selection
# ---------------------------------------------------------------------------


def test_select_macro_sequential():
    """omega=1, delta=3, gamma=0.2 → 'sequential' (low parallelism, shallow)."""
    features = DAGFeatures(omega=1, delta=3, gamma=0.2)
    result = select_macro_topology(features)
    assert result == "sequential"


def test_select_macro_parallel():
    """omega=4, delta=2, gamma=0.3 → 'parallel' (high parallelism, low coupling)."""
    features = DAGFeatures(omega=4, delta=2, gamma=0.3)
    result = select_macro_topology(features)
    assert result == "parallel"


def test_select_macro_hierarchical():
    """omega=2, delta=4, gamma=0.8 → 'hierarchical' (high coupling density)."""
    features = DAGFeatures(omega=2, delta=4, gamma=0.8)
    result = select_macro_topology(features)
    assert result == "hierarchical"


def test_select_macro_sequential_boundary():
    """omega=1, delta=5, gamma=0.0 is at the boundary — still sequential."""
    features = DAGFeatures(omega=1, delta=5, gamma=0.0)
    result = select_macro_topology(features)
    assert result == "sequential"


def test_select_macro_high_depth_low_coupling():
    """omega=1, delta=10 (deep chain) with low gamma → not sequential (delta > 5)."""
    features = DAGFeatures(omega=1, delta=10, gamma=0.1)
    result = select_macro_topology(features)
    # delta > _THETA_DELTA=5, omega=1, gamma<0.6 → hits parallel branch or hybrid
    assert result in ("parallel", "hybrid")


def test_select_macro_returns_string():
    """Return value is always a non-empty string."""
    features = DAGFeatures(omega=3, delta=3, gamma=0.5)
    result = select_macro_topology(features)
    assert isinstance(result, str)
    assert len(result) > 0
