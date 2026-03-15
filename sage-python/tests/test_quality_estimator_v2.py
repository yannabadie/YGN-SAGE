"""Tests for the new QualityEstimator (zero heuristic)."""
from __future__ import annotations
import pytest


class TestQualityEstimatorV2:
    def test_empty_response_returns_zero(self):
        from sage.quality_estimator import QualityEstimator
        qe = QualityEstimator()
        assert qe.estimate("task", "") == 0.0
        assert qe.estimate("task", "   ") == 0.0

    def test_whitespace_only_returns_zero(self):
        from sage.quality_estimator import QualityEstimator
        qe = QualityEstimator()
        assert qe.estimate("task", "\n\n") == 0.0

    def test_returns_float_or_none(self):
        from sage.quality_estimator import QualityEstimator
        qe = QualityEstimator()
        result = qe.estimate("write add", "def add(a,b): return a+b")
        assert result is None or isinstance(result, float)

    def test_no_heuristic_constants(self):
        """Verify zero heuristic: no QUALITY_* constants used."""
        import sage.quality_estimator as mod
        source = open(mod.__file__, "r").read()
        assert "QUALITY_BASELINE" not in source
        assert "QUALITY_LENGTH_WEIGHT" not in source
        assert "QUALITY_CODE_WEIGHT" not in source
        assert "QUALITY_ERROR_WEIGHT" not in source
