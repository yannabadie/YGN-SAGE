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

    def test_onnx_gate_off_by_default(self, monkeypatch):
        """Cycle-10 P7 (cgpro 2026-05-04): SAGE_QUALITY_ONNX must be off by default.

        Without the explicit opt-in, the ONNX load path must NOT activate
        even if a quality_estimator_v2.onnx file is present. This prevents
        an accidentally-dropped artifact from silently switching the runtime
        onto a learned-model path that has not been validated.
        """
        # Ensure no env var leaks from CI or prior tests.
        monkeypatch.delenv("SAGE_QUALITY_ONNX", raising=False)
        from sage.quality_estimator import QualityEstimator
        qe = QualityEstimator()
        # Backend must NOT be onnx when the gate is off; either Z3 labeler
        # (sage_core+smt available) or "none" (abstain).
        assert qe.backend_name != "onnx", (
            f"SAGE_QUALITY_ONNX off must not select onnx backend; got {qe.backend_name!r}"
        )

    def test_onnx_gate_explicit_off(self, monkeypatch):
        """Same as above but with SAGE_QUALITY_ONNX explicitly set to '0'."""
        monkeypatch.setenv("SAGE_QUALITY_ONNX", "0")
        from sage.quality_estimator import QualityEstimator
        qe = QualityEstimator()
        assert qe.backend_name != "onnx"

    def test_onnx_gate_on_but_artifact_missing_falls_through(self, monkeypatch):
        """SAGE_QUALITY_ONNX=1 + no artifact → still no onnx backend.

        Even with the gate opted in, missing artifact must fall through
        to Z3 labeler or abstain. The runtime never invents a learned
        score from nothing.
        """
        monkeypatch.setenv("SAGE_QUALITY_ONNX", "1")
        from sage.quality_estimator import QualityEstimator
        qe = QualityEstimator()
        # quality_estimator_v2.onnx is not in the repo (cycle-10 truth);
        # backend must be either z3_labeler (if sage_core+smt available)
        # or "none" — never "onnx".
        assert qe.backend_name != "onnx", (
            "SAGE_QUALITY_ONNX=1 with missing artifact must not "
            "select onnx backend"
        )
