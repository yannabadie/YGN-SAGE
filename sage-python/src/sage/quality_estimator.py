"""Quality estimation via formal verification (Z3) or learned model (ONNX).

Zero heuristics. Quality is either formally verified, model-predicted,
or unknown (None — bandit abstains from recording).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class QualityEstimator:
    """Estimate result quality without heuristics.

    Uses Rust QualityLabeler (Z3 formal verification) or Rust ONNX model.
    Returns None when quality cannot be assessed — never guesses.
    """

    def __init__(self) -> None:
        self._learned = self._try_load_onnx()
        self._labeler = self._try_load_labeler()

        if self._learned:
            log.info("QualityEstimator: ONNX learned model loaded")
        elif self._labeler:
            log.info("QualityEstimator: Z3 formal labeler active")
        else:
            log.warning("QualityEstimator: no backend available — will abstain")

    @staticmethod
    def _try_load_onnx():  # type: ignore[return]
        # Future capability: RustLearnedQualityEstimator from sage_core.
        # ONNX model not shipped; QualityLabeler (SMT) is the active backend.
        try:
            from sage_core import RustLearnedQualityEstimator  # noqa: F401 — future capability
            from pathlib import Path
            # ONNX model not shipped; QualityLabeler (SMT) is the active backend.
            # When quality_estimator_v2.onnx is trained and placed in models/,
            # this path will activate automatically.
            model_path = Path(__file__).parent.parent.parent / "models" / "quality_estimator_v2.onnx"
            tok_path = Path(__file__).parent.parent.parent / "models" / "tokenizer.json"
            if model_path.exists() and tok_path.exists():
                return RustLearnedQualityEstimator(str(model_path), str(tok_path))
        except (ImportError, Exception) as exc:
            log.debug("ONNX quality model not available: %s", exc)
        return None

    @staticmethod
    def _try_load_labeler():  # type: ignore[return]
        try:
            from sage_core import QualityLabeler
            return QualityLabeler()
        except ImportError:
            log.debug("QualityLabeler not available (sage_core not built with smt+tool-executor)")
        return None

    def estimate(
        self,
        task: str,
        result: str,
        latency_ms: float = 0.0,
        **kwargs,
    ) -> float | None:
        """Estimate quality. Returns float 0.0-1.0 or None (abstain)."""
        if not result or not result.strip():
            return 0.0

        if self._learned:
            try:
                return float(self._learned.estimate(task, result))
            except Exception as exc:
                log.debug("ONNX estimate failed: %s", exc)

        if self._labeler:
            try:
                label = self._labeler.label(task, result)
                if label is not None and label.assessable:
                    return float(label.score)
                return None
            except Exception as exc:
                log.debug("Z3 labeler failed: %s", exc)

        return None
