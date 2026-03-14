"""Multi-signal quality estimation for topology learning feedback.

Replaces the rudimentary `len(result) > 10` heuristic with a 5-signal
estimator that produces a 0.0-1.0 quality score.
"""
from __future__ import annotations

import re
from typing import Any

from sage.constants import (
    QUALITY_BASELINE,
    QUALITY_LENGTH_WEIGHT,
    QUALITY_CODE_WEIGHT,
    QUALITY_ERROR_WEIGHT,
    QUALITY_AVR_FAST,
    QUALITY_AVR_MEDIUM,
    QUALITY_AVR_SLOW,
    QUALITY_NONCODE_BASELINE,
    QUALITY_LENGTH_DENOM_SHORT,
    QUALITY_LENGTH_DENOM_LONG,
)


class QualityEstimator:
    """Estimate result quality from multiple signals (0.0-1.0).

    Signals:
    1. Non-empty response (baseline 0.3)
    2. Length adequacy relative to task complexity (0.0-0.2)
    3. Code presence when task requests code (0.0-0.2)
    4. Absence of error indicators (0.0-0.15)
    5. AVR convergence speed (0.0-0.15)
    """

    @staticmethod
    def estimate(
        task: str,
        result: str,
        latency_ms: float = 0.0,
        had_errors: bool = False,
        avr_iterations: int = 0,
    ) -> float:
        if not result or not result.strip():
            return 0.0

        score = QUALITY_BASELINE  # Signal 1: non-empty

        # Signal 2: Length adequacy (task-aware)
        task_words = len(task.split())
        result_words = len(result.split())
        task_wants_code = any(
            kw in task.lower()
            for kw in ("write", "code", "implement", "function", "class", "fix", "debug")
        )
        if task_wants_code:
            # Code tasks: expect substantial output
            if task_words < 10:
                score += min(result_words / QUALITY_LENGTH_DENOM_SHORT, 1.0) * QUALITY_LENGTH_WEIGHT
            else:
                score += min(result_words / QUALITY_LENGTH_DENOM_LONG, 1.0) * QUALITY_LENGTH_WEIGHT
        else:
            # Non-code tasks (math, Q&A): length is not a quality proxy.
            # A correct math answer ("42") or factual answer is valid at any length.
            if result_words >= 1:
                score += QUALITY_LENGTH_WEIGHT

        # Signal 3: Code task + code presence
        code_keywords = {"def ", "class ", "function ", "import ", "```"}
        result_has_code = any(kw in result for kw in code_keywords)
        if task_wants_code and result_has_code:
            score += QUALITY_CODE_WEIGHT
        elif not task_wants_code:
            score += QUALITY_NONCODE_BASELINE

        # Signal 4: No error indicators (pattern-matched to reduce false positives)
        error_patterns = (r"^error:", r"^traceback", r"^exception:", r"failed to", r"cannot ")
        if not had_errors and not any(re.search(p, result.lower(), re.MULTILINE) for p in error_patterns):
            score += QUALITY_ERROR_WEIGHT

        # Signal 5: AVR convergence
        if avr_iterations > 0:
            if avr_iterations <= 2:
                score += QUALITY_AVR_FAST
            elif avr_iterations <= 4:
                score += QUALITY_AVR_MEDIUM
            else:
                score += QUALITY_AVR_SLOW

        return min(score, 1.0)


# Rust acceleration (Phase 2 rationalization)
try:
    from sage_core import RustQualityEstimator as _RustQE
    _HAS_RUST_QE = True
except ImportError:
    _HAS_RUST_QE = False


def create_quality_estimator() -> Any:
    """Factory: returns Rust estimator when available, Python otherwise.

    Returns a ``QualityEstimator``-compatible object (duck-typed). The Rust
    implementation conforms to the same interface but is not a Python subclass.
    """
    if _HAS_RUST_QE:
        try:
            qe = _RustQE()
            return qe
        except Exception:
            pass
    return QualityEstimator()
