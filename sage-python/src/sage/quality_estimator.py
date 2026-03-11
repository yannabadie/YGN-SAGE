"""Multi-signal quality estimation for topology learning feedback.

Replaces the rudimentary `len(result) > 10` heuristic with a 5-signal
estimator that produces a 0.0-1.0 quality score.
"""
from __future__ import annotations


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

        score = 0.3  # Signal 1: non-empty

        # Signal 2: Length adequacy
        task_words = len(task.split())
        result_words = len(result.split())
        if task_words < 10:
            score += min(result_words / 20, 1.0) * 0.2
        else:
            score += min(result_words / 50, 1.0) * 0.2

        # Signal 3: Code task + code presence
        code_keywords = {"def ", "class ", "function ", "import ", "```"}
        task_wants_code = any(
            kw in task.lower()
            for kw in ("write", "code", "implement", "function", "class", "fix", "debug")
        )
        result_has_code = any(kw in result for kw in code_keywords)
        if task_wants_code and result_has_code:
            score += 0.2
        elif not task_wants_code:
            score += 0.1

        # Signal 4: No error indicators
        error_phrases = ("error", "exception", "traceback", "failed", "cannot")
        if not had_errors and not any(e in result.lower() for e in error_phrases):
            score += 0.15

        # Signal 5: AVR convergence
        if avr_iterations > 0:
            if avr_iterations <= 2:
                score += 0.15
            elif avr_iterations <= 4:
                score += 0.10
            else:
                score += 0.05

        return min(score, 1.0)
