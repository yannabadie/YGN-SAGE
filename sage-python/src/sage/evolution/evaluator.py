"""Evaluation cascade for scoring candidate solutions.

Inspired by AlphaEvolve's progressive evaluation: fast cheap tests
first, then expensive accurate tests only for promising candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class EvalResult:
    """Result of evaluating a candidate."""
    score: float
    passed: bool
    stage: str  # Which evaluation stage produced this
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalStage:
    """A single stage in the evaluation cascade."""
    name: str
    evaluator: Callable[[str], Awaitable[EvalResult]]
    threshold: float = 0.0  # Minimum score to pass to next stage
    weight: float = 1.0


class Evaluator:
    """Progressive evaluation cascade.

    Runs candidates through stages of increasing cost/accuracy.
    A candidate must pass each stage's threshold to advance.
    """

    def __init__(self):
        self._stages: list[EvalStage] = []

    def add_stage(
        self,
        name: str,
        evaluator: Callable[[str], Awaitable[EvalResult]],
        threshold: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        """Add an evaluation stage to the cascade."""
        self._stages.append(EvalStage(
            name=name,
            evaluator=evaluator,
            threshold=threshold,
            weight=weight,
        ))

    async def evaluate(self, code: str) -> EvalResult:
        """Run code through the evaluation cascade.

        Returns the final EvalResult with a weighted aggregate score.
        Stops early if a stage fails its threshold.
        """
        total_score = 0.0
        total_weight = 0.0

        for stage in self._stages:
            try:
                result = await stage.evaluator(code)
                result.stage = stage.name
            except Exception as e:
                result = EvalResult(
                    score=0.0,
                    passed=False,
                    stage=stage.name,
                    error=str(e),
                )

            total_score += result.score * stage.weight
            total_weight += stage.weight

            if result.score < stage.threshold:
                # Didn't pass this stage
                return EvalResult(
                    score=total_score / total_weight if total_weight > 0 else 0.0,
                    passed=False,
                    stage=stage.name,
                    details={"failed_at": stage.name, "stage_score": result.score},
                    error=result.error,
                )

        # Passed all stages
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return EvalResult(
            score=final_score,
            passed=True,
            stage=self._stages[-1].name if self._stages else "none",
            details={"stages_passed": len(self._stages)},
        )

    def stage_count(self) -> int:
        return len(self._stages)


# ---------------------------------------------------------------------------
# Statistical validation for EvolutionEngine
# ---------------------------------------------------------------------------

def validate_evolution(
    baseline_scores: list[float],
    evolved_scores: list[float],
) -> dict:
    """Compare baseline vs evolved topology scores with statistical rigor.

    Uses Wilcoxon signed-rank test (non-parametric, paired) and Cohen's d
    effect size to determine if evolution produced a genuine improvement.
    Blocks promotion to production if p > 0.05 or effect size <= 0.2.

    Parameters
    ----------
    baseline_scores : list[float]
        Quality scores from baseline topologies (N >= 10 required).
    evolved_scores : list[float]
        Quality scores from evolved topologies (same length, paired).

    Returns
    -------
    dict with keys: p_value, effect_size (Cohen's d), significant (bool),
    mean_improvement, n_runs, gate_passed.
    """
    import numpy as np

    if len(baseline_scores) != len(evolved_scores):
        return {"error": "Paired samples required (same length)", "significant": False, "gate_passed": False}

    n = len(baseline_scores)
    if n < 10:
        return {"error": f"Need N>=10 paired runs, got {n}", "significant": False, "gate_passed": False}

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"error": "scipy not installed", "significant": False, "gate_passed": False}

    diff = np.array(evolved_scores) - np.array(baseline_scores)

    # Wilcoxon signed-rank test (one-sided: evolved > baseline)
    try:
        _stat, p_value = wilcoxon(baseline_scores, evolved_scores, alternative="greater")
    except ValueError:
        # All differences are zero
        return {
            "p_value": 1.0,
            "effect_size": 0.0,
            "significant": False,
            "mean_improvement": 0.0,
            "n_runs": n,
            "gate_passed": False,
        }

    # Cohen's d effect size
    d = float(diff.mean() / (diff.std() + 1e-8))

    significant = p_value < 0.05

    return {
        "p_value": float(p_value),
        "effect_size": d,
        "significant": significant,
        "mean_improvement": float(diff.mean()),
        "n_runs": n,
        "gate_passed": significant and d > 0.2,
    }
