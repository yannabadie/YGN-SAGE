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
