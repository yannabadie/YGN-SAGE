"""Guardrails framework -- base classes for input/output validation.

Provides:
- GuardrailResult: outcome of a single guardrail check.
- Guardrail: abstract base class for all guardrails.
- GuardrailPipeline: runs a list of guardrails and aggregates results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """Outcome of a single guardrail check.

    Attributes:
        passed: True if the check succeeded.
        reason: Human-readable explanation (empty string when passed).
        severity: One of "info", "warn", "block".
    """

    passed: bool
    reason: str = ""
    severity: str = "info"


class Guardrail:
    """Abstract base class for guardrails.

    Subclasses must override :meth:`check`.
    """

    name: str = "base"

    async def check(
        self,
        input: str = "",
        output: str = "",
        context: dict | None = None,
    ) -> GuardrailResult:
        """Evaluate a guardrail condition.

        Args:
            input: The original input/prompt text.
            output: The model output text.
            context: Arbitrary metadata dict (e.g. cost_usd, model, step).

        Returns:
            GuardrailResult indicating pass/fail, reason, and severity.
        """
        return GuardrailResult(passed=True)


class GuardrailPipeline:
    """Runs a list of guardrails and aggregates their results.

    Args:
        guardrails: Ordered list of :class:`Guardrail` instances to evaluate.
    """

    def __init__(self, guardrails: list[Guardrail]) -> None:
        self.guardrails = list(guardrails)

    async def check_all(
        self,
        input: str = "",
        output: str = "",
        context: dict | None = None,
    ) -> list[GuardrailResult]:
        """Run every guardrail and return all results.

        Args:
            input: The original input/prompt text.
            output: The model output text.
            context: Arbitrary metadata dict.

        Returns:
            List of :class:`GuardrailResult`, one per guardrail, in order.
        """
        results: list[GuardrailResult] = []
        for guard in self.guardrails:
            result = await guard.check(input=input, output=output, context=context)
            results.append(result)
        return results

    def any_blocked(self, results: list[GuardrailResult]) -> bool:
        """Return True if any result has ``passed=False`` and ``severity='block'``."""
        return any(
            not r.passed and r.severity == "block"
            for r in results
        )
