"""Built-in guardrails -- cost budget, output validation, and schema validation.

Provides:
- CostGuardrail: blocks when accumulated cost exceeds a USD budget.
- OutputGuardrail: warns when free-text output is empty, too long, or a refusal.
- SchemaGuardrail: blocks when output is not valid JSON or lacks required fields.
"""

from __future__ import annotations

import json
import re

from sage.guardrails.base import Guardrail, GuardrailResult


class CostGuardrail(Guardrail):
    """Block if cumulative cost exceeds a USD budget.

    Reads ``context["cost_usd"]`` and compares against *max_usd*.

    Args:
        max_usd: Maximum allowed cost in US dollars (default 1.0).
    """

    name: str = "cost"

    def __init__(self, max_usd: float = 1.0) -> None:
        self.max_usd = max_usd

    async def check(
        self,
        input: str = "",
        output: str = "",
        context: dict | None = None,
    ) -> GuardrailResult:
        ctx = context or {}
        cost = ctx.get("cost_usd", 0.0)
        if cost > self.max_usd:
            return GuardrailResult(
                passed=False,
                reason=f"Cost ${cost:.2f} exceeds budget of ${self.max_usd:.2f}",
                severity="block",
            )
        return GuardrailResult(passed=True)


class OutputGuardrail(Guardrail):
    """Warn when free-text output is empty, too long, or looks like a refusal.

    Unlike :class:`SchemaGuardrail` (which expects JSON), this guardrail is
    designed for the common case where the agent returns plain text.

    Args:
        min_length: Minimum output length in characters (default 1).
        max_length: Maximum output length in characters (default 100 000).
        refusal_patterns: Case-insensitive substrings that indicate a refusal.
            Defaults to ``["I cannot", "I'm sorry", "I am unable"]``.
    """

    name: str = "output"

    _DEFAULT_REFUSAL_PATTERNS: list[str] = [
        "I cannot",
        "I'm sorry",
        "I am unable",
    ]

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 100_000,
        refusal_patterns: list[str] | None = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.refusal_patterns = (
            refusal_patterns
            if refusal_patterns is not None
            else list(self._DEFAULT_REFUSAL_PATTERNS)
        )
        # Pre-compile a single regex for refusal detection (case-insensitive)
        if self.refusal_patterns:
            escaped = [re.escape(p) for p in self.refusal_patterns]
            self._refusal_re = re.compile("|".join(escaped), re.IGNORECASE)
        else:
            self._refusal_re = None

    async def check(
        self,
        input: str = "",
        output: str = "",
        context: dict | None = None,
    ) -> GuardrailResult:
        # Skip when there is no output to validate (e.g., during input phase)
        if not output and not self.min_length:
            return GuardrailResult(passed=True)

        # Length checks
        if len(output) < self.min_length:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Output too short: {len(output)} chars "
                    f"(minimum {self.min_length})"
                ),
                severity="warn",
            )

        if len(output) > self.max_length:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Output too long: {len(output)} chars "
                    f"(maximum {self.max_length})"
                ),
                severity="warn",
            )

        # Refusal detection
        if self._refusal_re and self._refusal_re.search(output):
            return GuardrailResult(
                passed=False,
                reason="Output appears to be a refusal",
                severity="warn",
            )

        return GuardrailResult(passed=True)


class SchemaGuardrail(Guardrail):
    """Block if output is not valid JSON or lacks required fields.

    Args:
        required_fields: List of top-level keys that must be present in the
            parsed JSON object. If ``None``, only JSON validity is checked.
    """

    name: str = "schema"

    def __init__(self, required_fields: list[str] | None = None) -> None:
        self.required_fields = required_fields or []

    async def check(
        self,
        input: str = "",
        output: str = "",
        context: dict | None = None,
    ) -> GuardrailResult:
        # Skip when there is no output to validate (e.g., during input phase)
        if not output:
            return GuardrailResult(passed=True)

        # Parse JSON
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return GuardrailResult(
                passed=False,
                reason="Output is not valid JSON",
                severity="block",
            )

        # Check required fields
        if not isinstance(data, dict):
            return GuardrailResult(
                passed=False,
                reason="Output JSON is not an object",
                severity="block",
            )

        missing = [f for f in self.required_fields if f not in data]
        if missing:
            return GuardrailResult(
                passed=False,
                reason=f"Missing required fields: {', '.join(missing)}",
                severity="block",
            )

        return GuardrailResult(passed=True)
