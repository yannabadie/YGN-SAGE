"""Built-in guardrails -- cost budget and schema validation.

Provides:
- CostGuardrail: blocks when accumulated cost exceeds a USD budget.
- SchemaGuardrail: blocks when output is not valid JSON or lacks required fields.
"""

from __future__ import annotations

import json

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
