"""Tests for Guardrails Framework -- cost budget, schema validation, pipeline.

TDD: Tests written BEFORE implementation.
"""
import json

import pytest

from sage.guardrails.base import GuardrailResult, Guardrail, GuardrailPipeline
from sage.guardrails.builtin import CostGuardrail, SchemaGuardrail


# ---------------------------------------------------------------------------
# Test 1: GuardrailResult creation -- passed
# ---------------------------------------------------------------------------
def test_guardrail_result_passed():
    """A passing GuardrailResult has passed=True and default severity 'info'."""
    result = GuardrailResult(passed=True)
    assert result.passed is True
    assert result.reason == ""
    assert result.severity == "info"


# ---------------------------------------------------------------------------
# Test 2: GuardrailResult creation -- blocked
# ---------------------------------------------------------------------------
def test_guardrail_result_blocked():
    """A blocked GuardrailResult carries a reason and severity."""
    result = GuardrailResult(passed=False, reason="over budget", severity="block")
    assert result.passed is False
    assert result.reason == "over budget"
    assert result.severity == "block"


# ---------------------------------------------------------------------------
# Test 3: CostGuardrail passes under budget
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cost_guardrail_passes_under_budget():
    """CostGuardrail passes when cost_usd is under max_usd."""
    guard = CostGuardrail(max_usd=1.0)
    result = await guard.check(context={"cost_usd": 0.50})
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 4: CostGuardrail blocks over budget
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cost_guardrail_blocks_over_budget():
    """CostGuardrail blocks when cost_usd exceeds max_usd, with 'budget' in reason."""
    guard = CostGuardrail(max_usd=1.0)
    result = await guard.check(context={"cost_usd": 2.50})
    assert result.passed is False
    assert "budget" in result.reason.lower()
    assert result.severity == "block"


# ---------------------------------------------------------------------------
# Test 5: SchemaGuardrail passes valid JSON with required fields
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_schema_guardrail_passes_valid_json():
    """SchemaGuardrail passes when output is valid JSON containing required fields."""
    guard = SchemaGuardrail(required_fields=["name", "score"])
    output = json.dumps({"name": "test", "score": 42})
    result = await guard.check(output=output)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 6: SchemaGuardrail fails on missing field
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_schema_guardrail_fails_missing_field():
    """SchemaGuardrail blocks when a required field is absent."""
    guard = SchemaGuardrail(required_fields=["name", "score"])
    output = json.dumps({"name": "test"})
    result = await guard.check(output=output)
    assert result.passed is False
    assert result.severity == "block"


# ---------------------------------------------------------------------------
# Test 7: SchemaGuardrail fails on invalid JSON
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_schema_guardrail_fails_invalid_json():
    """SchemaGuardrail blocks when output is not valid JSON."""
    guard = SchemaGuardrail(required_fields=["name"])
    result = await guard.check(output="not json {{{")
    assert result.passed is False
    assert result.severity == "block"


# ---------------------------------------------------------------------------
# Test 8: Pipeline runs all guards and collects results
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_runs_all_guards():
    """GuardrailPipeline.check_all runs every guardrail and returns all results."""
    pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=5.0),
        SchemaGuardrail(required_fields=["answer"]),
    ])
    output = json.dumps({"answer": "42"})
    results = await pipeline.check_all(output=output, context={"cost_usd": 1.0})
    assert len(results) == 2
    assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Test 9: Pipeline reports correctly with mixed pass/fail
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_mixed_pass_fail():
    """any_blocked returns True when at least one guardrail blocks."""
    pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=0.10),   # will block
        SchemaGuardrail(required_fields=["answer"]),  # will pass
    ])
    output = json.dumps({"answer": "42"})
    results = await pipeline.check_all(output=output, context={"cost_usd": 5.0})
    assert len(results) == 2
    assert pipeline.any_blocked(results) is True
    # Exactly one should have failed
    blocked = [r for r in results if not r.passed]
    passed = [r for r in results if r.passed]
    assert len(blocked) == 1
    assert len(passed) == 1


# ---------------------------------------------------------------------------
# Test 10: Pipeline any_blocked returns False when all pass
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_any_blocked_false_when_all_pass():
    """any_blocked returns False when every guardrail passes."""
    pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=10.0),
    ])
    results = await pipeline.check_all(context={"cost_usd": 0.01})
    assert pipeline.any_blocked(results) is False
