"""Integration tests for guardrail pipeline (no mocks).

Every test uses real guardrail implementations.
No API keys required, no mocks, no patches.
"""
import pytest

from sage.guardrails.base import GuardrailPipeline
from sage.guardrails.builtin import CostGuardrail, OutputGuardrail, SchemaGuardrail


class TestGuardrailPipelineIntegration:
    """Test guardrail pipeline with real guardrails."""

    async def test_output_guardrail_passes_valid(self):
        pipeline = GuardrailPipeline([OutputGuardrail(min_length=1)])
        results = await pipeline.check_all(output="Hello, world!")
        assert not pipeline.any_blocked(results)
        assert all(r.passed for r in results)

    async def test_output_guardrail_warns_empty(self):
        pipeline = GuardrailPipeline([OutputGuardrail(min_length=1)])
        results = await pipeline.check_all(output="")
        assert any(not r.passed for r in results)
        # OutputGuardrail uses severity="warn", not "block"
        assert not pipeline.any_blocked(results)

    async def test_output_guardrail_warns_refusal(self):
        pipeline = GuardrailPipeline([OutputGuardrail()])
        results = await pipeline.check_all(output="I cannot help with that request")
        assert any(not r.passed for r in results)
        assert results[0].severity == "warn"
        assert "refusal" in results[0].reason.lower()

    async def test_output_guardrail_warns_too_long(self):
        pipeline = GuardrailPipeline([OutputGuardrail(max_length=10)])
        results = await pipeline.check_all(output="A" * 100)
        assert any(not r.passed for r in results)
        assert results[0].severity == "warn"

    async def test_cost_guardrail_blocks_over_budget(self):
        pipeline = GuardrailPipeline([CostGuardrail(max_usd=0.01)])
        results = await pipeline.check_all(context={"cost_usd": 100.0})
        assert pipeline.any_blocked(results)
        assert results[0].severity == "block"

    async def test_cost_guardrail_passes_under_budget(self):
        pipeline = GuardrailPipeline([CostGuardrail(max_usd=10.0)])
        results = await pipeline.check_all(context={"cost_usd": 0.5})
        assert not pipeline.any_blocked(results)
        assert results[0].passed is True

    async def test_schema_guardrail_passes_valid_json(self):
        pipeline = GuardrailPipeline([SchemaGuardrail(required_fields=["name"])])
        results = await pipeline.check_all(output='{"name": "test"}')
        assert not pipeline.any_blocked(results)
        assert results[0].passed is True

    async def test_schema_guardrail_blocks_invalid_json(self):
        pipeline = GuardrailPipeline([SchemaGuardrail()])
        results = await pipeline.check_all(output="not json at all")
        assert pipeline.any_blocked(results)
        assert results[0].severity == "block"

    async def test_schema_guardrail_blocks_missing_fields(self):
        pipeline = GuardrailPipeline([SchemaGuardrail(required_fields=["name", "age"])])
        results = await pipeline.check_all(output='{"name": "test"}')
        assert pipeline.any_blocked(results)
        assert "age" in results[0].reason

    async def test_multi_guardrail_pipeline(self):
        pipeline = GuardrailPipeline([
            CostGuardrail(max_usd=10.0),
            OutputGuardrail(min_length=1),
        ])
        results = await pipeline.check_all(
            output="Valid output",
            context={"cost_usd": 0.1},
        )
        assert not pipeline.any_blocked(results)
        assert len(results) == 2
        assert all(r.passed for r in results)

    async def test_multi_guardrail_one_blocks(self):
        pipeline = GuardrailPipeline([
            CostGuardrail(max_usd=0.01),
            OutputGuardrail(min_length=1),
        ])
        results = await pipeline.check_all(
            output="Valid output",
            context={"cost_usd": 100.0},
        )
        # Cost guardrail blocks, output guardrail passes
        assert pipeline.any_blocked(results)
        assert not results[0].passed  # cost
        assert results[1].passed  # output
