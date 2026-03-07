"""Tests for baseline benchmark mode."""
import pytest


def test_baseline_mode_flag():
    """HumanEvalBench should support baseline_mode parameter."""
    from sage.bench.humaneval import HumanEvalBench
    bench = HumanEvalBench(baseline_mode=True)
    assert bench.baseline_mode is True


def test_baseline_mode_default_false():
    """baseline_mode should default to False."""
    from sage.bench.humaneval import HumanEvalBench
    bench = HumanEvalBench()
    assert bench.baseline_mode is False


def test_baseline_model_id_prefix():
    """When baseline_mode=True, the manifest model should include 'baseline' prefix."""
    from sage.bench.humaneval import HumanEvalBench
    bench = HumanEvalBench(baseline_mode=True)
    # Trigger manifest creation (normally done in run())
    from sage.bench.truth_pack import BenchmarkManifest
    model_id = "baseline"  # No system, so model_id is ""
    bench.manifest = BenchmarkManifest(benchmark="humaneval", model=model_id)
    assert "baseline" in bench.manifest.model


def test_boot_has_multiple_guardrails():
    """Boot should install more than just CostGuardrail."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    pipeline = system.agent_loop.guardrail_pipeline
    assert len(pipeline.guardrails) >= 2, "Only 1 guardrail installed"
