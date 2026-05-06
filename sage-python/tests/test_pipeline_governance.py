"""Regression tests for A0b (2026-04-23, ALIRE2 §6 "fail-open governance").

``SAGE_STRICT_GOVERNANCE=1`` turns two previously-advisory failures into
hard aborts:

1. ``_build_write_gate`` exception → re-raise instead of returning None.
2. ``ctx.verification_passed == False`` in ``_stage_execute`` → raise
   and emit ``EXECUTE_HALTED_UNVERIFIED`` instead of logging a warning
   and continuing with ``EXECUTE_UNVERIFIED``.

Default behaviour (env unset) is unchanged — the dev-friendly fail-open
is preserved so existing smoke runs don't flip red on a flaky Rust
build or a soft Z3 signal.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sage.pipeline_v2.execute import execute


def _build_stub_pipeline(emit_mock):
    """Construct a minimally-viable Pipeline for _stage_execute bypass testing."""
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = None  # force the llm_provider or no-op branch
    pipeline.bandit = None
    pipeline.provider_pool = None
    pipeline.llm_provider = None
    pipeline._last_routing_decision = None
    pipeline._emit = emit_mock
    pipeline.event_bus = None
    pipeline.write_gate = None
    return pipeline


# ---------------------------------------------------------------------------
# Verification-failure gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verification_fail_default_continues():
    """Default (env unset): unverified provider assignment logs-and-continues.

    Locks the pre-A0b behaviour. We assert the pipeline does NOT raise
    and that it emits ``EXECUTE_UNVERIFIED``.
    """
    emit = MagicMock()
    pipeline = _build_stub_pipeline(emit)

    ctx = SimpleNamespace(
        task="t",
        topology=None,
        system=1,
        result=None,
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=False,
        bandit_decision_id=None,
    )

    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("SAGE_STRICT_GOVERNANCE", None)
        await execute(pipeline, ctx)

    kinds = [call.args[0] for call in emit.call_args_list]
    assert "EXECUTE_UNVERIFIED" in kinds, (
        f"expected EXECUTE_UNVERIFIED in default mode; got {kinds}"
    )
    assert "EXECUTE_HALTED_UNVERIFIED" not in kinds, (
        "strict-mode event must not fire when env is unset"
    )


@pytest.mark.asyncio
async def test_verification_fail_strict_mode_aborts():
    """SAGE_STRICT_GOVERNANCE=1: verification failure raises + emits HALTED."""
    emit = MagicMock()
    pipeline = _build_stub_pipeline(emit)

    ctx = SimpleNamespace(
        task="t",
        topology=None,
        system=1,
        result=None,
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=False,
        bandit_decision_id=None,
    )

    with patch.dict("os.environ", {"SAGE_STRICT_GOVERNANCE": "1"}, clear=False):
        with pytest.raises(RuntimeError, match="SAGE_STRICT_GOVERNANCE"):
            await execute(pipeline, ctx)

    kinds = [call.args[0] for call in emit.call_args_list]
    assert "EXECUTE_HALTED_UNVERIFIED" in kinds
    assert "EXECUTE_UNVERIFIED" not in kinds, (
        "strict mode emits HALTED, not the advisory UNVERIFIED"
    )


# ---------------------------------------------------------------------------
# Write-gate init failure gate
# ---------------------------------------------------------------------------

def test_write_gate_init_failure_default_returns_none():
    """Default: write-gate init failure returns None (ungated writes)."""
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._gate_config = {}  # empty — triggers failure in factory

    def _fail(**_kwargs):
        raise RuntimeError("simulated init failure")

    with patch(
        "sage.memory.write_gate.create_composite_write_gate", side_effect=_fail
    ):
        import os
        os.environ.pop("SAGE_STRICT_GOVERNANCE", None)
        result = pipeline._build_write_gate()

    assert result is None, "default mode should return None on init failure"


def test_write_gate_init_failure_strict_mode_raises():
    """SAGE_STRICT_GOVERNANCE=1: init failure re-raises for caller abort."""
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._gate_config = {}

    def _fail(**_kwargs):
        raise RuntimeError("simulated init failure")

    with patch(
        "sage.memory.write_gate.create_composite_write_gate", side_effect=_fail
    ):
        with patch.dict(
            "os.environ", {"SAGE_STRICT_GOVERNANCE": "1"}, clear=False
        ):
            with pytest.raises(RuntimeError, match="simulated init failure"):
                pipeline._build_write_gate()


# ---------------------------------------------------------------------------
# Env parser behaviour (truthy matrix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "val,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("garbage", False),
    ],
)
def test_is_strict_governance_parsing(val, expected):
    """Direct unit test for the env-var parser."""
    from sage.pipeline import _is_strict_governance

    with patch.dict("os.environ", {"SAGE_STRICT_GOVERNANCE": val}, clear=False):
        assert _is_strict_governance() is expected


def test_is_strict_governance_unset_is_false():
    """Unset env variable = default off."""
    from sage.pipeline import _is_strict_governance

    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("SAGE_STRICT_GOVERNANCE", None)
        assert _is_strict_governance() is False
