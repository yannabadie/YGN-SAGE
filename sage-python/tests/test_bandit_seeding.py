"""Test that bandit arms are seeded from registry, not hardcoded.

After the system.run() simplification, bandit seeding is no longer done
inline in system.run(). It is now handled by the pipeline's Stage 4
(model assignment). system.run() is a clean 3-path dispatcher:
mock bypass, pipeline, or direct fallback.

This test verifies:
1. No hardcoded model lists in system.run()
2. system.run() is clean (no inline bandit/routing logic)
"""
import pytest
from unittest.mock import MagicMock


def test_bandit_not_hardcoded_in_run():
    """system.run() should not contain hardcoded bandit model lists."""
    import inspect
    import sage.boot as boot_module
    source = inspect.getsource(boot_module.AgentSystem.run)

    # The old hardcoded 4-model bandit seeding list must not be present
    assert '"gemini-3.1-pro-preview", "gemini-2.5-flash-lite"' not in source, (
        "Hardcoded Gemini model list should not appear in run()"
    )
    # Bandit seeding logic should not be inline in run() — it's in the pipeline
    assert "register_arm" not in source, (
        "Bandit arm registration should be in pipeline, not in run()"
    )


def test_run_is_clean_dispatcher():
    """system.run() should be a clean 3-path dispatcher with no inline logic."""
    import inspect
    import sage.boot as boot_module
    source = inspect.getsource(boot_module.AgentSystem.run)

    # run() delegates to pipeline or agent_loop — no inline routing
    assert "pipeline" in source.lower(), "run() should delegate to pipeline"
    assert "agent_loop" in source.lower(), "run() should have agent_loop fallback"
    # No legacy routing artifacts
    assert "ModelRouter" not in source, "No ModelRouter in simplified run()"
    assert "complexity" not in source.lower(), "No inline complexity routing in run()"
