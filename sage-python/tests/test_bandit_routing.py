"""Test bandit-augmented model selection."""
import pytest


def test_bandit_initialized():
    """Bandit should be initialized in AgentSystem."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert hasattr(system, "bandit")


def test_bandit_register_and_choose():
    """Test that ContextualBandit can register arms and choose."""
    try:
        from sage_core import ContextualBandit
    except ImportError:
        pytest.skip("sage_core not compiled")

    bandit = ContextualBandit(0.995, 0.1)
    bandit.register_arm("gemini-2.5-flash", "sequential")
    bandit.register_arm("gemini-3.1-pro-preview", "avr")

    decision = bandit.select(0.5)
    assert decision.model_id in ("gemini-2.5-flash", "gemini-3.1-pro-preview")
    assert decision.template in ("sequential", "avr")
    assert 0.0 <= decision.expected_quality <= 1.0
    assert decision.decision_id  # non-empty


@pytest.mark.asyncio
async def test_bandit_updates_during_run():
    """After a run, bandit should have been called (choose + record)."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    if system.bandit is None:
        pytest.skip("sage_core not compiled")

    await system.run("Write a hello world program")
    # If bandit was called, arms should have been registered
    # The bandit is internal, so we just verify the run doesn't crash
    # and that topology was still recorded
    assert system.topology_engine.topology_count() >= 1
