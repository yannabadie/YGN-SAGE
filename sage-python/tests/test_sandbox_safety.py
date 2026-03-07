"""Tests for sandbox safety defaults -- audit response Task A1."""
import pytest
from sage.sandbox.manager import SandboxManager


def test_local_execution_blocked_by_default():
    """Local execution should be disabled by default."""
    mgr = SandboxManager()
    assert mgr._allow_local is False


@pytest.mark.asyncio
async def test_local_sandbox_refuses_without_opt_in():
    """Sandbox.execute should refuse local execution without explicit opt-in."""
    mgr = SandboxManager()  # allow_local=False by default
    sandbox = await mgr.create()
    result = await sandbox.execute("echo hello")
    assert result.exit_code != 0
    assert "disabled" in result.stderr.lower() or "not allowed" in result.stderr.lower()


@pytest.mark.asyncio
async def test_local_sandbox_works_with_opt_in():
    """Sandbox.execute should work when allow_local=True."""
    mgr = SandboxManager(allow_local=True)
    sandbox = await mgr.create()
    result = await sandbox.execute("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_boot_sandbox_has_local_disabled():
    """Boot sequence should create SandboxManager with local execution disabled."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.sandbox_manager._allow_local is False
