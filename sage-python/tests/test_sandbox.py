"""Tests for the sandbox system."""
import pytest
from sage.sandbox.manager import SandboxManager, SandboxConfig, SandboxResult


@pytest.mark.asyncio
async def test_create_sandbox():
    manager = SandboxManager(use_docker=False, allow_local=True)
    sandbox = await manager.create()
    assert sandbox.alive
    assert sandbox.id in manager.list_sandboxes()


@pytest.mark.asyncio
async def test_execute_in_sandbox():
    manager = SandboxManager(use_docker=False, allow_local=True)
    sandbox = await manager.create()
    result = await sandbox.execute("echo hello")
    assert isinstance(result, SandboxResult)
    assert "hello" in result.stdout
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_destroy_sandbox():
    manager = SandboxManager(use_docker=False, allow_local=True)
    sandbox = await manager.create()
    sid = sandbox.id
    assert await manager.destroy(sid)
    assert not sandbox.alive
    assert sid not in manager.list_sandboxes()


@pytest.mark.asyncio
async def test_destroy_nonexistent():
    manager = SandboxManager(use_docker=False, allow_local=True)
    assert not await manager.destroy("nonexistent")


@pytest.mark.asyncio
async def test_destroy_all():
    manager = SandboxManager(use_docker=False, allow_local=True)
    await manager.create()
    await manager.create()
    count = await manager.destroy_all()
    assert count == 2
    assert len(manager.list_sandboxes()) == 0


@pytest.mark.asyncio
async def test_sandbox_config_defaults():
    config = SandboxConfig()
    assert config.image == "python:3.13-slim"
    assert config.memory_limit == "512m"
    assert not config.network_enabled


@pytest.mark.asyncio
async def test_execute_after_destroy():
    manager = SandboxManager(use_docker=False, allow_local=True)
    sandbox = await manager.create()
    await sandbox.destroy()
    result = await sandbox.execute("echo test")
    assert result.exit_code == 1
    assert "not alive" in result.stderr
