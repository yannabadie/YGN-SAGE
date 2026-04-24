from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.memory.consolidator import ConsolidationResult, MemoryConsolidator


class SleepingEpisodic:
    def __init__(self, delay: float = 0.1) -> None:
        self.delay = delay
        self.calls = 0
        self.started = asyncio.Event()

    async def list_all(self, limit: int) -> list[dict[str, object]]:
        self.calls += 1
        self.started.set()
        await asyncio.sleep(self.delay)
        return []


class BlockingEpisodic:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def list_all(self, limit: int) -> list[dict[str, object]]:
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return []


class FirstPassBlocksEpisodic:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def list_all(self, limit: int) -> list[dict[str, object]]:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return []


def make_consolidator(episodic: object) -> MemoryConsolidator:
    return MemoryConsolidator(
        episodic=episodic,
        semantic=MagicMock(),
        causal=None,
        memory_agent=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_concurrent_consolidate_calls_share_single_in_flight_pass():
    episodic = SleepingEpisodic(delay=0.1)
    consolidator = make_consolidator(episodic)

    start = time.perf_counter()
    first, second = await asyncio.gather(
        consolidator.consolidate(),
        consolidator.consolidate(),
    )
    elapsed = time.perf_counter() - start

    assert episodic.calls == 1
    assert elapsed >= 0.1
    assert elapsed < 0.2
    assert isinstance(first, ConsolidationResult)
    assert first is second
    assert consolidator.is_running is False
    assert consolidator.last_error is None


@pytest.mark.asyncio
async def test_shutdown_during_in_flight_pass_waits_for_completion():
    episodic = SleepingEpisodic(delay=0.1)
    consolidator = make_consolidator(episodic)

    task = asyncio.create_task(consolidator.consolidate())
    await asyncio.wait_for(episodic.started.wait(), timeout=1.0)

    assert consolidator.is_running is True

    await consolidator.shutdown(timeout=1.0)
    result = await task

    assert result.processed == 0
    assert episodic.calls == 1
    assert task.cancelled() is False
    assert consolidator.is_running is False
    assert consolidator.last_error is None


@pytest.mark.asyncio
async def test_shutdown_timeout_cancels_in_flight_pass():
    episodic = BlockingEpisodic()
    consolidator = make_consolidator(episodic)

    task = asyncio.create_task(consolidator.consolidate())
    await asyncio.wait_for(episodic.started.wait(), timeout=1.0)

    await consolidator.shutdown(timeout=0.01)

    await asyncio.wait_for(episodic.cancelled.wait(), timeout=1.0)
    with suppress(asyncio.CancelledError):
        await task

    assert task.cancelled() is True
    assert consolidator.is_running is False
    assert isinstance(consolidator.last_error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_consolidator_is_reusable_after_shutdown():
    episodic = FirstPassBlocksEpisodic()
    consolidator = make_consolidator(episodic)

    task = asyncio.create_task(consolidator.consolidate())
    await asyncio.wait_for(episodic.started.wait(), timeout=1.0)
    await consolidator.shutdown(timeout=0.01)

    await asyncio.wait_for(episodic.cancelled.wait(), timeout=1.0)
    with suppress(asyncio.CancelledError):
        await task

    result = await consolidator.consolidate()

    assert result.processed == 0
    assert episodic.calls == 2
    assert consolidator.is_running is False
    assert consolidator.last_error is None
