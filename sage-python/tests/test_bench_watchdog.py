"""Tests for sage.bench.watchdog.

Three scenarios:

1. Normal completion: returns WatchdogResult with elapsed_wall_s.
2. Slow coro hits asyncio TimeoutError: re-raised as-is (not
   HostSuspendDetected — that's reserved for clock skew).
3. Simulated host suspend: ``time.time()`` jumps forward mid-await.
   The watchdog detects elapsed_wall > timeout * grace and raises.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from sage.bench.watchdog import (
    HostSuspendDetected,
    WatchdogResult,
    run_with_wallclock_watchdog,
)


@pytest.mark.asyncio
async def test_normal_completion_returns_result() -> None:
    async def fast() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    res = await run_with_wallclock_watchdog(fast(), timeout_s=1.0)
    assert isinstance(res, WatchdogResult)
    assert res.value == "ok"
    assert 0 <= res.elapsed_wall_s < 1.0


@pytest.mark.asyncio
async def test_asyncio_timeout_propagates_without_suspend_diagnosis() -> None:
    """When the loop IS ticking and coro is just slow, we get the normal asyncio.TimeoutError.

    HostSuspendDetected is reserved for the case where wait_for returns
    "successfully" but the wall clock shows suspend. The slow-task case
    must NOT be misdiagnosed as suspend.
    """
    async def slow() -> str:
        await asyncio.sleep(2.0)
        return "should-not-return"

    with pytest.raises(asyncio.TimeoutError):
        await run_with_wallclock_watchdog(slow(), timeout_s=0.05)


@pytest.mark.asyncio
async def test_simulated_suspend_raises_host_suspend_detected() -> None:
    """Simulate Modern Standby: wall clock jumps forward by hours during the await.

    ``time.time()`` jumps because the OS is awake when we call it.
    ``asyncio.wait_for`` saw only loop-time tick because the loop was
    suspended along with the process. So wait_for returns OK but
    elapsed_wall is huge.

    We patch ``time.time`` in the watchdog module to fake the jump
    after the await completes.
    """
    async def quick() -> str:
        await asyncio.sleep(0.01)
        return "result"

    real_time = time.time
    fake_t0 = real_time()
    # Jump forward by 5h 38min — the actual A3 BCB/273 elapsed.
    fake_t1 = fake_t0 + 20278.0
    sequence = iter([fake_t0, fake_t1])

    def fake_time() -> float:
        try:
            return next(sequence)
        except StopIteration:
            return fake_t1

    with patch("sage.bench.watchdog.time.time", side_effect=fake_time):
        with pytest.raises(HostSuspendDetected) as exc:
            await run_with_wallclock_watchdog(quick(), timeout_s=120.0)

    assert exc.value.elapsed_wall_s == pytest.approx(20278.0, rel=1e-3)
    assert exc.value.timeout_s == 120.0
    assert exc.value.grace_factor == 2.0
    msg = str(exc.value)
    assert "host_suspend_detected" in msg
    assert "20278" in msg


@pytest.mark.asyncio
async def test_grace_factor_threshold() -> None:
    """elapsed_wall just under timeout * grace must NOT raise; just above must raise."""
    async def quick() -> str:
        return "ok"

    real_time = time.time
    t0 = real_time()

    # Case 1: elapsed = 1.9 * timeout (under grace=2.0)
    seq1 = iter([t0, t0 + 19.0])

    def fake1() -> float:
        return next(seq1, t0 + 19.0)

    with patch("sage.bench.watchdog.time.time", side_effect=fake1):
        res = await run_with_wallclock_watchdog(quick(), timeout_s=10.0, grace_factor=2.0)
        assert res.value == "ok"
        assert res.elapsed_wall_s == pytest.approx(19.0)

    # Case 2: elapsed = 2.5 * timeout (over grace=2.0)
    seq2 = iter([t0, t0 + 25.0])

    def fake2() -> float:
        return next(seq2, t0 + 25.0)

    with patch("sage.bench.watchdog.time.time", side_effect=fake2):
        with pytest.raises(HostSuspendDetected):
            await run_with_wallclock_watchdog(quick(), timeout_s=10.0, grace_factor=2.0)


@pytest.mark.asyncio
async def test_invalid_timeout_rejected() -> None:
    async def noop() -> None:
        return None

    # Pre-create + close coroutines to avoid "coroutine was never awaited"
    # warnings on the rejection paths.
    c0 = noop()
    c0.close()
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        await run_with_wallclock_watchdog(c0, timeout_s=0)
    c1 = noop()
    c1.close()
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        await run_with_wallclock_watchdog(c1, timeout_s=-5)


@pytest.mark.asyncio
async def test_invalid_grace_factor_rejected() -> None:
    async def noop() -> None:
        return None

    c = noop()
    c.close()
    with pytest.raises(ValueError, match="grace_factor must be >= 1.0"):
        await run_with_wallclock_watchdog(c, timeout_s=10, grace_factor=0.5)


@pytest.mark.asyncio
async def test_real_time_no_suspend_passes_through() -> None:
    """Sanity: under real time.time() (no patch), a normal coro completes cleanly.

    Guards against breaking the watchdog by accidentally always
    raising HostSuspendDetected.
    """
    async def fast() -> int:
        await asyncio.sleep(0.001)
        return 42

    res = await run_with_wallclock_watchdog(fast(), timeout_s=5.0, grace_factor=2.0)
    assert res.value == 42
    assert res.elapsed_wall_s < 5.0
