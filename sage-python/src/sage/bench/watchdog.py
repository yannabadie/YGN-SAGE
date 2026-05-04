"""Wall-clock watchdog for long-running asyncio benches.

Companion to ``event_ledger.py``. Solves the failure mode that killed
A3 N=50: ``asyncio.wait_for(coro, timeout=120)`` does NOT enforce a
120s wall-clock cap when the asyncio event loop is itself suspended
(Windows Modern Standby S0 DRIPS, container freeze, OS scheduler
blackout). The internal timer counts loop ticks, not wall-clock; on
resume, the await sees buffered IO and returns "successfully" with
``elapsed_wall_ms`` of several hours.

This module wraps ``wait_for`` with a wall-clock check using
``time.time()`` (which advances during suspend on every platform we
target). If the wall elapsed exceeds ``timeout_s * grace_factor``
*before* the wait_for returns by detecting suspend at the Python
level, we raise ``HostSuspendDetected``. The bench callers are then
expected to:

1. Emit a ``TASK_ABORT`` event with ``reason="host_suspend_detected"``.
2. Exclude the task from pass/fail aggregation.
3. Mark the entire run as non-gate-quality unless rerun cleanly.

This is the runtime-side enforcement of the proposed Directive #9
"Timeout Enforcement Invariant" (see CLAUDE.md correction 2026-05-04).

Why not ``asyncio.timeout()`` (3.11+)? Same root cause: it also relies
on ``loop.call_later`` and is suspended along with the loop. The wall
check has to be done with a syscall the OS keeps incrementing during
suspend. ``time.time()`` (gettimeofday / GetSystemTime) does;
``loop.time()`` (monotonic) does NOT — that is part of why the
existing wait_for is broken under DRIPS.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable

__all__ = [
    "HostSuspendDetected",
    "WatchdogResult",
    "run_with_wallclock_watchdog",
]


class HostSuspendDetected(RuntimeError):
    """Raised when wall-clock elapsed exceeds ``timeout_s * grace_factor``.

    Carries ``elapsed_wall_s`` and ``timeout_s`` for ledger emission.
    """

    def __init__(self, elapsed_wall_s: float, timeout_s: float, grace_factor: float):
        self.elapsed_wall_s = elapsed_wall_s
        self.timeout_s = timeout_s
        self.grace_factor = grace_factor
        super().__init__(
            f"host_suspend_detected: elapsed_wall={elapsed_wall_s:.1f}s "
            f"> timeout={timeout_s:.1f}s * grace={grace_factor:.1f}",
        )


@dataclass
class WatchdogResult:
    """Result of ``run_with_wallclock_watchdog`` on the success path.

    ``elapsed_wall_s`` is the wall-clock duration including any OS
    suspend. Always populated, even when ``host_suspend_detected`` is
    True (in which case ``HostSuspendDetected`` was raised — see
    docstring of ``run_with_wallclock_watchdog`` for return semantics).
    """

    value: Any
    elapsed_wall_s: float


async def run_with_wallclock_watchdog(
    coro: Awaitable[Any],
    *,
    timeout_s: float,
    grace_factor: float = 2.0,
) -> WatchdogResult:
    """Run ``coro`` under ``asyncio.wait_for`` with a wall-clock backup check.

    Behaviour:

    - Normal completion within ``timeout_s``: returns ``WatchdogResult``
      with ``value`` and ``elapsed_wall_s``.
    - asyncio timeout fires (loop was ticking, coro slow): the
      ``asyncio.TimeoutError`` propagates. Caller should treat as a
      regular TIMEOUT.
    - Loop suspended past ``timeout_s * grace_factor``: ``HostSuspendDetected``
      raised. Caller MUST emit ``TASK_ABORT`` and exclude from stats.

    The watchdog does NOT cancel the suspended task — once Python is
    running again, the await may have already completed. We only
    surface the diagnosis. Cancellation under suspend is
    physically impossible.

    Note: ``grace_factor`` defaults to 2.0 (cgpro recommendation: tasks
    occasionally exceed timeout slightly due to cleanup; 2× is a clean
    "this is suspend, not slowness" threshold). Lower it to 1.2 for
    tight diagnostic runs; raise it to 3.0 for environments with known
    occasional GIL spikes.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    if grace_factor < 1.0:
        raise ValueError(f"grace_factor must be >= 1.0, got {grace_factor}")

    t0 = time.time()
    try:
        value = await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        # Normal asyncio timeout. Re-raise without wallclock check —
        # the caller treats as TIMEOUT, not as suspend. We only
        # diagnose suspend on the *success* path because that's the
        # pathology the wait_for missed.
        raise

    elapsed_wall_s = time.time() - t0
    if elapsed_wall_s > timeout_s * grace_factor:
        raise HostSuspendDetected(
            elapsed_wall_s=elapsed_wall_s,
            timeout_s=timeout_s,
            grace_factor=grace_factor,
        )
    return WatchdogResult(value=value, elapsed_wall_s=elapsed_wall_s)
