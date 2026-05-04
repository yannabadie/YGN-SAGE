"""OS-level keep-awake for long-running benches.

Cycle-9 recovery Step 4 (cgpro 2026-05-04). On Windows, ``powercfg``
plan timeouts (``standby-timeout-ac=0``) are NOT sufficient to prevent
process suspend under Modern Standby (S0 DRIPS). When the lid is
closed or the screen is off, the OS can suspend background processes
regardless of plan timeouts.

The reliable in-process blocker is ``SetThreadExecutionState`` with
``ES_CONTINUOUS | ES_SYSTEM_REQUIRED``, which Microsoft documents as
keeping the system in the working state until the calling thread
clears it (or exits).

Caveats acknowledged by cgpro:

- This does NOT block manual sleep, lid-close-forced sleep, critical
  battery, group policy, or update-driven restarts.
- ``ES_AWAYMODE_REQUIRED`` is explicitly documented as inappropriate
  on portable computers — DO NOT add it.
- Cloud VM is the gate-quality answer; this module is for
  diagnostic-grade local runs only.

Non-Windows: no-op context manager. The module imports cleanly on
Linux/macOS so callers can use ``with prevent_os_sleep():`` without
``platform.system()`` checks at the call site.
"""

from __future__ import annotations

import contextlib
import logging
import platform
from collections.abc import Iterator

__all__ = ["prevent_os_sleep"]

_log = logging.getLogger("sage.bench.keep_awake")

# Windows ``EXECUTION_STATE`` flags. Documented at
# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_thread_execution_state(flags: int) -> int | None:
    """Call SetThreadExecutionState. Returns previous flags on success, None on failure.

    Isolated for mockability — tests patch this symbol rather than
    ``ctypes.windll`` directly.
    """
    try:
        import ctypes  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return int(ctypes.windll.kernel32.SetThreadExecutionState(flags))  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:
        _log.warning("SetThreadExecutionState(0x%x) failed: %s", flags, exc)
        return None


@contextlib.contextmanager
def prevent_os_sleep(*, force: bool = False) -> Iterator[bool]:
    """Context manager that asks the OS to keep the system awake.

    Yields ``True`` if the keep-awake request was actually issued,
    ``False`` if no-op (non-Windows or ctypes failure). Callers can use
    the yielded value to decide whether to additionally warn the user
    (e.g. "keep-awake unavailable, do not leave laptop unattended").

    Args:
        force: ignored on Windows (always issues request); on non-
            Windows, has no effect (we never have an OS-level blocker
            to issue elsewhere). Reserved for future Linux/systemd
            ``inhibit`` integration.
    """
    is_windows = platform.system() == "Windows"
    issued = False
    if is_windows:
        prev = _set_thread_execution_state(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED,
        )
        issued = prev is not None
        if issued:
            _log.info(
                "prevent_os_sleep: SetThreadExecutionState(ES_CONTINUOUS | "
                "ES_SYSTEM_REQUIRED) issued (prev=0x%x)",
                prev or 0,
            )
        else:
            _log.warning(
                "prevent_os_sleep: SetThreadExecutionState failed; "
                "process may be suspended by Modern Standby DRIPS",
            )
    else:
        _log.info(
            "prevent_os_sleep: no-op on platform=%s (only Windows supported)",
            platform.system(),
        )
    try:
        yield issued
    finally:
        if is_windows and issued:
            # Clear by issuing ES_CONTINUOUS alone — Microsoft docs say
            # this resets the requirement set by previous calls.
            _set_thread_execution_state(_ES_CONTINUOUS)
            _log.info("prevent_os_sleep: cleared ES_SYSTEM_REQUIRED")
