"""Tests for sage.bench.keep_awake.

Mocks ``_set_thread_execution_state`` so tests run identically on
Linux CI runners and on the dev Windows host. Verifies:

1. On Windows: SetThreadExecutionState is called with ES_CONTINUOUS |
   ES_SYSTEM_REQUIRED on entry, and ES_CONTINUOUS alone on exit.
2. On non-Windows: the context yields False and never calls the
   ctypes layer.
3. Failure (None return from kernel call) yields False but does not
   raise — bench should still run, just without the keep-awake
   guarantee.
"""

from __future__ import annotations

from unittest.mock import patch

from sage.bench.keep_awake import (
    _ES_CONTINUOUS,
    _ES_SYSTEM_REQUIRED,
    prevent_os_sleep,
)


def test_noop_on_non_windows() -> None:
    """On Linux/macOS the context manager is a clean no-op."""
    with patch("sage.bench.keep_awake.platform.system", return_value="Linux"):
        with patch("sage.bench.keep_awake._set_thread_execution_state") as mock_set:
            with prevent_os_sleep() as issued:
                assert issued is False
            mock_set.assert_not_called()


def test_macos_is_noop() -> None:
    """Same for macOS."""
    with patch("sage.bench.keep_awake.platform.system", return_value="Darwin"):
        with patch("sage.bench.keep_awake._set_thread_execution_state") as mock_set:
            with prevent_os_sleep() as issued:
                assert issued is False
            mock_set.assert_not_called()


def test_windows_issues_keep_awake_then_clears() -> None:
    """On Windows: ES_SYSTEM_REQUIRED set on entry, cleared on exit."""
    expected_set_flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    expected_clear_flags = _ES_CONTINUOUS
    calls: list[int] = []

    def fake(flags: int) -> int:
        calls.append(flags)
        return 0x80000000  # Pretend prev state was just ES_CONTINUOUS

    with patch("sage.bench.keep_awake.platform.system", return_value="Windows"):
        with patch("sage.bench.keep_awake._set_thread_execution_state", side_effect=fake):
            with prevent_os_sleep() as issued:
                assert issued is True
            # Exit clears
    assert calls == [expected_set_flags, expected_clear_flags]


def test_windows_kernel_failure_yields_false() -> None:
    """If kernel32 returns None (ctypes import error or AttributeError), no crash."""
    with patch("sage.bench.keep_awake.platform.system", return_value="Windows"):
        with patch("sage.bench.keep_awake._set_thread_execution_state", return_value=None):
            with prevent_os_sleep() as issued:
                assert issued is False
            # Should not have raised


def test_does_not_use_away_mode_required() -> None:
    """Microsoft docs explicitly say ES_AWAYMODE_REQUIRED is inappropriate on laptops.

    Verify the module never sends it. The constant 0x00000040 corresponds
    to ES_AWAYMODE_REQUIRED.
    """
    away_mode_required = 0x00000040
    seen_flags: list[int] = []

    def fake(flags: int) -> int:
        seen_flags.append(flags)
        return 0x80000000

    with patch("sage.bench.keep_awake.platform.system", return_value="Windows"):
        with patch("sage.bench.keep_awake._set_thread_execution_state", side_effect=fake):
            with prevent_os_sleep():
                pass

    for f in seen_flags:
        assert (f & away_mode_required) == 0, (
            f"keep_awake set ES_AWAYMODE_REQUIRED bit (0x40), flags=0x{f:x}"
        )


def test_inner_exception_still_clears_keep_awake() -> None:
    """If the bench body raises, we must still call the clear path on the way out."""
    calls: list[int] = []

    def fake(flags: int) -> int:
        calls.append(flags)
        return 0

    with patch("sage.bench.keep_awake.platform.system", return_value="Windows"):
        with patch("sage.bench.keep_awake._set_thread_execution_state", side_effect=fake):
            try:
                with prevent_os_sleep():
                    raise RuntimeError("synthetic")
            except RuntimeError:
                pass

    # Two calls: set + clear
    assert len(calls) == 2
    assert calls[0] == _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    assert calls[1] == _ES_CONTINUOUS
