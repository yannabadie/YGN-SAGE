"""Lightweight circuit breaker for best-effort subsystems.

Replaces silent except:pass patterns with observable failure tracking.
Three states: CLOSED (normal) -> OPEN (skip calls) -> HALF_OPEN (probe one call).
After max_failures consecutive errors, the breaker opens. After cooldown_s
seconds, it transitions to half-open and allows one probe call. If the probe
succeeds, the circuit closes; if it fails, it re-opens.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Per-subsystem circuit breaker with half-open recovery.

    Parameters
    ----------
    name:
        Human-readable name for log messages (e.g. "semantic_memory").
    max_failures:
        Number of consecutive failures before the circuit opens.
    cooldown_s:
        Seconds to wait before transitioning OPEN -> HALF_OPEN.
    """

    def __init__(self, name: str, max_failures: int = 3, cooldown_s: float = 60.0):
        self.name = name
        self.max_failures = max_failures
        self.cooldown_s = cooldown_s
        self.failure_count = 0
        self._opened_at: float | None = None

    def is_closed(self) -> bool:
        return self.failure_count < self.max_failures

    def is_open(self) -> bool:
        return not self.is_closed() and not self._is_half_open()

    def _is_half_open(self) -> bool:
        if self.failure_count < self.max_failures:
            return False
        if self._opened_at is None:
            return False
        return (time.monotonic() - self._opened_at) >= self.cooldown_s

    def record_failure(self, error: Exception) -> None:
        was_half_open = self._is_half_open()
        self.failure_count += 1
        if was_half_open or self.failure_count == self.max_failures:
            # (Re-)open the circuit — reset cooldown timer
            self._opened_at = time.monotonic()
            if was_half_open:
                log.warning(
                    "Circuit RE-OPENED for %s — half-open probe failed: %s. "
                    "Will retry in %.0fs.",
                    self.name, error, self.cooldown_s,
                )
            else:
                log.warning(
                    "Circuit OPEN for %s after %d failures (last: %s). "
                    "Will probe in %.0fs.",
                    self.name, self.max_failures, error, self.cooldown_s,
                )
        elif self.failure_count < self.max_failures:
            log.debug("Failure %d/%d in %s: %s", self.failure_count, self.max_failures, self.name, error)

    def record_success(self) -> None:
        if self.failure_count > 0:
            if self._is_half_open():
                log.info("Circuit CLOSED for %s — half-open probe succeeded", self.name)
            self.failure_count = 0
            self._opened_at = None

    def should_skip(self) -> bool:
        """Check if the circuit is open (should skip the call).

        Returns False if CLOSED or HALF_OPEN (allows probe).
        Returns True only if OPEN and cooldown has not elapsed.
        """
        if self.is_closed():
            return False
        if self._is_half_open():
            log.info("Circuit HALF_OPEN for %s — allowing probe call", self.name)
            return False
        log.warning("Circuit OPEN for %s — skipping call", self.name)
        return True
