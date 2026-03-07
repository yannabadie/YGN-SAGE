"""Lightweight circuit breaker for best-effort subsystems.

Replaces silent except:pass patterns with observable failure tracking.
After max_failures consecutive errors, the breaker opens and skips
calls until record_success() resets it.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Per-subsystem circuit breaker.

    Parameters
    ----------
    name:
        Human-readable name for log messages (e.g. "semantic_memory").
    max_failures:
        Number of consecutive failures before the circuit opens.
    """

    def __init__(self, name: str, max_failures: int = 3):
        self.name = name
        self.max_failures = max_failures
        self.failure_count = 0

    def is_closed(self) -> bool:
        return self.failure_count < self.max_failures

    def is_open(self) -> bool:
        return not self.is_closed()

    def record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        if self.failure_count == self.max_failures:
            log.warning(
                "Circuit OPEN for %s after %d failures (last: %s)",
                self.name, self.max_failures, error,
            )
        elif self.failure_count < self.max_failures:
            log.debug("Failure %d/%d in %s: %s", self.failure_count, self.max_failures, self.name, error)

    def record_success(self) -> None:
        if self.failure_count > 0:
            self.failure_count = 0

    def should_skip(self) -> bool:
        """Check if the circuit is open (should skip the call)."""
        if self.is_open():
            log.warning("Circuit open for %s — skipping call", self.name)
            return True
        return False
