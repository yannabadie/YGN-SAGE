import sys
import types
import logging
import time
from unittest.mock import patch

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.resilience import CircuitBreaker


def test_circuit_breaker_allows_calls_initially():
    cb = CircuitBreaker("test", max_failures=3)
    assert cb.is_closed()
    assert not cb.is_open()


def test_circuit_breaker_opens_after_max_failures():
    cb = CircuitBreaker("test", max_failures=2)
    cb.record_failure(ValueError("fail 1"))
    assert cb.is_closed()
    cb.record_failure(ValueError("fail 2"))
    assert cb.is_open()


def test_circuit_breaker_skips_when_open(caplog):
    cb = CircuitBreaker("test", max_failures=1)
    cb.record_failure(ValueError("boom"))
    assert cb.is_open()
    with caplog.at_level(logging.WARNING):
        skipped = cb.should_skip()
    assert skipped is True
    assert any("circuit open" in r.message.lower() for r in caplog.records)


def test_circuit_breaker_records_success_resets():
    cb = CircuitBreaker("test", max_failures=2)
    cb.record_failure(ValueError("fail"))
    assert cb.failure_count == 1
    cb.record_success()
    assert cb.failure_count == 0


# ── Half-open recovery tests (Audit4 §5.2) ──────────────────────


def test_half_open_after_cooldown():
    """After cooldown elapses, circuit transitions to half-open and allows probe."""
    cb = CircuitBreaker("test", max_failures=1, cooldown_s=0.1)
    cb.record_failure(ValueError("fail"))
    assert cb.is_open()
    assert cb.should_skip() is True

    time.sleep(0.15)
    # After cooldown, should_skip returns False (half-open probe)
    assert cb.should_skip() is False


def test_half_open_probe_success_closes():
    """Successful probe in half-open state closes the circuit."""
    cb = CircuitBreaker("test", max_failures=1, cooldown_s=0.05)
    cb.record_failure(ValueError("fail"))
    time.sleep(0.1)
    assert cb.should_skip() is False  # half-open

    cb.record_success()
    assert cb.is_closed()
    assert cb.failure_count == 0
    assert cb._opened_at is None


def test_half_open_probe_failure_reopens():
    """Failed probe in half-open state re-opens the circuit."""
    cb = CircuitBreaker("test", max_failures=1, cooldown_s=0.05)
    cb.record_failure(ValueError("fail 1"))
    time.sleep(0.1)
    assert cb.should_skip() is False  # half-open

    cb.record_failure(ValueError("fail 2"))
    assert cb.is_open()
    # _opened_at should be refreshed
    assert cb._opened_at is not None


def test_cooldown_default_60s():
    """Default cooldown is 60 seconds."""
    cb = CircuitBreaker("test")
    assert cb.cooldown_s == 60.0
