import sys
import types
import logging

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
