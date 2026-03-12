"""Test that boot emits WARNING when no sandbox is available."""
import logging
import pytest


def test_sandbox_unavailable_warning(caplog):
    """When sandbox is unavailable, boot should emit a WARNING."""
    from sage.boot import _check_sandbox_availability

    with caplog.at_level(logging.WARNING):
        available = _check_sandbox_availability()

    if not available:
        assert any("sandbox" in r.message.lower() for r in caplog.records), (
            "Expected a WARNING about sandbox unavailability"
        )
