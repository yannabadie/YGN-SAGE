"""Tests that silent degradation is replaced with explicit warnings."""
import logging

import pytest

from sage.boot import boot_agent_system


def test_boot_warns_when_rust_unavailable(caplog):
    """Boot should emit WARNING when sage_core is a mock."""
    with caplog.at_level(logging.WARNING):
        system = boot_agent_system(use_mock_llm=True)
    # Should have a warning about Rust extension
    rust_warnings = [
        r
        for r in caplog.records
        if "sage_core" in r.message.lower() or "rust" in r.message.lower()
    ]
    assert len(rust_warnings) >= 1, "No warning emitted about missing Rust extension"


def test_boot_warns_episodic_volatile(caplog):
    """Boot should warn that episodic memory is volatile (no persistence)."""
    with caplog.at_level(logging.WARNING):
        system = boot_agent_system(use_mock_llm=True)
    ep_warnings = [
        r
        for r in caplog.records
        if "episodic" in r.message.lower()
        and ("volatile" in r.message.lower() or "in-memory" in r.message.lower())
    ]
    assert len(ep_warnings) >= 1, "No warning about volatile episodic memory"
