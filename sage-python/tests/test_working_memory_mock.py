"""Tests for _PyWorkingMemory mock S-MMU warning logging.

The _PyWorkingMemory class only exists when sage_core is NOT compiled
(it's defined inside an `if not _has_rust:` block at module level).
All tests are skipped when sage_core provides a real WorkingMemory.
"""
from __future__ import annotations

import logging

import pytest

import sage.memory.working as wm

pytestmark = pytest.mark.skipif(
    not hasattr(wm, "_PyWorkingMemory"),
    reason="Mock class only exists when sage_core is not compiled",
)

MOCK_LOGGER_NAME = "sage.memory.working"

# S-MMU methods that should produce a warning on first call.
SMMU_METHODS_WITH_WARNINGS = [
    "compact_to_arrow",
    "compact_to_arrow_with_meta",
    "retrieve_relevant_chunks",
    "get_page_out_candidates",
    "get_chunk_summary",
    "get_latest_arrow_chunk",
]


@pytest.fixture(autouse=True)
def _reset_mock_warned():
    """Clear the warn-once set before each test so warnings re-fire."""
    if hasattr(wm, "_PyWorkingMemory"):
        wm._PyWorkingMemory._mock_warned.clear()
    yield


def _call_smmu_method(mem: wm._PyWorkingMemory, method: str):
    """Invoke an S-MMU method with appropriate dummy arguments."""
    if method == "compact_to_arrow":
        return mem.compact_to_arrow()
    elif method == "compact_to_arrow_with_meta":
        return mem.compact_to_arrow_with_meta(["kw"])
    elif method == "retrieve_relevant_chunks":
        return mem.retrieve_relevant_chunks(0, 1)
    elif method == "get_page_out_candidates":
        return mem.get_page_out_candidates(0, 1, 5)
    elif method == "get_chunk_summary":
        return mem.get_chunk_summary(0)
    elif method == "get_latest_arrow_chunk":
        return mem.get_latest_arrow_chunk()
    else:
        raise ValueError(f"Unknown method: {method}")


@pytest.mark.parametrize("method", SMMU_METHODS_WITH_WARNINGS)
def test_smmu_mock_warns_on_first_call(method: str, caplog):
    """Each S-MMU mock method emits a WARNING the first time it is called."""
    mem = wm._PyWorkingMemory("test-agent")
    with caplog.at_level(logging.WARNING, logger=MOCK_LOGGER_NAME):
        _call_smmu_method(mem, method)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"Expected exactly 1 warning for {method}, got {len(warnings)}"
    assert method in warnings[0].message
    assert "sage_core not available" in warnings[0].message


@pytest.mark.parametrize("method", SMMU_METHODS_WITH_WARNINGS)
def test_smmu_mock_warns_only_once(method: str, caplog):
    """Repeated calls to the same mock method produce only one warning."""
    mem = wm._PyWorkingMemory("test-agent")
    with caplog.at_level(logging.WARNING, logger=MOCK_LOGGER_NAME):
        _call_smmu_method(mem, method)
        _call_smmu_method(mem, method)
        _call_smmu_method(mem, method)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"Expected 1 warning for {method} after 3 calls, got {len(warnings)}"


def test_smmu_chunk_count_no_warning(caplog):
    """smmu_chunk_count() is a simple counter and should NOT warn."""
    mem = wm._PyWorkingMemory("test-agent")
    with caplog.at_level(logging.WARNING, logger=MOCK_LOGGER_NAME):
        result = mem.smmu_chunk_count()

    assert result == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 0, "smmu_chunk_count should not emit a warning"


def test_all_smmu_methods_warn_independently(caplog):
    """Calling all S-MMU methods produces one warning per method."""
    mem = wm._PyWorkingMemory("test-agent")
    with caplog.at_level(logging.WARNING, logger=MOCK_LOGGER_NAME):
        for method in SMMU_METHODS_WITH_WARNINGS:
            _call_smmu_method(mem, method)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == len(SMMU_METHODS_WITH_WARNINGS)
    warned_methods = {r.message.split("()")[0] for r in warnings}
    assert warned_methods == set(SMMU_METHODS_WITH_WARNINGS)


def test_mock_return_values():
    """Mock S-MMU methods return the expected default values."""
    mem = wm._PyWorkingMemory("test-agent")
    assert mem.compact_to_arrow() == 0
    assert mem.compact_to_arrow_with_meta(["kw"]) == 0
    assert mem.retrieve_relevant_chunks(0, 1) == []
    assert mem.get_page_out_candidates(0, 1, 5) == []
    assert mem.smmu_chunk_count() == 0
    assert mem.get_chunk_summary(0) == ""
    assert mem.get_latest_arrow_chunk() is None


def test_warn_once_shared_across_instances(caplog):
    """The warn-once set is class-level, so two instances share it."""
    mem1 = wm._PyWorkingMemory("agent-1")
    mem2 = wm._PyWorkingMemory("agent-2")
    with caplog.at_level(logging.WARNING, logger=MOCK_LOGGER_NAME):
        mem1.compact_to_arrow()
        mem2.compact_to_arrow()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "Class-level _mock_warned should prevent duplicate warnings"
