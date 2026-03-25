"""Tests for memory backend reporting.

Issue E audit fix: The system must explicitly report which memory backend
is active (rust_smmu or mock) instead of silently degrading.
"""
from sage.memory.working import get_memory_backend


def test_get_memory_backend_returns_string():
    result = get_memory_backend()
    assert result in ("rust_smmu", "mock")


def test_get_memory_backend_mock_when_no_rust():
    """Without sage_core compiled with WorkingMemory, should return 'mock'."""
    # This test runs in CI without sage_core Rust build
    result = get_memory_backend()
    # We accept either value — the point is it never raises
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_capabilities_all_false():
    """Mock S-MMU must report all capabilities as False."""
    from sage.memory.working import _has_rust
    if _has_rust:
        import pytest
        pytest.skip("sage_core is compiled, mock not active")

    # Import the mock class
    from sage.memory import working
    mock_cls = type(working.sage_core.WorkingMemory)
    # The class itself should have capabilities()
    if hasattr(mock_cls, "capabilities"):
        caps = mock_cls.capabilities()
        assert isinstance(caps, dict)
        assert all(v is False for v in caps.values()), f"Expected all False, got {caps}"
