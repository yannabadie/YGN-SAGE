"""Test SnapBPF via sage_core (or mock if Rust not compiled)."""
import pytest


def test_snapbpf_available_as_class():
    """SnapBPF should be importable from sage_core."""
    try:
        import sage_core
    except ImportError:
        pytest.skip("sage_core not compiled — SnapBPF test skipped")

    if not hasattr(sage_core, "SnapBPF"):
        pytest.skip("sage_core compiled without SnapBPF — rebuild needed")

    snap = sage_core.SnapBPF()
    snap.snapshot("test", [1, 2, 3])
    restored = snap.restore("test")
    assert restored == [1, 2, 3]
