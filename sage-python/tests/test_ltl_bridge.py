"""Test LTL temporal verification bridge."""
from __future__ import annotations

import pytest

from sage.topology.ltl_bridge import verify_topology_ltl, check_reachability, _HAS_LTL


# ---------------------------------------------------------------------------
# Basic import / interface tests (always run)
# ---------------------------------------------------------------------------

def test_ltl_bridge_import():
    """LTL bridge module should be importable."""
    assert callable(verify_topology_ltl)
    assert callable(check_reachability)


def test_ltl_bridge_none_topology_returns_defaults():
    """verify_topology_ltl with None topology returns all-True defaults."""
    result = verify_topology_ltl(None)
    assert isinstance(result, dict)
    assert result["reachable"] is True
    assert result["safe"] is True
    assert result["live"] is True
    assert result["bounded_live"] is True
    assert result["warnings"] == []
    assert result["errors"] == []


def test_ltl_bridge_has_expected_keys():
    """Result dict has all expected keys."""
    result = verify_topology_ltl(None)
    expected_keys = {"reachable", "safe", "live", "bounded_live", "warnings", "errors"}
    assert set(result.keys()) == expected_keys


def test_ltl_bridge_none_returns_fresh_lists():
    """Each call returns independent list objects."""
    r1 = verify_topology_ltl(None)
    r2 = verify_topology_ltl(None)
    assert r1["warnings"] is not r2["warnings"]
    assert r1["errors"] is not r2["errors"]


def test_ltl_bridge_custom_max_depth():
    """max_depth parameter is accepted without error."""
    result = verify_topology_ltl(None, max_depth=100)
    assert result["bounded_live"] is True


# ---------------------------------------------------------------------------
# Tests that require sage_core (Rust LtlVerifier)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LTL, reason="sage_core not available")
class TestWithRustLtl:
    """Tests that exercise the real Rust LtlVerifier."""

    def _make_sequential(self):
        """Build a simple A->B->C topology via sage_core templates."""
        from sage_core import TopologyGraph, PyTemplateStore
        store = PyTemplateStore()
        return store.create("sequential", "m")

    def _make_parallel(self):
        """Build a parallel topology (source -> workers -> aggregator)."""
        from sage_core import PyTemplateStore
        store = PyTemplateStore()
        return store.create("parallel", "m")

    def test_verify_sequential_all_pass(self):
        """Sequential topology passes all LTL checks."""
        g = self._make_sequential()
        result = verify_topology_ltl(g)
        assert result["safe"] is True
        assert result["live"] is True
        assert result["bounded_live"] is True
        assert result["warnings"] == []
        assert result["errors"] == []

    def test_verify_parallel_all_pass(self):
        """Parallel topology passes all LTL checks."""
        g = self._make_parallel()
        result = verify_topology_ltl(g)
        assert result["safe"] is True
        assert result["live"] is True
        assert result["bounded_live"] is True

    def test_verify_bounded_liveness_tight_limit(self):
        """Sequential A->B->C with max_depth=1 fails bounded liveness."""
        g = self._make_sequential()
        result = verify_topology_ltl(g, max_depth=1)
        assert result["bounded_live"] is False
        assert len(result["warnings"]) > 0
        assert any("Bounded liveness" in w for w in result["warnings"])

    def test_reachability_forward(self):
        """Node 0 can reach node 2 in sequential topology."""
        g = self._make_sequential()
        assert check_reachability(g, 0, 2) is True

    def test_reachability_backward_fails(self):
        """Node 2 cannot reach node 0 in a DAG."""
        g = self._make_sequential()
        assert check_reachability(g, 2, 0) is False

    def test_reachability_self(self):
        """Node 0 can reach itself (BFS includes start node)."""
        g = self._make_sequential()
        assert check_reachability(g, 0, 0) is True

    def test_reachability_index_error(self):
        """Out-of-range index raises IndexError from Rust."""
        g = self._make_sequential()
        with pytest.raises(IndexError):
            check_reachability(g, 0, 999)
