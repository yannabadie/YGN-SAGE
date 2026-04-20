"""Tests for the Rust TopologyController port (Phase 2 of the Rust-First plan).

Plan 2.1 scaffold only — the Rust class exists, instantiates cleanly,
Python delegate holds a reference to it, but all decision paths still
run through the Python legacy `TopologyController.evaluate_and_decide`.
Commits 2.2..2.6 populate per-path delegation with 20-sample Rust-vs-
Python equivalence tests.

These tests skip cleanly when `sage_core` is not compiled — expected in
Python-only dev environments.
"""
from __future__ import annotations

import pytest

try:
    from sage_core import RustTopologyController, RustAdaptationDecision  # noqa: F401
    _HAS_SAGE_CORE = True
except ImportError:
    _HAS_SAGE_CORE = False


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_rust_topology_controller_imports_successfully():
    """The PyO3 scaffold must be importable and instantiable after `maturin develop`.

    Regression guard: if `lib.rs` drops the `add_class::<RustTopologyController>`
    line or `topology/mod.rs` loses the `pub mod controller;` declaration,
    this import fails at runtime even when the Rust lib compiles.
    """
    from sage_core import RustTopologyController
    ctrl = RustTopologyController()
    # State starts empty — must mirror Python TopologyController.__init__
    assert ctrl.reroute_count == 0
    assert ctrl.spawn_count == 0
    assert ctrl.abstain_count == 0


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_rust_topology_controller_evaluate_scaffold_returns_none():
    """Plan 2.1 contract: scaffold `evaluate_and_decide` returns None,
    signalling the Python delegate to fall through to the legacy path.
    This keeps pre-2.2..2.6 behavior identical while the Rust class
    exists alongside the Python logic."""
    from sage_core import RustTopologyController
    ctrl = RustTopologyController()
    decision = ctrl.evaluate_and_decide(
        node_idx=0,
        result="any output",
        task="any task",
    )
    assert decision is None, (
        "2.1 scaffold must return None — any Some(decision) here would "
        "start routing Python paths through an empty Rust brain"
    )


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_rust_adaptation_decision_roundtrip_from_python():
    """RustAdaptationDecision must be constructible + readable from Python.

    Once 2.2..2.6 start populating Rust decision paths, the Python
    delegate will receive RustAdaptationDecision instances back from
    Rust method calls. This test pins the field contract early so
    mismatches fail loudly (rather than producing silent AttributeError
    deep in pipeline.runner)."""
    from sage_core import RustAdaptationDecision
    d = RustAdaptationDecision(
        action="reroute_topology",
        target_node=3,
        reason="empty output",
    )
    assert d.action == "reroute_topology"
    assert d.target_node == 3
    assert d.reason == "empty output"
    # Defaults: Python None maps to Rust None for the optional fields.
    assert d.new_model_id is None
    assert d.invariant_feedback is None
    assert d.gate_source is None
    assert d.gate_target is None


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_python_controller_attaches_rust_companion_when_available():
    """TopologyController.__init__ must instantiate the Rust companion
    when sage_core is available — this wires the delegation hook that
    commits 2.2..2.6 will start using. Regression guard: silently
    dropping `self._rust_ctrl = RustTopologyController() if available
    else None` would leave every future port path with no place to
    delegate to."""
    from sage.topology_controller import TopologyController, _HAS_RUST_CTRL
    assert _HAS_RUST_CTRL, (
        "sage_core is importable above (test skip check passed) but "
        "topology_controller.py reported _HAS_RUST_CTRL = False — the "
        "import guard is mis-spelled or the PyO3 export name drifted"
    )

    tc = TopologyController()
    assert tc._rust_ctrl is not None
    assert hasattr(tc._rust_ctrl, "evaluate_and_decide")
