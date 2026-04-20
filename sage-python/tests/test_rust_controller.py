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
def test_rust_path1_empty_error_matches_python_on_20_samples():
    """Plan 2.2 equivalence: Rust path 1 (check_empty_error_reroute) must
    produce the same (action, target_node, reason) as Python legacy
    `TopologyController._is_empty_or_error` + reroute logic on 20 sample
    inputs. Rust-vs-Python drift here is the H4-class bypass pattern
    applied at port level: if Rust classifies "timed out" as safe but
    Python rejects it, silent behavior divergence.
    """
    from sage_core import RustTopologyController
    from sage.topology_controller import TopologyController, AdaptationDecision

    # 20 sample inputs exercising every branch of _is_empty_or_error
    # and the reroute-budget state machine.
    samples: list[tuple[str, int]] = [
        # Blank / whitespace-only → empty-output reroute
        ("", 0),
        ("   ", 1),
        ("\n\t ", 2),
        # Sentinel prefix (D3 audit fix) → sentinel reroute
        ("[sage: agent exited after 20 steps, no final content]", 3),
        ("[sage: agent exited after 5 steps, last_tool=execute_bash]", 4),
        # Error patterns at start of string (case-insensitive)
        ("Error: something went wrong", 5),
        ("ERROR: tool not found", 6),
        ("Exception during call", 7),
        ("Traceback (most recent call last):\n  File ...", 8),
        ("timeout after 60s", 9),
        ("failed to resolve model", 10),
        # Error phrases anywhere (not just prefix)
        ("Here is output, but the tool timed out during execution", 11),
        ("I got no output from the tool, retrying won't help", 12),
        ("The operation failed with code 42", 13),
        # Normal content that must pass through to quality cascade
        ("def solve(): return 42", 14),
        ("The answer is 42.", 15),
        ("Here is my analysis of the problem: ...", 16),
        # Edge cases: leading whitespace but non-empty
        ("    answer goes here", 17),
        # Pseudo-error terms embedded but not matching (e.g. "errored out" word-boundary won't match "error" with boundary)
        ("This commit errored-out the merge", 18),
        # Empty after strip of sentinel-like (false-positive candidate)
        ("error handling was improved", 19),
    ]

    rust_ctrl = RustTopologyController()
    py_ctrl = TopologyController()

    for result_text, node_idx in samples:
        rust_decision = rust_ctrl.check_empty_error_reroute(result_text, node_idx)

        # Invoke Python legacy via the same code path — replicate the
        # path-1 logic locally since evaluate_and_decide wraps it with
        # additional stages (quality cascade etc.) that we don't want
        # to mix in here.
        py_decision = _python_path1_reference(py_ctrl, result_text, node_idx)

        if rust_decision is None:
            assert py_decision is None, (
                f"Rust passed through {result_text!r}@{node_idx} but Python "
                f"classified it as {py_decision!r}. Divergence on "
                f"_is_empty_or_error classification."
            )
            continue

        assert py_decision is not None, (
            f"Rust issued {rust_decision.action!r} on {result_text!r}@{node_idx} "
            f"but Python passed through. Path 1 divergence."
        )

        assert rust_decision.action == py_decision.action, (
            f"action mismatch on {result_text!r}@{node_idx}: "
            f"rust={rust_decision.action!r} python={py_decision.action!r}"
        )
        assert rust_decision.target_node == py_decision.target_node, (
            f"target_node mismatch on {result_text!r}@{node_idx}: "
            f"rust={rust_decision.target_node} python={py_decision.target_node}"
        )
        assert rust_decision.reason == py_decision.reason, (
            f"reason mismatch on {result_text!r}@{node_idx}: "
            f"rust={rust_decision.reason!r} python={py_decision.reason!r}"
        )


def _python_path1_reference(py_ctrl, result, node_idx):
    """Mirror of the path-1 logic from topology_controller.py:134-149
    — isolated from the full evaluate_and_decide so the equivalence
    test only compares what 2.2 ports."""
    from sage.topology_controller import AdaptationDecision, TopologyController

    if not TopologyController._is_empty_or_error(result):
        return None
    if py_ctrl._reroute_count < TopologyController.MAX_REROUTES:
        py_ctrl._reroute_count += 1
        reason = "empty output" if not result.strip() else "error-like output"
        return AdaptationDecision(
            action="reroute_topology",
            target_node=node_idx,
            reason=reason,
        )
    return AdaptationDecision(
        action="continue",
        target_node=node_idx,
        reason="reroute budget exhausted",
    )


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_rust_path2_quality_cascade_matches_python_on_20_samples():
    """Plan 2.3 equivalence: Rust `check_quality_cascade` must produce
    the same (action, target_node) and equivalent reason on 20 samples
    spanning all three quality bands (good, critical, middle).
    `invariant_feedback` + `new_model_id` remain Python-resolved for
    now (scope-clipped per plan; 2.6 ports them) — equivalence here
    covers the threshold + retry state machine only.
    """
    from sage_core import RustTopologyController
    from sage.topology_controller import TopologyController

    # 20 samples across bands — force each of: good / middle / critical /
    # critical-retry-exhausted / per-node retry isolation.
    samples: list[tuple[float, int, int]] = [
        # Good band (>= THETA_GOOD=0.7) → continue
        (0.95, 0, 2), (0.80, 1, 2), (0.70, 2, 2), (1.00, 3, 2),
        # Middle band [0.3, 0.7) → None (gate candidate; Python falls through)
        (0.69, 4, 2), (0.50, 5, 2), (0.40, 6, 2), (0.30, 7, 2),
        # Critical (< 0.3) + retries available → upgrade_model, increment retry
        (0.25, 8, 2), (0.10, 9, 2), (0.00, 10, 2), (0.15, 11, 2),
        # Retry exhaustion (same node three times) — last one returns None
        (0.05, 12, 2), (0.05, 12, 2), (0.05, 12, 2),
        # Per-node retry isolation (node 13 fresh, node 12 exhausted)
        (0.05, 13, 2), (0.29, 14, 2),
        # Edge: exactly at THETA_CRITICAL=0.3 → middle band (None)
        (0.3, 15, 2),
        # Edge: just below THETA_CRITICAL → critical
        (0.299, 16, 2),
        # With retry_limit=0, critical never upgrades
        (0.1, 17, 0),
    ]

    rust_ctrl = RustTopologyController()
    py_ctrl = TopologyController()

    for quality, node_idx, retry_limit in samples:
        rust_decision = rust_ctrl.check_quality_cascade(quality, node_idx, retry_limit)
        py_decision = _python_path2_reference(py_ctrl, quality, node_idx, retry_limit)

        if rust_decision is None:
            assert py_decision is None, (
                f"rust passed through q={quality} n={node_idx} r={retry_limit} but "
                f"python returned {py_decision}"
            )
            continue

        assert py_decision is not None, (
            f"rust returned {rust_decision.action!r} on q={quality} n={node_idx} "
            f"r={retry_limit} but python passed through"
        )
        assert rust_decision.action == py_decision.action, (
            f"action mismatch q={quality} n={node_idx}: "
            f"rust={rust_decision.action!r} python={py_decision.action!r}"
        )
        assert rust_decision.target_node == py_decision.target_node
        # Reason format: Rust writes "quality=0.10 < 0.3", Python uses f"{q:.2f}"
        # Accept matching prefix (modulo rounding) on the upgrade path.
        if rust_decision.action == "upgrade_model":
            assert rust_decision.reason.startswith("quality="), rust_decision.reason
            assert py_decision.reason.startswith("quality="), py_decision.reason
            assert rust_decision.reason.endswith("< 0.3"), rust_decision.reason


def _python_path2_reference(py_ctrl, quality, node_idx, retry_limit):
    """Mirror of path-2 threshold + retry logic from
    topology_controller.py:175-199 — same scope as Rust check_quality_cascade.
    No debate gate, no upgrade-model resolution — 2.6 will cover those."""
    from sage.topology_controller import AdaptationDecision, TopologyController

    py_ctrl._node_qualities[node_idx] = quality

    if quality >= TopologyController.THETA_GOOD:
        return AdaptationDecision(action="continue", target_node=node_idx)

    if quality < TopologyController.THETA_CRITICAL:
        retries = py_ctrl._node_retries.get(node_idx, 0)
        if retries < retry_limit:
            py_ctrl._node_retries[node_idx] = retries + 1
            return AdaptationDecision(
                action="upgrade_model",
                target_node=node_idx,
                reason=f"quality={quality:.2f} < {TopologyController.THETA_CRITICAL}",
                # new_model_id + invariant_feedback deliberately left None —
                # Rust doesn't compute them either (2.6 scope).
            )
        return None

    # Middle band — caller checks debate gate
    return None


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
