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
def test_rust_path4_parallel_inconsistency_matches_python_on_20_samples():
    """Plan 2.4 equivalence for path 4 (parallel inconsistency reroute).
    Scoring stays in Python (embedder-backed). Rust takes pre-computed
    consistency score + is_debate flag and applies threshold + reroute
    state machine."""
    from sage_core import RustTopologyController
    from sage.topology_controller import TopologyController, AdaptationDecision

    # (node_idx, consistency, is_debate) — 20 samples
    cases: list[tuple[int, float, bool]] = [
        (0, 0.9, False),   # above threshold → None
        (0, 0.5, False),   # at threshold (exclusive) → None
        (1, 0.49, False),  # below → reroute
        (2, 0.1, False),   # way below → reroute (but budget already used)
        (3, 0.0, False),   # orthogonal
        (4, 0.3, True),    # debate → suppress
        (5, 0.0, True),    # debate → suppress even at 0
        (6, 0.8, False),   # above → None
        (7, 1.0, False),   # perfect → None
        (8, 0.49, False),  # budget may be exhausted from sample 1
        (9, 0.2, False),   # depends on budget
        (10, 0.5, True),   # debate
        (11, 0.6, False),  # above
        (12, 0.4, False),  # below (if budget available)
        (13, 0.55, False),
        (14, 0.45, False),
        (15, 0.7, False),
        (16, 0.99, True),
        (17, 0.15, True),
        (18, 0.5, False),  # exactly at — above-or-equal returns None
        (19, 0.0, False),  # final check
    ]

    rust_ctrl = RustTopologyController()
    py_ctrl = TopologyController()

    for node_idx, consistency, is_debate in cases[:20]:
        rust_decision = rust_ctrl.check_parallel_inconsistency(node_idx, consistency, is_debate)
        py_decision = _python_path4_reference(py_ctrl, node_idx, consistency, is_debate)

        if rust_decision is None:
            assert py_decision is None, (
                f"rust passed node={node_idx} consistency={consistency} debate={is_debate} "
                f"but python returned {py_decision}"
            )
            continue
        assert py_decision is not None
        assert rust_decision.action == py_decision.action == "reroute_topology"
        assert rust_decision.target_node == py_decision.target_node == node_idx
        # Reason: both f"consistency=X.XX < 0.5"
        assert rust_decision.reason.startswith("consistency=")
        assert py_decision.reason.startswith("consistency=")


def _python_path4_reference(py_ctrl, node_idx, consistency, is_debate):
    """Mirror of path-4 state machine from topology_controller.py:202-211."""
    from sage.topology_controller import AdaptationDecision, TopologyController

    if is_debate or py_ctrl._reroute_count >= TopologyController.MAX_REROUTES:
        return None
    if consistency >= TopologyController.THETA_CONSISTENCY:
        return None
    py_ctrl._reroute_count += 1
    return AdaptationDecision(
        action="reroute_topology",
        target_node=node_idx,
        reason=f"consistency={consistency:.2f} < {TopologyController.THETA_CONSISTENCY}",
    )


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_rust_path5_importance_prune_matches_python_on_20_samples():
    """Plan 2.4 equivalence for path 5 (importance prune)."""
    from sage_core import RustTopologyController
    from sage.topology_controller import TopologyController, AdaptationDecision

    # (node_idx, importance, is_debate, quality_is_known) — 20 samples
    cases: list[tuple[int, float, bool, bool]] = [
        (0, 0.1, False, True),    # prune (below THETA_PRUNE=0.2)
        (1, 0.19, False, True),   # just below → prune
        (2, 0.2, False, True),    # at threshold → None
        (3, 0.21, False, True),   # above → None
        (4, 0.9, False, True),    # way above → None
        (5, 0.0, False, True),    # zero → prune
        (6, 0.15, False, True),   # below → prune
        (7, 0.1, True, True),     # debate → suppress
        (8, 0.0, True, True),     # debate → suppress
        (9, 0.19, False, False),  # quality abstain → suppress
        (10, 0.0, False, False),  # quality abstain → suppress
        (11, 0.5, False, True),   # above → None
        (12, 0.199, False, True), # just below → prune
        (13, 0.25, False, True),
        (14, 0.05, False, True),
        (15, 0.18, True, False),  # both suppressed
        (16, 0.12, True, True),   # debate suppressed
        (17, 0.7, False, True),
        (18, 0.1, False, False),  # abstain suppressed
        (19, 0.0, False, True),   # prune
    ]

    rust_ctrl = RustTopologyController()
    py_ctrl = TopologyController()

    for node_idx, importance, is_debate, quality_is_known in cases:
        rust_decision = rust_ctrl.check_importance_prune(
            node_idx, importance, is_debate, quality_is_known
        )
        py_decision = _python_path5_reference(
            py_ctrl, node_idx, importance, is_debate, quality_is_known
        )

        if rust_decision is None:
            assert py_decision is None, (
                f"rust passed node={node_idx} importance={importance} debate={is_debate} "
                f"known={quality_is_known} but python returned {py_decision}"
            )
            continue
        assert py_decision is not None
        assert rust_decision.action == py_decision.action == "prune_node"
        assert rust_decision.target_node == py_decision.target_node == node_idx
        assert rust_decision.reason.startswith("importance=")


def _python_path5_reference(py_ctrl, node_idx, importance, is_debate, quality_is_known):
    """Mirror of path-5 state machine from topology_controller.py:213-222."""
    from sage.topology_controller import AdaptationDecision, TopologyController

    if is_debate or not quality_is_known:
        return None
    if importance >= TopologyController.THETA_PRUNE:
        return None
    return AdaptationDecision(
        action="prune_node",
        target_node=node_idx,
        reason=f"importance={importance:.2f} < {TopologyController.THETA_PRUNE}",
    )


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_rust_path6_emergent_spawn_matches_python_on_20_samples():
    """Plan 2.5 equivalence for path 6 (emergent subtask spawn). Rust's
    regex crate and Python's `re` module have compatible semantics for
    the three emergent patterns; any drift here (e.g. Unicode-class
    differences, \\s anchoring) would be caught by the 20 sample inputs."""
    from sage_core import RustTopologyController
    from sage.topology_controller import TopologyController, AdaptationDecision

    # Each sample is the LLM-output string a node might produce.
    # Wrapped in a named constant vector so it reads as a fixture, not
    # as work-in-progress markers (see feedback memory
    # feedback_no_todos_in_test_strings).
    llm_outputs: list[str] = [
        # Pattern 1 matches: "need to also|additionally|we should also|another step would be"
        "Done with the current step. Additionally we should handle the retry logic properly.",
        "Finished parsing. Another step would be validating the output schema against spec.",
        "Main fix done. We should also wire the observability counters for throughput.",
        "Fixed the race. Need to also update the migration script for the new schema.",
        # Pattern 2 matches: "TODO|FIXME|NOTE"
        "Refactored the loop. TODO: add unit tests for the edge cases of the parser.",
        "Added the feature. FIXME: this has a known off-by-one when the list is empty list.",
        "Shipped. NOTE: the caching strategy still needs a review by the infra team later.",
        # Pattern 3 matches: "this requires|prerequisite:"
        "Cannot proceed. Prerequisite: resolve the type-mismatch in the base config first.",
        "Blocked. This requires a schema migration before the new fields become readable.",
        # Plain content (no match)
        "The answer is 42.",
        "Function completed successfully with all tests passing.",
        "",
        # Edge: pattern match but content too short (< 10 chars) — Python rejects
        "TODO: fix",        # 3 chars captured → rejected
        "NOTE: lol",        # 3 chars captured → rejected
        # Match but content exactly 10 chars — accepted at boundary
        "TODO: 0123456789012",  # 13 chars after colon+space → accepted
        # Match near 200-char boundary
        "Additionally we should " + "x" * 180 + ".",  # content near 200
        # Content > 200 → truncated match (regex greedy-cap at 200)
        "TODO: " + "x" * 250,
        # Case-insensitive match
        "Wrapped up. todo: add more integration tests for the edge case of the decoder.",
        "Done. FIXME: this should really handle unicode normalization for the text input.",
        # Multi-pattern (Python for-loop takes first hit — pattern 1)
        "Additionally we should handle case X. TODO: also handle case Y properly.",
    ]

    rust_ctrl = RustTopologyController()
    py_ctrl = TopologyController()

    for idx, output in enumerate(llm_outputs):
        rust_decision = rust_ctrl.check_emergent_spawn(output, idx)
        py_decision = _python_path6_reference(py_ctrl, output, idx)

        if rust_decision is None:
            assert py_decision is None, (
                f"sample #{idx}: rust passed, python returned {py_decision!r}. "
                f"Output: {output!r}"
            )
            continue

        assert py_decision is not None, (
            f"sample #{idx}: rust returned {rust_decision!r}, python passed"
        )
        assert rust_decision.action == py_decision.action == "spawn_subagent"
        assert rust_decision.target_node == py_decision.target_node == idx
        # Reason is the captured emergent subtask text — must match exactly
        assert rust_decision.reason == py_decision.reason, (
            f"sample #{idx} reason mismatch:\n"
            f"  rust:   {rust_decision.reason!r}\n"
            f"  python: {py_decision.reason!r}\n"
            f"  output: {output!r}"
        )


def _python_path6_reference(py_ctrl, result, node_idx):
    """Mirror of path-6 logic from topology_controller.py:224-233."""
    from sage.topology_controller import AdaptationDecision, TopologyController

    emergent = TopologyController._detect_emergent_subtask(result)
    if not emergent or py_ctrl._spawn_count >= TopologyController.MAX_SPAWNS:
        return None
    py_ctrl._spawn_count += 1
    return AdaptationDecision(
        action="spawn_subagent",
        target_node=node_idx,
        reason=emergent,
    )


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
def test_h11_arithmetic_retry_syncs_onto_rust_state():
    """H11 regression (2026-04-20 advisor+Codex review). The cascade has
    two different branches that consume a retry on the same node:
      1. Python `_verify_arithmetic` fails on axis=depth and increments
         `self._node_retries[node_idx]` Python-side.
      2. Rust `check_quality_cascade` reads Rust-side `node_retries` on
         the NEXT evaluation of the same node; if the Python-side
         increment didn't mirror to Rust, Rust sees retries=0 and
         issues an EXTRA upgrade_model — bypassing the budget by one.

    Per-method equivalence tests missed this because they exercise one
    path at a time; the bug only appears in a cross-path trajectory.
    This test pins the contract.
    """
    from unittest.mock import MagicMock
    from sage.topology_controller import TopologyController

    qe = MagicMock()
    # qe.estimate is called ONCE per evaluate_and_decide (before the
    # depth branch runs, per the Rust-primary cascade structure).
    # Step 1: quality=0.5 (middle — path 2 returns None; arithmetic branch fires upgrade)
    # Step 2: quality=0.1 (critical — path 2 upgrade_model, Rust increments retries 1→2)
    # Step 3: quality=0.1 (critical — retries exhausted at 2/2 → path 2 returns None → default continue)
    qe.estimate.side_effect = [0.5, 0.1, 0.1]

    ctrl = TopologyController(assigner=None, quality_estimator=qe)
    assert ctrl._rust_ctrl is not None, "test requires sage_core"

    # Explicitly configure the topology mock so _max_retries_for_node
    # returns MAX_RETRIES=2 (not the auto-MagicMock coercion which
    # defaults node.max_retries to int(MagicMock())=1). Also set
    # `system = 1` so `_get_invariant_feedback`'s `< 3` int comparison
    # doesn't blow up on a bare MagicMock auto-attr.
    node = MagicMock()
    node.max_retries = 2
    node.system = 1
    node.model_id = ""
    node.required_capabilities = []
    node.role = "agent"
    node.max_cost_usd = 1.0
    topo = MagicMock()
    topo.get_node.return_value = node
    ctx_depth = MagicMock()
    ctx_depth.latency_ms = 100.0
    ctx_depth.get = lambda k, d=None: {"axis_hint": "depth"}.get(k, d)

    # Step 1: arithmetic branch fires (result contains wrong equation)
    bad_math = "The answer is 5 + 3 = 9. Done."
    d1 = ctrl.evaluate_and_decide(
        node_idx=7, result=bad_math, task="compute", topology=topo, ctx=ctx_depth,
    )
    assert d1.action == "upgrade_model"
    assert "arithmetic" in d1.reason
    assert ctrl._node_retries[7] == 1, "Python-side retries must increment"
    assert ctrl._rust_ctrl.quality_stats()["reroute_count"] == 0  # sanity: wrong counter
    # The critical invariant — Rust-side retry count for node 7 must also be 1
    rust_stats_retries = _rust_node_retries_for(ctrl._rust_ctrl, 7)
    assert rust_stats_retries == 1, (
        f"H11 regression: Python arithmetic branch incremented its dict to 1 but "
        f"Rust node_retries[7] = {rust_stats_retries}. Without the "
        f"set_node_retries mirror, check_quality_cascade sees retries=0 on "
        f"subsequent calls and over-issues upgrade_model decisions."
    )

    # Step 2: different evaluation of the SAME node hits critical quality
    # via the normal (non-depth) path. Rust should now see retries=1 (from
    # step 1's arithmetic), retry_limit=2 → one more upgrade available.
    ctx_normal = MagicMock()
    ctx_normal.latency_ms = 100.0
    ctx_normal.get = lambda k, d=None: {}.get(k, d)
    d2 = ctrl.evaluate_and_decide(
        node_idx=7, result="some answer text here", task="compute",
        topology=topo, ctx=ctx_normal,
    )
    # d2 should upgrade (retry still available: 1 of 2 used). Rust increments to 2.
    assert d2.action == "upgrade_model"
    rust_stats_retries = _rust_node_retries_for(ctrl._rust_ctrl, 7)
    assert rust_stats_retries == 2, (
        f"After step 2, Rust retries[7] should be 2 (1 from arithmetic + "
        f"1 from critical-quality), got {rust_stats_retries}"
    )

    # Step 3: third evaluation — budget now exhausted at 2. No upgrade.
    d3 = ctrl.evaluate_and_decide(
        node_idx=7, result="answer text", task="compute",
        topology=topo, ctx=ctx_normal,
    )
    assert d3.action != "upgrade_model", (
        f"After 2 retries used, a third critical eval must NOT upgrade_model; "
        f"got {d3.action!r}. This is the H11 behavioral regression — "
        f"without the mirror, Rust thinks it has retries still available."
    )


def _rust_node_retries_for(rust_ctrl, node_idx):
    """Direct Rust-side retry-count observation — H11 audit exposes
    `get_node_retries(node_idx) -> u32` for this purpose so the
    cross-path trajectory assertions can check invariants without
    reaching into HashMap internals."""
    return rust_ctrl.get_node_retries(node_idx)


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
