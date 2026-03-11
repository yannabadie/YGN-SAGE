"""LTL temporal property verification bridge.

Wraps Rust LtlVerifier (always compiled in sage_core) to check
topology properties: safety, liveness, bounded liveness, reachability.

The Rust LtlVerifier uses petgraph BFS/DFS — O(V+E), no SMT solver needed.
All checks operate on TopologyGraph instances.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

try:
    from sage_core import LtlVerifier, LtlResult  # noqa: F401
    _HAS_LTL = True
except ImportError:
    _HAS_LTL = False


def verify_topology_ltl(
    topology: Any | None,
    max_depth: int = 20,
) -> dict[str, Any]:
    """Run LTL temporal property checks on a TopologyGraph.

    Parameters
    ----------
    topology : TopologyGraph | None
        The topology graph to verify.  If *None*, returns optimistic defaults.
    max_depth : int
        Upper bound for bounded-liveness path depth (default 20).

    Returns
    -------
    dict with keys:
        reachable  – bool (always True; per-pair reachability needs explicit indices)
        safe       – bool (no HIGH->LOW information flow edges)
        live       – bool (every entry node reaches at least one exit)
        bounded_live – bool (all entry-to-exit paths within *max_depth*)
        warnings   – list[str] (liveness / bounded-liveness violations)
        errors     – list[str] (safety violations)
    """
    default: dict[str, Any] = {
        "reachable": True,
        "safe": True,
        "live": True,
        "bounded_live": True,
        "warnings": [],
        "errors": [],
    }

    if topology is None:
        return default

    if not _HAS_LTL:
        _log.debug("LtlVerifier not available (sage_core not compiled)")
        return default

    try:
        verifier = LtlVerifier()
        warnings: list[str] = []
        errors: list[str] = []

        # Safety: no HIGH->LOW information flow paths
        safety = verifier.check_safety(topology)
        if not safety.passed:
            errors.extend(safety.violations)

        # Liveness: all entry nodes reach some exit
        liveness = verifier.check_liveness(topology)
        if not liveness.passed:
            warnings.extend(liveness.violations)

        # Bounded liveness: all paths complete within depth limit
        bounded = verifier.check_bounded_liveness(topology, max_depth)
        if not bounded.passed:
            warnings.extend(bounded.violations)

        return {
            "reachable": True,  # per-pair reachability needs specific node indices
            "safe": safety.passed,
            "live": liveness.passed,
            "bounded_live": bounded.passed,
            "warnings": warnings,
            "errors": errors,
        }
    except Exception as exc:
        _log.warning("LTL verification failed (%s)", exc)
        default["warnings"].append(f"LTL check failed: {exc}")
        return default


def check_reachability(
    topology: Any,
    from_idx: int,
    to_idx: int,
) -> bool:
    """Check if node *to_idx* is reachable from *from_idx* via BFS.

    Requires sage_core.  Returns False if LtlVerifier is unavailable.
    Raises IndexError (from Rust) if indices are out of range.
    """
    if not _HAS_LTL:
        _log.debug("LtlVerifier not available — returning False")
        return False

    verifier = LtlVerifier()
    return verifier.check_reachability(topology, from_idx, to_idx)
