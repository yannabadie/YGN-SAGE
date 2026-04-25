"""B1.b dev-machine smoke: count distinct Rust span names emitted from
a Python sage_span scope that exercises the SystemRouter routing path.

Acceptance criterion §8.D: ≥ 8 distinct Rust span names in routing path.

Usage:
    cd sage-python
    SAGE_OTEL_EXPORTER=console python scripts/b1b_smoke.py 2>spans.log
    # then: grep -oE 'name: "[^"]+"' spans.log | sort -u
"""
from __future__ import annotations

import os
import sys

# Force exporter env BEFORE imports cache anything.
os.environ.setdefault("SAGE_OTEL_EXPORTER", "console")

from sage.observability import _init_tracer  # noqa: E402
from sage.observability.spans import sage_span  # noqa: E402

import sage_core  # noqa: E402


def main() -> int:
    _init_tracer()

    # Verify the bridge is live — init_otel is idempotent so this returns
    # False on a freshly-rebuilt session if Python already mirrored.
    print(
        f"sage_core.init_otel(console) idempotent: "
        f"{sage_core.init_otel('console', None)}",
        file=sys.stderr,
    )

    if not hasattr(sage_core, "SystemRouter"):
        print("sage_core.SystemRouter not exposed; cannot exercise routing.", file=sys.stderr)
        return 1

    # Resolve cards.toml from the repo root regardless of cwd.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cards_path = os.path.join(repo_root, "sage-core", "config", "cards.toml")
    registry = sage_core.ModelRegistry.from_toml_file(cards_path)
    print(f"loaded {registry.len()} models from {cards_path}", file=sys.stderr)
    router = sage_core.SystemRouter(registry)

    # Bandit + topology-engine instances exercise additional Rust span sites.
    bandit = sage_core.ContextualBandit(0.5)
    bandit.register_arm("arm_a", "sequential")
    bandit.register_arm("arm_b", "parallel")
    bandit.register_arm("arm_c", "debate")

    # Routing path under a Python sage.assign span — production-shaped.
    with sage_span("sage.assign", op="assign_models"):
        for task in [
            "Compute the fibonacci of n",
            "Refactor this Python class into separate modules",
            "Solve this LeetCode binary tree problem",
            "Translate the docstring to French",
            "Generate unit tests for the parser",
        ]:
            try:
                decision = router.route(task, 1.0)
                print(
                    f"  routed '{task[:40]}...' "
                    f"system={decision.system} model={decision.model_id}",
                    file=sys.stderr,
                )
                router.record_outcome(decision.decision_id, 0.85, 0.001, 250.0)
            except Exception as e:  # pylint: disable=broad-except
                print(f"  routing call failed: {e}", file=sys.stderr)

    # Bandit selection + record under a sage.bandit_demo span.
    # Exercises bandit.select, bandit.select_contextual, bandit.record.
    with sage_span("sage.bandit_demo", op="bandit_demo"):
        try:
            d1 = bandit.select(0.5)
            print(f"  bandit.select model={d1.model_id}", file=sys.stderr)
            bandit.record(d1.decision_id, 0.8, 0.001, 100.0)
            d2 = bandit.select_with_context(0.5, [0.1, 0.2, 0.3])
            print(f"  bandit.select_with_context model={d2.model_id}", file=sys.stderr)
            bandit.record(d2.decision_id, 0.7, 0.002, 120.0)
        except Exception as e:  # pylint: disable=broad-except
            print(f"  bandit calls failed: {e}", file=sys.stderr)

    # TopologyEngine generate + record_outcome under a sage.topology_select span.
    # Exercises topology_engine.generate + path-specific spans + record_outcome.
    # Embedding dim 384 matches arctic-embed-m used elsewhere in sage.
    if hasattr(sage_core, "TopologyEngine"):
        try:
            engine = sage_core.TopologyEngine()
            embedding = [0.1] * 384
            with sage_span("sage.topology_select", op="topology_select"):
                gen_result = engine.generate("Compute fibonacci", embedding, 1, 0.5)
                print(
                    f"  topology generate source={gen_result.source} "
                    f"topology_id={gen_result.topology_id}",
                    file=sys.stderr,
                )
                engine.record_outcome(gen_result.topology_id, 0.85, 0.005, 250.0)
        except Exception as e:  # pylint: disable=broad-except
            print(f"  topology calls failed: {e}", file=sys.stderr)

    print("smoke complete", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
