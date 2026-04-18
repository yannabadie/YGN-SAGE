"""Candidate: pin MiniMax on coder nodes via provider hints.

Hypothesis: in v5c-e smoke runs MiniMax was the strongest model at
SWE-bench-style coding tasks (it produced the 567/1386/451 char patches
on astropy-14995, django-10924, astropy-6938). The Apr 18 diversity
penalty in Rust ModelAssigner correctly spread load, but some of the
dilution may have cost quality on the coder node specifically. This
candidate explicitly prefers MiniMax on coder-style roles while letting
the planner and synthesizer float on the default scoring.

Axis: routing
Expected impact: +0.1 to +0.3 val_score vs baseline IF MiniMax is
genuinely the best coder. Otherwise flat or negative — in which case
the candidate is rejected and the frontier stays unchanged.

Structural vs hyperparameter: this is a routing-policy change, not a
knob tune. The baseline has no way to express "always use provider X
for role Y" without a new mechanism; this candidate adds that mechanism.
"""
from __future__ import annotations

from typing import Any

from reference_examples.ygn_sage.sage_candidate import SageCandidate

_CODER_ROLE_KEYWORDS = ("coder", "programmer", "implementer", "actor")


def _is_coder(role: str) -> bool:
    if not isinstance(role, str):
        return False
    rl = role.lower()
    return any(k in rl for k in _CODER_ROLE_KEYWORDS)


class MinimaxPinnedCoder(SageCandidate):
    name = "minimax_pinned"
    hypothesis = (
        "MiniMax is the observed best coder on SWE-bench Lite tasks "
        "3..7; force it on coder nodes while letting other roles float."
    )
    axis = "routing"

    def build_system(self, hints: dict[str, Any] | None = None) -> Any:
        from sage.boot import boot_agent_system
        system = boot_agent_system()

        pipeline = getattr(system, "pipeline", None)
        if pipeline is None:
            # Nothing to patch — fall back to baseline behaviour silently.
            return system

        _original_assign = pipeline._stage_assign_models

        def _patched_assign_models(ctx: Any) -> Any:
            """Inject provider_hints=[(node_idx, 'minimax'), ...] for every
            coder-role node before ModelAssigner scores candidates."""
            hints_list = []
            topology = getattr(ctx, "topology", None)
            if topology is not None:
                node_count = int(getattr(topology, "node_count", lambda: 0)() or 0)
                for idx in range(node_count):
                    try:
                        node = topology.get_node(idx)
                    except (AttributeError, Exception):
                        continue
                    if _is_coder(getattr(node, "role", "")):
                        hints_list.append((idx, "minimax"))
            if hints_list:
                existing = getattr(ctx, "provider_hints", None) or []
                ctx.provider_hints = list(existing) + hints_list
            return _original_assign(ctx)

        pipeline._stage_assign_models = _patched_assign_models
        return system


CANDIDATE = MinimaxPinnedCoder()
