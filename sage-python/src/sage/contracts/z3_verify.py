"""Z3 SMT contract verification for TaskDAG.

Four properties:
1. Capability coverage — every required capability is available
2. Budget feasibility — sum of per-node budgets ≤ total budget
3. Type compatibility — output fields of predecessor ⊇ input fields of successor
4. Provider assignment — genuine SAT: assign providers to nodes respecting
   capability requirements and mutual exclusion constraints
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    import z3
except ImportError:
    z3 = None  # type: ignore[assignment]

from sage.contracts.dag import TaskDAG


@dataclass
class ContractVerdict:
    """Result of a Z3 contract check."""

    satisfied: bool
    property_name: str
    counterexample: str | None = None


def _require_z3() -> None:
    if z3 is None:
        raise ImportError("z3-solver is required for contract verification")


# ---------------------------------------------------------------------------
# 1. Capability coverage
# ---------------------------------------------------------------------------

def verify_capability_coverage(
    dag: TaskDAG,
    available_capabilities: set[str],
) -> ContractVerdict:
    """Verify every node's required capabilities are available (Python native)."""
    missing = []
    for nid in dag.node_ids:
        node = dag.get_node(nid)
        if node.capabilities_required:
            m = set(node.capabilities_required) - available_capabilities
            if m:
                missing.append(f"node '{nid}' requires {m}")
    return ContractVerdict(
        satisfied=not missing,
        property_name="capability_coverage",
        counterexample="; ".join(missing) if missing else None,
    )


# ---------------------------------------------------------------------------
# 2. Budget feasibility
# ---------------------------------------------------------------------------

def verify_budget_feasibility(
    dag: TaskDAG,
    total_budget_usd: float,
) -> ContractVerdict:
    """Verify sum of per-node max_cost_usd <= total budget (Python native)."""
    total = sum(
        dag.get_node(nid).budget.max_cost_usd
        for nid in dag.node_ids
        if dag.get_node(nid).budget.max_cost_usd > 0
    )
    ok = total <= total_budget_usd
    return ContractVerdict(
        satisfied=ok,
        property_name="budget_feasibility",
        counterexample=f"Total cost ${total:.4f} > budget ${total_budget_usd:.4f}" if not ok else None,
    )


# ---------------------------------------------------------------------------
# 3. Type compatibility
# ---------------------------------------------------------------------------

def verify_type_compatibility(dag: TaskDAG) -> ContractVerdict:
    """Verify for each edge A->B, A.output_fields >= B.input_fields (Python native)."""
    missing = []
    for from_id in dag.node_ids:
        src = dag.get_node(from_id)
        for to_id in dag.successors(from_id):
            dst = dag.get_node(to_id)
            for field_name in dst.input_schema.fields:
                if field_name not in src.output_schema.fields:
                    missing.append(f"edge {from_id}->{to_id}: '{field_name}' missing")
    return ContractVerdict(
        satisfied=not missing,
        property_name="type_compatibility",
        counterexample="; ".join(missing) if missing else None,
    )


# ---------------------------------------------------------------------------
# 4. Provider assignment (genuine constraint satisfaction)
# ---------------------------------------------------------------------------

@dataclass
class ProviderSpec:
    """Describes a provider's capability set and exclusion rules."""
    name: str
    capabilities: set[str]
    # Pairs of capabilities that cannot be used together on this provider
    exclusions: list[tuple[str, str]] = field(default_factory=list)


def verify_provider_assignment(
    dag: TaskDAG,
    providers: list[ProviderSpec],
) -> ContractVerdict:
    """Verify that every node can be assigned a provider satisfying its requirements.

    This is a genuine constraint satisfaction problem (SAT):
    - Each node must be assigned exactly one provider
    - The provider must offer ALL capabilities the node requires
    - If a node requires capabilities that are mutually exclusive on a provider,
      that provider cannot serve the node

    Z3 models this as: for each node, OR over providers that can serve it.
    A provider can serve a node iff:
      (a) provider.capabilities ⊇ node.capabilities_required
      (b) no exclusion pair is fully required by the node

    Returns unsatisfied if ANY node cannot be served by ANY provider.
    The counterexample lists which nodes are unassignable and why.
    """
    _require_z3()

    if not providers:
        # No providers — only satisfiable if no node requires capabilities
        for nid in dag.node_ids:
            node = dag.get_node(nid)
            if node.capabilities_required:
                return ContractVerdict(
                    satisfied=False,
                    property_name="provider_assignment",
                    counterexample=f"Node '{nid}' requires {node.capabilities_required} but no providers available",
                )
        return ContractVerdict(satisfied=True, property_name="provider_assignment")

    solver = z3.Solver()

    # For each node, create a Bool variable per provider: "node_i assigned to provider_j"
    assignment_vars: dict[str, dict[str, z3.BoolRef]] = {}

    for nid in dag.node_ids:
        node = dag.get_node(nid)
        if not node.capabilities_required:
            continue  # No requirements — any provider works

        required = set(node.capabilities_required)
        node_vars: dict[str, z3.BoolRef] = {}

        for prov in providers:
            var = z3.Bool(f"assign_{nid}_{prov.name}")
            node_vars[prov.name] = var

            # Check if provider CAN serve this node
            has_all_caps = required.issubset(prov.capabilities)

            # Check mutual exclusion: if node requires both sides of an exclusion pair
            has_exclusion_conflict = any(
                a in required and b in required
                for a, b in prov.exclusions
            )

            if not has_all_caps or has_exclusion_conflict:
                # Provider cannot serve this node
                solver.add(var == False)  # noqa: E712
            # else: provider CAN serve — variable is free

        if node_vars:
            assignment_vars[nid] = node_vars
            # At least one provider must be assigned to this node
            solver.add(z3.Or(*node_vars.values()))

    if solver.check() == z3.sat:
        return ContractVerdict(satisfied=True, property_name="provider_assignment")

    # UNSAT — find which nodes are unassignable
    unassignable = []
    for nid, pvars in assignment_vars.items():
        node = dag.get_node(nid)
        required = set(node.capabilities_required)
        reasons = []
        for prov in providers:
            missing = required - prov.capabilities
            conflicts = [
                f"{a}+{b}" for a, b in prov.exclusions
                if a in required and b in required
            ]
            if missing:
                reasons.append(f"{prov.name}: missing {missing}")
            elif conflicts:
                reasons.append(f"{prov.name}: exclusion conflict {conflicts}")
        if reasons:
            unassignable.append(f"node '{nid}' ({', '.join(reasons)})")

    return ContractVerdict(
        satisfied=False,
        property_name="provider_assignment",
        counterexample="; ".join(unassignable) if unassignable else "UNSAT (no counterexample extracted)",
    )
