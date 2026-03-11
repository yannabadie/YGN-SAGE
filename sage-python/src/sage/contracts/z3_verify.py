"""SMT contract verification for TaskDAG.

Four properties:
1. Capability coverage — every required capability is available
2. Budget feasibility — sum of per-node budgets ≤ total budget
3. Type compatibility — output fields of predecessor ⊇ input fields of successor
4. Provider assignment — genuine SAT: assign providers to nodes respecting
   capability requirements and mutual exclusion constraints

Backend priority: Rust OxiZ (sage_core.SmtVerifier) > Python z3-solver.
Properties 1-3 are Python-native (no SMT needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Try Rust OxiZ backend first
try:
    from sage_core import SmtVerifier as _RustSmtVerifier
    _RUST_SMT_AVAILABLE = True
except ImportError:
    _RUST_SMT_AVAILABLE = False

# Fallback: Python z3-solver
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


def _require_smt() -> None:
    if not _RUST_SMT_AVAILABLE and z3 is None:
        raise ImportError("No SMT backend: install sage_core[smt] or z3-solver")


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

    Backend: Rust OxiZ (sage_core.SmtVerifier) when available, Python z3-solver fallback.
    """
    _require_smt()

    # --- Rust OxiZ path (preferred) ---
    if _RUST_SMT_AVAILABLE:
        return _verify_provider_assignment_rust(dag, providers)

    # --- Python Z3 fallback ---
    return _verify_provider_assignment_z3(dag, providers)


def _verify_provider_assignment_rust(
    dag: TaskDAG,
    providers: list[ProviderSpec],
) -> ContractVerdict:
    """Provider assignment via Rust OxiZ SmtVerifier."""
    verifier = _RustSmtVerifier()
    nodes = []
    for nid in dag.node_ids:
        node = dag.get_node(nid)
        caps = list(node.capabilities_required) if node.capabilities_required else []
        nodes.append((nid, caps))
    provs = [
        (p.name, list(p.capabilities), list(p.exclusions))
        for p in providers
    ]
    sat, errors = verifier.verify_provider_assignment(nodes, provs)
    return ContractVerdict(
        satisfied=sat,
        property_name="provider_assignment",
        counterexample="; ".join(errors) if errors else None,
    )


def _verify_provider_assignment_z3(
    dag: TaskDAG,
    providers: list[ProviderSpec],
) -> ContractVerdict:
    """Provider assignment via Python z3-solver (fallback)."""
    if not providers:
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

    assignment_vars: dict[str, dict[str, z3.BoolRef]] = {}

    for nid in dag.node_ids:
        node = dag.get_node(nid)
        if not node.capabilities_required:
            continue

        required = set(node.capabilities_required)
        node_vars: dict[str, z3.BoolRef] = {}

        for prov in providers:
            var = z3.Bool(f"assign_{nid}_{prov.name}")
            node_vars[prov.name] = var

            has_all_caps = required.issubset(prov.capabilities)
            has_exclusion_conflict = any(
                a in required and b in required
                for a, b in prov.exclusions
            )

            if not has_all_caps or has_exclusion_conflict:
                solver.add(var == False)  # noqa: E712

        if node_vars:
            assignment_vars[nid] = node_vars
            solver.add(z3.PbEq([(v, 1) for v in node_vars.values()], 1))

    if solver.check() == z3.sat:
        return ContractVerdict(satisfied=True, property_name="provider_assignment")

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
