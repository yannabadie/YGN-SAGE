import sys, tempfile, os, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.evolution.llm_mutator import (
    AdaptiveMutator,
    LLMMutator,
    MutationRequest,
    MutationResponse,
)


def test_mutation_request_structure():
    req = MutationRequest(
        code="def sort(arr): return sorted(arr)",
        objective="Optimize sorting",
        context="Previous best: O(n log n)"
    )
    assert req.code is not None
    assert req.objective is not None


def test_mutation_response_structure():
    from sage.evolution.llm_mutator import MutationItem
    resp = MutationResponse(
        mutations=[MutationItem(search="sorted(arr)", replace="arr.sort()", description="in-place")],
        features=[3, 7],
        reasoning="In-place sorting reduces memory",
    )
    assert len(resp.mutations) == 1
    assert len(resp.features) == 2


def test_mutator_builds_prompt():
    mutator = LLMMutator(llm_tier="budget")
    prompt = mutator._build_mutation_prompt("x = 1", "optimize", "")
    assert "SEARCH" in prompt or "Source Code" in prompt
    assert "optimize" in prompt.lower() or "Objective" in prompt


# ── AdaptiveMutator persistence (cgpro 2026-04-27 verdict: keep + persist) ──


def test_adaptive_mutator_state_dict_round_trip():
    """state_dict() / load_state_dict() preserve Beta posteriors + selection counts.

    Drives the bandit through a few record() calls (mix of improved/not),
    serialises, restores into a fresh instance, asserts every field is
    byte-identical. Regression for the bandit::restore_arm class —
    Thompson posteriors must survive process boundaries.
    """
    am = AdaptiveMutator(tiers=["budget", "fast", "reasoner"])
    # Seed real history
    am.record("budget", improved=True)
    am.record("budget", improved=True)
    am.record("budget", improved=False)
    am.record("fast", improved=True)
    am.record("reasoner", improved=False)
    am.record("reasoner", improved=False)
    am._total_selections["budget"] = 7
    am._total_selections["fast"] = 3
    am._total_selections["reasoner"] = 2

    state = am.state_dict()
    am2 = AdaptiveMutator(tiers=["budget", "fast", "reasoner"])
    am2.load_state_dict(state)

    for tier in am.tiers:
        assert am2._successes[tier] == am._successes[tier], f"alpha({tier}) drift"
        assert am2._failures[tier] == am._failures[tier], f"beta({tier}) drift"
        assert am2._total_selections[tier] == am._total_selections[tier], (
            f"selections({tier}) drift"
        )
    assert am2.tiers == am.tiers


def test_adaptive_mutator_save_load_sqlite_round_trip():
    """save() / load() via SQLite preserve state across process boundaries.

    Drives a real bandit, persists to a temp DB, loads into a fresh
    instance, asserts byte-identical state. Mirrors
    ``test_engine_extras_survives_save_load`` for the
    ``TopologyEngine`` extras file.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        am = AdaptiveMutator()  # default 4 tiers
        am.record("budget", improved=True)
        am.record("budget", improved=True)
        am.record("fast", improved=False)
        am.record("mutator", improved=True)
        am.record("reasoner", improved=False)
        am._total_selections["budget"] = 5
        am._total_selections["fast"] = 2
        am._total_selections["mutator"] = 1
        am._total_selections["reasoner"] = 4
        am.save(db_path)

        am2 = AdaptiveMutator()
        # Sanity: fresh instance has uniform priors
        for tier in am2.tiers:
            assert am2._successes[tier] == 1.0
            assert am2._failures[tier] == 1.0
            assert am2._total_selections[tier] == 0

        am2.load(db_path)
        for tier in am.tiers:
            assert am2._successes[tier] == am._successes[tier]
            assert am2._failures[tier] == am._failures[tier]
            assert am2._total_selections[tier] == am._total_selections[tier]
    finally:
        os.unlink(db_path)


def test_adaptive_mutator_load_missing_file_is_cold_start():
    """load() on a missing file is a no-op; state stays at defaults."""
    am = AdaptiveMutator()
    am.load("/tmp/this_file_definitely_does_not_exist_123456789.db")
    for tier in am.tiers:
        assert am._successes[tier] == 1.0
        assert am._failures[tier] == 1.0


def test_adaptive_mutator_load_widens_tier_list():
    """load_state_dict() must adopt tiers from state that aren't yet configured.

    Covers the migration path where the saved state has more tiers than
    the loading instance. Tests should not silently drop history.
    """
    am = AdaptiveMutator(tiers=["budget", "fast"])
    state = {
        "tiers": ["budget", "fast", "experimental"],
        "successes": {"budget": 5.0, "fast": 3.0, "experimental": 2.0},
        "failures": {"budget": 1.0, "fast": 4.0, "experimental": 1.0},
        "total_selections": {"budget": 6, "fast": 7, "experimental": 3},
    }
    am.load_state_dict(state)
    assert "experimental" in am.tiers
    assert am._successes["experimental"] == 2.0
    assert am._failures["experimental"] == 1.0
    assert am._total_selections["experimental"] == 3
