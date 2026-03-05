import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.evolution.llm_mutator import LLMMutator, MutationRequest, MutationResponse


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
