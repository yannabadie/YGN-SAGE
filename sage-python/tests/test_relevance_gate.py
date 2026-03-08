"""Tests for CRAG-style RelevanceGate."""

from sage.memory.relevance_gate import RelevanceGate


def test_relevance_gate_passes_relevant():
    """Python coding context should be relevant to a Python task."""
    gate = RelevanceGate(threshold=0.3)
    task = "Write a Python function that sorts a list using quicksort algorithm"
    context = (
        "Python quicksort implementation: partition the list around a pivot, "
        "recursively sort sublists. The function uses list comprehensions "
        "for concise partitioning."
    )
    assert gate.is_relevant(task, context) is True


def test_relevance_gate_rejects_irrelevant():
    """Cooking context should be irrelevant to a Python task."""
    gate = RelevanceGate(threshold=0.3)
    task = "Write a Python function that sorts a list using quicksort algorithm"
    context = (
        "To make a perfect sourdough bread, mix flour and water to create "
        "the starter. Let it ferment for 24 hours at room temperature. "
        "Knead the dough and bake at 450 degrees."
    )
    assert gate.is_relevant(task, context) is False


def test_relevance_gate_rejects_empty():
    """Empty context should always be rejected."""
    gate = RelevanceGate(threshold=0.3)
    task = "Write a Python function"
    assert gate.is_relevant(task, "") is False
    assert gate.is_relevant(task, "   ") is False
    assert gate.is_relevant(task, None) is False  # type: ignore[arg-type]


def test_relevance_gate_high_threshold_strict():
    """A high threshold should reject context that a low threshold accepts."""
    task = "Write a Python function that sorts a list"
    context = "Python list sorting is done with the sort method or sorted builtin"

    lenient = RelevanceGate(threshold=0.1)
    strict = RelevanceGate(threshold=0.9)

    # Lenient should pass, strict should reject
    assert lenient.is_relevant(task, context) is True
    assert strict.is_relevant(task, context) is False


def test_relevance_gate_score_between_0_and_1():
    """Score should always be normalized in [0.0, 1.0]."""
    gate = RelevanceGate()
    task = "Implement binary search in Python for a sorted array"
    context = "Binary search divides the sorted array in half each iteration"

    score = gate.score(task, context)
    assert 0.0 <= score <= 1.0

    # Empty inputs
    assert gate.score("", context) == 0.0
    assert gate.score(task, "") == 0.0
    assert gate.score("", "") == 0.0
