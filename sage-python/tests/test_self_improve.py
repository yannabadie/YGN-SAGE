"""Tests for the self-improvement loop (sage.evolution.self_improve)."""

import pytest

from sage.evolution.self_improve import ImprovementCycle, SelfImprovementLoop


# ---------------------------------------------------------------------------
# ImprovementCycle dataclass
# ---------------------------------------------------------------------------

def test_improvement_cycle_creation():
    cycle = ImprovementCycle(
        cycle=1,
        before_score=0.6,
        after_score=0.8,
        changes=["fixed prompt", "added retry"],
        improved=True,
    )
    assert cycle.cycle == 1
    assert cycle.before_score == 0.6
    assert cycle.after_score == 0.8
    assert cycle.changes == ["fixed prompt", "added retry"]
    assert cycle.improved is True


def test_improvement_cycle_no_change():
    cycle = ImprovementCycle(
        cycle=2,
        before_score=0.7,
        after_score=0.7,
        changes=[],
        improved=False,
    )
    assert cycle.improved is False
    assert cycle.changes == []


# ---------------------------------------------------------------------------
# SelfImprovementLoop — init
# ---------------------------------------------------------------------------

def test_loop_starts_empty():
    loop = SelfImprovementLoop()
    assert loop.history == []
    assert loop.improvement_rate() == 0.0


# ---------------------------------------------------------------------------
# SelfImprovementLoop — run_cycle with mock functions
# ---------------------------------------------------------------------------

class MockBenchResult:
    """Simulates a benchmark result with pass_rate and results."""
    def __init__(self, pass_rate, results=None):
        self.pass_rate = pass_rate
        self.results = results or []


class MockTaskResult:
    """Simulates a single task result."""
    def __init__(self, passed):
        self.passed = passed


@pytest.mark.asyncio
async def test_run_cycle_improvement():
    """Cycle where evolve_fn actually improves the score."""
    call_count = 0

    async def benchmark_fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockBenchResult(0.5, [MockTaskResult(True), MockTaskResult(False)])
        return MockBenchResult(0.8, [MockTaskResult(True), MockTaskResult(True)])

    async def diagnose_fn(failures):
        return ["weak prompt detected"]

    async def evolve_fn(diagnosis):
        return ["improved prompt template"]

    loop = SelfImprovementLoop()
    cycle = await loop.run_cycle(benchmark_fn, diagnose_fn, evolve_fn)

    assert cycle.cycle == 1
    assert cycle.before_score == 0.5
    assert cycle.after_score == 0.8
    assert cycle.improved is True
    assert cycle.changes == ["improved prompt template"]
    assert len(loop.history) == 1


@pytest.mark.asyncio
async def test_run_cycle_no_improvement():
    """Cycle where score stays the same."""
    async def benchmark_fn():
        return MockBenchResult(0.5, [MockTaskResult(True), MockTaskResult(False)])

    async def diagnose_fn(failures):
        return ["issue found"]

    async def evolve_fn(diagnosis):
        return ["attempted fix"]

    loop = SelfImprovementLoop()
    cycle = await loop.run_cycle(benchmark_fn, diagnose_fn, evolve_fn)

    assert cycle.before_score == 0.5
    assert cycle.after_score == 0.5
    assert cycle.improved is False


@pytest.mark.asyncio
async def test_run_cycle_no_failures():
    """When benchmark has no failures, diagnose and evolve get empty inputs."""
    async def benchmark_fn():
        return MockBenchResult(1.0, [MockTaskResult(True)])

    async def diagnose_fn(failures):
        # Should not be called when there are no failures
        return ["should not happen"]

    async def evolve_fn(diagnosis):
        return ["should not happen"]

    loop = SelfImprovementLoop()
    cycle = await loop.run_cycle(benchmark_fn, diagnose_fn, evolve_fn)

    assert cycle.before_score == 1.0
    assert cycle.after_score == 1.0
    assert cycle.improved is False
    assert cycle.changes == []


@pytest.mark.asyncio
async def test_run_cycle_no_diagnosis():
    """When diagnosis returns empty, evolve should not be called."""
    async def benchmark_fn():
        return MockBenchResult(0.5, [MockTaskResult(False)])

    async def diagnose_fn(failures):
        return []  # No actionable diagnosis

    async def evolve_fn(diagnosis):
        return ["should not happen"]

    loop = SelfImprovementLoop()
    cycle = await loop.run_cycle(benchmark_fn, diagnose_fn, evolve_fn)

    assert cycle.changes == []


# ---------------------------------------------------------------------------
# SelfImprovementLoop — multiple cycles + improvement_rate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_cycles_tracking():
    """Run multiple cycles and verify history tracking."""
    cycle_scores = [(0.5, 0.7), (0.7, 0.7), (0.7, 0.9)]
    cycle_idx = 0

    async def make_benchmark(before, after):
        call_count = 0
        async def fn():
            nonlocal call_count
            call_count += 1
            score = before if call_count == 1 else after
            results = [MockTaskResult(False)] if score < 1.0 else [MockTaskResult(True)]
            return MockBenchResult(score, results)
        return fn

    async def diagnose_fn(failures):
        return ["issue"]

    async def evolve_fn(diagnosis):
        return ["fix"]

    loop = SelfImprovementLoop()

    for before, after in cycle_scores:
        bench_fn = await make_benchmark(before, after)
        await loop.run_cycle(bench_fn, diagnose_fn, evolve_fn)

    assert len(loop.history) == 3
    assert loop.history[0].cycle == 1
    assert loop.history[1].cycle == 2
    assert loop.history[2].cycle == 3


@pytest.mark.asyncio
async def test_improvement_rate_calculation():
    """Test improvement_rate returns correct proportion."""
    loop = SelfImprovementLoop()

    # Manually populate history
    loop.history = [
        ImprovementCycle(1, 0.5, 0.7, ["fix1"], True),
        ImprovementCycle(2, 0.7, 0.7, [], False),
        ImprovementCycle(3, 0.7, 0.9, ["fix2"], True),
        ImprovementCycle(4, 0.9, 0.85, ["regression"], False),
    ]

    rate = loop.improvement_rate()
    assert rate == pytest.approx(0.5)  # 2 out of 4


def test_improvement_rate_empty():
    loop = SelfImprovementLoop()
    assert loop.improvement_rate() == 0.0


def test_improvement_rate_all_improved():
    loop = SelfImprovementLoop()
    loop.history = [
        ImprovementCycle(1, 0.3, 0.5, ["a"], True),
        ImprovementCycle(2, 0.5, 0.8, ["b"], True),
    ]
    assert loop.improvement_rate() == 1.0


def test_improvement_rate_none_improved():
    loop = SelfImprovementLoop()
    loop.history = [
        ImprovementCycle(1, 0.5, 0.5, [], False),
        ImprovementCycle(2, 0.5, 0.3, ["bad"], False),
    ]
    assert loop.improvement_rate() == 0.0


# ---------------------------------------------------------------------------
# SelfImprovementLoop — benchmark_fn without pass_rate attribute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_cycle_benchmark_without_pass_rate():
    """Handles benchmark results that lack pass_rate/results attributes."""
    async def benchmark_fn():
        return object()  # No pass_rate, no results

    async def diagnose_fn(failures):
        return []

    async def evolve_fn(diagnosis):
        return []

    loop = SelfImprovementLoop()
    cycle = await loop.run_cycle(benchmark_fn, diagnose_fn, evolve_fn)

    assert cycle.before_score == 0.0
    assert cycle.after_score == 0.0
    assert cycle.improved is False
