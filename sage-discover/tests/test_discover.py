"""Tests for the sage-discover flagship agent."""
import pytest
from discover.researcher import ResearchAgent, Hypothesis
from discover.workflow import DiscoverWorkflow, DiscoverConfig


# --- ResearchAgent Tests ---

def test_researcher_add_hypothesis():
    agent = ResearchAgent(domain="math", goal="Find a better sorting algorithm")
    h = agent.add_hypothesis("Binary insertion sort is optimal for small arrays")
    assert h.id == "H1"
    assert h.status == "proposed"
    assert len(agent.hypotheses) == 1


def test_researcher_confirm_hypothesis():
    agent = ResearchAgent(domain="math", goal="test")
    h = agent.add_hypothesis("Test hypothesis")
    discovery = agent.confirm_hypothesis(h.id, code="def sort(): pass", score=0.95)
    assert discovery.score == 0.95
    assert h.status == "confirmed"
    assert len(agent.discoveries) == 1


def test_researcher_reject_hypothesis():
    agent = ResearchAgent(domain="math", goal="test")
    h = agent.add_hypothesis("Bad hypothesis")
    agent.reject_hypothesis(h.id, reason="No evidence")
    assert h.status == "rejected"


def test_researcher_pending():
    agent = ResearchAgent(domain="math", goal="test")
    agent.add_hypothesis("A")
    agent.add_hypothesis("B")
    h = agent.add_hypothesis("C")
    agent.reject_hypothesis(h.id)
    assert len(agent.pending_hypotheses) == 2


def test_researcher_stats():
    agent = ResearchAgent(domain="optimization", goal="test")
    agent.add_hypothesis("A")
    h2 = agent.add_hypothesis("B")
    agent.confirm_hypothesis(h2.id, code="x=1", score=0.8)
    stats = agent.stats()
    assert stats["total_hypotheses"] == 2
    assert stats["confirmed"] == 1
    assert stats["best_score"] == 0.8


# --- Workflow Tests ---

@pytest.mark.asyncio
async def test_workflow_exploration():
    config = DiscoverConfig(domain="math", goal="Find primes faster")
    workflow = DiscoverWorkflow(config)

    async def mock_explore(goal: str) -> list[str]:
        return ["Sieve of Eratosthenes is O(n log log n)", "Wheel factorization improves constants"]

    findings = await workflow.run_exploration(mock_explore)
    assert len(findings) == 2
    assert workflow.phase == "exploring"


@pytest.mark.asyncio
async def test_workflow_hypothesis_generation():
    config = DiscoverConfig(domain="math", goal="test")
    workflow = DiscoverWorkflow(config)

    async def mock_generate(goal: str, findings: list[str]) -> list[str]:
        return ["Hypothesis A: faster primes", "Hypothesis B: better memory"]

    hypotheses = await workflow.run_hypothesis_generation(mock_generate, ["finding1"])
    assert len(hypotheses) == 2
    assert workflow.researcher.hypotheses[0].statement == "Hypothesis A: faster primes"


@pytest.mark.asyncio
async def test_workflow_full_iteration():
    config = DiscoverConfig(domain="optimization", goal="Find sorting improvement")
    workflow = DiscoverWorkflow(config)

    async def explore(goal: str) -> list[str]:
        return ["quicksort is average O(n log n)", "timsort uses runs"]

    async def hypothesize(goal: str, findings: list[str]) -> list[str]:
        return ["Hybrid approach combining quicksort and timsort"]

    async def evolve(h: Hypothesis) -> tuple[str, float]:
        return "def hybrid_sort(arr): return sorted(arr)", 0.85

    async def evaluate(code: str) -> float:
        return 0.9

    discoveries = await workflow.run_iteration(explore, hypothesize, evolve, evaluate)
    assert len(discoveries) == 1
    assert discoveries[0].score == 0.9
    assert workflow.iteration == 1
    assert workflow.phase == "complete"


@pytest.mark.asyncio
async def test_workflow_rejection():
    config = DiscoverConfig(domain="test", goal="test")
    workflow = DiscoverWorkflow(config)

    async def explore(goal: str) -> list[str]:
        return ["fact"]

    async def hypothesize(goal: str, findings: list[str]) -> list[str]:
        return ["Bad hypothesis"]

    async def evolve(h: Hypothesis) -> tuple[str, float]:
        return "bad code", 0.1

    async def evaluate(code: str) -> float:
        return 0.2  # Below threshold

    discoveries = await workflow.run_iteration(explore, hypothesize, evolve, evaluate)
    assert len(discoveries) == 0
    assert workflow.researcher.hypotheses[0].status == "rejected"


@pytest.mark.asyncio
async def test_workflow_stats():
    config = DiscoverConfig(domain="test", goal="test")
    workflow = DiscoverWorkflow(config)
    stats = workflow.stats()
    assert stats["iteration"] == 0
    assert stats["phase"] == "idle"
    assert stats["domain"] == "test"
