"""Tests for the sage-discover flagship agent."""
import pytest
from unittest.mock import AsyncMock
from discover.researcher import ResearchAgent, Hypothesis, Discovery
from discover.workflow import DiscoverWorkflow, DiscoverConfig
from sage.llm.base import LLMProvider, LLMResponse

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

@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMProvider)
    llm.generate.return_value = LLMResponse(content="- Finding 1\n- Finding 2")
    return llm

@pytest.mark.asyncio
async def test_workflow_exploration(mock_llm):
    config = DiscoverConfig(domain="math", goal="Find primes faster")
    workflow = DiscoverWorkflow(config, llm_provider=mock_llm)
    workflow.main_agent.run = AsyncMock(return_value="- Sieve of Eratosthenes is O(n log log n)\n- Wheel factorization improves constants")

    findings = await workflow.run_exploration()
    assert len(findings) == 2
    assert workflow.phase == "exploring"


@pytest.mark.asyncio
async def test_workflow_hypothesis_generation(mock_llm):
    config = DiscoverConfig(domain="math", goal="test")
    workflow = DiscoverWorkflow(config, llm_provider=mock_llm)
    workflow.main_agent.run = AsyncMock(return_value="- Hypothesis A: faster primes\n- Hypothesis B: better memory")

    hypotheses = await workflow.run_hypothesis_generation(["finding1"])
    assert len(hypotheses) == 2
    assert workflow.researcher.hypotheses[0].statement == "Hypothesis A: faster primes"


@pytest.mark.asyncio
async def test_workflow_full_iteration(mock_llm):
    config = DiscoverConfig(domain="optimization", goal="Find sorting improvement", evolution_generations=1)
    workflow = DiscoverWorkflow(config, llm_provider=mock_llm)

    # Mock exploration
    workflow.run_exploration = AsyncMock(return_value=["quicksort is average O(n log n)", "timsort uses runs"])
    
    # Mock hypothesis generation
    h = workflow.researcher.add_hypothesis("Hybrid approach")
    workflow.run_hypothesis_generation = AsyncMock(return_value=[h])
    
    # Mock evolution to return a high score
    workflow.run_evolution = AsyncMock(return_value=("def hybrid_sort(arr): return sorted(arr)", 0.85))

    discoveries = await workflow.run_iteration()
    assert len(discoveries) == 1
    assert discoveries[0].score == 0.85
    assert workflow.iteration == 1
    assert workflow.phase == "complete"


@pytest.mark.asyncio
async def test_workflow_rejection(mock_llm):
    config = DiscoverConfig(domain="test", goal="test", evolution_generations=1)
    workflow = DiscoverWorkflow(config, llm_provider=mock_llm)

    # Mock exploration
    workflow.run_exploration = AsyncMock(return_value=["fact"])
    
    # Mock hypothesis generation
    h = workflow.researcher.add_hypothesis("Bad hypothesis")
    workflow.run_hypothesis_generation = AsyncMock(return_value=[h])
    
    # Mock evolution to return a low score
    workflow.run_evolution = AsyncMock(return_value=("bad code", 0.2))

    discoveries = await workflow.run_iteration()
    assert len(discoveries) == 0
    assert workflow.researcher.hypotheses[0].status == "rejected"


@pytest.mark.asyncio
async def test_workflow_stats(mock_llm):
    config = DiscoverConfig(domain="test", goal="test")
    workflow = DiscoverWorkflow(config, llm_provider=mock_llm)
    stats = workflow.stats()
    assert stats["iteration"] == 0
    assert stats["phase"] == "idle"
    assert stats["domain"] == "test"
