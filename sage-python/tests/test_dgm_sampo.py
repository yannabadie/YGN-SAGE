import pytest
import asyncio
from unittest.mock import Mock

from sage.evolution.engine import EvolutionEngine, EvolutionConfig
from sage.evolution.population import Individual
from sage.evolution.evaluator import EvalResult

@pytest.mark.asyncio
async def test_dgm_self_modification():
    config = EvolutionConfig(population_size=10, mutations_per_generation=2, hard_warm_start_threshold=1)
    
    # Mock evaluator that always returns 1.0 (improving over default 0.0)
    mock_evaluator = Mock()
    mock_evaluator.evaluate = Mock()
    # Need to return an awaitable for async mock
    async def mock_eval(*args, **kwargs):
        return EvalResult(score=1.0, passed=True, stage="test")
    mock_evaluator.evaluate.side_effect = mock_eval
    
    engine = EvolutionEngine(config=config, evaluator=mock_evaluator)
    
    # Force the initial policy to definitely choose action 3 (Mutate clip_epsilon)
    engine._dgm_solver._policy = [0.0, 0.0, 0.0, 1.0, 0.0]
    
    initial_epsilon = engine._dgm_solver.clip_epsilon
    
    # Seed population
    engine.seed([Individual(code="print('parent')", score=0.5, features=(0,), generation=0)])
    
    async def fake_mutate(code, dgm_context=None):
        return ("print('child')", (1,))
        
    await engine.evolve_step(fake_mutate)
    
    # Because action 3 was forced, clip_epsilon should be reduced
    assert engine._dgm_solver.clip_epsilon < initial_epsilon
    assert engine.total_mutations > 0
    
    # The batch needs 5 trajectories to update the solver, so let's run a few more
    for _ in range(4):
        await engine.evolve_step(fake_mutate)
        
    # Check if trajectories were processed (list should be empty or reset)
    assert len(engine._trajectories) == 0


@pytest.mark.asyncio
async def test_dgm_context_passed_to_mutate_fn():
    """DGM action context must be passed to mutate_fn."""
    config = EvolutionConfig(
        population_size=10, mutations_per_generation=2,
        hard_warm_start_threshold=1,
    )

    mock_evaluator = Mock()
    async def mock_eval(*args, **kwargs):
        return EvalResult(score=1.0, passed=True, stage="test")
    mock_evaluator.evaluate.side_effect = mock_eval

    engine = EvolutionEngine(config=config, evaluator=mock_evaluator)
    engine.seed([Individual(code="print('parent')", score=0.5, features=(0,), generation=0)])

    received_contexts = []

    async def fake_mutate(code, dgm_context=None):
        received_contexts.append(dgm_context)
        return ("print('child')", (1,))

    await engine.evolve_step(fake_mutate)

    assert len(received_contexts) > 0
    ctx = received_contexts[0]
    assert "action" in ctx
    assert "description" in ctx
    assert isinstance(ctx["description"], str)
    assert "parent_score" in ctx
    assert "generation" in ctx

