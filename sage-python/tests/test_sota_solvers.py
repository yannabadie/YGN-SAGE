import pytest
import numpy as np
from sage.strategy.solvers import VolatilityAdaptiveSolver, SHORPSROSolver
from sage.strategy.engine import StrategyEngine

def test_vad_cfr_initialization():
    n_actions = 3
    solver = VolatilityAdaptiveSolver(n_actions)
    assert solver.n_actions == n_actions
    assert np.allclose(solver.get_strategy(), np.ones(n_actions) / n_actions)

def test_vad_cfr_volatility_tracking():
    solver = VolatilityAdaptiveSolver(n_actions=2)
    # High volatility update
    solver.update([10.0, 0.0], 1)
    assert solver._ewma_volatility > 0
    
    # Check that discounting is happening
    v1 = solver._cumulative_regret.copy()
    solver.update([0.0, 0.0], 0)
    # Regrets should be discounted
    assert np.all(np.abs(solver._cumulative_regret) <= np.abs(v1) + 1e-12)

def test_vad_cfr_warm_start():
    # Set threshold low for testing
    solver = VolatilityAdaptiveSolver(n_actions=2, warm_start_threshold=5)
    
    for i in range(4):
        solver.update([1.0, 0.0], 1)
        assert np.sum(solver._cumulative_policy) == 0.0
        
    solver.update([1.0, 0.0], 1) # 5th iteration
    assert np.sum(solver._cumulative_policy) > 0.0

def test_shor_psro_annealing():
    solver = SHORPSROSolver(n_actions=3, total_iters=10)
    
    # Start params
    b0, t0, d0, m0 = solver._get_params()
    assert b0 == pytest.approx(0.30)
    assert t0 == pytest.approx(0.50)
    assert m0 == pytest.approx(0.50)
    
    # Update
    for _ in range(10):
        solver.update([1.0, 0.5, 0.0], 0)
        
    # End params
    b1, t1, d1, m1 = solver._get_params()
    assert b1 == pytest.approx(0.05)
    assert t1 == pytest.approx(0.01)
    assert d1 == pytest.approx(0.001)
    assert m1 == pytest.approx(0.50)

def test_shor_psro_strategy_blending():
    solver = SHORPSROSolver(n_actions=2)
    payoffs = [10.0, 0.0]
    
    strategy = solver.get_strategy(payoffs)
    # Strategy should be biased towards the high payoff action (Softmax component)
    assert strategy[0] > strategy[1]

def test_strategy_engine_integration():
    engine = StrategyEngine(["a", "b"], solver_type="vad_cfr", warm_start_threshold=2)
    
    # Initial uniform
    assert np.allclose(engine.get_strategy(), [0.5, 0.5])
    
    # Report outcomes
    engine.report_outcome(1, [1.0, 0.0])
    engine.report_outcome(1, [1.0, 0.0])
    
    # Strategy should shift towards "a" (index 0) which has higher utility
    strategy = engine.get_strategy()
    assert strategy[0] > strategy[1]
    
    # Check allocations
    allocs = engine.get_allocations()
    assert allocs[0].strategy_name == "a"
    assert allocs[0].weight > 0.5
