"""Tests for the strategy engine."""
import pytest
import numpy as np
from sage.strategy.solvers import RegretMatcher, SAMPOSolver
from sage.strategy.allocator import ResourceAllocator
from sage.strategy.engine import StrategyEngine
from sage.strategy.metacognition import ComplexityRouter


# --- RegretMatcher Tests ---

def test_regret_matcher_initial_uniform():
    rm = RegretMatcher(3)
    strategy = rm.get_strategy()
    assert len(strategy) == 3
    assert abs(sum(strategy) - 1.0) < 1e-6
    assert all(abs(s - 1/3) < 1e-6 for s in strategy)


def test_regret_matcher_update():
    rm = RegretMatcher(3)
    # Action 0 is best
    rm.update([1.0, 0.5, 0.2], chosen_action=1)
    strategy = rm.get_strategy()
    # Action 0 should have highest weight (regret for not choosing it)
    assert strategy[0] > strategy[1]


def test_regret_matcher_converges():
    rm = RegretMatcher(2)
    # Alternate choosing suboptimal action to build regret for action 0
    for i in range(100):
        rm.update([1.0, 0.2], chosen_action=i % 2)
    avg = rm.average_strategy()
    # Action 0 has more regret when not chosen, so it should dominate
    assert avg[0] > avg[1]


# --- SAMPOSolver Tests ---

def test_sampo_initial_uniform():
    sampo = SAMPOSolver(3)
    strategy = sampo.get_strategy()
    assert abs(sum(strategy) - 1.0) < 1e-6


def test_sampo_update_favors_better():
    sampo = SAMPOSolver(3)
    # Action 0 has highest payoff
    for _ in range(20):
        sampo.update([{"actions": [0, 1], "rewards": [1.0, 0.0]}])
    strategy = sampo.get_strategy()
    assert strategy[0] > strategy[1]


def test_sampo_numerical_stability_large_rewards():
    sampo = SAMPOSolver(2, clip_epsilon=0.1)
    for _ in range(50):
        sampo.update([{"actions": [0, 1], "rewards": [1e9, -1e9]}])
    strategy = sampo.get_strategy()
    assert np.isfinite(strategy).all()
    assert abs(float(np.sum(strategy)) - 1.0) < 1e-9
    assert np.all(strategy >= 0.0)


def test_sampo_adaptive_lr_reduces_after_high_variance():
    sampo = SAMPOSolver(2, base_lr=0.05, min_lr=0.001, max_lr=0.2, lr_decay=0.999)
    low_var_traj = [{"actions": [0, 1], "rewards": [0.51, 0.49]}]
    high_var_traj = [{"actions": [0, 1], "rewards": [1000.0, -1000.0]}]

    sampo.update(low_var_traj)
    lr_low = sampo.stats()["learning_rate"]

    sampo.update(high_var_traj)
    lr_high = sampo.stats()["learning_rate"]

    assert lr_high < lr_low


def test_sampo_ignores_malformed_trajectories_without_crash():
    sampo = SAMPOSolver(3)
    before = sampo.get_strategy().copy()
    sampo.update(
        [
            {"actions": [0, 1], "rewards": [1.0]},  # length mismatch
            {"actions": [99], "rewards": [0.5]},  # invalid action index
            {"actions": [0], "rewards": [float("nan")]},  # non-finite reward
        ]
    )
    after = sampo.get_strategy()
    assert np.allclose(before, after)


def test_sampo_mixed_precision_gradient_scaling_stays_finite():
    sampo = SAMPOSolver(
        2,
        clip_epsilon=0.1,
        mixed_precision=True,
        grad_scale_init=1024.0,
        grad_scale_growth_interval=1,
    )
    for _ in range(20):
        sampo.update([{"actions": [0, 1], "rewards": [1e4, -1e4]}])
    strategy = sampo.get_strategy()
    stats = sampo.stats()
    assert np.isfinite(strategy).all()
    assert abs(float(np.sum(strategy)) - 1.0) < 1e-9
    assert stats["grad_scale"] >= 1.0


# --- ResourceAllocator Tests ---

def test_allocator_proportional():
    alloc = ResourceAllocator(total_tokens=1000, total_agents=4, total_time=100.0)
    allocations = alloc.allocate(
        ["strategy_a", "strategy_b"],
        [0.75, 0.25],
    )
    assert len(allocations) == 2
    assert allocations[0].tokens == 750
    assert allocations[1].tokens == 250
    assert abs(allocations[0].time_budget - 75.0) < 0.01


def test_allocator_zero_weights():
    alloc = ResourceAllocator(total_tokens=100)
    allocations = alloc.allocate(["a", "b"], [0.0, 0.0])
    # Should fall back to uniform
    assert allocations[0].tokens == 50


# --- StrategyEngine Tests ---

def test_engine_initial_allocations():
    engine = StrategyEngine(["explore", "exploit", "evolve"])
    allocs = engine.get_allocations()
    assert len(allocs) == 3
    # Initially uniform
    assert all(abs(a.weight - 1/3) < 0.01 for a in allocs)


def test_engine_report_outcome():
    engine = StrategyEngine(["a", "b"], solver_type="regret")
    engine.report_outcome(0, [1.0, 0.5])
    engine.report_outcome(0, [1.0, 0.3])
    stats = engine.stats()
    assert stats["rounds"] == 2


def test_engine_sampo_solver():
    engine = StrategyEngine(["a", "b", "c"], solver_type="sampo")
    for _ in range(20):
        engine.report_outcome(0, [1.0, 0.3, 0.1])
    strategy = engine.get_strategy()
    assert strategy[0] > strategy[1]


def test_engine_stats_dominant():
    engine = StrategyEngine(["alpha", "beta"])
    for _ in range(10):
        engine.report_outcome(0, [1.0, 0.0])
    stats = engine.stats()
    assert stats["dominant"] == "alpha"


def test_engine_invalid_solver():
    with pytest.raises(ValueError):
        StrategyEngine(["a"], solver_type="invalid")


# --- ComplexityRouter heuristic routing tests ---

def test_routing_simple_factual_to_s1():
    """Simple factual question should route to S1 (fast/cheap)."""
    router = ComplexityRouter()
    profile = router.assess_complexity("What is the capital of France?")
    decision = router.route(profile)
    assert decision.system == 1, (
        f"Expected S1 for simple factual, got S{decision.system} "
        f"(c={profile.complexity:.2f}, u={profile.uncertainty:.2f})"
    )


def test_routing_code_generation_to_s2():
    """Tasks with multiple complex keywords should route to S2+."""
    router = ComplexityRouter()
    # "implement" + "algorithm" = 2 hits → 0.67
    profile = router.assess_complexity("Implement an algorithm to optimize sorting")
    decision = router.route(profile)
    assert decision.system >= 2, (
        f"Expected S2+ for complex code generation, got S{decision.system} "
        f"(c={profile.complexity:.2f}, u={profile.uncertainty:.2f})"
    )


def test_routing_simple_code_to_s1():
    """Simple tasks with no complex keywords route to S1 (degraded heuristic)."""
    router = ComplexityRouter()
    profile = router.assess_complexity("Write a hello world function")
    decision = router.route(profile)
    assert decision.system == 1, (
        f"Expected S1 for simple task, got S{decision.system} "
        f"(c={profile.complexity:.2f}, u={profile.uncertainty:.2f})"
    )


def test_routing_complex_debug_to_s3():
    """Complex debug task with race condition and long description should route to S3."""
    router = ComplexityRouter()
    task = (
        "Debug the race condition in the distributed worker pool that causes "
        "intermittent deadlocks under high load. The system uses a shared mutex "
        "across multiple threads and the lock ordering appears inconsistent. "
        "Workers sometimes hang indefinitely when processing concurrent batch "
        "jobs, and the error manifests only under production-level traffic with "
        "at least fifty parallel connections hitting the queue simultaneously."
    )
    profile = router.assess_complexity(task)
    decision = router.route(profile)
    assert decision.system == 3, (
        f"Expected S3 for complex debug, got S{decision.system} "
        f"(c={profile.complexity:.2f}, u={profile.uncertainty:.2f})"
    )


def test_routing_fibonacci_to_s1_or_s2():
    """Simple algorithm request should route to S1 or S2, never S3."""
    router = ComplexityRouter()
    profile = router.assess_complexity(
        "Write a function that returns the nth Fibonacci number"
    )
    decision = router.route(profile)
    assert decision.system in (1, 2), (
        f"Expected S1 or S2 for simple algorithm, got S{decision.system} "
        f"(c={profile.complexity:.2f}, u={profile.uncertainty:.2f})"
    )


def test_routing_long_task_complexity_boost():
    """Longer task descriptions should have higher complexity than short ones."""
    router = ComplexityRouter()
    short_task = "Add two numbers"
    long_task = " ".join(["Add two numbers and verify the result is correct."] * 12)
    short_profile = router.assess_complexity(short_task)
    long_profile = router.assess_complexity(long_task)
    assert long_profile.complexity > short_profile.complexity, (
        f"Long task complexity ({long_profile.complexity:.2f}) should exceed "
        f"short task ({short_profile.complexity:.2f})"
    )
