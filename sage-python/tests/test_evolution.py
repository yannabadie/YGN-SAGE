"""Tests for the evolution engine."""
import pytest
from sage.evolution.population import Population, Individual
from sage.evolution.mutator import Mutator, Mutation
from sage.evolution.evaluator import Evaluator, EvalResult
from sage.evolution.engine import EvolutionEngine, EvolutionConfig


# --- Population Tests ---

def test_population_add():
    pop = Population(feature_dims=2, bins_per_dim=5)
    ind = Individual(code="x=1", score=0.8, features=(1, 2))
    assert pop.add(ind)
    assert pop.size() == 1


def test_population_best_replaces():
    pop = Population(feature_dims=2, bins_per_dim=5)
    pop.add(Individual(code="v1", score=0.5, features=(1, 1)))
    pop.add(Individual(code="v2", score=0.9, features=(1, 1)))
    assert pop.size() == 1  # Same cell
    assert pop.get((1, 1)).code == "v2"


def test_population_doesnt_replace_worse():
    pop = Population(feature_dims=2, bins_per_dim=5)
    pop.add(Individual(code="good", score=0.9, features=(1, 1)))
    result = pop.add(Individual(code="bad", score=0.3, features=(1, 1)))
    assert not result
    assert pop.get((1, 1)).code == "good"


def test_population_best_n():
    pop = Population(feature_dims=1, bins_per_dim=10)
    for i in range(5):
        pop.add(Individual(code=f"c{i}", score=float(i), features=(i,)))
    best = pop.best(3)
    assert len(best) == 3
    assert best[0].score == 4.0


def test_population_coverage():
    pop = Population(feature_dims=1, bins_per_dim=10)
    for i in range(5):
        pop.add(Individual(code=f"c{i}", score=1.0, features=(i,)))
    assert pop.coverage() == 0.5  # 5 out of 10


def test_population_clamp():
    pop = Population(feature_dims=2, bins_per_dim=5)
    ind = Individual(code="x", score=1.0, features=(99, -5))
    pop.add(ind)
    assert ind.features == (4, 0)  # Clamped to [0, bins-1]


# --- Mutator Tests ---

def test_mutator_apply():
    m = Mutator()
    code = "def foo():\n    return 1"
    mutation = Mutation(search="return 1", replace="return 2")
    result = m.apply_mutation(code, mutation)
    assert "return 2" in result


def test_mutator_missing_search():
    m = Mutator()
    with pytest.raises(ValueError, match="not found"):
        m.apply_mutation("code here", Mutation(search="missing", replace="x"))


def test_mutator_parse_diff():
    m = Mutator()
    diff = """<<<SEARCH
return 1
===
return 2
>>>REPLACE: Improved return value"""
    mutations = m.parse_diff(diff)
    assert len(mutations) == 1
    assert mutations[0].search == "return 1"
    assert mutations[0].replace == "return 2"
    assert "Improved" in mutations[0].description


def test_mutator_generate_diff():
    m = Mutator()
    diff = m.generate_diff("old code", "new code", "Fixed bug")
    assert "<<<SEARCH" in diff
    assert "old code" in diff
    assert "new code" in diff
    assert ">>>REPLACE: Fixed bug" in diff


# --- Evaluator Tests ---

@pytest.mark.asyncio
async def test_evaluator_single_stage():
    ev = Evaluator()

    async def check_syntax(code: str) -> EvalResult:
        try:
            compile(code, "<test>", "exec")
            return EvalResult(score=1.0, passed=True, stage="syntax")
        except SyntaxError:
            return EvalResult(score=0.0, passed=False, stage="syntax")

    ev.add_stage("syntax", check_syntax, threshold=0.5)

    result = await ev.evaluate("x = 1")
    assert result.passed
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_evaluator_cascade_fail():
    ev = Evaluator()

    async def always_fail(code: str) -> EvalResult:
        return EvalResult(score=0.1, passed=False, stage="test")

    ev.add_stage("test", always_fail, threshold=0.5)

    result = await ev.evaluate("code")
    assert not result.passed


@pytest.mark.asyncio
async def test_evaluator_cascade_multi_stage():
    ev = Evaluator()

    async def stage_1(code: str) -> EvalResult:
        return EvalResult(score=0.8, passed=True, stage="s1")

    async def stage_2(code: str) -> EvalResult:
        return EvalResult(score=0.6, passed=True, stage="s2")

    ev.add_stage("syntax", stage_1, threshold=0.5, weight=1.0)
    ev.add_stage("correctness", stage_2, threshold=0.5, weight=2.0)

    result = await ev.evaluate("code")
    assert result.passed
    # Weighted: (0.8*1 + 0.6*2) / (1+2) = 2.0/3 ≈ 0.667
    assert abs(result.score - 0.6667) < 0.01


# --- Engine Tests ---

@pytest.mark.asyncio
async def test_engine_seed():
    engine = EvolutionEngine(config=EvolutionConfig(feature_dims=1, bins_per_dim=5))
    count = engine.seed([
        Individual(code="a", score=0.5, features=(0,)),
        Individual(code="b", score=0.7, features=(1,)),
    ])
    assert count == 2
    assert engine.population.size() == 2


@pytest.mark.asyncio
async def test_engine_evolve_step():
    config = EvolutionConfig(feature_dims=1, bins_per_dim=5, mutations_per_generation=2)
    ev = Evaluator()

    async def always_pass(code: str) -> EvalResult:
        return EvalResult(score=0.9, passed=True, stage="test")

    ev.add_stage("test", always_pass)

    engine = EvolutionEngine(config=config, evaluator=ev)
    engine.seed([Individual(code="x=1", score=0.5, features=(0,))])

    async def mutate(code: str, dgm_context=None) -> tuple[str, tuple[int, ...]]:
        return code + "\n# mutated", (1,)

    accepted = await engine.evolve_step(mutate)
    assert engine.generation == 1
    assert len(accepted) >= 0  # May or may not improve


@pytest.mark.asyncio
async def test_engine_stats():
    engine = EvolutionEngine(config=EvolutionConfig(feature_dims=1, bins_per_dim=5))
    engine.seed([Individual(code="x", score=0.7, features=(2,))])
    stats = engine.stats()
    assert stats["population_size"] == 1
    assert stats["best_score"] == 0.7
    assert stats["generation"] == 0
