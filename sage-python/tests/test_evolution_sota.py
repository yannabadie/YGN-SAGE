import pytest
from unittest.mock import AsyncMock

from sage.evolution.population import Population, Individual
from sage.evolution.mutator import Mutator, Mutation
from sage.evolution.llm_mutator import LLMMutator
from sage.llm.base import LLMProvider, LLMResponse

@pytest.fixture
def population():
    return Population(feature_dims=2, bins_per_dim=10)

def test_population_add_and_best(population):
    # Features mapped to bins: e.g. (5, 5)
    ind1 = Individual(code="print('v1')", score=0.5, features=(5, 5))
    ind2 = Individual(code="print('v2')", score=0.8, features=(5, 5))  # Same bin, better score
    ind3 = Individual(code="print('v3')", score=0.6, features=(2, 2))  # Different bin
    
    assert population.add(ind1) is True
    assert population.add(ind2) is True  # Replaces ind1
    assert population.add(ind3) is True
    
    assert population.size() == 2
    best = population.best(1)
    assert best[0].code == "print('v2')"

def test_mutator_parse_diff():
    mutator = Mutator()
    diff = """Some intro text
<<<SEARCH
old code
===
new code
>>>REPLACE: fixed bug
"""
    mutations = mutator.parse_diff(diff)
    assert len(mutations) == 1
    assert mutations[0].search == "old code"
    assert mutations[0].replace == "new code"

@pytest.mark.asyncio
async def test_llm_mutator():
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = LLMMutator_MockResponse()
    
    llm_mutator = LLMMutator(llm=mock_llm)
    code = "def add(a, b):\n    return a - b"
    
    new_code, features = await llm_mutator.mutate(code, "Fix the addition bug")
    assert "return a + b" in new_code
    assert features == (2, 8)  # Parsed from mock response

class LLMMutator_MockResponse:
    content = """Here is the fix:
<<<SEARCH
    return a - b
===
    return a + b
>>>REPLACE: fixed minus to plus

FEATURES: Complexity=2, Creativity=8
"""
    tool_calls = []
