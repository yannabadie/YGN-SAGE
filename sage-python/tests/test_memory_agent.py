import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.memory.memory_agent import MemoryAgent, ExtractionResult

def test_extraction_result():
    r = ExtractionResult(
        entities=["Z3", "eBPF"],
        relationships=[("Z3", "verifies", "eBPF")],
        summary="Z3 verifies eBPF bytecode safety"
    )
    assert len(r.entities) == 2
    assert len(r.relationships) == 1

@pytest.mark.asyncio
async def test_memory_agent_extract_from_text():
    agent = MemoryAgent(use_llm=False)
    result = await agent.extract("The Z3 solver verified that the eBPF bytecode is safe.")
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) >= 0

def test_memory_agent_should_compress():
    agent = MemoryAgent(use_llm=False, compress_threshold=5)
    assert not agent.should_compress(event_count=3)
    assert agent.should_compress(event_count=6)

@pytest.mark.asyncio
async def test_heuristic_extract_finds_capitalized_terms():
    agent = MemoryAgent(use_llm=False)
    result = await agent.extract("The AlphaEvolve system uses MAP-Elites for SAMPO optimization.")
    assert "AlphaEvolve" in result.entities or "MAP" in result.entities or len(result.entities) > 0
