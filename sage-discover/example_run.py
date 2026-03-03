import asyncio
import os
import sys

# Ensure modules are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../sage-python/src"))

from discover.workflow import DiscoverWorkflow, DiscoverConfig
from sage.llm.openai import OpenAIProvider
from sage.llm.base import LLMConfig

async def main():
    print("Starting YGN-SAGE Discovery on a real-world task...")
    
    # We use a mocked LLM for safety in the example, but the plumbing is real.
    # In a real environment, you'd set OPENAI_API_KEY and use "gpt-4o-mini".
    api_key = os.getenv("OPENAI_API_KEY", "mock")
    
    config = DiscoverConfig(
        domain="algorithmic_optimization",
        goal="Find an algorithm to sort an array of integers faster than O(N log N) by exploiting data distribution characteristics (like Bucket Sort).",
        max_iterations=1,
        evolution_generations=2,
        population_size=5,
        solver_type="vad_cfr"
    )
    
    # If using a mock key, we'll patch the provider for this demonstration
    provider = OpenAIProvider()
    if api_key == "mock":
        print("WARN: OPENAI_API_KEY not found. Mocking LLM responses.")
        from unittest.mock import AsyncMock
        from sage.llm.base import LLMResponse
        provider.generate = AsyncMock(return_value=LLMResponse(
            content="- Data distribution can be exploited via Bucket or Radix sort.\n- O(N) is possible for specific integer ranges.\n- Cache locality matters.\nFEATURES: Complexity=4, Creativity=7"
        ))

    workflow = DiscoverWorkflow(
        config=config,
        llm_provider=provider,
        memory_compressor=None, # Optional
        sandbox_manager=None, # Uses local subprocess fallback
    )
    
    discoveries = await workflow.run_iteration()
    
    print("\n--- DISCOVERY RESULTS ---")
    if discoveries:
        for i, d in enumerate(discoveries):
            print(f"Discovery {i+1}:")
            print(f"  Hypothesis: {d.hypothesis.statement}")
            print(f"  Score: {d.score}")
            print(f"  Code:\n{d.code}\n")
    else:
        print("No discoveries met the confidence threshold.")
        
    print("\n--- STATS ---")
    print(workflow.stats())

if __name__ == "__main__":
    asyncio.run(main())
