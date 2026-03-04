import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Load env
load_dotenv()

# Add paths
sys.path.append(os.path.join(os.getcwd(), "sage-discover/src"))
sys.path.append(os.path.join(os.getcwd(), "sage-python/src"))

from discover.workflow import DiscoverWorkflow, DiscoverConfig
from discover.researcher import Hypothesis
from sage.llm.google import GoogleProvider

async def test_score_signal():
    print("🚀 Validating FINAL Dynamic Score Signal (Mars 2026)")
    
    config = DiscoverConfig(
        domain="hardware_optimized_sorting",
        goal="Optimize speed.",
        max_iterations=1,
        evolution_generations=1,
        population_size=2
    )
    
    provider = GoogleProvider()
    workflow = DiscoverWorkflow(config=config, llm_provider=provider)
    
    # Manually create hypothesis to bypass Pro generation issues in test
    h = Hypothesis(id="H_TEST", statement="Optimize H96 using block-size tuning.", status="proposed", confidence=0.5, evidence=[])
    
    print(f"\n🧬 Running evolution for: {h.statement}...")
    code, score = await workflow.run_evolution(h)
    
    print(f"\n📊 FINAL Score (relative to H96): {score}")
    if score >= 0.9:
        print("\n✅ SUCCESS: Signal is 1.0 (H96 Baseline). Ready for increasing evolution!")
    else:
        print(f"\n❌ FAILURE: Score is {score}. Check sandbox.")

if __name__ == "__main__":
    asyncio.run(test_score_signal())
