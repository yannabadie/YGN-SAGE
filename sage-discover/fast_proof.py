import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure modules are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../sage-python/src"))

from discover.workflow import DiscoverWorkflow, DiscoverConfig
from sage.llm.codex import CodexProvider
from sage.llm.base import LLMConfig

async def main():
    print("⚡ Starting REFACTORED YGN-SAGE Sprint Run (SOTA Proof)...")
    print("🧠 Engine: Codex CLI 5.3 (SOTA 2026)")

    # SOTA configuration - Forced precision
    config = DiscoverConfig(
        domain="sorting",
        goal="Produce a functional Python function 'solution(arr)' that implements an optimized sorting logic.",
        max_iterations=1,
        evolution_generations=1, 
        population_size=2,       
        solver_type="vad_cfr"
    )
    
    # Codex configuration
    llm_config = LLMConfig(
        provider="codex",
        model="codex", 
        max_tokens=4096,
        temperature=0.3
    )
    
    provider = CodexProvider()
    
    workflow = DiscoverWorkflow(
        config=config,
        llm_provider=provider,
    )
    
    # Propagate SOTA model to main agent
    workflow.main_agent.config.llm = llm_config
    
    print(f"📡 SOTA Run (Code Mode): {config.goal}")
    
    try:
        discoveries = await workflow.run_iteration()
        print("\n✅ Sprint Run Complete!")
        
        if os.path.exists('latest_discovery.json'):
            with open('latest_discovery.json', 'r') as f:
                data = json.load(f)
                print(f"Final Phase: {data['phase']}")
                print(f"Total Hypotheses: {data['total_hypotheses']}")
                print(f"Confirmed Discoveries: {data['confirmed']}")
                if data['confirmed'] > 0:
                    print("🏆 SUCCESS: Breakthrough achieved and verified!")
        else:
            print("⚠️ Checkpoint file not found.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
