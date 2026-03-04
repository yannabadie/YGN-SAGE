import asyncio
import os
import sys
import json
import time
from dotenv import load_dotenv

# Load env
load_dotenv()

# Add paths
sys.path.append(os.path.join(os.getcwd(), "sage-discover/src"))
sys.path.append(os.path.join(os.getcwd(), "sage-python/src"))

from discover.workflow import DiscoverWorkflow, DiscoverConfig
from sage.llm.google import GoogleProvider
from sage.llm.router import ModelRouter

async def main():
    print("🚀 YGN-SAGE: Starting LONG-TERM Autonomous Discovery (Mars 2026)")
    print(f"🧠 Routing: {ModelRouter.MODELS['reasoner']} (Strategy) | {ModelRouter.MODELS['fast']} (Evolution)")
    
    config = DiscoverConfig(
        domain="hardware_optimized_sorting",
        goal="Design a hybrid sorting algorithm that leverages SIMD-like block moves and branchless partitioning to maximize throughput on modern CPUs.",
        max_iterations=15,
        evolution_generations=5,
        population_size=10,
        solver_type="vad_cfr"
    )
    
    provider = GoogleProvider()
    workflow = DiscoverWorkflow(config=config, llm_provider=provider)
    
    total_start = time.perf_counter()
    all_discoveries = []

    print("\n🎯 Objective: " + config.goal)
    print("📡 Grounding: Using NotebookLM SOTA Synthesis")

    for i in range(config.max_iterations):
        iter_start = time.perf_counter()
        print("\n🔄 --- ITERATION " + str(i+1) + "/" + str(config.max_iterations) + " ---")
        
        try:
            discoveries = await workflow.run_iteration()
            if discoveries:
                print("⭐ Found " + str(len(discoveries)) + " new discoveries!")
                all_discoveries.extend(discoveries)
            else:
                print("📉 No breakthrough in this cycle.")
        except Exception as e:
            print("⚠️ Error: " + str(e))
            await asyncio.sleep(5)
            
        iter_time = time.perf_counter() - iter_start
        print("⏱️ Iteration time: " + f"{iter_time:.2f}" + "s")
        print("📊 Best Score: " + str(workflow.stats().get('best_score', 0.0)))

    total_time = time.perf_counter() - total_start
    print("\n🏁 --- SESSION COMPLETE ---")
    print("⏱️ Total Time: " + f"{total_time/60:.2f}" + " minutes")

if __name__ == "__main__":
    asyncio.run(main())
