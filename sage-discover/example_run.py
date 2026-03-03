import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# SOTA SSL Handling (Mars 2026)
# If custom cert exists, use it, otherwise rely on system certs (since OOD)
cert_path = os.path.join(os.path.dirname(__file__), "..", "Cert", "ca-bundle.pem")
if os.path.exists(cert_path):
    os.environ["SSL_CERT_FILE"] = cert_path
    os.environ["HTTpx_CA_BUNDLE"] = cert_path
    print(f"🔒 SSL: Using custom CA bundle at {cert_path}")
else:
    print("🌐 SSL: Relying on system certificates (OOD mode).")

# Ensure modules are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../sage-python/src"))

from discover.workflow import DiscoverWorkflow, DiscoverConfig
from sage.llm.google import GoogleProvider
from sage.llm.base import LLMConfig

async def main():
    print("🚀 Starting YGN-SAGE HIGH-POWER SOTA Discovery Run (Mars 2026)...")
    print("🧠 Brain Engine: Gemini 3.1 Pro Preview")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in .env")
        return

    # SOTA High-power configuration
    config = DiscoverConfig(
        domain="adaptive_hybrid_sorting",
        goal="Design a hybrid sorting algorithm that optimizes cache-locality and branch-prediction by switching between 'Block QuickSort' and 'Partition-aware Insertion Sort' based on data entropy.",
        max_iterations=1,
        evolution_generations=5, # Slightly reduced for faster proof
        population_size=10,
        solver_type="vad_cfr"
    )
    
    # Force LLM configuration for all components
    llm_config = LLMConfig(
        provider="google",
        model="gemini-3.1-pro-preview",
        max_tokens=8192,
        temperature=0.8 
    )
    
    provider = GoogleProvider()
    
    workflow = DiscoverWorkflow(
        config=config,
        llm_provider=provider,
        memory_compressor=None,
        sandbox_manager=None, 
    )
    
    # Propagate SOTA model to main agent
    workflow.main_agent.config.llm = llm_config
    
    print(f"📡 Investigating SOTA domain: {config.domain}")
    print(f"🎯 Objective: {config.goal}")
    
    discoveries = await workflow.run_iteration()
    
    print("\n🏆 --- SOTA DISCOVERY RESULTS ---")
    if discoveries:
        for i, d in enumerate(discoveries):
            print(f"⭐ Discovery {i+1}:")
            print(f"  Hypothesis: {d.hypothesis.statement}")
            print(f"  Confidence Score: {d.score}")
            print(f"  Evolved SOTA Code:\n{d.code}\n")
    else:
        print("⚠️ No breakthroughs reached the confidence threshold. The SOTA threshold is high.")
        
    print("\n📊 --- ENGINE STATS ---")
    print(json.dumps(workflow.stats(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
