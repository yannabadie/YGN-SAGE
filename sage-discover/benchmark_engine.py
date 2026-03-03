import asyncio
import os
import sys
import time
import json
from dotenv import load_dotenv

load_dotenv()

# SOTA SSL Handling
cert_path = os.path.join(os.path.dirname(__file__), "..", "Cert", "ca-bundle.pem")
if os.path.exists(cert_path):
    os.environ["SSL_CERT_FILE"] = cert_path
    os.environ["HTTpx_CA_BUNDLE"] = cert_path

sys.path.append(os.path.join(os.path.dirname(__file__), "../sage-python/src"))
from sage.agent import Agent, AgentConfig
from sage.llm.google import GoogleProvider
from sage.llm.base import LLMConfig

async def run_benchmark(name: str, model: str, use_docker: bool = False):
    print(f"\n📊 Running SOTA Benchmark: {name}")
    print(f"🧠 Engine: {model}")
    
    provider = GoogleProvider()
    llm_config = LLMConfig(provider="google", model=model, max_tokens=4096, temperature=0.7)
    agent_config = AgentConfig(name=f"Bench_{name}", llm=llm_config, use_docker_sandbox=use_docker)
    
    agent = Agent(config=agent_config, llm_provider=provider)
    
    task = "Design a novel recursive sorting algorithm that reduces swaps by predicting element positions using a local density heuristic. Provide Python code."
    
    print("📡 Executing complex reasoning task...")
    await agent.run(task)
    
    stats = agent.get_aio_stats()
    
    print("-" * 40)
    print(f"⏱️  Total Wall-Clock Time: {stats['total_wall_time']:.4f}s")
    print(f"🧠 LLM Inference Time:    {stats['llm_inference_time']:.4f}s")
    print(f"⚙️  Infra Overhead:        {stats['infrastructure_overhead_time']:.4f}s")
    print(f"📈 AIO Ratio:             {stats['aio_ratio']:.2%}")
    print("-" * 40)
    
    if stats['aio_ratio'] < 0.05:
        print("✅ ASI STATUS: EXCELLENT. System is reasoning-bound.")
    else:
        print("❌ ASI STATUS: BOTTLENECK detected.")
        
    return stats

async def main():
    print("🚀 YGN-SAGE SOTA Benchmarking Engine (Mars 2026) 🚀")
    
    # Run with the frontier model
    await run_benchmark("SOTA_Reasoning_Performance", "gemini-3.1-pro-preview", use_docker=False)

if __name__ == "__main__":
    asyncio.run(main())
