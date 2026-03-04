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
from sage.memory.python_backend import PythonWorkingMemory

async def run_benchmark(name: str, model: str, use_rust: bool = True):
    print(f"\n📊 Running SOTA Benchmark: {name} (Backend: {'Rust/Arrow' if use_rust else 'Pure Python'})")
    
    provider = GoogleProvider()
    llm_config = LLMConfig(provider="google", model=model, max_tokens=4096, temperature=0.7)
    agent_config = AgentConfig(name=f"Bench_{name}", llm=llm_config)
    
    agent = Agent(config=agent_config, llm_provider=provider)
    
    if not use_rust:
        # Swap out the memory for the pure python version
        agent.working_memory = PythonWorkingMemory(agent_id=agent_config.name)
    
    # Stress test: Add many events to see infrastructure overhead
    for i in range(2000): 
        agent.working_memory.add_event("debug", f"Synthetic memory event {i} to test overhead.")
    
    task = "Explique en une phrase courte pourquoi l'architecture YGN-SAGE est efficace pour l'ASI."
    
    print(f"📡 Executing task with {model} and stress-loaded memory...")
    await agent.run(task)
    
    # Also test Arrow export time
    export_start = time.perf_counter()
    batch = agent.working_memory.to_arrow()
    export_time = time.perf_counter() - export_start
    print(f"🏹 Arrow Export Time: {export_time:.6f}s")
    
    stats = agent.get_aio_stats()
    stats["export_time"] = export_time
    
    print("-" * 40)
    print(f"⏱️  Total Wall-Clock Time: {stats['total_wall_time']:.4f}s")
    print(f"🧠 LLM Inference Time:    {stats['llm_inference_time']:.4f}s")
    print(f"⚙️  Infra Overhead:        {stats['infrastructure_overhead_time']:.4f}s")
    print(f"📈 AIO Ratio:             {stats['aio_ratio']:.2%}")
    print("-" * 40)
    
    return stats

async def main():
    print("🚀 YGN-SAGE SOTA Benchmarking Engine (Mars 2026) 🚀")
    
    # Utilisation du dernier modèle Flash Lite pour un benchmark rapide et précis
    model = "gemini-3.1-flash-lite-preview" 

    
    try:
        rust_stats = await run_benchmark("Rust_Memory", model, use_rust=True)
        python_stats = await run_benchmark("Python_Memory", model, use_rust=False)
        
        print("\n🏁 FINAL COMPARISON 🏁")
        print(f"Rust Infra Overhead:   {rust_stats['infrastructure_overhead_time']:.4f}s")
        print(f"Python Infra Overhead: {python_stats['infrastructure_overhead_time']:.4f}s")
        
        speedup = python_stats['infrastructure_overhead_time'] / rust_stats['infrastructure_overhead_time']
        print(f"🚀 Rust memory is {speedup:.2f}x faster than Python memory.")
        
        export_speedup = python_stats['export_time'] / rust_stats['export_time']
        print(f"🏹 Rust Arrow export is {export_speedup:.2f}x faster (zero-copy).")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("Note: Si 'gemini-3-flash' n'est pas encore déployé sur votre endpoint, essayez 'gemini-3.1-pro'.")

if __name__ == "__main__":
    asyncio.run(main())
