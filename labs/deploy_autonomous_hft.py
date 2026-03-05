import asyncio
import time
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))
from sage.evolution.ebpf_evaluator import EbpfEvaluator
from sage.topology.kg_rlvr import ProcessRewardModel
from sage.llm.google import GoogleProvider
from sage.llm.base import LLMConfig, Message, Role

async def write_stream(event_type: str, content: str, meta: dict = None):
    """Write structured JSON logs for the Real-Time UI."""
    log_file = "docs/plans/agent_stream.jsonl"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "content": content,
        "meta": meta or {}
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    
    # Also print to terminal for CLI users
    prefix = f"[{event_type}]"
    print(f"{prefix} {content}")

async def autonomous_trading_loop():
    # Clear previous stream
    open("docs/plans/agent_stream.jsonl", "w").close()
    
    await write_stream("SYSTEM", "VIGIL RUNTIME INITIATED: Sovereign Autonomous Mode")
    await write_stream("SYSTEM", "Target: High-Frequency Trading (HFT) Order Book Exploitation")
    
    # Initialize the "System 2" LLM for periodic reflection
    provider = GoogleProvider()
    config = LLMConfig(provider="google", model="gemini-3.1-flash-lite-preview", temperature=0.7)
    
    # The optimal eBPF bytecode discovered by the DGM (Threshold 36)
    optimal_bytecode = b"\xb7\x00\x00\x00\x24\x00\x00\x00\x95\x00\x00\x00\x00\x00\x00\x00"
    evaluator = EbpfEvaluator()
    
    total_revenue = 0.0
    cycle_count = 0
    
    await write_stream("ACTION", "Deploying Policy-as-Code execution envelope. eBPF bytecode loaded into Kernel Space.")
    
    try:
        while True:
            cycle_count += 1
            start_time = time.perf_counter()
            
            # Execute the verified eBPF trading logic
            result = await evaluator.evaluate(optimal_bytecode)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Simulated market capture based on latency advantage
            revenue = 880.0 if latency_ms < 1.0 else 400.0
            total_revenue += revenue
            
            metrics = {
                "cycle": cycle_count,
                "latency_ms": round(latency_ms, 4),
                "revenue": revenue,
                "total_aum": total_revenue
            }
            
            await write_stream("METRIC", f"Cycle {cycle_count} completed. Latency: {latency_ms:.4f}ms", metrics)
            
            # Every 5 cycles, trigger System 2 Reflection (CoT)
            if cycle_count % 5 == 0:
                await write_stream("ACTION", "Invoking System 2 (LLM) for Metacognitive Reflection on recent performance...")
                
                prompt = f"Analyze the last 5 trading cycles. Average latency is around {latency_ms:.2f}ms. Total AUM is ${total_revenue:.2f}. Generate a brief System 3 reflection wrapped in <think> tags confirming if the execution is within optimal bounds and if the strategy should continue."
                
                response = await provider.generate([Message(role=Role.USER, content=prompt)], config=config)
                
                await write_stream("THINK", response.content)
                
            await asyncio.sleep(2) # Simulate waiting for the next market tick
            
    except asyncio.CancelledError:
        await write_stream("SYSTEM", "Daemon halted gracefully.")
    except Exception as e:
        await write_stream("ERROR", f"Autonomous loop interrupted: {e}")

if __name__ == "__main__":
    asyncio.run(autonomous_trading_loop())
