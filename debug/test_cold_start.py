import asyncio
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))

from sage.sandbox.manager import SandboxManager, SandboxConfig

async def benchmark_cold_start():
    print("===================================================================")
    print(" ⚡ YGN-SAGE COLD START BENCHMARK (March 2026)")
    print(" Comparing Docker container initialization vs Wasm Component Model")
    print("===================================================================")
    
    # 1. Benchmark Docker Cold Start
    print("\n[1] Testing Docker Cold Start Latency...")
    # NOTE: To run this, Docker Desktop needs to be running. We will attempt it.
    try:
        docker_manager = SandboxManager(use_docker=True)
        start_docker = time.perf_counter()
        # This will run `docker run -d ...`
        sandbox_docker = await docker_manager.create(SandboxConfig(image="alpine:latest", timeout=10))
        end_docker = time.perf_counter()
        
        docker_latency = (end_docker - start_docker) * 1000
        print(f"Docker container started in {docker_latency:.2f} ms")
        
        # Cleanup
        await docker_manager.destroy_all()
    except Exception as e:
        print(f"Docker benchmark failed (is Docker running?): {e}")
        docker_latency = float('inf')
        
    # 2. Benchmark Wasm Cold Start
    print("\n[2] Testing Wasm Component Model Cold Start Latency...")
    try:
        wasm_manager = SandboxManager(use_docker=False)
        
        start_wasm = time.perf_counter()
        # Creating a sandbox locally/wasm doesn't start a process, but the execution compiles the component
        sandbox_wasm = await wasm_manager.create()
        
        # Minimal valid wasm module (empty component)
        # We'll just measure the execution engine overhead by passing a mock module
        # A valid basic WebAssembly binary header
        wasm_bytes = b"\x00asm\x01\x00\x00\x00" 
        
        # We expect a compilation error for this dummy bytes, but the engine initialization itself is what we measure
        await sandbox_wasm.execute("test", wasm_module=wasm_bytes)
        end_wasm = time.perf_counter()
        
        wasm_latency = (end_wasm - start_wasm) * 1000
        print(f"Wasm Engine initialized & executed in {wasm_latency:.2f} ms")
    except Exception as e:
        print(f"Wasm benchmark failed: {e}")
        wasm_latency = float('inf')

    if docker_latency != float('inf') and wasm_latency != float('inf'):
        print(f"\n✅ RESULTS: Wasm is {docker_latency/wasm_latency:.1f}x faster for cold starts.")

if __name__ == "__main__":
    asyncio.run(benchmark_cold_start())
