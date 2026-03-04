import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))

from sage.evolution.ebpf_evaluator import EbpfEvaluator

async def main():
    evaluator = EbpfEvaluator()
    print("Testing EbpfEvaluator with mock string (representing DGM mutation)...")
    result = await evaluator.evaluate("def mock_dgm_mutation(): pass")
    print(f"Result (String Mock): Score={result.score}, Passed={result.passed}, Details={result.details}")
    
    print("\nTesting EbpfEvaluator with raw bytes (representing compiled ELF)...")
    # A dummy byte array to see if solana_rbpf accepts/rejects it and returns in <1ms
    # Since it's not a real ELF, it should fail parsing, but we want to see the error and latency.
    result_bytes = await evaluator.evaluate(b"bad elf header")
    print(f"Result (Raw Bytes): Score={result_bytes.score}, Passed={result_bytes.passed}, Stage={result_bytes.stage}")
    if result_bytes.error:
        print(f"Expected Error: {result_bytes.error}")

if __name__ == "__main__":
    asyncio.run(main())
