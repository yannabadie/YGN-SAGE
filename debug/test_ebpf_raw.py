import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))

from sage.evolution.ebpf_evaluator import EbpfEvaluator
import sage_core

async def main():
    sandbox = sage_core.EbpfSandbox()
    
    # Simple eBPF bytecode:
    # 1. mov64 r0, 42 (b7 00 00 00 2a 00 00 00)
    # 2. exit (95 00 00 00 00 00 00 00)
    bytecode = b"\xb7\x00\x00\x00\x2a\x00\x00\x00\x95\x00\x00\x00\x00\x00\x00\x00"
    
    print("Loading RAW eBPF bytecode...")
    try:
        sandbox.load_raw(bytecode)
        mem = bytearray(1024 * 1024)
        print("Executing in solana_rbpf VM...")
        instruction_count, res = sandbox.execute(list(mem))
        print(f"✅ Execution Success!")
        print(f"Instruction Count: {instruction_count}")
        print(f"Result Code (R0): {res}")
    except Exception as e:
        print(f"❌ Execution Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
