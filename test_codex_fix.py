import asyncio
import sys
import os

# Add paths to sys.path
sys.path.append(os.path.join(os.getcwd(), "sage-python", "src"))

from sage.llm.codex import CodexExecProvider

async def main():
    print("🧪 Testing CodexExecProvider Fix...")
    provider = CodexExecProvider()
    
    code = """
import numpy as np
def solution(arr):
    arr = np.array(arr)
    if len(arr) <= 1: return arr.tolist()
    pivot = arr[len(arr)//2]
    # Vectorized Partitioning (SOTA)
    return solution(arr[arr < pivot]) + arr[arr == pivot].tolist() + solution(arr[arr > pivot])
"""
    objective = "Implement branchless partitioning via vectorized boolean masking for SOTA sorting."
    
    print("🔍 Calling review_code...")
    result = await provider.review_code(code, objective)
    
    print("\n✅ Result:")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
