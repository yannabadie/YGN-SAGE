import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))
import sage_core

def benchmark_simd_retrieval():
    print("===================================================================")
    print(" 🚀 YGN-SAGE SIMD RETRIEVAL BENCHMARK (March 2026)")
    print(" Comparing Pure Python/NumPy vs Rust AVX-512 Zero-Copy (H96)")
    print("===================================================================")
    
    # Generate 10 Million random embeddings/scores
    size = 10_000_000
    print(f"\nGenerating {size:,} float32 data points...")
    data = np.random.rand(size).astype(np.float32)
    
    # 1. Baseline: NumPy partition
    print("\n[1] Testing NumPy Partition (Top-K Retrieval Baseline)...")
    data_copy1 = data.copy()
    start_numpy = time.perf_counter()
    # Find top 1000 items
    np.partition(data_copy1, size - 1000)
    end_numpy = time.perf_counter()
    numpy_time = (end_numpy - start_numpy) * 1000
    print(f"NumPy Partition took: {numpy_time:.2f} ms")
    
    # 2. Rust SIMD (AVX-512) Zero-Copy
    print("\n[2] Testing Rust AVX-512 Zero-Copy (H96)...")
    data_copy2 = data.copy()
    start_simd = time.perf_counter()
    # H96 vectorized partition exposed via PyO3 buffer protocol
    sage_core.vectorized_partition_h96(data_copy2, 0.9)
    end_simd = time.perf_counter()
    simd_time = (end_simd - start_simd) * 1000
    print(f"Rust SIMD Partition took: {simd_time:.2f} ms")
    
    speedup = numpy_time / simd_time
    print(f"\n✅ RESULTS: Rust AVX-512 (H96) is {speedup:.1f}x faster than NumPy for Agentic Memory Retrieval.")

if __name__ == "__main__":
    benchmark_simd_retrieval()
