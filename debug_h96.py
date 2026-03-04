
import numpy as np
import time

def solution(arr):
    arr = np.array(arr)
    if len(arr) <= 1: return arr.tolist()
    pivot = arr[len(arr)//2]
    # Vectorized Partitioning
    return solution(arr[arr < pivot]) + arr[arr == pivot].tolist() + solution(arr[arr > pivot])

def benchmark():
    print("🔍 Debugging H96 Code...")
    try:
        n = 1000
        test_arr = np.random.randint(0, 10000, n).tolist()
        
        start = time.perf_counter()
        result = solution(test_arr)
        end = time.perf_counter()
        
        print(f"✅ Success! Time: {end-start:.4f}s")
        print(f"✅ Correctness: {result == sorted(test_arr)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark()
