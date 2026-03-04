import numpy as np
import sage_core
import time

def test_simd_sort():
    print("Testing SIMD Quicksort (Zero-copy)")
    # Create a random float32 array
    size = 100_000
    arr = np.random.rand(size).astype(np.float32)
    
    # Check if sorted initially (unlikely)
    is_sorted = np.all(arr[:-1] <= arr[1:])
    print(f"Initially sorted: {is_sorted}")
    
    start = time.perf_counter()
    sage_core.h96_quicksort_zerocopy(arr)
    end = time.perf_counter()
    
    # Check if sorted now
    is_sorted = np.all(arr[:-1] <= arr[1:])
    print(f"Sorted after h96_quicksort_zerocopy: {is_sorted}")
    print(f"Time taken: {end - start:.6f}s")
    
    if not is_sorted:
        raise ValueError("Sort failed!")

    print("\nTesting vectorized_partition_h96")
    arr_list = [5.0, 1.0, 8.0, 2.0, 7.0, 3.0]
    pivot = 4.0
    left, right = sage_core.vectorized_partition_h96(arr_list, pivot)
    print(f"Pivot: {pivot}")
    print(f"Left: {left}")
    print(f"Right: {right}")
    
    assert all(x < pivot for x in left)
    assert all(x >= pivot for x in right)
    print("Partition test passed!")

if __name__ == "__main__":
    try:
        test_simd_sort()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
