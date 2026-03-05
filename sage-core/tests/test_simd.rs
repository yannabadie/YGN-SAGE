#[test]
fn test_h96_quicksort_empty() {
    let result = sage_core::simd_sort::h96_quicksort(vec![]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_h96_quicksort_sorted() {
    let result = sage_core::simd_sort::h96_quicksort(vec![5.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_h96_quicksort_large() {
    let data: Vec<f32> = (0..10_000).rev().map(|x| x as f32).collect();
    let sorted = sage_core::simd_sort::h96_quicksort(data).unwrap();
    for i in 1..sorted.len() {
        assert!(sorted[i - 1] <= sorted[i], "Not sorted at index {}", i);
    }
}

#[test]
fn test_partition() {
    let (left, right) = sage_core::simd_sort::vectorized_partition_h96(
        vec![5.0, 1.0, 3.0, 2.0, 4.0], 3.0
    ).unwrap();
    assert!(left.iter().all(|&x| x < 3.0));
    assert!(right.iter().all(|&x| x >= 3.0));
    assert_eq!(left.len() + right.len(), 5);
}

#[test]
fn test_argsort() {
    let indices = sage_core::simd_sort::h96_argsort(vec![30.0, 10.0, 20.0]).unwrap();
    assert_eq!(indices, vec![1, 2, 0]);
}
