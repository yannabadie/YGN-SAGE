use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// H96 Quicksort — uses Rust's pdqsort (pattern-defeating quicksort).
/// When vqsort-rs supports Windows, swap the sort() call for vqsort_rs::sort().
#[pyfunction]
pub fn h96_quicksort(mut arr: Vec<f32>) -> PyResult<Vec<f32>> {
    arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(arr)
}

/// Zero-copy in-place sort on a NumPy array.
#[pyfunction]
pub fn h96_quicksort_zerocopy(arr: &Bound<'_, PyArray1<f32>>) -> PyResult<()> {
    let mut view = arr.readwrite();
    let slice = view
        .as_slice_mut()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Array is not contiguous"))?;

    slice.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(())
}

/// Partition array around a pivot. Returns (left < pivot, right >= pivot).
#[pyfunction]
pub fn vectorized_partition_h96(mut arr: Vec<f32>, pivot: f32) -> PyResult<(Vec<f32>, Vec<f32>)> {
    arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let split = arr.partition_point(|&x| x < pivot);
    let right = arr.split_off(split);
    Ok((arr, right))
}

/// Argsort: returns indices that would sort the array (ascending).
/// Essential for MCTS UCB node selection in TopologyPlanner.
#[pyfunction]
pub fn h96_argsort(arr: Vec<f32>) -> PyResult<Vec<usize>> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        arr[a]
            .partial_cmp(&arr[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(indices)
}
