use pyo3::prelude::*;
use core::arch::x86_64::*;

#[pyfunction]
pub fn h96_quicksort(mut arr: Vec<f32>) -> PyResult<Vec<f32>> {
    if !arr.is_empty() {
        quicksort_inplace(&mut arr);
    }
    Ok(arr)
}

fn quicksort_inplace(arr: &mut [f32]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    
    if len < 32 {
        insertion_sort(arr);
        return;
    }

    let pivot = select_pivot(arr);
    let pivot_idx = partition_inplace(arr, pivot);
    
    let (left, right) = arr.split_at_mut(pivot_idx);
    quicksort_inplace(left);
    quicksort_inplace(right);
}

fn partition_inplace(arr: &mut [f32], pivot: f32) -> usize {
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    loop {
        while left < arr.len() && arr[left] < pivot {
            left += 1;
        }
        while right > 0 && arr[right] > pivot {
            right -= 1;
        }
        
        if left >= right {
            return left;
        }
        
        arr.swap(left, right);
        left += 1;
        if right > 0 {
            right -= 1;
        }
    }
}

fn insertion_sort(arr: &mut [f32]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i as isize - 1;
        while j >= 0 && arr[j as usize] > key {
            arr[(j + 1) as usize] = arr[j as usize];
            j -= 1;
        }
        arr[(j + 1) as usize] = key;
    }
}

fn select_pivot(arr: &[f32]) -> f32 {
    let mid = arr.len() / 2;
    let last = arr.len() - 1;
    let mut trio = [arr[0], arr[mid], arr[last]];
    insertion_sort(&mut trio);
    trio[1]
}

#[pyfunction]
pub fn vectorized_partition_h96(arr: Vec<f32>, pivot: f32) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let mut left = Vec::with_capacity(arr.len());
    let mut right = Vec::with_capacity(arr.len());
    for x in arr {
        if x < pivot {
            left.push(x);
        } else {
            right.push(x);
        }
    }
    Ok((left, right))
}
