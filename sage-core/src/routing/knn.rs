//! Rust kNN router for pre-computed exemplar embeddings.
//!
//! Hot-path: L2-normalize, dot product (cosine similarity), top-k partial sort,
//! distance-weighted majority vote, OOD rejection.
//!
//! Always compiled — no feature gate required.

use pyo3::prelude::*;
use tracing::instrument;

/// Rust kNN router for pre-computed exemplar embeddings.
///
/// Hot-path: L2-normalize, dot product, top-k, distance-weighted vote.
#[pyclass]
pub struct RustKnnRouter {
    /// Row-major exemplar embeddings (N x dim)
    embeddings: Vec<f32>,
    /// Labels for each exemplar
    labels: Vec<i32>,
    /// Number of exemplars
    n: usize,
    /// Embedding dimension
    dim: usize,
    /// Number of neighbors
    k: usize,
    /// OOD rejection threshold (cosine similarity)
    distance_threshold: f32,
}

#[pymethods]
impl RustKnnRouter {
    #[new]
    #[pyo3(signature = (k=5, distance_threshold=0.3))]
    fn new(k: usize, distance_threshold: f32) -> Self {
        Self {
            embeddings: Vec::new(),
            labels: Vec::new(),
            n: 0,
            dim: 0,
            k,
            distance_threshold,
        }
    }

    /// Load pre-computed exemplars.
    ///
    /// `embeddings` is flat row-major (N*dim), `labels` is (N,).
    #[pyo3(signature = (embeddings, labels, dim))]
    fn load_exemplars(
        &mut self,
        embeddings: Vec<f32>,
        labels: Vec<i32>,
        dim: usize,
    ) -> PyResult<()> {
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("dim must be > 0"));
        }
        if !embeddings.len().is_multiple_of(dim) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "embeddings length must be a multiple of dim",
            ));
        }
        let n = embeddings.len() / dim;
        if n != labels.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "number of embedding rows must match labels length",
            ));
        }
        self.embeddings = embeddings;
        self.labels = labels;
        self.n = n;
        self.dim = dim;
        Ok(())
    }

    /// Returns true if exemplars have been loaded.
    fn has_exemplars(&self) -> bool {
        self.n > 0
    }

    /// Returns the number of loaded exemplars.
    fn exemplar_count(&self) -> usize {
        self.n
    }

    /// Route a query embedding.
    ///
    /// Returns `(system, confidence, nearest_distance)` or `None` when:
    /// - No exemplars are loaded
    /// - Query dimension mismatches
    /// - Nearest neighbor is below OOD threshold
    ///
    /// `query` must be L2-normalized and have length == `dim`.
    #[instrument(skip(self, query))]
    #[pyo3(signature = (query,))]
    fn route(&self, query: Vec<f32>) -> Option<(i32, f32, f32)> {
        if self.n == 0 || query.len() != self.dim {
            return None;
        }

        // Compute cosine similarities (dot products — both sides are L2-normalized)
        let mut similarities: Vec<f32> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let start = i * self.dim;
            let end = start + self.dim;
            let dot: f32 = self.embeddings[start..end]
                .iter()
                .zip(query.iter())
                .map(|(a, b)| a * b)
                .sum();
            similarities.push(dot);
        }

        // Partial sort: bring top-k largest similarities to indices [0..k)
        let k = self.k.min(self.n);
        let mut indices: Vec<usize> = (0..self.n).collect();
        // select_nth_unstable_by puts the k-1-th largest at position k-1;
        // elements [0..k) are the top-k (unsorted among themselves).
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            similarities[b]
                .partial_cmp(&similarities[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort the top-k slice descending by similarity
        let mut top_k: Vec<usize> = indices[..k].to_vec();
        top_k.sort_by(|&a, &b| {
            similarities[b]
                .partial_cmp(&similarities[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // OOD rejection: nearest neighbor must exceed threshold
        let nearest_dist = similarities[top_k[0]];
        if nearest_dist < self.distance_threshold {
            return None;
        }

        // Distance-weighted majority vote (only positive similarities count)
        let mut votes: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
        for &idx in &top_k {
            let weight = similarities[idx].max(0.0);
            let label = self.labels[idx];
            *votes.entry(label).or_insert(0.0) += weight;
        }

        let total_weight: f32 = votes.values().sum();
        if total_weight <= 0.0 {
            return None;
        }

        let (&winner, &winner_weight) = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        let confidence = winner_weight / total_weight;
        Some((winner, confidence, nearest_dist))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[test]
    fn test_empty_returns_none() {
        let router = RustKnnRouter::new(5, 0.3);
        let query = make_unit(&[1.0_f32, 0.0, 0.0]);
        assert!(router.route(query).is_none());
    }

    #[test]
    fn test_single_exemplar() {
        let mut router = RustKnnRouter::new(1, 0.0);
        let v = make_unit(&[1.0_f32, 0.0, 0.0]);
        router
            .load_exemplars(v.clone(), vec![2i32], 3)
            .expect("load ok");

        let result = router.route(v.clone()).expect("should route");
        assert_eq!(result.0, 2);
        // Cosine similarity of a vector with itself = 1.0
        assert!((result.2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_three_exemplars_majority_vote() {
        // Two label=1 exemplars close to e1, one label=2 exemplar pointing elsewhere.
        let e1 = make_unit(&[1.0_f32, 0.1, 0.0]);
        let e2 = make_unit(&[0.9_f32, 0.2, 0.0]);
        let e3 = make_unit(&[0.0_f32, 0.0, 1.0]); // orthogonal

        let mut embeddings = Vec::new();
        embeddings.extend_from_slice(&e1);
        embeddings.extend_from_slice(&e2);
        embeddings.extend_from_slice(&e3);

        let labels = vec![1i32, 1i32, 2i32];

        let mut router = RustKnnRouter::new(3, 0.0);
        router
            .load_exemplars(embeddings, labels, 3)
            .expect("load ok");

        let query = make_unit(&[1.0_f32, 0.05, 0.0]);
        let result = router.route(query).expect("should route");
        // Both label=1 exemplars are closest → winner must be 1
        assert_eq!(result.0, 1);
        assert!(result.1 > 0.5, "confidence should be >0.5 for majority label");
    }

    #[test]
    fn test_ood_rejection() {
        let e1 = make_unit(&[1.0_f32, 0.0, 0.0]);

        let mut router = RustKnnRouter::new(1, 0.9); // high threshold
        router
            .load_exemplars(e1, vec![1i32], 3)
            .expect("load ok");

        // Query is roughly orthogonal to the exemplar → low similarity
        let query = make_unit(&[0.0_f32, 1.0, 0.0]);
        assert!(
            router.route(query).is_none(),
            "OOD query should be rejected"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let e1 = make_unit(&[1.0_f32, 0.0, 0.0]);

        let mut router = RustKnnRouter::new(1, 0.0);
        router
            .load_exemplars(e1, vec![1i32], 3)
            .expect("load ok");

        // Query has wrong dimension (2 instead of 3)
        let query = vec![1.0_f32, 0.0];
        assert!(
            router.route(query).is_none(),
            "dimension mismatch should return None"
        );
    }

    #[test]
    fn test_load_exemplars_validates_inputs() {
        let mut router = RustKnnRouter::new(5, 0.3);

        // Misaligned length
        let result = router.load_exemplars(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1i32], 3);
        assert!(result.is_err(), "4 floats / dim=3 → error");

        // Mismatched label count
        let result = router.load_exemplars(
            vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![1i32], // only 1 label for 2 exemplars
            3,
        );
        assert!(result.is_err(), "2 exemplars / 1 label → error");

        // dim=0
        let result = router.load_exemplars(vec![], vec![], 0);
        assert!(result.is_err(), "dim=0 → error");
    }
}
