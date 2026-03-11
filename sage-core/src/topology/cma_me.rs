//! CMA-ME (Covariance Matrix Adaptation MAP-Elites) emitter for continuous
//! parameter optimization of topology budgets and edge weights.
//!
//! Optimises 3 continuous parameters per topology:
//! - `max_cost_usd` (f64)
//! - `max_wall_time_s` (f64)
//! - `edge_weight` (f64)
//!
//! Uses a simplified diagonal covariance (no full matrix needed for 3D).

/// CMA-ME emitter — samples continuous parameter vectors and adapts the
/// search distribution based on elite fitness feedback.
pub struct CmaEmitter {
    dim: usize,
    mean: Vec<f64>,
    sigma: f64,
    /// Diagonal covariance (simplified — no full matrix needed for 3D).
    cov_diag: Vec<f64>,
    pub generation: u32,
}

impl CmaEmitter {
    /// Create a new emitter with `dim` dimensions and initial step size `initial_sigma`.
    ///
    /// Mean is initialised at `[0.5; dim]`, covariance diagonal at `[1.0; dim]`.
    pub fn new(dim: usize, initial_sigma: f64) -> Self {
        Self {
            dim,
            mean: vec![0.5; dim],
            sigma: initial_sigma,
            cov_diag: vec![1.0; dim],
            generation: 0,
        }
    }

    /// Number of dimensions.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Current distribution mean.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Sample `n` parameter vectors from `N(mean, sigma² * diag(cov_diag))`.
    ///
    /// Uses a deterministic spread: for each sample `i`, compute an offset
    /// proportional to `(i - n/2) / n * sigma`, scaled by `sqrt(cov_diag[j])`.
    /// All values are clamped to `[0.01, 10.0]`.
    pub fn ask(&self, n: usize) -> Vec<Vec<f64>> {
        let half_n = n as f64 / 2.0;
        (0..n)
            .map(|i| {
                let offset = (i as f64 - half_n) / n as f64 * self.sigma;
                (0..self.dim)
                    .map(|j| {
                        let v = self.mean[j] + offset * self.cov_diag[j].sqrt();
                        v.clamp(0.01, 10.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Update the distribution from evaluated samples.
    ///
    /// Sorts by fitness (descending), takes the top `μ = n/2` elites, then:
    /// - Updates `mean` as the weighted average of elites.
    /// - Updates `cov_diag` from the elite variance.
    /// - Increments `generation`.
    pub fn tell(&mut self, samples: &[Vec<f64>], fitnesses: &[f64]) {
        assert_eq!(
            samples.len(),
            fitnesses.len(),
            "samples and fitnesses must have the same length"
        );
        if samples.is_empty() {
            return;
        }

        // Sort indices by fitness descending.
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.sort_by(|&a, &b| {
            fitnesses[b]
                .partial_cmp(&fitnesses[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top μ = n/2 (at least 1).
        let mu = (samples.len() / 2).max(1);
        let elites: Vec<&Vec<f64>> = indices[..mu].iter().map(|&i| &samples[i]).collect();

        // Weights: linearly decreasing, normalised to sum to 1.
        let raw_weights: Vec<f64> = (0..mu).map(|i| (mu - i) as f64).collect();
        let weight_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / weight_sum).collect();

        // Update mean: weighted average of elites.
        let mut new_mean = vec![0.0; self.dim];
        for (elite, &w) in elites.iter().zip(weights.iter()) {
            for j in 0..self.dim {
                new_mean[j] += w * elite[j];
            }
        }

        // Update cov_diag: weighted variance of elites around new mean.
        let mut new_cov = vec![0.0; self.dim];
        for (elite, &w) in elites.iter().zip(weights.iter()) {
            for j in 0..self.dim {
                let diff = elite[j] - new_mean[j];
                new_cov[j] += w * diff * diff;
            }
        }
        // Floor the covariance to avoid collapse.
        for v in &mut new_cov {
            if *v < 1e-8 {
                *v = 1e-8;
            }
        }

        self.mean = new_mean;
        self.cov_diag = new_cov;
        self.generation += 1;
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let e = CmaEmitter::new(3, 0.3);
        assert_eq!(e.dimension(), 3);
        assert_eq!(e.mean(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_ask_returns_correct_count() {
        let e = CmaEmitter::new(3, 0.3);
        let samples = e.ask(5);
        assert_eq!(samples.len(), 5);
        for s in &samples {
            assert_eq!(s.len(), 3);
        }
    }

    #[test]
    fn test_ask_samples_near_mean() {
        let e = CmaEmitter::new(3, 0.1);
        let samples = e.ask(10);
        for s in &samples {
            for &v in s {
                assert!(v > 0.0 && v < 2.0);
            }
        }
    }

    #[test]
    fn test_tell_shifts_mean() {
        let mut e = CmaEmitter::new(3, 0.3);
        let samples = e.ask(6);
        // Higher fitness for samples with higher values
        let fitnesses: Vec<f64> = samples.iter().map(|s| s.iter().sum()).collect();
        let old_mean = e.mean().to_vec();
        e.tell(&samples, &fitnesses);
        // Mean should shift toward higher-sum samples
        assert_ne!(e.mean(), &old_mean[..]);
    }

    #[test]
    fn test_tell_increments_generation() {
        let mut e = CmaEmitter::new(3, 0.3);
        assert_eq!(e.generation, 0);
        let samples = e.ask(4);
        let fitnesses = vec![0.1, 0.5, 0.9, 0.3];
        e.tell(&samples, &fitnesses);
        assert_eq!(e.generation, 1);
    }

    #[test]
    fn test_multiple_generations_converge() {
        let mut e = CmaEmitter::new(1, 0.5);
        // Target: high fitness at x=2.0
        for _ in 0..10 {
            let samples = e.ask(8);
            let fitnesses: Vec<f64> = samples.iter().map(|s| -(s[0] - 2.0).powi(2)).collect();
            e.tell(&samples, &fitnesses);
        }
        // Mean should be closer to 2.0 than initial 0.5
        assert!((e.mean()[0] - 2.0).abs() < (0.5_f64 - 2.0).abs());
    }
}
