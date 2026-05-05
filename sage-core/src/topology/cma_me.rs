//! CMA-ME (Covariance Matrix Adaptation MAP-Elites) emitter for continuous
//! parameter optimization of topology budgets and edge weights.
//!
//! Optimises 3 continuous parameters per topology:
//! - `max_cost_usd` (f64)
//! - `max_wall_time_s` (f64)
//! - `edge_weight` (f64)
//!
//! Uses a simplified diagonal covariance (no full matrix needed for 3D).

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Per-dimension bounds for clamping samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionBounds {
    pub min: f64,
    pub max: f64,
}

impl DimensionBounds {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Clamp a value to this dimension's bounds.
    pub fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }
}

/// CMA-ME emitter — samples continuous parameter vectors and adapts the
/// search distribution based on elite fitness feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaEmitter {
    dim: usize,
    mean: Vec<f64>,
    sigma: f64,
    /// Diagonal covariance (simplified — no full matrix needed for 3D).
    cov_diag: Vec<f64>,
    /// Per-dimension bounds for clamping.
    bounds: Vec<DimensionBounds>,
    /// Sigma decay factor applied after each generation — subject to ablation.
    /// 1.0 = no decay (default). <1.0 decays sigma over time to refine search.
    sigma_decay: f64,
    pub generation: u32,
}

impl CmaEmitter {
    /// Create a new emitter with `dim` dimensions and initial step size `initial_sigma`.
    ///
    /// Mean is initialised at `[0.5; dim]`, covariance diagonal at `[1.0; dim]`.
    /// Bounds default to `[0.01, 10.0]` per dimension.
    pub fn new(dim: usize, initial_sigma: f64) -> Self {
        Self {
            dim,
            mean: vec![0.5; dim],
            sigma: initial_sigma,
            cov_diag: vec![1.0; dim],
            bounds: vec![DimensionBounds::new(0.01, 10.0); dim],
            sigma_decay: 1.0,
            generation: 0,
        }
    }

    /// Create an emitter with per-dimension bounds and optional sigma decay.
    ///
    /// For topology parameters: dim 0 = max_cost_usd [0.001, 5.0],
    /// dim 1 = max_wall_time_s [1.0, 300.0], dim 2 = edge_weight [0.1, 5.0].
    pub fn with_bounds(
        dim: usize,
        initial_sigma: f64,
        bounds: Vec<DimensionBounds>,
        sigma_decay: f64,
    ) -> Self {
        assert_eq!(bounds.len(), dim, "bounds length must match dim");
        // Initialize mean at the center of each dimension's range.
        let mean: Vec<f64> = bounds.iter().map(|b| (b.min + b.max) / 2.0).collect();
        Self {
            dim,
            mean,
            sigma: initial_sigma,
            cov_diag: vec![1.0; dim],
            bounds,
            sigma_decay: sigma_decay.clamp(0.9, 1.0),
            generation: 0,
        }
    }

    /// Warm-start the emitter mean from observed elite values.
    ///
    /// Useful at small scale (<1000 archives) to avoid wasting early
    /// generations exploring around an arbitrary initial mean.
    pub fn warm_start(&mut self, center: &[f64]) {
        assert_eq!(center.len(), self.dim, "center length must match dim");
        for (j, &c) in center.iter().enumerate().take(self.dim) {
            self.mean[j] = self.bounds[j].clamp(c);
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

    /// Current sigma (step size).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Sample `n` parameter vectors from `N(mean, sigma^2 * diag(cov_diag))`.
    ///
    /// Uses Box-Muller transform to sample from a Gaussian distribution
    /// centered on `mean` with variance `sigma^2 * cov_diag[j]`.
    /// Values are clamped to per-dimension bounds.
    pub fn ask(&self, n: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        self.ask_with_rng(n, &mut rng)
    }

    pub fn ask_with_rng<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| {
                (0..self.dim)
                    .map(|j| {
                        // Box-Muller Gaussian: N(mean[j], sigma^2 * cov_diag[j])
                        let u1: f64 = rng.random::<f64>().max(1e-15); // avoid ln(0)
                        let u2: f64 = rng.random::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        let v = self.mean[j] + self.sigma * self.cov_diag[j].sqrt() * z;
                        self.bounds[j].clamp(v)
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
    /// - Applies sigma decay.
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
        self.sigma *= self.sigma_decay;
        self.generation += 1;
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn assert_close(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{label}: expected {expected:.17}, got {actual:.17}"
        );
    }

    #[test]
    fn test_new_defaults() {
        let e = CmaEmitter::new(3, 0.3);
        assert_eq!(e.dimension(), 3);
        assert_eq!(e.mean(), &[0.5, 0.5, 0.5]);
        assert!((e.sigma() - 0.3).abs() < 1e-10);
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
    fn test_ask_with_rng_same_seed_same_samples() {
        let e = CmaEmitter::new(3, 0.3);
        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);

        let left = e.ask_with_rng(8, &mut rng_a);
        let right = e.ask_with_rng(8, &mut rng_b);

        assert_eq!(left, right);
    }

    #[test]
    fn test_ask_with_rng_different_seed_different_samples() {
        let e = CmaEmitter::new(3, 0.3);
        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let mut rng_b = ChaCha8Rng::seed_from_u64(456);

        let left = e.ask_with_rng(8, &mut rng_a);
        let right = e.ask_with_rng(8, &mut rng_b);

        assert_ne!(left, right);
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
    fn tell_updates_mean_and_cov_diag_from_synthetic_elites_exact() {
        let mut e = CmaEmitter::new(1, 0.5);
        let samples = vec![vec![0.0], vec![2.0], vec![4.0], vec![6.0]];
        let fitnesses = vec![0.0, 10.0, 1.0, -1.0];

        e.tell(&samples, &fitnesses);

        assert_close(e.mean[0], 8.0 / 3.0, "mean[0]");
        assert_close(e.cov_diag[0], 8.0 / 9.0, "cov_diag[0]");
    }

    #[test]
    fn seeded_cma_one_generation_snapshot_exact() {
        let mut e = CmaEmitter::new(1, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FF_EE26);
        let samples = e.ask_with_rng(8, &mut rng);
        let fitnesses: Vec<f64> = samples.iter().map(|s| -(s[0] - 2.0).powi(2)).collect();

        e.tell(&samples, &fitnesses);

        assert_close(e.mean[0], 1.644_390_300_095_77, "mean[0]");
        assert_close(e.sigma, 1.0, "sigma");
        assert_close(e.cov_diag[0], 0.581_829_367_328_581_9, "cov_diag[0]");
    }

    #[test]
    fn seeded_cma_three_generation_snapshot_exact() {
        let mut e = CmaEmitter::with_bounds(1, 1.0, vec![DimensionBounds::new(0.0, 5.0)], 0.98);
        e.warm_start(&[1.0]);
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FF_EE27);

        for _ in 0..3 {
            let samples = e.ask_with_rng(8, &mut rng);
            let fitnesses: Vec<f64> = samples.iter().map(|s| -(s[0] - 2.0).powi(2)).collect();
            e.tell(&samples, &fitnesses);
        }

        assert_close(e.mean[0], 1.644_923_696_826_093, "mean[0]");
        assert_close(e.sigma, 0.941_191_999_999_999_9, "sigma");
        assert_close(e.cov_diag[0], 0.00453737301757347, "cov_diag[0]");
    }

    #[test]
    #[ignore = "empirical CMA convergence; run by stochastic-empirical workflow"]
    fn empirical_cma_multiple_generations_converge_many_seeds() {
        // Same Thompson/CMA flake pattern as test_small_scale_convergence:
        // CmaEmitter::new(1, 0.5) starts at mean=0.5, sigma=0.5, so the
        // sample cloud is [0, 1] and the fitness signal at x=2.0 is
        // invisible until sigma decays. 10×8 = 80 evals had a flake tail
        // on CI run 24954171389; bumped to sigma=1.0 + 30 generations × 16
        // samples = 480 fitness evals so the mean reliably crosses 1.0
        // toward 2.0 within the budget.
        let mut successes = 0;
        for seed in 0..50 {
            let mut e = CmaEmitter::new(1, 1.0);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..30 {
                let samples = e.ask_with_rng(16, &mut rng);
                let fitnesses: Vec<f64> = samples.iter().map(|s| -(s[0] - 2.0).powi(2)).collect();
                e.tell(&samples, &fitnesses);
            }
            if e.mean[0] > 1.0 {
                successes += 1;
            }
        }
        assert!(successes >= 45, "successes={successes}/50");
    }

    #[test]
    fn test_with_bounds_topology_params() {
        // Topology CMA-ME: cost [0.001, 5.0], time [1.0, 300.0], weight [0.1, 5.0]
        let bounds = vec![
            DimensionBounds::new(0.001, 5.0),
            DimensionBounds::new(1.0, 300.0),
            DimensionBounds::new(0.1, 5.0),
        ];
        let e = CmaEmitter::with_bounds(3, 0.3, bounds, 0.99);
        // Mean should be centered in each range
        assert!((e.mean()[0] - 2.5005).abs() < 0.01);
        assert!((e.mean()[1] - 150.5).abs() < 0.01);
        assert!((e.mean()[2] - 2.55).abs() < 0.01);

        let samples = e.ask(10);
        for s in &samples {
            assert!(s[0] >= 0.001 && s[0] <= 5.0, "cost out of bounds: {}", s[0]);
            assert!(s[1] >= 1.0 && s[1] <= 300.0, "time out of bounds: {}", s[1]);
            assert!(s[2] >= 0.1 && s[2] <= 5.0, "weight out of bounds: {}", s[2]);
        }
    }

    #[test]
    fn test_sigma_decay() {
        let bounds = vec![DimensionBounds::new(0.0, 10.0); 3];
        let mut e = CmaEmitter::with_bounds(3, 1.0, bounds, 0.95);
        let samples = e.ask(4);
        let fitnesses = vec![0.1, 0.5, 0.9, 0.3];
        e.tell(&samples, &fitnesses);
        assert!(
            (e.sigma() - 0.95).abs() < 1e-10,
            "sigma should decay by 0.95"
        );
        // After 10 generations
        for _ in 0..9 {
            let s = e.ask(4);
            e.tell(&s, &fitnesses);
        }
        assert!(
            e.sigma() < 0.65,
            "sigma should have decayed after 10 gens: {}",
            e.sigma()
        );
    }

    #[test]
    fn test_warm_start() {
        let mut e = CmaEmitter::new(3, 0.3);
        assert_eq!(e.mean(), &[0.5, 0.5, 0.5]);
        e.warm_start(&[2.0, 3.0, 1.5]);
        assert!((e.mean()[0] - 2.0).abs() < 1e-10);
        assert!((e.mean()[1] - 3.0).abs() < 1e-10);
        assert!((e.mean()[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_warm_start_clamps_to_bounds() {
        let bounds = vec![
            DimensionBounds::new(0.001, 5.0),
            DimensionBounds::new(1.0, 300.0),
            DimensionBounds::new(0.1, 5.0),
        ];
        let mut e = CmaEmitter::with_bounds(3, 0.3, bounds, 1.0);
        e.warm_start(&[100.0, -50.0, 3.0]); // extreme values should be clamped
        assert!((e.mean()[0] - 5.0).abs() < 1e-10, "should clamp to max");
        assert!((e.mean()[1] - 1.0).abs() < 1e-10, "should clamp to min");
        assert!(
            (e.mean()[2] - 3.0).abs() < 1e-10,
            "within bounds, unchanged"
        );
    }

    #[test]
    #[ignore = "empirical CMA convergence; run by stochastic-empirical workflow"]
    fn empirical_cma_small_scale_convergence_many_seeds() {
        // CMA-ME at small scale. The non-deterministic `ask()` uses
        // `rand::rng()`, and at small populations the sample noise can
        // dominate the fitness gradient: we observed both 4×5 (CI flake
        // 0.82, regressed) and 8×10 (CI flake 0.9999, near-no-move).
        // 32 samples × 20 generations = 640 fitness evals is the regime
        // where the gradient toward x=2.0 reliably overwhelms sample
        // noise on a 1D problem with sigma=1.0. Assertion relaxed to
        // 0.5 (instead of 1.0) so the budget bump produces a real
        // statement about convergence rather than just non-regression.
        let mut successes = 0;
        for seed in 0..50 {
            let bounds = vec![DimensionBounds::new(0.0, 5.0)];
            let mut e = CmaEmitter::with_bounds(1, 1.0, bounds, 0.98);
            e.warm_start(&[1.0]);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);

            for _ in 0..20 {
                let samples = e.ask_with_rng(32, &mut rng);
                let fitnesses: Vec<f64> = samples.iter().map(|s| -(s[0] - 2.0).powi(2)).collect();
                e.tell(&samples, &fitnesses);
            }
            if (e.mean()[0] - 2.0).abs() < 0.5 {
                successes += 1;
            }
        }
        assert!(successes >= 45, "successes={successes}/50");
    }
}
