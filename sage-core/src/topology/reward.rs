//! Verified dense reward for topology RL training.
//!
//! Combines execution feedback with formal verification signals.
//! Each component is formally grounded — no heuristic weights.
//!
//! Signals:
//! - **execution**: binary ground truth from sandbox (pass@1)
//! - **structural**: HybridVerifier formal checks (0.0-1.0)
//! - **density**: S_complex mathematical function (0.0-1.0)
//! - **temporal**: LTL model checking (0.0-1.0, optional)
//!
//! Combination: equal weighting across available signals. Each signal
//! is formally grounded, so no learned or tuned weights are needed.

use pyo3::prelude::*;
use tracing::instrument;

// ---------------------------------------------------------------------------
// RewardScore
// ---------------------------------------------------------------------------

/// Multi-signal reward score from topology execution.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RewardScore {
    /// Combined reward (0.0-1.0+).
    #[pyo3(get)]
    pub total: f32,
    /// Task pass@1 (0.0 or 1.0).
    #[pyo3(get)]
    pub execution: f32,
    /// HybridVerifier score (0.0-1.0).
    #[pyo3(get)]
    pub structural: f32,
    /// S_complex score (0.0-1.0).
    #[pyo3(get)]
    pub density: f32,
    /// LTL temporal score (0.0-1.0, 0.0 if not provided).
    #[pyo3(get)]
    pub temporal: f32,
    /// Number of signals that contributed.
    #[pyo3(get)]
    pub n_signals: u32,
    /// Resilience score: bonus for topologies that survived adaptation.
    #[pyo3(get)]
    pub resilience: f32,
    /// Cost efficiency score: 1.0 - tanh(cost / budget).
    #[pyo3(get)]
    pub cost_efficiency: f32,
}

#[pymethods]
impl RewardScore {
    fn __repr__(&self) -> String {
        format!(
            "RewardScore(total={:.4}, execution={:.1}, structural={:.4}, density={:.4}, temporal={:.4}, resilience={:.4}, cost_eff={:.4}, n_signals={})",
            self.total, self.execution, self.structural, self.density, self.temporal,
            self.resilience, self.cost_efficiency, self.n_signals
        )
    }
}

// ---------------------------------------------------------------------------
// TopologyReward
// ---------------------------------------------------------------------------

/// Computes verified dense reward for topology RL training.
///
/// Combines execution feedback with formal verification signals.
/// Equal weighting across available signals (no heuristic weights).
#[pyclass]
pub struct TopologyReward;

#[pymethods]
impl TopologyReward {
    #[new]
    pub fn new() -> Self {
        TopologyReward
    }

    /// Compute verified dense reward for a topology execution.
    ///
    /// Arguments:
    /// - `execution_passed`: whether the task passed (bool)
    /// - `structural_score`: HybridVerifier score (0.0-1.0, from Python)
    /// - `density_score`: S_complex score (from TopologyDensity.compute())
    /// - `temporal_score`: LtlVerifier score (0.0-1.0, from Python, optional)
    ///
    /// Returns RewardScore with full breakdown.
    #[instrument(skip(self))]
    #[pyo3(signature = (execution_passed, structural_score, density_score, temporal_score=None))]
    pub fn compute(
        &self,
        execution_passed: bool,
        structural_score: f32,
        density_score: f32,
        temporal_score: Option<f32>,
    ) -> RewardScore {
        let execution = if execution_passed { 1.0 } else { 0.0 };
        let mut n_signals = 3u32; // execution + structural + density always present

        let temporal = temporal_score.unwrap_or(0.0);
        if temporal_score.is_some() {
            n_signals += 1;
        }

        // Equal weighting across available signals (no heuristic weights).
        // Each signal is formally grounded:
        // - execution: binary ground truth from sandbox
        // - structural: HybridVerifier formal checks
        // - density: S_complex mathematical function
        // - temporal: LTL model checking
        let total = (execution + structural_score + density_score + temporal) / n_signals as f32;

        RewardScore {
            total,
            execution,
            structural: structural_score,
            density: density_score,
            temporal,
            n_signals,
            resilience: 0.0,
            cost_efficiency: 0.0,
        }
    }

    /// Compute reward with all 6 signals including resilience and cost efficiency.
    ///
    /// The resilience and cost_efficiency values are computed in Python
    /// (from trace analysis and provider costs) and passed in directly.
    /// Weights are initial values subject to ablation (see spec C2).
    #[instrument(skip(self))]
    #[pyo3(signature = (execution_passed, structural_score, density_score, temporal_score=None, resilience=0.0, cost_efficiency=1.0))]
    pub fn compute_full(
        &self,
        execution_passed: bool,
        structural_score: f32,
        density_score: f32,
        temporal_score: Option<f32>,
        resilience: f32,
        cost_efficiency: f32,
    ) -> RewardScore {
        let base = self.compute(execution_passed, structural_score, density_score, temporal_score);
        // Return with resilience and cost_efficiency filled in.
        // The Python reward.py handles the final weighted combination.
        RewardScore {
            resilience,
            cost_efficiency,
            ..base
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_signals_present() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 0.8, 0.6, Some(0.9));

        assert_eq!(score.n_signals, 4, "all 4 signals present");
        assert_eq!(score.execution, 1.0);
        assert!((score.structural - 0.8).abs() < 1e-6);
        assert!((score.density - 0.6).abs() < 1e-6);
        assert!((score.temporal - 0.9).abs() < 1e-6);

        // total = (1.0 + 0.8 + 0.6 + 0.9) / 4 = 3.3 / 4 = 0.825
        let expected = (1.0 + 0.8 + 0.6 + 0.9) / 4.0;
        assert!(
            (score.total - expected).abs() < 1e-6,
            "total should be {}, got {}",
            expected,
            score.total
        );
    }

    #[test]
    fn test_execution_only() {
        let reward = TopologyReward::new();
        // Execution passed, but structural and density are 0.
        let score = reward.compute(true, 0.0, 0.0, None);

        assert_eq!(score.n_signals, 3, "3 signals (no temporal)");
        assert_eq!(score.execution, 1.0);
        assert_eq!(score.structural, 0.0);
        assert_eq!(score.density, 0.0);
        assert_eq!(score.temporal, 0.0);

        // total = (1.0 + 0.0 + 0.0) / 3 = 0.333...
        let expected = 1.0 / 3.0;
        assert!(
            (score.total - expected).abs() < 1e-6,
            "total should be {}, got {}",
            expected,
            score.total
        );
    }

    #[test]
    fn test_failed_execution() {
        let reward = TopologyReward::new();
        // Execution failed, but structural is perfect.
        let score = reward.compute(false, 1.0, 0.5, None);

        assert_eq!(score.n_signals, 3);
        assert_eq!(score.execution, 0.0);
        assert!((score.structural - 1.0).abs() < 1e-6);
        assert!((score.density - 0.5).abs() < 1e-6);

        // total = (0.0 + 1.0 + 0.5) / 3 = 0.5
        let expected = 1.5 / 3.0;
        assert!(
            (score.total - expected).abs() < 1e-6,
            "total should be {}, got {}",
            expected,
            score.total
        );
    }

    #[test]
    fn test_with_temporal_signal() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 0.5, 0.5, Some(1.0));

        assert_eq!(score.n_signals, 4, "temporal present -> 4 signals");
        assert!((score.temporal - 1.0).abs() < 1e-6);

        // total = (1.0 + 0.5 + 0.5 + 1.0) / 4 = 0.75
        let expected = 3.0 / 4.0;
        assert!(
            (score.total - expected).abs() < 1e-6,
            "total should be {}, got {}",
            expected,
            score.total
        );
    }

    #[test]
    fn test_without_temporal_signal() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 0.8, 0.6, None);

        assert_eq!(score.n_signals, 3, "no temporal -> 3 signals");
        assert_eq!(score.temporal, 0.0, "temporal defaults to 0.0 when absent");

        // total = (1.0 + 0.8 + 0.6) / 3 = 2.4 / 3 = 0.8
        let expected = 2.4 / 3.0;
        assert!(
            (score.total - expected).abs() < 1e-6,
            "total should be {}, got {}",
            expected,
            score.total
        );
    }

    #[test]
    fn test_all_zeros() {
        let reward = TopologyReward::new();
        let score = reward.compute(false, 0.0, 0.0, Some(0.0));

        assert_eq!(score.n_signals, 4);
        assert_eq!(score.total, 0.0);
        assert_eq!(score.execution, 0.0);
        assert_eq!(score.structural, 0.0);
        assert_eq!(score.density, 0.0);
        assert_eq!(score.temporal, 0.0);
    }

    #[test]
    fn test_all_perfect() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 1.0, 1.0, Some(1.0));

        assert_eq!(score.n_signals, 4);
        assert!((score.total - 1.0).abs() < 1e-6, "perfect score should be 1.0");
    }

    #[test]
    fn test_repr() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 0.8, 0.6, None);
        let repr = score.__repr__();
        assert!(repr.starts_with("RewardScore("), "repr should start with class name");
        assert!(repr.contains("n_signals=3"), "repr should show n_signals");
    }

    #[test]
    fn test_compute_full_with_resilience() {
        let reward = TopologyReward::new();
        let score = reward.compute_full(true, 0.8, 0.6, Some(0.9), 0.5, 0.7);
        assert!((score.resilience - 0.5).abs() < 1e-6);
        assert!((score.cost_efficiency - 0.7).abs() < 1e-6);
        // Base total unchanged (compute_full delegates to compute for base signals)
        assert_eq!(score.n_signals, 4);
    }

    #[test]
    fn test_compute_full_defaults() {
        let reward = TopologyReward::new();
        let score = reward.compute_full(true, 0.8, 0.6, None, 0.0, 1.0);
        assert_eq!(score.resilience, 0.0);
        assert!((score.cost_efficiency - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_backward_compat() {
        let reward = TopologyReward::new();
        let score = reward.compute(true, 0.8, 0.6, None);
        // Existing compute() sets resilience=0, cost_efficiency=0
        assert_eq!(score.resilience, 0.0);
        assert_eq!(score.cost_efficiency, 0.0);
    }
}
