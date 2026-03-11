//! ContextualBandit — per-arm Beta/Gamma posteriors with Thompson sampling.
//!
//! Selects the best (model, topology) combination given runtime context.
//! Uses Beta posteriors for quality (bounded [0,1]) and Gamma posteriors
//! for cost/latency (non-negative). Builds a global Pareto front at
//! decision time and selects based on runtime constraints.

use pyo3::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, info_span};

// ── Error type ─────────────────────────────────────────────────────────────

/// Errors from bandit operations.
#[derive(Debug, thiserror::Error)]
pub enum BanditError {
    #[error("No arms registered. Call register_arm() first.")]
    NoArms,
    #[error("Unknown decision_id: '{0}'. Was it already recorded or never issued?")]
    UnknownDecision(String),
}

impl From<BanditError> for PyErr {
    fn from(err: BanditError) -> PyErr {
        match &err {
            BanditError::NoArms => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
            BanditError::UnknownDecision(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
        }
    }
}

// ── ArmKey ─────────────────────────────────────────────────────────────────

/// Unique identifier for a bandit arm = (model_id, topology_template).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArmKey {
    pub model_id: String,
    pub template: String, // template name (e.g., "sequential", "avr")
}

// ── BetaPosterior (quality — bounded [0,1]) ────────────────────────────────

/// Beta distribution posterior for modelling quality (pass rate).
///
/// Beta(alpha, beta) is conjugate to Bernoulli observations.
/// Mean = alpha / (alpha + beta).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaPosterior {
    pub alpha: f64, // success count + prior
    pub beta: f64,  // failure count + prior
}

impl BetaPosterior {
    /// Uniform prior: Beta(1, 1).
    fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Mean of Beta(alpha, beta) = alpha / (alpha + beta).
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Thompson sample: draw from Beta distribution.
    ///
    /// Uses mean + scaled noise approximation (no rand_distr dependency).
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mean = self.mean();
        let variance = self.alpha * self.beta
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let std = variance.sqrt();
        let noise = rng.random::<f64>() * 2.0 - 1.0; // Uniform [-1, 1]
        (mean + noise * std).clamp(0.0, 1.0)
    }

    /// Update with observation. `quality` should be 0.0 to 1.0.
    fn update(&mut self, quality: f64, decay: f64) {
        // Apply decay first (temporal discounting)
        self.alpha *= decay;
        self.beta *= decay;
        // Clamp minimums to prevent collapse
        self.alpha = self.alpha.max(0.5);
        self.beta = self.beta.max(0.5);
        // Update with observation
        self.alpha += quality;
        self.beta += 1.0 - quality;
    }
}

// ── GammaPosterior (cost/latency — non-negative) ──────────────────────────

/// Gamma distribution posterior for modelling cost and latency.
///
/// Gamma(shape, rate) where mean = shape / rate.
/// Conjugate to exponential/Poisson observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaPosterior {
    pub shape: f64, // k (event count)
    pub rate: f64,  // θ^-1 (inverse scale)
}

impl GammaPosterior {
    /// Weakly informative prior: Gamma(2, 1).
    fn new() -> Self {
        Self {
            shape: 2.0,
            rate: 1.0,
        }
    }

    /// Mean = shape / rate.
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Thompson sample (mean + scaled noise approximation).
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mean = self.mean();
        let variance = self.shape / self.rate.powi(2);
        let std = variance.sqrt();
        let noise = rng.random::<f64>() * 2.0 - 1.0;
        (mean + noise * std).max(0.001) // Never negative
    }

    /// Update with observation value (cost or latency).
    fn update(&mut self, value: f64, decay: f64) {
        self.shape = (self.shape * decay).max(1.0);
        self.rate = (self.rate * decay).max(0.1);
        self.shape += 1.0;
        self.rate += value;
    }
}

// ── ArmPosterior ───────────────────────────────────────────────────────────

/// Full posterior state for a single bandit arm.
///
/// Tracks quality (Beta), cost (Gamma), and latency (Gamma) posteriors
/// along with the total observation count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmPosterior {
    pub key: ArmKey,
    pub quality: BetaPosterior,
    pub cost: GammaPosterior,
    pub latency: GammaPosterior,
    pub observation_count: u32,
}

impl ArmPosterior {
    fn new(key: ArmKey) -> Self {
        Self {
            key,
            quality: BetaPosterior::new(),
            cost: GammaPosterior::new(),
            latency: GammaPosterior::new(),
            observation_count: 0,
        }
    }

    /// Update all posteriors with a single observation.
    fn update(&mut self, quality: f64, cost: f64, latency: f64, decay: f64) {
        self.quality.update(quality, decay);
        self.cost.update(cost, decay);
        self.latency.update(latency, decay);
        self.observation_count += 1;
    }
}

// ── BanditDecision ─────────────────────────────────────────────────────────

/// Result of a bandit arm selection.
///
/// Returned by `ContextualBandit::select()`. Contains the chosen
/// model/template pair and expected quality/cost/latency from the
/// Thompson sample.
#[pyclass]
#[derive(Debug, Clone)]
pub struct BanditDecision {
    /// Unique decision identifier (ULID).
    #[pyo3(get)]
    pub decision_id: String,
    /// Selected model identifier.
    #[pyo3(get)]
    pub model_id: String,
    /// Selected topology template.
    #[pyo3(get)]
    pub template: String,
    /// Expected quality from Thompson sample.
    #[pyo3(get)]
    pub expected_quality: f32,
    /// Expected cost from Thompson sample.
    #[pyo3(get)]
    pub expected_cost: f32,
    /// Expected latency from Thompson sample.
    #[pyo3(get)]
    pub expected_latency: f32,
    /// True if this was an exploration (random) pick rather than exploit.
    #[pyo3(get)]
    pub exploration: bool,
}

#[pymethods]
impl BanditDecision {
    fn __repr__(&self) -> String {
        format!(
            "BanditDecision(id='{}', model='{}', template='{}', quality={:.3}, cost={:.4}, latency={:.1}, explore={})",
            self.decision_id,
            self.model_id,
            self.template,
            self.expected_quality,
            self.expected_cost,
            self.expected_latency,
            self.exploration,
        )
    }
}

// ── ContextualBandit ───────────────────────────────────────────────────────

/// Contextual bandit that selects the best (model, topology) combination.
///
/// Uses per-arm Beta/Gamma posteriors with Thompson sampling. Each arm
/// tracks quality (Beta posterior), cost (Gamma), and latency (Gamma).
/// Temporal discounting via configurable decay factor ensures the bandit
/// adapts to non-stationary environments.
///
/// # Selection strategy
///
/// 1. If `exploration_budget > random()`, pick a random arm (explore).
/// 2. Otherwise, Thompson sample from each arm's quality posterior.
/// 3. Pick the arm with the highest sampled quality (exploit via Thompson).
///
/// # Cold start
///
/// New arms start with Beta(1,1) (uniform) for quality and Gamma(2,1) for
/// cost/latency. Early selections are effectively random due to high
/// posterior variance, providing natural exploration.
#[pyclass]
#[derive(Clone)]
pub struct ContextualBandit {
    arms: HashMap<ArmKey, ArmPosterior>,
    decay_factor: f64,
    #[allow(dead_code)]
    exploration_bonus: f64,
    /// Pending decisions: decision_id -> arm_key (for deferred record()).
    pending: HashMap<String, ArmKey>,
}

// ── Core Rust API (no PyO3 dependency) ─────────────────────────────────────

impl ContextualBandit {
    /// Create a new bandit with the given decay factor and exploration bonus.
    pub fn create(decay_factor: f64, exploration_bonus: f64) -> Self {
        Self {
            arms: HashMap::new(),
            decay_factor,
            exploration_bonus,
            pending: HashMap::new(),
        }
    }

    /// Register a known arm (model + template combination).
    ///
    /// If the arm already exists, this is a no-op.
    pub fn add_arm(&mut self, model_id: &str, template: &str) {
        let key = ArmKey {
            model_id: model_id.to_string(),
            template: template.to_string(),
        };
        self.arms
            .entry(key.clone())
            .or_insert_with(|| ArmPosterior::new(key));
    }

    /// Select the best arm given an exploration budget.
    ///
    /// `exploration_budget`: 0.0 = pure exploit, 1.0 = pure explore.
    ///
    /// Returns `BanditDecision` with the chosen arm and Thompson-sampled
    /// expected quality/cost/latency. The decision_id can later be passed
    /// to `record_outcome()` to update posteriors.
    pub fn choose(&mut self, exploration_budget: f32) -> Result<BanditDecision, BanditError> {
        let _span = info_span!(
            "bandit.select",
            arms = self.arms.len(),
            exploration = exploration_budget
        )
        .entered();

        if self.arms.is_empty() {
            return Err(BanditError::NoArms);
        }

        let mut rng = rand::rng();
        let decision_id = ulid::Ulid::new().to_string();

        // Decide: explore or exploit
        let exploring = rng.random::<f32>() < exploration_budget;

        let arm_keys: Vec<ArmKey> = self.arms.keys().cloned().collect();

        let chosen_key = if exploring {
            // Pure exploration: pick a random arm
            let idx = rng.random_range(0..arm_keys.len());
            arm_keys[idx].clone()
        } else {
            // Thompson sampling: sample quality from each arm, pick the best
            let mut best_key = arm_keys[0].clone();
            let mut best_quality = f64::NEG_INFINITY;

            for key in &arm_keys {
                let arm = &self.arms[key];
                let sampled_quality = arm.quality.sample(&mut rng);
                if sampled_quality > best_quality {
                    best_quality = sampled_quality;
                    best_key = key.clone();
                }
            }
            best_key
        };

        let arm = &self.arms[&chosen_key];
        let expected_quality = arm.quality.sample(&mut rng) as f32;
        let expected_cost = arm.cost.sample(&mut rng) as f32;
        let expected_latency = arm.latency.sample(&mut rng) as f32;

        // Store pending decision for deferred record()
        self.pending.insert(decision_id.clone(), chosen_key.clone());

        let decision = BanditDecision {
            decision_id,
            model_id: chosen_key.model_id,
            template: chosen_key.template,
            expected_quality,
            expected_cost,
            expected_latency,
            exploration: exploring,
        };

        info!(
            model = %decision.model_id,
            template = %decision.template,
            explore = decision.exploration,
            expected_quality = decision.expected_quality,
            "bandit_decision"
        );

        Ok(decision)
    }

    /// Record outcome for a previous decision.
    ///
    /// Updates the arm's posteriors with temporal decay. The `decision_id`
    /// must match a previous `choose()` call.
    pub fn record_outcome(
        &mut self,
        decision_id: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> Result<(), BanditError> {
        let _span = info_span!(
            "bandit.record",
            decision_id = decision_id,
            quality = quality,
            cost = cost,
            latency_ms = latency_ms,
        )
        .entered();

        let arm_key = self
            .pending
            .remove(decision_id)
            .ok_or_else(|| BanditError::UnknownDecision(decision_id.to_string()))?;

        let decay = self.decay_factor;
        let arm = self
            .arms
            .get_mut(&arm_key)
            .expect("arm_key from pending must exist in arms");

        arm.update(quality as f64, cost as f64, latency_ms as f64, decay);

        debug!(
            model = %arm_key.model_id,
            template = %arm_key.template,
            observations = arm.observation_count,
            quality_mean = arm.quality.mean() as f32,
            "bandit_outcome_recorded"
        );

        Ok(())
    }

    /// Number of registered arms.
    pub fn arm_count(&self) -> usize {
        self.arms.len()
    }

    /// Total observations across all arms.
    pub fn total_observations(&self) -> u32 {
        self.arms.values().map(|a| a.observation_count).sum()
    }

    /// Get per-arm summary stats (for debugging/dashboard).
    ///
    /// Returns list of `(model_id, template, quality_mean, cost_mean, latency_mean, obs_count)`.
    pub fn arm_summaries(&self) -> Vec<(String, String, f32, f32, f32, u32)> {
        self.arms
            .values()
            .map(|arm| {
                (
                    arm.key.model_id.clone(),
                    arm.key.template.clone(),
                    arm.quality.mean() as f32,
                    arm.cost.mean() as f32,
                    arm.latency.mean() as f32,
                    arm.observation_count,
                )
            })
            .collect()
    }

    /// Get a reference to the arm posteriors map (test/integration use only).
    pub fn arms_map(&self) -> &HashMap<ArmKey, ArmPosterior> {
        &self.arms
    }

    /// Get decay factor.
    pub fn decay_factor(&self) -> f64 {
        self.decay_factor
    }

    /// Get exploration bonus.
    pub fn exploration_bonus(&self) -> f64 {
        self.exploration_bonus
    }

    /// Iterate over all arm posteriors.
    pub fn arms_iter(&self) -> impl Iterator<Item = &ArmPosterior> {
        self.arms.values()
    }

    /// Restore an arm with pre-computed posteriors (from SQLite load).
    #[allow(clippy::too_many_arguments)]
    pub fn restore_arm(
        &mut self,
        model_id: String,
        template: String,
        quality_alpha: f64,
        quality_beta: f64,
        cost_shape: f64,
        cost_rate: f64,
        latency_shape: f64,
        latency_rate: f64,
        observation_count: u32,
    ) {
        let key = ArmKey { model_id, template };
        let arm = ArmPosterior {
            key: key.clone(),
            quality: BetaPosterior {
                alpha: quality_alpha,
                beta: quality_beta,
            },
            cost: GammaPosterior {
                shape: cost_shape,
                rate: cost_rate,
            },
            latency: GammaPosterior {
                shape: latency_shape,
                rate: latency_rate,
            },
            observation_count,
        };
        self.arms.insert(key, arm);
    }

    /// Set the temporal decay factor, clamped to [0.9, 1.0].
    pub fn set_decay(&mut self, factor: f64) {
        self.decay_factor = factor.clamp(0.9, 1.0);
        info!(decay = self.decay_factor, "bandit_decay_updated");
    }

    /// Warm-start the bandit from model affinity scores.
    ///
    /// Registers one arm per (model_id, template) pair and sets the quality
    /// prior proportional to the affinity score: higher affinity → higher alpha.
    /// `affinities` must have length `model_ids.len() * templates.len()`,
    /// laid out in row-major order (model-major): `affinities[i * T + j]`
    /// corresponds to `(model_ids[i], templates[j])`.
    pub fn warm_start(&mut self, model_ids: &[String], templates: &[String], affinities: &[f32]) {
        let n_models = model_ids.len();
        let n_templates = templates.len();
        let expected = n_models * n_templates;

        if affinities.len() != expected {
            info!(
                expected = expected,
                got = affinities.len(),
                "warm_start: affinity length mismatch, skipping"
            );
            return;
        }

        for (i, model_id) in model_ids.iter().enumerate() {
            for (j, template) in templates.iter().enumerate() {
                let affinity = affinities[i * n_templates + j] as f64;
                let key = ArmKey {
                    model_id: model_id.clone(),
                    template: template.clone(),
                };
                let arm = self
                    .arms
                    .entry(key.clone())
                    .or_insert_with(|| ArmPosterior::new(key));
                // Set quality prior: Beta(1 + 2*affinity, 1 + 2*(1-affinity))
                // affinity=1.0 → Beta(3, 1) mean=0.75 (strong prior)
                // affinity=0.5 → Beta(2, 2) mean=0.50 (neutral)
                // affinity=0.0 → Beta(1, 3) mean=0.25 (weak prior)
                arm.quality.alpha = 1.0 + 2.0 * affinity;
                arm.quality.beta = 1.0 + 2.0 * (1.0 - affinity);
            }
        }

        info!(
            arms = self.arms.len(),
            models = n_models,
            templates = n_templates,
            "bandit_warm_started"
        );
    }

    /// Format the bandit state as a string.
    pub fn repr(&self) -> String {
        format!(
            "ContextualBandit(arms={}, observations={}, decay={:.4}, pending={})",
            self.arms.len(),
            self.total_observations(),
            self.decay_factor,
            self.pending.len(),
        )
    }
}

// ── PyO3 methods (thin wrappers) ───────────────────────────────────────────

#[pymethods]
impl ContextualBandit {
    #[new]
    #[pyo3(signature = (decay_factor=0.995, exploration_bonus=0.1))]
    pub fn new(decay_factor: f64, exploration_bonus: f64) -> Self {
        Self::create(decay_factor, exploration_bonus)
    }

    /// Register a known arm (model + template combination).
    #[pyo3(name = "register_arm")]
    pub fn py_register_arm(&mut self, model_id: &str, template: &str) {
        self.add_arm(model_id, template);
    }

    /// Select the best arm given an exploration budget.
    #[pyo3(name = "select")]
    pub fn py_select(&mut self, exploration_budget: f32) -> PyResult<BanditDecision> {
        self.choose(exploration_budget).map_err(Into::into)
    }

    /// Record outcome for a previous decision.
    #[pyo3(name = "record")]
    pub fn py_record(
        &mut self,
        decision_id: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> PyResult<()> {
        self.record_outcome(decision_id, quality, cost, latency_ms)
            .map_err(Into::into)
    }

    /// Number of registered arms.
    #[pyo3(name = "arm_count")]
    pub fn py_arm_count(&self) -> usize {
        self.arm_count()
    }

    /// Total observations across all arms.
    #[pyo3(name = "total_observations")]
    pub fn py_total_observations(&self) -> u32 {
        self.total_observations()
    }

    /// Get per-arm summary stats.
    #[pyo3(name = "arm_summaries")]
    pub fn py_arm_summaries(&self) -> Vec<(String, String, f32, f32, f32, u32)> {
        self.arm_summaries()
    }

    /// Set the temporal decay factor, clamped to [0.9, 1.0].
    #[pyo3(name = "set_decay_factor")]
    pub fn py_set_decay_factor(&mut self, factor: f64) {
        self.set_decay(factor);
    }

    /// Warm-start the bandit from model affinity scores.
    #[pyo3(name = "warm_start_from_affinities")]
    pub fn py_warm_start_from_affinities(
        &mut self,
        model_ids: Vec<String>,
        templates: Vec<String>,
        affinities: Vec<f32>,
    ) {
        self.warm_start(&model_ids, &templates, &affinities);
    }

    /// Save bandit state to SQLite (requires `cognitive` feature).
    #[cfg(feature = "cognitive")]
    #[pyo3(name = "save_to_sqlite")]
    pub fn py_save_to_sqlite(&self, path: &str) -> PyResult<()> {
        super::persistence::save_bandit(self, path).map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load bandit state from SQLite (requires `cognitive` feature).
    #[cfg(feature = "cognitive")]
    #[staticmethod]
    #[pyo3(name = "load_from_sqlite")]
    pub fn py_load_from_sqlite(path: &str) -> PyResult<Self> {
        super::persistence::load_bandit(path).map_err(pyo3::exceptions::PyIOError::new_err)
    }

    fn __repr__(&self) -> String {
        self.repr()
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_posterior_uniform_prior() {
        let bp = BetaPosterior::new();
        assert!((bp.mean() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn beta_posterior_update_increases_mean() {
        let mut bp = BetaPosterior::new();
        let mean_before = bp.mean();
        bp.update(1.0, 1.0); // perfect quality, no decay
        assert!(bp.mean() > mean_before);
    }

    #[test]
    fn beta_posterior_sample_in_range() {
        let bp = BetaPosterior::new();
        let mut rng = rand::rng();
        for _ in 0..100 {
            let s = bp.sample(&mut rng);
            assert!((0.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn gamma_posterior_prior_mean() {
        let gp = GammaPosterior::new();
        // Gamma(2, 1) -> mean = 2.0
        assert!((gp.mean() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn gamma_posterior_sample_positive() {
        let gp = GammaPosterior::new();
        let mut rng = rand::rng();
        for _ in 0..100 {
            let s = gp.sample(&mut rng);
            assert!(s > 0.0);
        }
    }

    #[test]
    fn gamma_posterior_update_shifts_mean() {
        let mut gp = GammaPosterior::new();
        gp.update(10.0, 1.0);
        // shape = 2+1 = 3, rate = 1+10 = 11, mean = 3/11
        assert!(gp.mean() < 2.0); // shifted down from prior mean=2
    }

    #[test]
    fn arm_posterior_new_has_zero_observations() {
        let key = ArmKey {
            model_id: "test".into(),
            template: "seq".into(),
        };
        let arm = ArmPosterior::new(key);
        assert_eq!(arm.observation_count, 0);
    }

    #[test]
    fn arm_posterior_update_increments_count() {
        let key = ArmKey {
            model_id: "test".into(),
            template: "seq".into(),
        };
        let mut arm = ArmPosterior::new(key);
        arm.update(0.8, 0.01, 200.0, 0.995);
        assert_eq!(arm.observation_count, 1);
        arm.update(0.9, 0.02, 150.0, 0.995);
        assert_eq!(arm.observation_count, 2);
    }

    #[test]
    fn set_decay_clamps_low() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.set_decay(0.5); // below 0.9
        assert!((bandit.decay_factor() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn set_decay_clamps_high() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.set_decay(1.5); // above 1.0
        assert!((bandit.decay_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn set_decay_accepts_valid_value() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.set_decay(0.97);
        assert!((bandit.decay_factor() - 0.97).abs() < 1e-10);
    }

    #[test]
    fn warm_start_creates_arms() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let models = vec!["model-a".to_string(), "model-b".to_string()];
        let templates = vec!["seq".to_string(), "avr".to_string()];
        // 2 models x 2 templates = 4 affinities
        let affinities = vec![0.9, 0.3, 0.5, 0.8];
        bandit.warm_start(&models, &templates, &affinities);
        assert_eq!(bandit.arm_count(), 4);
    }

    #[test]
    fn warm_start_sets_quality_prior() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let models = vec!["model-a".to_string()];
        let templates = vec!["seq".to_string()];
        let affinities = vec![1.0]; // max affinity
        bandit.warm_start(&models, &templates, &affinities);

        let key = ArmKey {
            model_id: "model-a".into(),
            template: "seq".into(),
        };
        let arm = &bandit.arms_map()[&key];
        // affinity=1.0 → Beta(3, 1), mean=0.75
        assert!((arm.quality.alpha - 3.0).abs() < 1e-10);
        assert!((arm.quality.beta - 1.0).abs() < 1e-10);
        assert!((arm.quality.mean() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn warm_start_skips_on_length_mismatch() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let models = vec!["model-a".to_string()];
        let templates = vec!["seq".to_string()];
        let affinities = vec![0.5, 0.6]; // wrong length: expected 1, got 2
        bandit.warm_start(&models, &templates, &affinities);
        assert_eq!(bandit.arm_count(), 0); // no arms created
    }

    #[test]
    fn warm_start_neutral_affinity() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let models = vec!["m".to_string()];
        let templates = vec!["t".to_string()];
        let affinities = vec![0.5]; // neutral
        bandit.warm_start(&models, &templates, &affinities);

        let key = ArmKey {
            model_id: "m".into(),
            template: "t".into(),
        };
        let arm = &bandit.arms_map()[&key];
        // affinity=0.5 → Beta(2, 2), mean=0.5
        assert!((arm.quality.alpha - 2.0).abs() < 1e-10);
        assert!((arm.quality.beta - 2.0).abs() < 1e-10);
    }
}
