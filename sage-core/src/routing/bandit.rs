//! ContextualBandit — per-arm Thompson sampling with optional context bias.
//!
//! Each arm is a `(model_id, topology_template)` pair with Beta quality and
//! Gamma cost/latency posteriors plus restorable running context statistics.
//! `choose()` explores randomly or selects the arm with the highest sampled
//! quality. `choose_contextual()` uses the same quality sample multiplied by
//! `1 + max(cosine_similarity(context, arm_context_mean), 0)`.
//!
//! Cost and latency posteriors are updated and sampled into `BanditDecision`
//! for telemetry, but they do not currently affect selection. No global
//! Pareto front or constraint-aware selection is built here; that belongs to
//! the roadmap-A24 multi-objective routing follow-up.

use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
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
    #[error(
        "Off-policy outcome for decision '{decision_id}': selected ({selected_model_id}, {selected_template}) but executed ({executed_model_id}, {executed_template})"
    )]
    OffPolicyOutcome {
        decision_id: String,
        selected_model_id: String,
        selected_template: String,
        executed_model_id: String,
        executed_template: String,
    },
}

impl From<BanditError> for PyErr {
    fn from(err: BanditError) -> PyErr {
        match &err {
            BanditError::NoArms => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
            BanditError::UnknownDecision(_) | BanditError::OffPolicyOutcome { .. } => {
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
    /// Uses Box-Muller Gaussian approximation (no rand_distr dependency).
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mean = self.mean();
        let variance = self.alpha * self.beta
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let std = variance.sqrt();
        let u1: f64 = rng.random::<f64>().max(1e-15); // avoid ln(0)
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (mean + std * z).clamp(0.0, 1.0)
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

    /// Thompson sample (Box-Muller Gaussian approximation).
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mean = self.mean();
        let variance = self.shape / self.rate.powi(2);
        let std = variance.sqrt();
        let u1: f64 = rng.random::<f64>().max(1e-15); // avoid ln(0)
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (mean + std * z).max(0.001) // Never negative
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
/// along with the total observation count and aggregated task-context
/// statistics for contextual arm selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmPosterior {
    pub key: ArmKey,
    pub quality: BetaPosterior,
    pub cost: GammaPosterior,
    pub latency: GammaPosterior,
    pub observation_count: u32,
    /// Running sum of context feature vectors (for computing mean context).
    pub context_sum: Vec<f64>,
    /// Number of context-bearing observations (may differ from observation_count
    /// if some calls omitted context).
    pub context_count: u32,
}

impl ArmPosterior {
    fn new(key: ArmKey) -> Self {
        Self {
            key,
            quality: BetaPosterior::new(),
            cost: GammaPosterior::new(),
            latency: GammaPosterior::new(),
            observation_count: 0,
            context_sum: Vec::new(),
            context_count: 0,
        }
    }

    /// Update all posteriors with a single observation.
    fn update(&mut self, quality: f64, cost: f64, latency: f64, decay: f64) {
        self.quality.update(quality, decay);
        self.cost.update(cost, decay);
        self.latency.update(latency, decay);
        self.observation_count += 1;
    }

    /// Update the running context statistics with a new context vector.
    ///
    /// Accumulates into `context_sum` for later mean computation.
    fn update_context(&mut self, context: &[f32]) {
        if context.is_empty() {
            return;
        }
        if self.context_sum.is_empty() {
            self.context_sum = vec![0.0; context.len()];
        }
        // If dimensions changed, reset (defensive — shouldn't happen in practice)
        if self.context_sum.len() != context.len() {
            self.context_sum = vec![0.0; context.len()];
            self.context_count = 0;
        }
        for (s, &c) in self.context_sum.iter_mut().zip(context.iter()) {
            *s += c as f64;
        }
        self.context_count += 1;
    }

    /// Compute the mean context vector, or `None` if no context has been recorded.
    fn context_mean(&self) -> Option<Vec<f64>> {
        if self.context_count == 0 || self.context_sum.is_empty() {
            return None;
        }
        let n = self.context_count as f64;
        Some(self.context_sum.iter().map(|s| s / n).collect())
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
    /// Task context features used for this decision (empty if no context).
    #[pyo3(get)]
    pub context: Vec<f32>,
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

// ── Cosine similarity ─────────────────────────────────────────────────────

/// Cosine similarity between a context vector (f32) and a mean vector (f64).
///
/// Returns a value in [-1, 1]. Returns 0.0 if either vector is zero-length
/// or has zero norm.
fn cosine_similarity_f64(a: &[f32], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        dot += xf * y;
        norm_a += xf * xf;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    dot / denom
}

fn contextual_score(sampled_quality: f64, context: &[f32], arm: &ArmPosterior) -> f64 {
    let similarity = arm
        .context_mean()
        .map_or(0.0, |mean| cosine_similarity_f64(context, &mean).max(0.0));
    sampled_quality * (1.0 + similarity)
}

/// Info stored for a pending decision (between choose and record_outcome).
#[derive(Debug, Clone)]
struct PendingInfo {
    arm_key: ArmKey,
    /// Context features from the choose call (empty if no context was provided).
    context: Vec<f32>,
}

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
/// # Contextual selection (`choose_contextual`)
///
/// When a task-feature context vector is provided, the bandit computes
/// cosine similarity between the input context and each arm's historical
/// mean context. This similarity acts as a multiplicative bonus on the
/// Thompson-sampled quality score, biasing selection toward arms that
/// have historically performed well on similar tasks.
///
/// # Cold start
///
/// New arms start with Beta(1,1) (uniform) for quality and Gamma(2,1) for
/// cost/latency. Early selections are effectively random due to high
/// posterior variance, providing natural exploration. Arms with no context
/// history receive no context bonus (similarity = 0).
#[pyclass]
#[derive(Clone)]
pub struct ContextualBandit {
    arms: HashMap<ArmKey, ArmPosterior>,
    decay_factor: f64,
    #[allow(dead_code)]
    exploration_bonus: f64,
    /// Pending decisions: decision_id -> pending info (for deferred record()).
    pending: HashMap<String, PendingInfo>,
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

    fn sorted_arm_keys(&self) -> Vec<ArmKey> {
        let mut arm_keys: Vec<ArmKey> = self.arms.keys().cloned().collect();
        arm_keys.sort_by(|a, b| {
            a.model_id
                .cmp(&b.model_id)
                .then_with(|| a.template.cmp(&b.template))
        });
        arm_keys
    }

    /// Select the best arm given an exploration budget.
    ///
    /// `exploration_budget`: 0.0 = pure exploit, 1.0 = pure explore.
    ///
    /// Returns `BanditDecision` with the chosen arm and Thompson-sampled
    /// expected quality/cost/latency. The decision_id can later be passed
    /// to `record_outcome()` to update posteriors.
    pub fn choose(&mut self, exploration_budget: f32) -> Result<BanditDecision, BanditError> {
        let mut rng = rand::rng();
        self.choose_with_rng(exploration_budget, &mut rng)
    }

    pub fn choose_with_rng<R: Rng>(
        &mut self,
        exploration_budget: f32,
        rng: &mut R,
    ) -> Result<BanditDecision, BanditError> {
        let _span = info_span!(
            "bandit.select",
            arms = self.arms.len(),
            exploration = exploration_budget
        )
        .entered();

        let arm_keys = self.sorted_arm_keys();
        let decision =
            self.choose_from_candidates_with_rng(exploration_budget, &[], arm_keys, rng)?;

        info!(
            model = %decision.model_id,
            template = %decision.template,
            explore = decision.exploration,
            expected_quality = decision.expected_quality,
            "bandit_decision"
        );

        Ok(decision)
    }

    /// Select the best arm with task-context bias.
    ///
    /// Like `choose()`, but biases Thompson-sampled arm selection using
    /// cosine similarity between the provided `context` and each arm's
    /// historical mean context. Arms that have performed well on similar
    /// tasks (by context) receive a multiplicative bonus.
    ///
    /// `context`: a small feature vector (e.g., `[system_tier, task_length, node_count]`).
    /// `exploration_budget`: 0.0 = pure exploit, 1.0 = pure explore.
    ///
    /// Falls back to standard `choose()` if `context` is empty.
    pub fn choose_contextual(
        &mut self,
        exploration_budget: f32,
        context: &[f32],
    ) -> Result<BanditDecision, BanditError> {
        let mut rng = rand::rng();
        self.choose_contextual_with_rng(exploration_budget, context, &mut rng)
    }

    pub fn choose_contextual_with_rng<R: Rng>(
        &mut self,
        exploration_budget: f32,
        context: &[f32],
        rng: &mut R,
    ) -> Result<BanditDecision, BanditError> {
        // If no context provided, fall back to standard Thompson sampling
        if context.is_empty() {
            return self.choose_with_rng(exploration_budget, rng);
        }

        let _span = info_span!(
            "bandit.select_contextual",
            arms = self.arms.len(),
            exploration = exploration_budget,
            context_dim = context.len(),
        )
        .entered();

        let arm_keys = self.sorted_arm_keys();
        let decision =
            self.choose_from_candidates_with_rng(exploration_budget, context, arm_keys, rng)?;

        info!(
            model = %decision.model_id,
            template = %decision.template,
            explore = decision.exploration,
            expected_quality = decision.expected_quality,
            context_dim = context.len(),
            "bandit_contextual_decision"
        );

        Ok(decision)
    }

    fn choose_from_candidates_with_rng<R: Rng>(
        &mut self,
        exploration_budget: f32,
        context: &[f32],
        mut arm_keys: Vec<ArmKey>,
        rng: &mut R,
    ) -> Result<BanditDecision, BanditError> {
        arm_keys.sort_by(|a, b| {
            a.model_id
                .cmp(&b.model_id)
                .then_with(|| a.template.cmp(&b.template))
        });

        if arm_keys.is_empty() {
            return Err(BanditError::NoArms);
        }

        let decision_id = ulid::Ulid::new().to_string();
        let exploring = rng.random::<f32>() < exploration_budget;

        let chosen_key = if exploring {
            let idx = rng.random_range(0..arm_keys.len());
            arm_keys[idx].clone()
        } else {
            let mut best_key = arm_keys[0].clone();
            let mut best_score = f64::NEG_INFINITY;

            for key in &arm_keys {
                let arm = &self.arms[key];
                let sampled_quality = arm.quality.sample(rng);
                let score = if context.is_empty() {
                    sampled_quality
                } else {
                    contextual_score(sampled_quality, context, arm)
                };

                if score > best_score {
                    best_score = score;
                    best_key = key.clone();
                }
            }
            best_key
        };

        let arm = &self.arms[&chosen_key];
        let expected_quality = arm.quality.sample(rng) as f32;
        let expected_cost = arm.cost.sample(rng) as f32;
        let expected_latency = arm.latency.sample(rng) as f32;

        if self.pending.len() > 10_000 {
            let drain_count = self.pending.len() - 5_000;
            let keys_to_remove: Vec<String> =
                self.pending.keys().take(drain_count).cloned().collect();
            for key in keys_to_remove {
                self.pending.remove(&key);
            }
        }

        self.pending.insert(
            decision_id.clone(),
            PendingInfo {
                arm_key: chosen_key.clone(),
                context: context.to_vec(),
            },
        );

        Ok(BanditDecision {
            decision_id,
            model_id: chosen_key.model_id,
            template: chosen_key.template,
            expected_quality,
            expected_cost,
            expected_latency,
            exploration: exploring,
            context: context.to_vec(),
        })
    }

    /// Select the best arm for one executed template.
    pub fn choose_for_template(
        &mut self,
        exploration_budget: f32,
        template: &str,
    ) -> Result<BanditDecision, BanditError> {
        let mut rng = rand::rng();
        self.choose_for_template_with_rng(exploration_budget, template, &mut rng)
    }

    pub fn choose_for_template_with_rng<R: Rng>(
        &mut self,
        exploration_budget: f32,
        template: &str,
        rng: &mut R,
    ) -> Result<BanditDecision, BanditError> {
        let _span = info_span!(
            "bandit.select_template",
            arms = self.arms.len(),
            exploration = exploration_budget,
            template = template,
        )
        .entered();

        let arm_keys: Vec<ArmKey> = self
            .arms
            .keys()
            .filter(|key| key.template == template)
            .cloned()
            .collect();
        let decision =
            self.choose_from_candidates_with_rng(exploration_budget, &[], arm_keys, rng)?;

        debug_assert_eq!(decision.template, template);
        info!(
            model = %decision.model_id,
            template = %decision.template,
            explore = decision.exploration,
            expected_quality = decision.expected_quality,
            "bandit_template_decision"
        );
        Ok(decision)
    }

    /// Select the best arm for one executed template with task-context bias.
    pub fn choose_contextual_for_template(
        &mut self,
        exploration_budget: f32,
        context: &[f32],
        template: &str,
    ) -> Result<BanditDecision, BanditError> {
        let mut rng = rand::rng();
        self.choose_contextual_for_template_with_rng(
            exploration_budget,
            context,
            template,
            &mut rng,
        )
    }

    pub fn choose_contextual_for_template_with_rng<R: Rng>(
        &mut self,
        exploration_budget: f32,
        context: &[f32],
        template: &str,
        rng: &mut R,
    ) -> Result<BanditDecision, BanditError> {
        if context.is_empty() {
            return self.choose_for_template_with_rng(exploration_budget, template, rng);
        }

        let _span = info_span!(
            "bandit.select_contextual_template",
            arms = self.arms.len(),
            exploration = exploration_budget,
            context_dim = context.len(),
            template = template,
        )
        .entered();

        let arm_keys: Vec<ArmKey> = self
            .arms
            .keys()
            .filter(|key| key.template == template)
            .cloned()
            .collect();
        let decision =
            self.choose_from_candidates_with_rng(exploration_budget, context, arm_keys, rng)?;

        debug_assert_eq!(decision.template, template);
        info!(
            model = %decision.model_id,
            template = %decision.template,
            explore = decision.exploration,
            expected_quality = decision.expected_quality,
            context_dim = context.len(),
            "bandit_contextual_template_decision"
        );
        Ok(decision)
    }

    /// Record outcome for a previous decision.
    ///
    /// Updates the arm's posteriors with temporal decay. The `decision_id`
    /// must match a previous `choose()` or `choose_contextual()` call.
    /// If the original decision carried context features, they are folded
    /// into the arm's running context statistics.
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

        let pending_info = self
            .pending
            .remove(decision_id)
            .ok_or_else(|| BanditError::UnknownDecision(decision_id.to_string()))?;

        let arm_key = pending_info.arm_key;
        let decay = self.decay_factor;
        let arm = match self.arms.get_mut(&arm_key) {
            Some(a) => a,
            None => {
                return Err(BanditError::UnknownDecision(format!(
                    "arm {:?} removed between choose() and record()",
                    arm_key
                )))
            }
        };

        arm.update(quality as f64, cost as f64, latency_ms as f64, decay);

        // Update context statistics if the decision carried context features
        if !pending_info.context.is_empty() {
            arm.update_context(&pending_info.context);
        }

        debug!(
            model = %arm_key.model_id,
            template = %arm_key.template,
            observations = arm.observation_count,
            quality_mean = arm.quality.mean() as f32,
            context_count = arm.context_count,
            "bandit_outcome_recorded"
        );

        Ok(())
    }

    /// Record an outcome only if it matches the selected full arm.
    pub fn record_outcome_checked(
        &mut self,
        decision_id: &str,
        executed_model_id: &str,
        executed_template: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> Result<(), BanditError> {
        let pending_info = self
            .pending
            .get(decision_id)
            .ok_or_else(|| BanditError::UnknownDecision(decision_id.to_string()))?;
        let selected = &pending_info.arm_key;

        if selected.model_id != executed_model_id || selected.template != executed_template {
            return Err(BanditError::OffPolicyOutcome {
                decision_id: decision_id.to_string(),
                selected_model_id: selected.model_id.clone(),
                selected_template: selected.template.clone(),
                executed_model_id: executed_model_id.to_string(),
                executed_template: executed_template.to_string(),
            });
        }

        self.record_outcome(decision_id, quality, cost, latency_ms)
    }

    /// Cancel a pending decision without updating posteriors.
    pub fn cancel_decision(&mut self, decision_id: &str) -> bool {
        self.pending.remove(decision_id).is_some()
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

    /// Get quality posterior mean for a specific (model, template) arm.
    ///
    /// Returns `None` if the arm doesn't exist or has no observations.
    pub fn get_quality_mean(&self, model_id: &str, template: &str) -> Option<f64> {
        let key = ArmKey {
            model_id: model_id.to_string(),
            template: template.to_string(),
        };
        self.arms.get(&key).map(|arm| arm.quality.mean())
    }

    /// Thompson-sample quality for a specific (model, template) arm.
    ///
    /// Returns a single draw from the arm's Beta posterior. This preserves
    /// exploration: bad models get low draws (~0.05) 99% of the time but
    /// retain a small chance (~0.1%) of being retried — enabling automatic
    /// rediscovery if the model improves server-side.
    /// Based on MonoScale (arXiv 2601.23219): trust-region Thompson sampling.
    pub fn sample_quality(&self, model_id: &str, template: &str) -> f64 {
        let key = ArmKey {
            model_id: model_id.to_string(),
            template: template.to_string(),
        };
        match self.arms.get(&key) {
            Some(arm) => {
                let mut rng = rand::rng();
                arm.quality.sample(&mut rng)
            }
            None => 0.5, // neutral prior for unknown arms
        }
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
    /// Reconstruct an arm from persisted state.
    ///
    /// `context_sum` and `context_count` are required so the bandit's
    /// cosine-similarity context bias survives save/load. Pre-2026-04-26
    /// this method silently dropped both fields (initialised to empty
    /// vec / 0), erasing all contextual learning across restarts. See
    /// `persistence::migrate_add_context_columns` for the SQLite-side
    /// schema-evolution path.
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
        context_sum: Vec<f64>,
        context_count: u32,
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
            context_sum,
            context_count,
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

    #[pyo3(name = "select_with_seed")]
    #[pyo3(signature = (exploration_budget, seed))]
    pub fn py_select_with_seed(
        &mut self,
        exploration_budget: f32,
        seed: u64,
    ) -> PyResult<BanditDecision> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.choose_with_rng(exploration_budget, &mut rng)
            .map_err(Into::into)
    }

    /// Select the best arm with task-context bias.
    ///
    /// `context` is a small feature vector (e.g., `[system_tier, task_length, node_count]`).
    /// When context is provided, arms whose historical contexts are similar get a boost.
    /// Falls back to standard Thompson sampling if context is empty.
    #[pyo3(name = "select_with_context")]
    #[pyo3(signature = (exploration_budget, context=vec![]))]
    pub fn py_select_with_context(
        &mut self,
        exploration_budget: f32,
        context: Vec<f32>,
    ) -> PyResult<BanditDecision> {
        if context.is_empty() {
            self.choose(exploration_budget).map_err(Into::into)
        } else {
            self.choose_contextual(exploration_budget, &context)
                .map_err(Into::into)
        }
    }

    #[pyo3(name = "select_with_context_with_seed")]
    #[pyo3(signature = (exploration_budget, seed, context=vec![]))]
    pub fn py_select_with_context_with_seed(
        &mut self,
        exploration_budget: f32,
        seed: u64,
        context: Vec<f32>,
    ) -> PyResult<BanditDecision> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.choose_contextual_with_rng(exploration_budget, &context, &mut rng)
            .map_err(Into::into)
    }

    /// Select the best arm for a template with optional task-context bias.
    #[pyo3(name = "select_with_context_for_template")]
    #[pyo3(signature = (exploration_budget, template, context=vec![]))]
    pub fn py_select_with_context_for_template(
        &mut self,
        exploration_budget: f32,
        template: &str,
        context: Vec<f32>,
    ) -> PyResult<BanditDecision> {
        self.choose_contextual_for_template(exploration_budget, &context, template)
            .map_err(Into::into)
    }

    #[pyo3(name = "select_with_context_for_template_with_seed")]
    #[pyo3(signature = (exploration_budget, template, seed, context=vec![]))]
    pub fn py_select_with_context_for_template_with_seed(
        &mut self,
        exploration_budget: f32,
        template: &str,
        seed: u64,
        context: Vec<f32>,
    ) -> PyResult<BanditDecision> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.choose_contextual_for_template_with_rng(
            exploration_budget,
            &context,
            template,
            &mut rng,
        )
        .map_err(Into::into)
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

    /// Record outcome only if the executed full arm matches the pending decision.
    #[pyo3(name = "record_outcome_checked")]
    pub fn py_record_outcome_checked(
        &mut self,
        decision_id: &str,
        executed_model_id: &str,
        executed_template: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> PyResult<()> {
        self.record_outcome_checked(
            decision_id,
            executed_model_id,
            executed_template,
            quality,
            cost,
            latency_ms,
        )
        .map_err(Into::into)
    }

    /// Cancel a pending decision without updating posteriors.
    #[pyo3(name = "cancel_decision")]
    pub fn py_cancel_decision(&mut self, decision_id: &str) -> bool {
        self.cancel_decision(decision_id)
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

    /// Get quality posterior mean for a specific (model, template) arm.
    #[pyo3(name = "get_quality_mean")]
    pub fn py_get_quality_mean(&self, model_id: &str, template: &str) -> Option<f64> {
        self.get_quality_mean(model_id, template)
    }

    /// Thompson-sample quality from arm's Beta posterior (preserves exploration).
    #[pyo3(name = "sample_quality")]
    pub fn py_sample_quality(&self, model_id: &str, template: &str) -> f64 {
        self.sample_quality(model_id, template)
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn bandit_with_arms(order: &[(&str, &str)]) -> ContextualBandit {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        for (model_id, template) in order {
            bandit.add_arm(model_id, template);
        }
        bandit
    }

    fn train_arm_on_context(
        bandit: &mut ContextualBandit,
        model_id: &str,
        template: &str,
        context: &[f32],
    ) {
        let key = ArmKey {
            model_id: model_id.to_string(),
            template: template.to_string(),
        };
        let decay = bandit.decay_factor();
        let arm = bandit
            .arms
            .get_mut(&key)
            .expect("test arm should be registered before training");
        for _ in 0..16 {
            arm.update(0.95, 0.01, 100.0, decay);
            arm.update_context(context);
        }
    }

    fn assert_same_selection(left: &BanditDecision, right: &BanditDecision) {
        assert_eq!(left.model_id, right.model_id);
        assert_eq!(left.template, right.template);
        assert_eq!(left.exploration, right.exploration);
        assert_eq!(left.expected_quality, right.expected_quality);
        assert_eq!(left.expected_cost, right.expected_cost);
        assert_eq!(left.expected_latency, right.expected_latency);
    }

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

    // ── Cosine similarity tests ──────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = &[1.0_f32, 2.0, 3.0];
        let b = &[1.0_f64, 2.0, 3.0];
        let sim = cosine_similarity_f64(a, b);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "identical vectors should have sim=1.0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = &[1.0_f32, 0.0];
        let b = &[0.0_f64, 1.0];
        let sim = cosine_similarity_f64(a, b);
        assert!(
            sim.abs() < 1e-10,
            "orthogonal vectors should have sim=0.0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = &[1.0_f32, 2.0];
        let b = &[-1.0_f64, -2.0];
        let sim = cosine_similarity_f64(a, b);
        assert!(
            (sim - (-1.0)).abs() < 1e-10,
            "opposite vectors should have sim=-1.0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a = &[0.0_f32, 0.0];
        let b = &[1.0_f64, 2.0];
        assert_eq!(cosine_similarity_f64(a, b), 0.0);
    }

    #[test]
    fn cosine_similarity_empty_returns_zero() {
        let a: &[f32] = &[];
        let b: &[f64] = &[];
        assert_eq!(cosine_similarity_f64(a, b), 0.0);
    }

    #[test]
    fn cosine_similarity_mismatched_length_returns_zero() {
        let a = &[1.0_f32, 2.0];
        let b = &[1.0_f64, 2.0, 3.0];
        assert_eq!(cosine_similarity_f64(a, b), 0.0);
    }

    #[test]
    fn cosine_similarity_mechanics_exact() {
        assert_eq!(cosine_similarity_f64(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity_f64(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity_f64(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
        assert_eq!(cosine_similarity_f64(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn contextual_score_bonus_mechanics_exact() {
        let key = ArmKey {
            model_id: "arm-s1".into(),
            template: "single_agent".into(),
        };
        let mut arm = ArmPosterior::new(key);
        arm.update_context(&[1.0, 0.0]);

        let sampled_quality = 0.25;
        let matching = contextual_score(sampled_quality, &[1.0, 0.0], &arm);
        let orthogonal = contextual_score(sampled_quality, &[0.0, 1.0], &arm);
        let opposite = contextual_score(sampled_quality, &[-1.0, 0.0], &arm);

        assert!((matching - 0.5).abs() < 1e-12);
        assert!((orthogonal - 0.25).abs() < 1e-12);
        assert!((opposite - 0.25).abs() < 1e-12);
    }

    // ── Arm context tracking tests ───────────────────────────────────────

    #[test]
    fn arm_context_starts_empty() {
        let key = ArmKey {
            model_id: "m".into(),
            template: "t".into(),
        };
        let arm = ArmPosterior::new(key);
        assert_eq!(arm.context_count, 0);
        assert!(arm.context_sum.is_empty());
        assert!(arm.context_mean().is_none());
    }

    #[test]
    fn arm_context_update_accumulates() {
        let key = ArmKey {
            model_id: "m".into(),
            template: "t".into(),
        };
        let mut arm = ArmPosterior::new(key);
        arm.update_context(&[2.0, 4.0, 6.0]);
        arm.update_context(&[4.0, 6.0, 8.0]);
        assert_eq!(arm.context_count, 2);
        let mean = arm.context_mean().unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 5.0).abs() < 1e-10);
        assert!((mean[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn arm_context_empty_input_is_noop() {
        let key = ArmKey {
            model_id: "m".into(),
            template: "t".into(),
        };
        let mut arm = ArmPosterior::new(key);
        arm.update_context(&[]);
        assert_eq!(arm.context_count, 0);
        assert!(arm.context_mean().is_none());
    }

    #[test]
    fn arm_keys_sort_by_model_id_then_template() {
        let bandit = bandit_with_arms(&[
            ("z-model", "sequential"),
            ("a-model", "parallel"),
            ("a-model", "avr"),
            ("m-model", "single_agent"),
        ]);

        let sorted: Vec<(String, String)> = bandit
            .sorted_arm_keys()
            .into_iter()
            .map(|key| (key.model_id, key.template))
            .collect();

        assert_eq!(
            sorted,
            vec![
                ("a-model".to_string(), "avr".to_string()),
                ("a-model".to_string(), "parallel".to_string()),
                ("m-model".to_string(), "single_agent".to_string()),
                ("z-model".to_string(), "sequential".to_string()),
            ]
        );
    }

    #[test]
    fn choose_with_rng_is_deterministic_across_registration_order() {
        let order_a = [
            ("z-model", "sequential"),
            ("a-model", "parallel"),
            ("a-model", "avr"),
            ("m-model", "single_agent"),
        ];
        let order_b = [
            ("m-model", "single_agent"),
            ("a-model", "avr"),
            ("z-model", "sequential"),
            ("a-model", "parallel"),
        ];
        let mut left = bandit_with_arms(&order_a);
        let mut right = bandit_with_arms(&order_b);
        let mut left_rng = ChaCha8Rng::seed_from_u64(42);
        let mut right_rng = ChaCha8Rng::seed_from_u64(42);

        let left_decision = left.choose_with_rng(0.0, &mut left_rng).unwrap();
        let right_decision = right.choose_with_rng(0.0, &mut right_rng).unwrap();

        assert_ne!(left_decision.decision_id, right_decision.decision_id);
        assert_same_selection(&left_decision, &right_decision);
    }

    #[test]
    fn choose_contextual_with_rng_is_deterministic_across_registration_order() {
        let order_a = [
            ("z-model", "sequential"),
            ("a-model", "parallel"),
            ("a-model", "avr"),
            ("m-model", "single_agent"),
        ];
        let order_b = [
            ("a-model", "avr"),
            ("m-model", "single_agent"),
            ("a-model", "parallel"),
            ("z-model", "sequential"),
        ];
        let mut left = bandit_with_arms(&order_a);
        let mut right = bandit_with_arms(&order_b);
        let mut left_rng = ChaCha8Rng::seed_from_u64(42);
        let mut right_rng = ChaCha8Rng::seed_from_u64(42);

        let left_decision = left
            .choose_contextual_with_rng(0.0, &[1.0, 10.0, 1.0], &mut left_rng)
            .unwrap();
        let right_decision = right
            .choose_contextual_with_rng(0.0, &[1.0, 10.0, 1.0], &mut right_rng)
            .unwrap();

        assert_same_selection(&left_decision, &right_decision);
        assert_eq!(left_decision.context, right_decision.context);
    }

    #[test]
    fn choose_contextual_for_template_with_rng_filters_and_sorts_candidates() {
        let order_a = [
            ("z-model", "single_agent"),
            ("a-model", "single_agent"),
            ("other-model", "sequential"),
        ];
        let order_b = [
            ("other-model", "sequential"),
            ("a-model", "single_agent"),
            ("z-model", "single_agent"),
        ];
        let mut left = bandit_with_arms(&order_a);
        let mut right = bandit_with_arms(&order_b);
        let mut left_rng = ChaCha8Rng::seed_from_u64(123);
        let mut right_rng = ChaCha8Rng::seed_from_u64(123);

        let left_decision = left
            .choose_contextual_for_template_with_rng(
                0.0,
                &[2.0, 0.0, 1.0],
                "single_agent",
                &mut left_rng,
            )
            .unwrap();
        let right_decision = right
            .choose_contextual_for_template_with_rng(
                0.0,
                &[2.0, 0.0, 1.0],
                "single_agent",
                &mut right_rng,
            )
            .unwrap();

        assert_eq!(left_decision.template, "single_agent");
        assert_eq!(right_decision.template, "single_agent");
        assert_ne!(left_decision.model_id, "other-model");
        assert_ne!(right_decision.model_id, "other-model");
        assert_same_selection(&left_decision, &right_decision);
    }

    #[test]
    fn py_seeded_selection_wrappers_are_reproducible() {
        let order_a = [
            ("z-model", "sequential"),
            ("a-model", "parallel"),
            ("a-model", "avr"),
            ("m-model", "single_agent"),
        ];
        let order_b = [
            ("m-model", "single_agent"),
            ("a-model", "avr"),
            ("z-model", "sequential"),
            ("a-model", "parallel"),
        ];
        let mut left = bandit_with_arms(&order_a);
        let mut right = bandit_with_arms(&order_b);

        let left_decision = left.py_select_with_seed(0.0, 7).unwrap();
        let right_decision = right.py_select_with_seed(0.0, 7).unwrap();

        assert_ne!(left_decision.decision_id, right_decision.decision_id);
        assert_same_selection(&left_decision, &right_decision);
    }

    // ── choose_contextual tests ──────────────────────────────────────────

    #[test]
    fn choose_contextual_empty_context_falls_back() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "seq");
        let decision = bandit.choose_contextual(0.0, &[]).unwrap();
        assert!(!decision.decision_id.is_empty());
        assert!(decision.context.is_empty());
    }

    #[test]
    fn choose_contextual_returns_decision_with_context() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "seq");
        bandit.add_arm("model-b", "avr");
        let ctx = vec![1.0_f32, 100.0, 3.0];
        let decision = bandit.choose_contextual(0.0, &ctx).unwrap();
        assert!(!decision.decision_id.is_empty());
        assert_eq!(decision.context, ctx);
    }

    #[test]
    fn choose_contextual_no_arms_errors() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let result = bandit.choose_contextual(0.0, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn seeded_contextual_template_selection_sequence_is_exact() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("a-model", "single_agent");
        bandit.add_arm("z-model", "single_agent");
        bandit.add_arm("other-model", "sequential");
        let mut rng = ChaCha8Rng::seed_from_u64(0xA25_5EED);
        let mut selected = Vec::new();

        for _ in 0..10 {
            let decision = bandit
                .choose_contextual_for_template_with_rng(
                    1.0,
                    &[1.0, 42.0, 0.0],
                    "single_agent",
                    &mut rng,
                )
                .unwrap();

            assert_eq!(decision.template, "single_agent");
            assert_ne!(decision.model_id, "other-model");
            selected.push(decision.model_id);
        }

        // Fixed by ChaCha8Rng seed + lexicographic candidate ordering.
        let expected: Vec<&str> = vec![
            "a-model", "a-model", "a-model", "z-model", "a-model", "a-model", "z-model", "a-model",
            "z-model", "z-model",
        ];
        assert_eq!(selected, expected);
    }

    #[test]
    fn checked_record_rejects_mismatched_model_without_update() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "single_agent");
        let before = bandit
            .get_quality_mean("model-a", "single_agent")
            .expect("registered arm should have a prior mean");

        let decision = bandit
            .choose_contextual_for_template(0.0, &[1.0, 10.0, 0.0], "single_agent")
            .unwrap();
        let err = bandit
            .record_outcome_checked(
                &decision.decision_id,
                "model-b",
                "single_agent",
                1.0,
                1.0,
                100.0,
            )
            .unwrap_err();

        match err {
            BanditError::OffPolicyOutcome {
                decision_id,
                selected_model_id,
                selected_template,
                executed_model_id,
                executed_template,
            } => {
                assert_eq!(decision_id, decision.decision_id);
                assert_eq!(selected_model_id, "model-a");
                assert_eq!(selected_template, "single_agent");
                assert_eq!(executed_model_id, "model-b");
                assert_eq!(executed_template, "single_agent");
            }
            other => panic!("expected OffPolicyOutcome, got {other:?}"),
        }
        assert_eq!(bandit.total_observations(), 0);
        let after = bandit
            .get_quality_mean("model-a", "single_agent")
            .expect("registered arm should remain present");
        assert_eq!(after, before);
    }

    #[test]
    fn checked_record_rejects_mismatched_template_without_update() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "single_agent");

        let decision = bandit
            .choose_contextual_for_template(0.0, &[1.0, 10.0, 0.0], "single_agent")
            .unwrap();
        let err = bandit
            .record_outcome_checked(
                &decision.decision_id,
                "model-a",
                "sequential",
                1.0,
                1.0,
                100.0,
            )
            .unwrap_err();

        match err {
            BanditError::OffPolicyOutcome {
                selected_model_id,
                selected_template,
                executed_model_id,
                executed_template,
                ..
            } => {
                assert_eq!(selected_model_id, "model-a");
                assert_eq!(selected_template, "single_agent");
                assert_eq!(executed_model_id, "model-a");
                assert_eq!(executed_template, "sequential");
            }
            other => panic!("expected OffPolicyOutcome, got {other:?}"),
        }
        assert_eq!(bandit.total_observations(), 0);
    }

    #[test]
    fn checked_record_updates_matching_arm() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "single_agent");
        let before = bandit
            .get_quality_mean("model-a", "single_agent")
            .expect("registered arm should have a prior mean");

        let decision = bandit
            .choose_contextual_for_template(0.0, &[1.0, 10.0, 0.0], "single_agent")
            .unwrap();
        bandit
            .record_outcome_checked(
                &decision.decision_id,
                "model-a",
                "single_agent",
                1.0,
                1.0,
                100.0,
            )
            .unwrap();

        assert_eq!(bandit.total_observations(), 1);
        let after = bandit
            .get_quality_mean("model-a", "single_agent")
            .expect("registered arm should remain present");
        assert!(after > before, "quality mean should increase after success");
    }

    #[test]
    fn record_outcome_updates_context_stats() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "seq");

        let ctx = vec![2.0_f32, 50.0, 3.0];
        let decision = bandit.choose_contextual(0.0, &ctx).unwrap();
        bandit
            .record_outcome(&decision.decision_id, 0.9, 0.01, 100.0)
            .unwrap();

        let key = ArmKey {
            model_id: "model-a".into(),
            template: "seq".into(),
        };
        let arm = &bandit.arms_map()[&key];
        assert_eq!(arm.context_count, 1);
        let mean = arm.context_mean().unwrap();
        assert!((mean[0] - 2.0).abs() < 1e-10);
        assert!((mean[1] - 50.0).abs() < 1e-10);
        assert!((mean[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn choose_without_context_does_not_update_context_stats() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("model-a", "seq");

        let decision = bandit.choose(0.0).unwrap();
        bandit
            .record_outcome(&decision.decision_id, 0.9, 0.01, 100.0)
            .unwrap();

        let key = ArmKey {
            model_id: "model-a".into(),
            template: "seq".into(),
        };
        let arm = &bandit.arms_map()[&key];
        assert_eq!(arm.context_count, 0);
        assert!(arm.context_mean().is_none());
    }

    #[test]
    #[ignore = "empirical contextual bandit convergence; run by stochastic-empirical workflow"]
    fn empirical_contextual_bandit_matching_context_wins_many_seeds() {
        let mut passing_seeds = 0;

        for seed in 0..100 {
            let mut bandit = ContextualBandit::create(0.999, 0.1);
            bandit.add_arm("arm-s1", "single_agent");
            bandit.add_arm("arm-s3", "single_agent");
            train_arm_on_context(&mut bandit, "arm-s1", "single_agent", &[1.0, 0.0]);
            train_arm_on_context(&mut bandit, "arm-s3", "single_agent", &[0.0, 1.0]);

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut arm_s1_wins = 0;
            for _ in 0..50 {
                let decision = bandit
                    .choose_contextual_for_template_with_rng(
                        0.0,
                        &[1.0, 0.0],
                        "single_agent",
                        &mut rng,
                    )
                    .unwrap();
                if decision.model_id == "arm-s1" {
                    arm_s1_wins += 1;
                }
            }

            if arm_s1_wins >= 35 {
                passing_seeds += 1;
            }
        }

        assert!(passing_seeds >= 90, "passing_seeds={passing_seeds}/100");
    }

    #[test]
    fn seeded_contextual_selection_prefers_matching_arm_exact() {
        let mut bandit = ContextualBandit::create(0.999, 0.1);
        bandit.add_arm("arm-s1", "single_agent");
        bandit.add_arm("arm-s3", "single_agent");
        train_arm_on_context(&mut bandit, "arm-s1", "single_agent", &[1.0, 0.0]);
        train_arm_on_context(&mut bandit, "arm-s3", "single_agent", &[0.0, 1.0]);

        const BANDIT_A25_SEED: u64 = 0xBAD_1A25;
        let mut rng = ChaCha8Rng::seed_from_u64(BANDIT_A25_SEED);
        let decision = bandit
            .choose_contextual_for_template_with_rng(0.0, &[1.0, 0.0], "single_agent", &mut rng)
            .unwrap();

        assert_eq!(decision.model_id, "arm-s1");
        assert_eq!(decision.template, "single_agent");
        assert!(!decision.exploration);
    }
}
