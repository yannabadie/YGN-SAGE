//! ModelRegistry — manages ModelCards and selects best models per CognitiveSystem.

use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use tracing::info;

use super::model_card::{CognitiveSystem, ModelCard};

/// Running telemetry record for a single model.
#[derive(Debug, Clone, Default)]
pub struct TelemetryRecord {
    pub quality_sum: f64,
    pub cost_sum: f64,
    pub count: u32,
    /// Recent latencies for P95 estimation (bounded ring buffer, last 100).
    /// Uses VecDeque for O(1) pop_front.
    pub latencies: VecDeque<f32>,
}

impl TelemetryRecord {
    /// Observed average quality (0.0 if no observations).
    pub fn avg_quality(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.quality_sum / self.count as f64
        }
    }

    /// Approximate P95 latency from recent samples.
    pub fn latency_p95(&self) -> f32 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.latencies.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() - 1) as f64 * 0.95).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    cards: HashMap<String, ModelCard>,
    telemetry: HashMap<String, TelemetryRecord>,
}

#[pymethods]
impl ModelRegistry {
    #[staticmethod]
    pub fn from_toml_file(path: &str) -> PyResult<Self> {
        let cards_vec = ModelCard::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let cards = cards_vec.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok(Self {
            cards,
            telemetry: HashMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    #[pyo3(name = "is_empty")]
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<ModelCard> {
        self.cards.get(id).cloned()
    }

    pub fn register(&mut self, card: ModelCard) {
        self.cards.insert(card.id.clone(), card);
    }

    pub fn unregister(&mut self, id: &str) {
        self.cards.remove(id);
    }

    /// Return all cards sorted by affinity for the given system (descending).
    pub fn select_for_system(&self, system: CognitiveSystem) -> Vec<ModelCard> {
        let mut candidates: Vec<_> = self.cards.values().cloned().collect();
        candidates.sort_by(|a, b| {
            b.affinity_for(system)
                .partial_cmp(&a.affinity_for(system))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Return all registered models (unordered).
    pub fn all_models(&self) -> Vec<ModelCard> {
        self.cards.values().cloned().collect()
    }

    pub fn list_ids(&self) -> Vec<String> {
        self.cards.keys().cloned().collect()
    }

    /// Record a telemetry observation (quality + cost) for a model.
    #[pyo3(name = "record_telemetry")]
    pub fn py_record_telemetry(&mut self, model_id: &str, quality: f32, cost: f32) {
        self.record_telemetry(model_id, quality, cost);
    }

    /// Record telemetry with latency (full version, exposed to Python).
    #[pyo3(name = "record_telemetry_full")]
    pub fn py_record_telemetry_full(&mut self, model_id: &str, quality: f32, cost: f32, latency_ms: f32) {
        self.record_telemetry_full(model_id, quality, cost, latency_ms);
    }

    /// Get observed P95 latency for a model (0.0 if no observations).
    #[pyo3(name = "observed_latency_p95")]
    pub fn py_observed_latency_p95(&self, model_id: &str) -> f32 {
        self.observed_latency_p95(model_id)
    }

    /// Select best model for a given task domain within budget.
    #[pyo3(name = "select_best_for_domain")]
    pub fn py_select_best_for_domain(&self, domain: &str, max_cost_usd: f32) -> Option<ModelCard> {
        self.select_best_for_domain(domain, max_cost_usd)
    }

    /// Get calibrated affinity blending card prior with telemetry observations.
    ///
    /// Weight w = min(count / 50, 0.8). Returns (1 - w) * card_affinity + w * observed_quality.
    /// Falls back to card affinity if model_id is unknown.
    #[pyo3(name = "calibrated_affinity")]
    pub fn py_calibrated_affinity(&self, model_id: &str, system: CognitiveSystem) -> f32 {
        self.calibrated_affinity(model_id, system)
    }

    fn __repr__(&self) -> String {
        format!("ModelRegistry(models={})", self.cards.len())
    }

    fn __len__(&self) -> usize {
        self.cards.len()
    }
}

impl ModelRegistry {
    /// Load from a TOML string (for tests and embedding).
    pub fn from_toml_str(toml_str: &str) -> Result<Self, toml::de::Error> {
        let cards_vec = ModelCard::parse_toml(toml_str)?;
        let cards = cards_vec.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok(Self {
            cards,
            telemetry: HashMap::new(),
        })
    }

    /// Record a telemetry observation for a model (without latency).
    pub fn record_telemetry(&mut self, model_id: &str, quality: f32, cost: f32) {
        self.record_telemetry_full(model_id, quality, cost, 0.0);
    }

    /// Record telemetry with latency (full version).
    pub fn record_telemetry_full(&mut self, model_id: &str, quality: f32, cost: f32, latency_ms: f32) {
        let record = self.telemetry.entry(model_id.to_string()).or_default();
        record.quality_sum += quality as f64;
        record.cost_sum += cost as f64;
        record.count += 1;
        // Only record real latencies — guard against 0.0 from record_telemetry() delegation
        if latency_ms > 0.0 {
            if record.latencies.len() >= 100 {
                record.latencies.pop_front(); // O(1) with VecDeque
            }
            record.latencies.push_back(latency_ms);
        }
        info!(model = model_id, count = record.count, "telemetry_recorded");
    }

    /// Get observed P95 latency for a model (0.0 if no observations).
    pub fn observed_latency_p95(&self, model_id: &str) -> f32 {
        self.telemetry.get(model_id).map(|r| r.latency_p95()).unwrap_or(0.0)
    }

    /// Infer CognitiveSystem from a domain name.
    fn system_for_domain(domain: &str) -> CognitiveSystem {
        match domain {
            "math" | "formal" => CognitiveSystem::S3,
            "code" | "reasoning" | "tool_use" => CognitiveSystem::S2,
            _ => CognitiveSystem::S1,
        }
    }

    fn domain_routing_score(&self, card: &ModelCard, domain: &str, max_cost: f32) -> f32 {
        let domain_score = card.domain_score(domain);
        let system = Self::system_for_domain(domain);
        let affinity = self.calibrated_affinity(&card.id, system);
        let cost_norm = card.estimate_cost(1000, 500) / max_cost;
        domain_score * 0.6 + affinity * 0.3 + (1.0 - cost_norm) * 0.1
    }

    /// Select best model for a given task domain within budget.
    ///
    /// Scoring: domain_score * 0.6 + calibrated_system_affinity * 0.3 + (1 - cost_norm) * 0.1
    /// System inferred from domain: math/formal → S3, code/reasoning → S2, else → S1.
    pub fn select_best_for_domain(&self, domain: &str, max_cost_usd: f32) -> Option<ModelCard> {
        let mut candidates: Vec<_> = self.cards.values()
            .filter(|c| max_cost_usd <= 0.0 || c.estimate_cost(1000, 500) <= max_cost_usd)
            .cloned()
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let max_cost = candidates.iter()
            .map(|c| c.estimate_cost(1000, 500))
            .fold(0.001_f32, f32::max);

        candidates.sort_by(|a, b| {
            let score_a = self.domain_routing_score(a, domain, max_cost);
            let score_b = self.domain_routing_score(b, domain, max_cost);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.into_iter().next()
    }

    /// Calibrated affinity blending card prior with telemetry observations.
    ///
    /// Weight w = min(count / 50, 0.8). Returns (1 - w) * card_affinity + w * observed_quality.
    pub fn calibrated_affinity(&self, model_id: &str, system: CognitiveSystem) -> f32 {
        let card_affinity = self
            .cards
            .get(model_id)
            .map(|c| c.affinity_for(system))
            .unwrap_or(0.0);

        let Some(record) = self.telemetry.get(model_id) else {
            return card_affinity;
        };

        if record.count == 0 {
            return card_affinity;
        }

        let w = (record.count as f32 / 50.0).min(0.8);
        let observed = record.avg_quality() as f32;
        (1.0 - w) * card_affinity + w * observed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_toml() -> &'static str {
        r#"
            [[models]]
            id = "fast-model"
            provider = "test"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.3
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.9
            s2_affinity = 0.5
            s3_affinity = 0.2
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
        "#
    }

    #[test]
    fn telemetry_record_default() {
        let rec = TelemetryRecord::default();
        assert_eq!(rec.count, 0);
        assert!((rec.avg_quality() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn record_telemetry_increments_count() {
        let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        reg.record_telemetry("fast-model", 0.8, 0.01);
        reg.record_telemetry("fast-model", 0.9, 0.02);
        let rec = &reg.telemetry["fast-model"];
        assert_eq!(rec.count, 2);
        assert!((rec.avg_quality() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn calibrated_affinity_no_telemetry_returns_card() {
        let reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        let aff = reg.calibrated_affinity("fast-model", CognitiveSystem::S1);
        assert!((aff - 0.9).abs() < 0.001); // pure card affinity
    }

    #[test]
    fn calibrated_affinity_blends_with_telemetry() {
        let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        // Record 25 observations of quality=0.5
        for _ in 0..25 {
            reg.record_telemetry("fast-model", 0.5, 0.01);
        }
        // w = min(25/50, 0.8) = 0.5
        // calibrated = (1-0.5)*0.9 + 0.5*0.5 = 0.45 + 0.25 = 0.70
        let aff = reg.calibrated_affinity("fast-model", CognitiveSystem::S1);
        assert!((aff - 0.70).abs() < 0.01);
    }

    #[test]
    fn calibrated_affinity_caps_weight_at_80_percent() {
        let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        // Record 100 observations (above 50 threshold) of quality=1.0
        for _ in 0..100 {
            reg.record_telemetry("fast-model", 1.0, 0.01);
        }
        // w = min(100/50, 0.8) = 0.8
        // calibrated = (1-0.8)*0.9 + 0.8*1.0 = 0.18 + 0.80 = 0.98
        let aff = reg.calibrated_affinity("fast-model", CognitiveSystem::S1);
        assert!((aff - 0.98).abs() < 0.01);
    }

    #[test]
    fn calibrated_affinity_unknown_model_returns_zero() {
        let reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        let aff = reg.calibrated_affinity("nonexistent", CognitiveSystem::S1);
        assert!((aff - 0.0).abs() < 0.001);
    }

    #[test]
    fn telemetry_tracks_latency() {
        let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        // Need enough samples for P95 to be meaningful
        for i in 0..20 {
            reg.record_telemetry_full("fast-model", 0.8, 0.01, 100.0 + i as f32 * 10.0);
        }
        // 20 samples: 100..290, P95 idx = floor(19*0.95) = 18 → sorted[18] = 280
        let latency = reg.observed_latency_p95("fast-model");
        assert!(latency > 200.0, "P95 of 100..290 should be > 200, got {}", latency);
    }

    #[test]
    fn observed_latency_unknown_model_returns_zero() {
        let reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        assert!((reg.observed_latency_p95("nonexistent") - 0.0).abs() < 0.001);
    }

    #[test]
    fn record_telemetry_without_latency_does_not_corrupt_p95() {
        let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
        reg.record_telemetry("fast-model", 0.8, 0.01);
        reg.record_telemetry("fast-model", 0.9, 0.02);
        reg.record_telemetry_full("fast-model", 0.7, 0.01, 200.0);
        let latency = reg.observed_latency_p95("fast-model");
        assert!((latency - 200.0).abs() < 0.001);
    }

    fn two_model_toml() -> &'static str {
        r#"
            [[models]]
            id = "math-model"
            provider = "test"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.5
            cost_input_per_m = 1.0
            cost_output_per_m = 2.0
            latency_ttft_ms = 500.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.5
            s3_affinity = 0.9
            recommended_topologies = []
            supports_tools = true
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
            safety_rating = 0.8
            [models.domain_scores]
            math = 0.94

            [[models]]
            id = "general-model"
            provider = "test"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.5
            cost_input_per_m = 0.5
            cost_output_per_m = 1.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 200.0
            s1_affinity = 0.9
            s2_affinity = 0.5
            s3_affinity = 0.2
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
        "#
    }

    #[test]
    fn select_best_for_domain_uses_domain_score() {
        let reg = ModelRegistry::from_toml_str(two_model_toml()).unwrap();
        let best = reg.select_best_for_domain("math", 10.0);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "math-model");
    }

    #[test]
    fn select_best_for_domain_unknown_domain() {
        let reg = ModelRegistry::from_toml_str(two_model_toml()).unwrap();
        let best = reg.select_best_for_domain("unknown", 10.0);
        assert!(best.is_some());
    }

    #[test]
    fn select_best_for_domain_no_budget_limit() {
        let reg = ModelRegistry::from_toml_str(two_model_toml()).unwrap();
        let best = reg.select_best_for_domain("math", 0.0);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "math-model");
    }
}
