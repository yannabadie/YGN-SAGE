//! ModelRegistry — manages ModelCards and selects best models per CognitiveSystem.

use pyo3::prelude::*;
use std::collections::HashMap;
use tracing::info;

use super::model_card::{CognitiveSystem, ModelCard};

/// Running telemetry record for a single model.
#[derive(Debug, Clone, Default)]
pub struct TelemetryRecord {
    pub quality_sum: f64,
    pub cost_sum: f64,
    pub count: u32,
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

    /// Record a telemetry observation for a model.
    pub fn record_telemetry(&mut self, model_id: &str, quality: f32, cost: f32) {
        let record = self.telemetry.entry(model_id.to_string()).or_default();
        record.quality_sum += quality as f64;
        record.cost_sum += cost as f64;
        record.count += 1;
        info!(
            model = model_id,
            count = record.count,
            avg_quality = record.avg_quality() as f32,
            "telemetry_recorded"
        );
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
}
