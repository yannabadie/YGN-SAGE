//! ModelCard — structured capability descriptor for LLM models.
//!
//! Inspired by Google A2A Agent Cards, specialized for cognitive routing.
//! Each card captures benchmark scores, cost/latency profiles, and S1/S2/S3
//! affinity scores that drive the adaptive router's model selection.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

fn default_safety() -> f32 {
    0.5
}

// ── CognitiveSystem ─────────────────────────────────────────────────────────

/// Kahneman-inspired cognitive modes for task routing.
///
/// - **S1** — Fast / Intuitive: simple lookups, factual recall, low-latency.
/// - **S2** — Deliberate / Tools: multi-step reasoning, code generation, tool use.
/// - **S3** — Formal / Reasoning: proofs, Z3 verification, deep chain-of-thought.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveSystem {
    S1 = 1,
    S2 = 2,
    S3 = 3,
}

impl fmt::Display for CognitiveSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CognitiveSystem::S1 => write!(f, "S1"),
            CognitiveSystem::S2 => write!(f, "S2"),
            CognitiveSystem::S3 => write!(f, "S3"),
        }
    }
}

#[pymethods]
impl CognitiveSystem {
    fn __repr__(&self) -> String {
        format!("CognitiveSystem.{}", self)
    }
}

// ── ModelCard ────────────────────────────────────────────────────────────────

/// Structured capability card for an LLM model.
///
/// Combines benchmark scores, cost/latency profiles, capability flags,
/// and S1/S2/S3 affinity scores. Parsed from TOML `[[models]]` arrays.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Unique model identifier (e.g. "gemini-2.5-flash").
    #[pyo3(get)]
    pub id: String,

    /// Provider name (e.g. "google", "openai").
    #[pyo3(get)]
    pub provider: String,

    /// Model family (e.g. "gemini-2.5", "gpt-5").
    #[pyo3(get)]
    pub family: String,

    // ── Benchmark scores (0.0–1.0) ──────────────────────────────────────
    /// Code generation benchmark score.
    #[pyo3(get)]
    pub code_score: f32,

    /// Reasoning / chain-of-thought benchmark score.
    #[pyo3(get)]
    pub reasoning_score: f32,

    /// Tool use / function calling benchmark score.
    #[pyo3(get)]
    pub tool_use_score: f32,

    /// Math benchmark score.
    #[pyo3(get)]
    pub math_score: f32,

    /// Z3 / formal verification strength score.
    #[pyo3(get)]
    pub formal_z3_strength: f32,

    // ── Cost & latency ──────────────────────────────────────────────────
    /// Input token cost per million tokens (USD).
    #[pyo3(get)]
    pub cost_input_per_m: f32,

    /// Output token cost per million tokens (USD).
    #[pyo3(get)]
    pub cost_output_per_m: f32,

    /// Time to first token (milliseconds).
    #[pyo3(get)]
    pub latency_ttft_ms: f32,

    /// Output throughput (tokens per second).
    #[pyo3(get)]
    pub tokens_per_sec: f32,

    // ── Cognitive affinities (0.0–1.0) ──────────────────────────────────
    /// Affinity for S1 (fast/intuitive) tasks.
    #[pyo3(get)]
    pub s1_affinity: f32,

    /// Affinity for S2 (deliberate/tools) tasks.
    #[pyo3(get)]
    pub s2_affinity: f32,

    /// Affinity for S3 (formal/reasoning) tasks.
    #[pyo3(get)]
    pub s3_affinity: f32,

    // ── Topology & capabilities ─────────────────────────────────────────
    /// Recommended topology types (e.g. ["sequential", "avr", "parallel"]).
    #[pyo3(get)]
    pub recommended_topologies: Vec<String>,

    /// Whether the model supports tool/function calling.
    #[pyo3(get)]
    pub supports_tools: bool,

    /// Whether the model supports structured JSON output mode.
    #[pyo3(get)]
    pub supports_json_mode: bool,

    /// Whether the model supports vision/image inputs.
    #[pyo3(get)]
    pub supports_vision: bool,

    /// Maximum context window size (tokens).
    #[pyo3(get)]
    pub context_window: u32,

    // ── Domain-specific scores (0.0–1.0) ──────────────────────────────
    /// Per-domain benchmark scores (e.g. "math" → 0.94, "code" → 0.87).
    /// Parsed from TOML `[models.domain_scores]` sub-table.
    #[pyo3(get)]
    #[serde(default)]
    pub domain_scores: HashMap<String, f32>,

    /// Safety rating (0.0–1.0). Higher = safer output, fewer refusals.
    #[pyo3(get)]
    #[serde(default = "default_safety")]
    pub safety_rating: f32,
}

#[pymethods]
impl ModelCard {
    /// Return the cognitive system with the highest affinity.
    ///
    /// Ties are broken in order: S1 > S2 > S3 (prefer cheaper/faster).
    pub fn best_system(&self) -> CognitiveSystem {
        if self.s1_affinity >= self.s2_affinity && self.s1_affinity >= self.s3_affinity {
            CognitiveSystem::S1
        } else if self.s2_affinity >= self.s3_affinity {
            CognitiveSystem::S2
        } else {
            CognitiveSystem::S3
        }
    }

    /// Return the affinity score for a given cognitive system.
    pub fn affinity_for(&self, system: CognitiveSystem) -> f32 {
        match system {
            CognitiveSystem::S1 => self.s1_affinity,
            CognitiveSystem::S2 => self.s2_affinity,
            CognitiveSystem::S3 => self.s3_affinity,
        }
    }

    /// Estimate the cost (USD) for a request with the given token counts.
    ///
    /// Uses per-million pricing: `(input * cost_input + output * cost_output) / 1_000_000`.
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f32 {
        (input_tokens as f32 * self.cost_input_per_m
            + output_tokens as f32 * self.cost_output_per_m)
            / 1_000_000.0
    }

    /// Get domain-specific score. Returns 0.5 (neutral) if domain unknown.
    pub fn domain_score(&self, domain: &str) -> f32 {
        self.domain_scores.get(domain).copied().unwrap_or(0.5)
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelCard(id='{}', provider='{}', s1={:.2}, s2={:.2}, s3={:.2})",
            self.id, self.provider, self.s1_affinity, self.s2_affinity, self.s3_affinity,
        )
    }
}

// ── TOML deserialization ─────────────────────────────────────────────────────

/// Wrapper for deserializing a TOML file containing a `[[models]]` array.
#[derive(Deserialize)]
struct CardsFile {
    models: Vec<ModelCard>,
}

impl ModelCard {
    /// Parse a `[[models]]` TOML string into a list of `ModelCard`s.
    pub fn parse_toml(toml_str: &str) -> Result<Vec<Self>, toml::de::Error> {
        let file: CardsFile = toml::from_str(toml_str)?;
        Ok(file.models)
    }

    /// Read a TOML file from disk and parse its `[[models]]` array.
    pub fn load_from_file(path: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let cards = Self::parse_toml(&content)?;
        Ok(cards)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_card(s1: f32, s2: f32, s3: f32) -> ModelCard {
        ModelCard {
            id: "test".into(),
            provider: "test".into(),
            family: "test".into(),
            code_score: 0.5,
            reasoning_score: 0.5,
            tool_use_score: 0.5,
            math_score: 0.5,
            formal_z3_strength: 0.5,
            cost_input_per_m: 1.0,
            cost_output_per_m: 2.0,
            latency_ttft_ms: 500.0,
            tokens_per_sec: 100.0,
            s1_affinity: s1,
            s2_affinity: s2,
            s3_affinity: s3,
            recommended_topologies: vec![],
            supports_tools: true,
            supports_json_mode: false,
            supports_vision: false,
            context_window: 128000,
            domain_scores: HashMap::new(),
            safety_rating: 0.5,
        }
    }

    #[test]
    fn best_system_s1() {
        let card = make_test_card(0.9, 0.5, 0.3);
        assert_eq!(card.best_system(), CognitiveSystem::S1);
    }

    #[test]
    fn best_system_s2() {
        let card = make_test_card(0.3, 0.9, 0.5);
        assert_eq!(card.best_system(), CognitiveSystem::S2);
    }

    #[test]
    fn best_system_s3() {
        let card = make_test_card(0.2, 0.4, 0.8);
        assert_eq!(card.best_system(), CognitiveSystem::S3);
    }

    #[test]
    fn best_system_tie_favors_s1() {
        let card = make_test_card(0.7, 0.7, 0.7);
        assert_eq!(card.best_system(), CognitiveSystem::S1);
    }

    #[test]
    fn affinity_for_returns_correct_value() {
        let card = make_test_card(0.1, 0.5, 0.9);
        assert!((card.affinity_for(CognitiveSystem::S1) - 0.1).abs() < 0.001);
        assert!((card.affinity_for(CognitiveSystem::S2) - 0.5).abs() < 0.001);
        assert!((card.affinity_for(CognitiveSystem::S3) - 0.9).abs() < 0.001);
    }

    #[test]
    fn estimate_cost_calculation() {
        let card = make_test_card(0.5, 0.5, 0.5);
        // 1000 input tokens at $1/M + 500 output tokens at $2/M
        // = (1000 * 1.0 + 500 * 2.0) / 1_000_000 = 2000 / 1_000_000 = 0.002
        let cost = card.estimate_cost(1000, 500);
        assert!((cost - 0.002).abs() < 0.0001);
    }

    #[test]
    fn cognitive_system_display() {
        assert_eq!(format!("{}", CognitiveSystem::S1), "S1");
        assert_eq!(format!("{}", CognitiveSystem::S2), "S2");
        assert_eq!(format!("{}", CognitiveSystem::S3), "S3");
    }

    #[test]
    fn cognitive_system_repr() {
        assert_eq!(CognitiveSystem::S1.__repr__(), "CognitiveSystem.S1");
        assert_eq!(CognitiveSystem::S2.__repr__(), "CognitiveSystem.S2");
        assert_eq!(CognitiveSystem::S3.__repr__(), "CognitiveSystem.S3");
    }

    #[test]
    fn parse_toml_single_model() {
        let toml_str = r#"
            [[models]]
            id = "gemini-2.5-flash"
            provider = "google"
            family = "gemini-2.5"
            code_score = 0.85
            reasoning_score = 0.80
            tool_use_score = 0.90
            math_score = 0.75
            formal_z3_strength = 0.60
            cost_input_per_m = 0.075
            cost_output_per_m = 0.30
            latency_ttft_ms = 200.0
            tokens_per_sec = 200.0
            s1_affinity = 0.70
            s2_affinity = 0.85
            s3_affinity = 0.40
            recommended_topologies = ["sequential", "avr"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 1048576
        "#;
        let cards = ModelCard::parse_toml(toml_str).unwrap();
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "gemini-2.5-flash");
        assert_eq!(cards[0].provider, "google");
        assert_eq!(cards[0].family, "gemini-2.5");
        assert!((cards[0].code_score - 0.85).abs() < 0.001);
        assert!((cards[0].s2_affinity - 0.85).abs() < 0.001);
        assert_eq!(cards[0].context_window, 1048576);
        assert_eq!(cards[0].recommended_topologies, vec!["sequential", "avr"]);
        assert!(cards[0].supports_tools);
        assert!(cards[0].supports_vision);
    }

    #[test]
    fn parse_toml_multiple_models() {
        let toml_str = r#"
            [[models]]
            id = "model-a"
            provider = "provider-a"
            family = "family-a"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.5
            cost_input_per_m = 1.0
            cost_output_per_m = 1.0
            latency_ttft_ms = 100.0
            tokens_per_sec = 100.0
            s1_affinity = 0.9
            s2_affinity = 0.3
            s3_affinity = 0.1
            recommended_topologies = []
            supports_tools = false
            supports_json_mode = false
            supports_vision = false
            context_window = 8192

            [[models]]
            id = "model-b"
            provider = "provider-b"
            family = "family-b"
            code_score = 0.9
            reasoning_score = 0.9
            tool_use_score = 0.9
            math_score = 0.9
            formal_z3_strength = 0.9
            cost_input_per_m = 10.0
            cost_output_per_m = 30.0
            latency_ttft_ms = 500.0
            tokens_per_sec = 50.0
            s1_affinity = 0.2
            s2_affinity = 0.7
            s3_affinity = 0.95
            recommended_topologies = ["parallel", "avr", "z3"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 200000
        "#;
        let cards = ModelCard::parse_toml(toml_str).unwrap();
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0].id, "model-a");
        assert_eq!(cards[0].best_system(), CognitiveSystem::S1);
        assert_eq!(cards[1].id, "model-b");
        assert_eq!(cards[1].best_system(), CognitiveSystem::S3);
    }

    #[test]
    fn model_card_repr() {
        let card = make_test_card(0.3, 0.8, 0.5);
        let repr = card.__repr__();
        assert!(repr.contains("test"));
        assert!(repr.contains("0.30"));
        assert!(repr.contains("0.80"));
        assert!(repr.contains("0.50"));
    }

    #[test]
    fn domain_scores_default_empty() {
        let card = make_test_card(0.5, 0.5, 0.5);
        assert!(card.domain_scores.is_empty());
        assert!((card.safety_rating - 0.5).abs() < 0.001);
    }

    #[test]
    fn domain_score_for_known_domain() {
        let mut card = make_test_card(0.5, 0.5, 0.5);
        card.domain_scores.insert("math".to_string(), 0.94);
        card.domain_scores.insert("code".to_string(), 0.87);
        assert!((card.domain_score("math") - 0.94).abs() < 0.001);
        assert!((card.domain_score("code") - 0.87).abs() < 0.001);
        assert!((card.domain_score("unknown") - 0.5).abs() < 0.001);
    }

    #[test]
    fn parse_toml_with_domain_scores() {
        let toml_str = r#"
            [[models]]
            id = "test-model"
            provider = "test"
            family = "test"
            code_score = 0.85
            reasoning_score = 0.80
            tool_use_score = 0.90
            math_score = 0.75
            formal_z3_strength = 0.60
            cost_input_per_m = 0.075
            cost_output_per_m = 0.30
            latency_ttft_ms = 200.0
            tokens_per_sec = 200.0
            s1_affinity = 0.70
            s2_affinity = 0.85
            s3_affinity = 0.40
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 1048576
            safety_rating = 0.85

            [models.domain_scores]
            math = 0.94
            code = 0.87
            reasoning = 0.80
        "#;
        let cards = ModelCard::parse_toml(toml_str).unwrap();
        assert_eq!(cards.len(), 1);
        assert!((cards[0].safety_rating - 0.85).abs() < 0.001);
        assert!((cards[0].domain_score("math") - 0.94).abs() < 0.001);
        assert!((cards[0].domain_score("code") - 0.87).abs() < 0.001);
        assert!((cards[0].domain_score("reasoning") - 0.80).abs() < 0.001);
    }
}
