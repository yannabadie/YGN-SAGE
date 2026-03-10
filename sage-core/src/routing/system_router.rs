//! SystemRouter — decides which cognitive System (S1/S2/S3) to activate.
//!
//! Decision process:
//! 1. Structural analysis via `StructuralFeatures::extract_from(task)`
//! 2. System decision based on feature values + formal keyword detection
//! 3. Model selection — best model for that system from `ModelRegistry`
//! 4. Budget constraint — downgrade to cheapest available if over budget

use pyo3::prelude::*;

use super::features::StructuralFeatures;
use super::model_card::{CognitiveSystem, ModelCard};
use super::model_registry::ModelRegistry;

// ── Formal reasoning keywords ────────────────────────────────────────────────

/// Keywords that explicitly indicate formal reasoning / S3 tasks,
/// regardless of overall complexity score.
const FORMAL_KEYWORDS: &[&str] = &[
    "prove",
    "theorem",
    "induction",
    "formal",
    "z3",
    "constraint",
    "invariant",
    "verification",
];

/// Check whether the task contains any formal reasoning keyword.
fn has_formal_keywords(task: &str) -> bool {
    let lower = task.to_lowercase();
    FORMAL_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

// ── RoutingDecision ──────────────────────────────────────────────────────────

/// The result of routing a task: which system, which model, and at what cost.
#[pyclass]
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    #[pyo3(get)]
    pub system: CognitiveSystem,
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub estimated_cost: f32,
}

#[pymethods]
impl RoutingDecision {
    fn __repr__(&self) -> String {
        format!(
            "RoutingDecision(system={}, model='{}', confidence={:.2}, cost={:.4})",
            self.system, self.model_id, self.confidence, self.estimated_cost
        )
    }
}

// ── SystemRouter ─────────────────────────────────────────────────────────────

/// Cognitive system decision engine.
///
/// Routes a task to S1 (fast), S2 (deliberate/tools), or S3 (formal/reasoning)
/// and selects the best model from the registry within a budget constraint.
#[pyclass]
pub struct SystemRouter {
    registry: ModelRegistry,
}

#[pymethods]
impl SystemRouter {
    #[new]
    pub fn new(registry: ModelRegistry) -> Self {
        Self { registry }
    }

    /// Route a task to the best cognitive system + model.
    pub fn route(&self, task: &str, budget: f32) -> RoutingDecision {
        // Step 1: Structural analysis
        let features = StructuralFeatures::extract_from(task);
        let (system, confidence) = self.decide_system(task, &features);

        // Step 2: Select best model for chosen system
        let candidates = self.registry.select_for_system(system);

        // Step 3: Budget-constrained selection
        let (model, estimated_cost) = self.select_within_budget(&candidates, budget);

        // If budget forced a downgrade, recalculate system from model's best affinity
        let final_system = if model.id != candidates[0].id {
            model.best_system()
        } else {
            system
        };

        RoutingDecision {
            system: final_system,
            model_id: model.id.clone(),
            confidence,
            estimated_cost,
        }
    }

    fn __repr__(&self) -> String {
        format!("SystemRouter(models={})", self.registry.len())
    }
}

impl SystemRouter {
    /// Decide cognitive system from structural features and raw task text.
    ///
    /// Priority order:
    /// 1. Formal keywords (prove, theorem, induction, etc.) -> S3
    /// 2. High complexity + low uncertainty -> S3
    /// 3. Tool required, code block, or medium complexity -> S2
    /// 4. Everything else -> S1
    fn decide_system(&self, task: &str, features: &StructuralFeatures) -> (CognitiveSystem, f32) {
        let complexity = features.keyword_complexity;
        let uncertainty = features.keyword_uncertainty;

        // S3: formal reasoning keywords explicitly indicate need for deep reasoning.
        // This catches cases like "prove by induction" where keyword complexity alone
        // (base 0.2 + design 0.20 = 0.40) would be below the 0.65 threshold.
        if has_formal_keywords(task) {
            return (CognitiveSystem::S3, 0.85);
        }

        // S3: high complexity + low uncertainty (algorithmic / system-design)
        if complexity >= 0.65 && uncertainty < 0.3 {
            return (CognitiveSystem::S3, 0.7 + complexity * 0.3);
        }

        // S2: medium complexity or tool/code required
        if features.tool_required || features.has_code_block || complexity >= 0.35 {
            return (CognitiveSystem::S2, 0.6 + complexity * 0.3);
        }

        // S1: simple, fast, intuitive
        let confidence = (1.0 - complexity).clamp(0.5, 0.95);
        (CognitiveSystem::S1, confidence)
    }

    /// Select best model within budget. Returns (model, estimated_cost).
    /// Assumes ~1000 input + ~2000 output tokens as estimate.
    fn select_within_budget<'a>(
        &self,
        candidates: &'a [ModelCard],
        budget: f32,
    ) -> (&'a ModelCard, f32) {
        let est_input = 1000_u32;
        let est_output = 2000_u32;

        for card in candidates {
            let cost = card.estimate_cost(est_input, est_output);
            if cost <= budget {
                return (card, cost);
            }
        }

        // Fallback: cheapest model overall
        let mut cheapest = &candidates[0];
        let mut min_cost = cheapest.estimate_cost(est_input, est_output);
        for card in candidates.iter().skip(1) {
            let cost = card.estimate_cost(est_input, est_output);
            if cost < min_cost {
                cheapest = card;
                min_cost = cost;
            }
        }
        (cheapest, min_cost)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_formal_keywords_detects_prove() {
        assert!(has_formal_keywords("Prove that x > 0"));
        assert!(has_formal_keywords("Use induction to show..."));
        assert!(has_formal_keywords("Write a Z3 constraint"));
        assert!(!has_formal_keywords("What is the capital of France?"));
    }

    #[test]
    fn has_formal_keywords_case_insensitive() {
        assert!(has_formal_keywords("PROVE by INDUCTION"));
        assert!(has_formal_keywords("Formal Verification of Safety"));
    }
}
