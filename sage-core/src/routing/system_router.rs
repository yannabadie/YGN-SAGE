//! SystemRouter — decides which cognitive System (S1/S2/S3) to activate.
//!
//! Decision process:
//! 1. Structural analysis via `StructuralFeatures::extract_from(task)`
//! 2. System decision based on feature values + formal keyword detection
//! 3. Model selection — best model for that system from `ModelRegistry`
//! 4. Budget constraint — downgrade to cheapest available if over budget

use pyo3::prelude::*;
use tracing::{info, info_span};

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

// ── RoutingConstraints ───────────────────────────────────────────────────────

/// Runtime constraints for constrained routing.
///
/// All fields default to "unconstrained" (0.0 / empty).
/// A value of 0.0 for numeric fields means "no limit".
#[pyclass]
#[derive(Debug, Clone)]
pub struct RoutingConstraints {
    /// Maximum acceptable cost in USD (0.0 = no limit).
    #[pyo3(get, set)]
    pub max_cost_usd: f32,
    /// Maximum acceptable time-to-first-token in ms (0.0 = no limit).
    #[pyo3(get, set)]
    pub max_latency_ms: f32,
    /// Minimum quality score (0.0 = no minimum).
    #[pyo3(get, set)]
    pub min_quality: f32,
    /// Required capability flags, e.g. ["tools", "json_mode", "vision"].
    #[pyo3(get, set)]
    pub required_capabilities: Vec<String>,
    /// Security label constraint (empty = no constraint).
    #[pyo3(get, set)]
    pub security_label: String,
    /// Exploration budget: 0.0 = pure exploit, 1.0 = pure explore.
    #[pyo3(get, set)]
    pub exploration_budget: f32,
}

#[pymethods]
impl RoutingConstraints {
    #[new]
    #[pyo3(signature = (max_cost_usd=0.0, max_latency_ms=0.0, min_quality=0.0, required_capabilities=vec![], security_label=String::new(), exploration_budget=0.0))]
    pub fn new(
        max_cost_usd: f32,
        max_latency_ms: f32,
        min_quality: f32,
        required_capabilities: Vec<String>,
        security_label: String,
        exploration_budget: f32,
    ) -> Self {
        Self {
            max_cost_usd,
            max_latency_ms,
            min_quality,
            required_capabilities,
            security_label,
            exploration_budget,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingConstraints(max_cost={:.4}, max_latency={:.0}ms, min_quality={:.2}, caps={:?}, security='{}', explore={:.2})",
            self.max_cost_usd,
            self.max_latency_ms,
            self.min_quality,
            self.required_capabilities,
            self.security_label,
            self.exploration_budget,
        )
    }
}

// ── RoutingDecision ──────────────────────────────────────────────────────────

/// The result of routing a task: which system, which model, and at what cost.
#[pyclass]
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Unique decision identifier (ULID, 26-char Crockford base32).
    #[pyo3(get)]
    pub decision_id: String,
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
            "RoutingDecision(id='{}', system={}, model='{}', confidence={:.2}, cost={:.4})",
            self.decision_id, self.system, self.model_id, self.confidence, self.estimated_cost
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

    /// Route a task to the best cognitive system + model (legacy API).
    pub fn route(&self, task: &str, budget: f32) -> RoutingDecision {
        let _span =
            info_span!("system_router.route", task_len = task.len(), budget = budget).entered();

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

        let decision = RoutingDecision {
            decision_id: ulid::Ulid::new().to_string(),
            system: final_system,
            model_id: model.id.clone(),
            confidence,
            estimated_cost,
        };

        info!(
            system = %decision.system,
            model = %decision.model_id,
            confidence = decision.confidence,
            cost = decision.estimated_cost,
            "routing_decision"
        );

        decision
    }

    /// Route a task with explicit constraints.
    ///
    /// Applies hard constraint filtering before budget-based selection.
    /// Falls back to progressively wider candidate pools if constraints
    /// filter out all models for the chosen system.
    pub fn route_constrained(
        &self,
        task: &str,
        constraints: &RoutingConstraints,
    ) -> RoutingDecision {
        let _span = info_span!(
            "system_router.route_constrained",
            task_len = task.len(),
            max_cost = constraints.max_cost_usd,
            max_latency = constraints.max_latency_ms,
            min_quality = constraints.min_quality,
            explore = constraints.exploration_budget,
        )
        .entered();

        let features = StructuralFeatures::extract_from(task);
        let (system, confidence) = self.decide_system(task, &features);

        // Step 1: Get candidates for the chosen system
        let mut candidates = self.registry.select_for_system(system);

        // Step 2: Hard constraint filter
        candidates = self.apply_constraints(&candidates, constraints);

        // Step 3: If all candidates filtered out, widen to all models and re-filter
        if candidates.is_empty() {
            candidates = self.registry.all_models();
            candidates = self.apply_constraints(&candidates, constraints);
        }

        // Step 4: If still empty, use all models (no constraint can be satisfied)
        if candidates.is_empty() {
            candidates = self.registry.all_models();
        }

        // Step 5: Budget-constrained selection
        let budget = if constraints.max_cost_usd > 0.0 {
            constraints.max_cost_usd
        } else {
            f32::MAX
        };
        let (model, estimated_cost) = self.select_within_budget(&candidates, budget);

        let final_system = if model.id != candidates[0].id {
            model.best_system()
        } else {
            // Derive from model if widened beyond original system
            model.best_system()
        };

        let decision = RoutingDecision {
            decision_id: ulid::Ulid::new().to_string(),
            system: final_system,
            model_id: model.id.clone(),
            confidence,
            estimated_cost,
        };

        info!(
            system = %decision.system,
            model = %decision.model_id,
            confidence = decision.confidence,
            cost = decision.estimated_cost,
            "constrained_routing_decision"
        );

        decision
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

    /// Apply hard constraints to a candidate list, filtering out models
    /// that violate any active constraint.
    fn apply_constraints(
        &self,
        candidates: &[ModelCard],
        constraints: &RoutingConstraints,
    ) -> Vec<ModelCard> {
        let est_input = 1000_u32;
        let est_output = 2000_u32;

        candidates
            .iter()
            .filter(|card| {
                // Cost filter
                if constraints.max_cost_usd > 0.0 {
                    let est_cost = card.estimate_cost(est_input, est_output);
                    if est_cost > constraints.max_cost_usd {
                        return false;
                    }
                }

                // Latency filter
                if constraints.max_latency_ms > 0.0
                    && card.latency_ttft_ms > constraints.max_latency_ms
                {
                    return false;
                }

                // Quality filter (max of code_score, reasoning_score as proxy)
                if constraints.min_quality > 0.0 {
                    let quality = card.code_score.max(card.reasoning_score);
                    if quality < constraints.min_quality {
                        return false;
                    }
                }

                // Capability filter
                for cap in &constraints.required_capabilities {
                    match cap.as_str() {
                        "tools" => {
                            if !card.supports_tools {
                                return false;
                            }
                        }
                        "json_mode" => {
                            if !card.supports_json_mode {
                                return false;
                            }
                        }
                        "vision" => {
                            if !card.supports_vision {
                                return false;
                            }
                        }
                        _ => {} // Unknown capabilities ignored (forward-compat)
                    }
                }

                true
            })
            .cloned()
            .collect()
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
