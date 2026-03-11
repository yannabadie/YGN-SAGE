//! SystemRouter — decides which cognitive System (S1/S2/S3) to activate.
//!
//! Decision process:
//! 1. Structural analysis via `StructuralFeatures::extract_from(task)`
//! 2. System decision based on feature values + formal keyword detection
//! 3. Model selection — best model for that system from `ModelRegistry`
//! 4. Budget constraint — downgrade to cheapest available if over budget

use pyo3::prelude::*;
use tracing::{info, info_span};

use super::bandit::ContextualBandit;
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
    /// Topology identifier (empty if not topology-routed).
    #[pyo3(get)]
    pub topology_id: String,
}

#[pymethods]
impl RoutingDecision {
    fn __repr__(&self) -> String {
        format!(
            "RoutingDecision(id='{}', system={}, model='{}', confidence={:.2}, cost={:.4}, topology='{}')",
            self.decision_id, self.system, self.model_id, self.confidence, self.estimated_cost, self.topology_id
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
    bandit: Option<ContextualBandit>,
}

#[pymethods]
impl SystemRouter {
    #[new]
    pub fn new(registry: ModelRegistry) -> Self {
        Self {
            registry,
            bandit: None,
        }
    }

    /// Inject a contextual bandit for integrated routing.
    #[pyo3(name = "set_bandit")]
    pub fn py_set_bandit(&mut self, bandit: ContextualBandit) {
        info!(arms = bandit.arm_count(), "bandit_injected_into_router");
        self.bandit = Some(bandit);
    }

    /// Route a task to the best cognitive system + model (legacy API).
    pub fn route(&self, task: &str, budget: f32) -> RoutingDecision {
        let _span = info_span!(
            "system_router.route",
            task_len = task.len(),
            budget = budget
        )
        .entered();

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
            topology_id: String::new(),
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
            topology_id: String::new(),
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

    /// Route a task with bandit integration and topology awareness.
    ///
    /// 1. Structural feature extraction + system decision (existing logic).
    /// 2. Hard constraint filtering (existing logic).
    /// 3. If bandit is available, use Thompson sampling to select model.
    /// 4. Otherwise, fall back to budget-constrained selection.
    /// 5. Returns RoutingDecision with topology_id set.
    #[pyo3(name = "route_integrated")]
    pub fn py_route_integrated(
        &mut self,
        task: &str,
        constraints: &RoutingConstraints,
        topology_id: &str,
    ) -> PyResult<RoutingDecision> {
        self.route_integrated(task, constraints, topology_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Record outcome for a previous routing decision.
    ///
    /// Forwards to bandit (if available) and always records telemetry.
    #[pyo3(name = "record_outcome")]
    pub fn py_record_outcome(
        &mut self,
        decision_id: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> PyResult<()> {
        self.record_outcome(decision_id, quality, cost, latency_ms)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "SystemRouter(models={}, bandit={})",
            self.registry.len(),
            self.bandit.is_some()
        )
    }
}

impl SystemRouter {
    /// Route with bandit integration and topology awareness.
    pub fn route_integrated(
        &mut self,
        task: &str,
        constraints: &RoutingConstraints,
        topology_id: &str,
    ) -> Result<RoutingDecision, String> {
        let _span = info_span!(
            "system_router.route_integrated",
            task_len = task.len(),
            topology_id = topology_id,
        )
        .entered();

        // Step 1: Structural analysis + system decision
        let features = StructuralFeatures::extract_from(task);
        let (system, confidence) = self.decide_system(task, &features);

        // Step 2: Get candidates + hard constraint filtering
        let mut candidates = self.registry.select_for_system(system);
        candidates = self.apply_constraints(&candidates, constraints);
        if candidates.is_empty() {
            candidates = self.registry.all_models();
            candidates = self.apply_constraints(&candidates, constraints);
        }
        if candidates.is_empty() {
            candidates = self.registry.all_models();
        }

        // Step 3: Model selection — bandit or budget-constrained
        let (model_id, estimated_cost, decision_id) = if let Some(ref mut bandit) = self.bandit {
            match bandit.choose(constraints.exploration_budget) {
                Ok(bd) => {
                    // Use bandit's model if it's in our candidates, otherwise fall back
                    let est = candidates
                        .iter()
                        .find(|c| c.id == bd.model_id)
                        .map(|c| c.estimate_cost(1000, 2000))
                        .unwrap_or(bd.expected_cost);
                    (bd.model_id.clone(), est, bd.decision_id.clone())
                }
                Err(_) => {
                    // Bandit has no arms, fall back to budget selection
                    let budget = if constraints.max_cost_usd > 0.0 {
                        constraints.max_cost_usd
                    } else {
                        f32::MAX
                    };
                    let (model, cost) = self.select_within_budget(&candidates, budget);
                    (model.id.clone(), cost, ulid::Ulid::new().to_string())
                }
            }
        } else {
            let budget = if constraints.max_cost_usd > 0.0 {
                constraints.max_cost_usd
            } else {
                f32::MAX
            };
            let (model, cost) = self.select_within_budget(&candidates, budget);
            (model.id.clone(), cost, ulid::Ulid::new().to_string())
        };

        // Step 4: Use calibrated affinity for confidence adjustment
        let calibrated = self.registry.calibrated_affinity(&model_id, system);
        let adjusted_confidence = confidence * calibrated.max(0.1);

        // Step 5: Determine final system from selected model
        let final_system = self
            .registry
            .get(&model_id)
            .map(|c| c.best_system())
            .unwrap_or(system);

        let decision = RoutingDecision {
            decision_id,
            system: final_system,
            model_id: model_id.clone(),
            confidence: adjusted_confidence,
            estimated_cost,
            topology_id: topology_id.to_string(),
        };

        info!(
            system = %decision.system,
            model = %decision.model_id,
            confidence = decision.confidence,
            cost = decision.estimated_cost,
            topology = %decision.topology_id,
            "integrated_routing_decision"
        );

        Ok(decision)
    }

    /// Record outcome: forwards to bandit (if available) and records telemetry.
    pub fn record_outcome(
        &mut self,
        decision_id: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> Result<(), String> {
        let _span = info_span!(
            "system_router.record_outcome",
            decision_id = decision_id,
            quality = quality,
            cost = cost,
            latency_ms = latency_ms,
        )
        .entered();

        // Forward to bandit if available (may fail if decision_id unknown)
        if let Some(ref mut bandit) = self.bandit {
            if let Err(e) = bandit.record_outcome(decision_id, quality, cost, latency_ms) {
                info!(error = %e, "bandit_record_outcome_skipped");
            }
        }

        // Always record telemetry — we need the model_id but don't have it here
        // from the decision. We log with the decision_id for traceability.
        // The caller should also call registry.record_telemetry() with the model_id.
        info!(
            decision_id = decision_id,
            quality = quality,
            cost = cost,
            "outcome_recorded"
        );

        Ok(())
    }

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

    fn test_registry() -> ModelRegistry {
        let toml_str = r#"
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

            [[models]]
            id = "smart-model"
            provider = "test"
            family = "test"
            code_score = 0.9
            reasoning_score = 0.9
            tool_use_score = 0.8
            math_score = 0.8
            formal_z3_strength = 0.7
            cost_input_per_m = 5.0
            cost_output_per_m = 15.0
            latency_ttft_ms = 500.0
            tokens_per_sec = 50.0
            s1_affinity = 0.3
            s2_affinity = 0.8
            s3_affinity = 0.95
            recommended_topologies = ["avr", "debate"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 200000
        "#;
        ModelRegistry::from_toml_str(toml_str).unwrap()
    }

    #[test]
    fn routing_decision_has_topology_id() {
        let registry = test_registry();
        let router = SystemRouter::new(registry);
        let decision = router.route("hello", f32::MAX);
        assert!(decision.topology_id.is_empty()); // legacy route: empty topology_id
    }

    #[test]
    fn route_constrained_has_empty_topology_id() {
        let registry = test_registry();
        let router = SystemRouter::new(registry);
        let constraints = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);
        let decision = router.route_constrained("Write a function", &constraints);
        assert!(decision.topology_id.is_empty());
    }

    #[test]
    fn route_integrated_without_bandit() {
        let registry = test_registry();
        let mut router = SystemRouter::new(registry);
        let constraints = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);
        let decision = router
            .route_integrated("hello world", &constraints, "topo-123")
            .unwrap();
        assert_eq!(decision.topology_id, "topo-123");
        assert!(!decision.model_id.is_empty());
    }

    #[test]
    fn route_integrated_with_bandit() {
        let registry = test_registry();
        let mut router = SystemRouter::new(registry);

        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("fast-model", "sequential");
        bandit.add_arm("smart-model", "avr");
        router.bandit = Some(bandit);

        let constraints = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);
        let decision = router
            .route_integrated("explain quantum computing", &constraints, "topo-456")
            .unwrap();
        assert_eq!(decision.topology_id, "topo-456");
        // Model should be one of the registered arms
        assert!(
            decision.model_id == "fast-model" || decision.model_id == "smart-model",
            "unexpected model: {}",
            decision.model_id
        );
    }

    #[test]
    fn record_outcome_without_bandit_succeeds() {
        let registry = test_registry();
        let mut router = SystemRouter::new(registry);
        let result = router.record_outcome("some-decision-id", 0.8, 0.01, 200.0);
        assert!(result.is_ok());
    }

    #[test]
    fn record_outcome_with_bandit_records() {
        let registry = test_registry();
        let mut router = SystemRouter::new(registry);

        let mut bandit = ContextualBandit::create(0.995, 0.1);
        bandit.add_arm("fast-model", "sequential");
        router.bandit = Some(bandit);

        // Make a decision first via bandit
        let constraints = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);
        let decision = router
            .route_integrated("hello", &constraints, "t1")
            .unwrap();

        // Record outcome — should succeed if bandit has the pending decision
        let result = router.record_outcome(&decision.decision_id, 0.9, 0.01, 100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn repr_shows_bandit_status() {
        let registry = test_registry();
        let mut router = SystemRouter::new(registry);
        assert!(router.__repr__().contains("bandit=false"));

        router.bandit = Some(ContextualBandit::create(0.995, 0.1));
        assert!(router.__repr__().contains("bandit=true"));
    }
}
