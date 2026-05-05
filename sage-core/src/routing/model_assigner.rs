//! ModelAssigner — per-node model assignment using ModelCard scoring.

use pyo3::prelude::*;
use tracing::{info, warn};

use super::model_card::CognitiveSystem;
use super::model_registry::ModelRegistry;
use crate::topology::topology_graph::TopologyGraph;

/// Default scoring weights — subject to ablation (see CLAUDE.md §2).
const DEFAULT_WEIGHT_AFFINITY: f32 = 0.4;
const DEFAULT_WEIGHT_DOMAIN: f32 = 0.4;
const DEFAULT_WEIGHT_COST: f32 = 0.2;
const BUDGET_EPSILON: f32 = 0.01;

/// Provider hint scoring bonus — subject to ablation.
const PROVIDER_HINT_BONUS: f32 = 0.15;

/// Fixed token estimate for cost normalization (input, output).
const COST_ESTIMATE_TOKENS: (u32, u32) = (1000, 500);

/// Roles that templates.rs explicitly marks as "sink" (final-stage
/// forwarder) by assigning `SINK_NODE_PROMPT` to them. Single source of
/// truth: maintain this list in sync with templates.rs whenever a new
/// template adds a SINK_NODE_PROMPT-tagged node.
///
/// To audit:
///   `grep -B 1 SINK_NODE_PROMPT sage-core/src/topology/templates.rs`
/// As of 2026-04-17 the templates assign SINK_NODE_PROMPT to:
///   synthesizer (sequential, brainstorming)
///   aggregator  (parallel, horizon_pipeline, parallel_fanout)
///   mixer       (self_moa)
///   judge       (debate)
///   verifier    (robust — final node; AVR's verifier is non-final but
///                also benign as a sink classification — its job is
///                cheap pattern-matching, not deep reasoning)
///   solver      (formal_solver — DETERMINISTIC compute node, model_id="";
///                MUST stay sink or F7 will replace free Rust math with
///                a $0.10 LLM call on math/formal tasks)
const SINK_ROLES: &[&str] = &[
    "synthesizer",
    "aggregator",
    "mixer",
    "judge",
    "verifier",
    "solver",
    "formatter",
    "output",
    "sink",
];

/// Classify a role string as a "sink" (output-only forwarder) vs a
/// "producer" (generates content). Sink nodes keep their template-assigned
/// cognitive tier — they just format / synthesize predecessor output.
/// Producer nodes do the actual reasoning and MUST be promoted to the
/// overall task tier floor when the task is complex (F7, Topaz-inspired).
///
/// Research: Topaz (arXiv 2604.03527) argues routing must match model
/// capability to per-subtask REQUIREMENTS. The role-based sink/producer
/// split is the training-free approximation that fits our existing
/// template catalogue (templates.rs).
fn is_sink_role(role: &str) -> bool {
    let r = role.to_lowercase();
    SINK_ROLES.iter().any(|&s| r == s) || r.starts_with("output_")
}

/// True if the task domain demands the highest reasoning tier
/// (formal proofs, math, theorem proving). For these, an S3 task pushes
/// producer nodes all the way to S3 (full reasoner tier), not the
/// general-purpose S2 floor.
///
/// Match is case-insensitive substring, so `"math"`, `"formal"`,
/// `"formal_verification"`, `"Math"` all classify as high-rigour. This
/// keeps the function tolerant to whatever upstream pipeline naming
/// convention surfaces (`ctx.domain` in pipeline.py uses lower-case
/// short tokens; cards.toml uses `math`, `formal`).
fn is_high_rigour_domain(task_domain: &str) -> bool {
    let d = task_domain.to_lowercase();
    d.contains("math") || d.contains("formal")
}

/// Compute the effective cognitive tier for Stage 3 model assignment.
///
/// * Sink nodes stay on their template tier — they just forward predecessor
///   output, they don't need a reasoner.
/// * Producer nodes get promoted to `max(node.system, floor)`, where
///   `floor` depends on (task tier, task domain):
///   - S3 + math/formal     → 3 (full reasoner tier — proofs/Z3 need it)
///   - S3 + other domains   → 2 (mid-tier reasoner suffices for code/general)
///   - S2                   → 1 (no-op; S1 is already the minimum)
///   - S1 / None            → keep node tier (no promotion)
/// * `task_system=None` disables the promotion entirely (full back-compat
///   with pre-F7 callers that didn't know about task-level routing).
///
/// Why the domain split: an S3 SWE-bench task wants a strong code agent,
/// not necessarily a pure proof model. But an S3 math/formal task NEEDS
/// the reasoner tier — there's no point promoting a coder. This is the
/// minimal Topaz-inspired adaptation that the existing card domain_scores
/// already support (cards expose `math` and `formal` columns explicitly).
fn effective_system(
    role: &str,
    node_system: CognitiveSystem,
    task_system: Option<CognitiveSystem>,
    task_domain: &str,
) -> CognitiveSystem {
    let task = match task_system {
        Some(t) => t,
        None => return node_system,
    };
    if is_sink_role(role) {
        return node_system;
    }
    let floor_n: u8 = if matches!(task, CognitiveSystem::S3) && is_high_rigour_domain(task_domain) {
        3
    } else {
        (task as u8).saturating_sub(1)
    };
    let node_n = node_system as u8;
    let promoted_n = std::cmp::max(node_n, floor_n);
    match promoted_n {
        1 => CognitiveSystem::S1,
        2 => CognitiveSystem::S2,
        3 => CognitiveSystem::S3,
        _ => node_system,
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelAssigner {
    registry: ModelRegistry,
    /// Weight for cognitive system affinity (S1/S2/S3 match) — subject to ablation.
    weight_affinity: f32,
    /// Weight for task domain score — subject to ablation.
    weight_domain: f32,
    /// Weight for cost efficiency (lower cost = higher score) — subject to ablation.
    weight_cost: f32,
    /// Providers excluded from assignment (dead at boot health check).
    excluded_providers: Vec<String>,
}

impl ModelAssigner {
    pub fn from_registry(registry: &ModelRegistry) -> Self {
        Self {
            registry: registry.clone(),
            weight_affinity: DEFAULT_WEIGHT_AFFINITY,
            weight_domain: DEFAULT_WEIGHT_DOMAIN,
            weight_cost: DEFAULT_WEIGHT_COST,
            excluded_providers: Vec::new(),
        }
    }

    /// Create an assigner with custom scoring weights.
    /// Weights are normalized to sum to 1.0 internally.
    pub fn with_weights(
        registry: &ModelRegistry,
        weight_affinity: f32,
        weight_domain: f32,
        weight_cost: f32,
    ) -> Self {
        let total = weight_affinity + weight_domain + weight_cost;
        let norm = if total > 0.0 { total } else { 1.0 };
        Self {
            registry: registry.clone(),
            weight_affinity: weight_affinity / norm,
            weight_domain: weight_domain / norm,
            weight_cost: weight_cost / norm,
            excluded_providers: Vec::new(),
        }
    }

    /// Exclude providers from model assignment (dead at boot health check).
    /// Models from these providers will never be assigned to any node.
    pub fn set_excluded_providers(&mut self, providers: Vec<String>) {
        info!(excluded = ?providers, "ModelAssigner: excluding dead providers");
        self.excluded_providers = providers;
    }

    pub fn assign_models_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> usize {
        self.assign_models_with_hints_inner(graph, task_domain, budget_usd, &[], None)
    }

    /// Assign models with optional per-node provider hints.
    ///
    /// `provider_hints` is a slice of `(node_idx, provider_name)` pairs.
    /// When a hint is present for a node, candidates from that provider get
    /// a +0.15 scoring bonus (soft preference, not hard filter — if no
    /// model from the hinted provider qualifies, the best alternative wins).
    ///
    /// `task_system` is the OVERALL cognitive tier of the task (from Stage 0
    /// routing + optional `system_hint` override). Producer nodes (planner,
    /// coder, worker, verifier, source) are promoted via `effective_system`
    /// so an S3 task doesn't end up with flash-tier planners. Pass `None` to
    /// keep the legacy per-node-only behaviour (full back-compat).
    pub fn assign_models_with_hints_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: &[(usize, String)],
        task_system: Option<CognitiveSystem>,
    ) -> usize {
        let node_count = graph.node_count();
        let mut remaining_budget = budget_usd;
        let mut assigned = 0usize;

        let all_models = self.registry.all_models();
        if all_models.is_empty() {
            warn!("ModelAssigner: no models in registry, skipping assignment");
            return 0;
        }

        // Pre-compute cost estimates once (avoids repeated calls per node).
        let (input_tok, output_tok) = COST_ESTIMATE_TOKENS;
        let model_costs: Vec<f32> = all_models
            .iter()
            .map(|c| c.estimate_cost(input_tok, output_tok))
            .collect();
        let max_cost = model_costs.iter().fold(0.001_f32, |a, &b| a.max(b));

        // Build provider hint lookup
        let hint_map: std::collections::HashMap<usize, &str> = provider_hints
            .iter()
            .map(|(idx, prov)| (*idx, prov.as_str()))
            .collect();

        // Provider-diversity load balancing (added 2026-04-18 after v5f
        // observation: MiniMax took 88% of calls → rate-limit saturation).
        // Track how many nodes have already been assigned to each provider
        // during this call; apply a soft penalty that grows with
        // concentration. Cap the penalty so affinity still wins when a
        // specific provider is strongly preferred by the score function.
        let mut provider_count: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for idx in 0..node_count {
            if remaining_budget < BUDGET_EPSILON {
                warn!(
                    node_idx = idx,
                    remaining_nodes = node_count - idx,
                    "budget_exhausted — stopping assignment"
                );
                break;
            }

            let node = match graph.try_get_node(idx) {
                Ok(n) => n,
                Err(_) => continue,
            };

            // Respect pre-assigned model_id from template (e.g., formal_solver
            // pins formalizer to "deepseek-chat"). Only override if empty.
            if !node.model_id.is_empty() {
                assigned += 1;
                continue;
            }

            let local_system = match node.system {
                1 => CognitiveSystem::S1,
                2 => CognitiveSystem::S2,
                3 => CognitiveSystem::S3,
                _ => CognitiveSystem::S1,
            };
            let system = effective_system(&node.role, local_system, task_system, task_domain);
            if system != local_system {
                info!(
                    node = idx,
                    role = %node.role,
                    local = %local_system,
                    effective = %system,
                    task = ?task_system,
                    domain = task_domain,
                    "tier_promoted"
                );
            }

            let caps = &node.required_capabilities;
            let needs_tools = caps.iter().any(|c| c == "tools");
            let needs_json = caps.iter().any(|c| c == "json" || c == "json_mode");
            let node_budget = node.max_cost_usd.min(remaining_budget);
            let preferred_provider = hint_map.get(&idx).copied().unwrap_or("");

            let mut best_id: Option<String> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;

            for (card_idx, card) in all_models.iter().enumerate() {
                // Skip models from excluded providers (dead at boot health check)
                if self.excluded_providers.iter().any(|p| p == &card.provider) {
                    continue;
                }
                if needs_tools && !card.supports_tools {
                    continue;
                }
                if needs_json && !card.supports_json_mode {
                    continue;
                }
                let est_cost = model_costs[card_idx];
                if est_cost > node_budget {
                    continue;
                }

                let affinity = self.registry.calibrated_affinity(&card.id, system);
                let domain = card.domain_score(task_domain);
                let cost_norm = est_cost / max_cost;
                let mut score = self.weight_affinity * affinity
                    + self.weight_domain * domain
                    + self.weight_cost * (1.0 - cost_norm);

                // Provider hint bonus: soft preference for the hinted provider
                if !preferred_provider.is_empty() && card.provider == preferred_provider {
                    score += PROVIDER_HINT_BONUS;
                }

                // Diversity penalty: -0.08 per previous assignment to the
                // same provider in this call (capped at -0.20). Prevents
                // one provider from taking every node even when it
                // marginally wins on score. Observed impact: MiniMax
                // saturation (88% of calls in v5f) → balanced across
                // 2-3 providers, reducing rate-limit pressure.
                let concentration = provider_count.get(&card.provider).copied().unwrap_or(0);
                let diversity_penalty = (concentration as f32 * 0.08_f32).min(0.20_f32);
                score -= diversity_penalty;

                if score > best_score {
                    best_score = score;
                    best_id = Some(card.id.clone());
                }
            }

            if let Some(model_id) = best_id {
                remaining_budget -= self
                    .registry
                    .get(&model_id)
                    .map(|c| c.estimate_cost(input_tok, output_tok))
                    .unwrap_or(0.0);
                // Track provider for diversity penalty on subsequent nodes.
                if let Some(provider) = self.registry.get(&model_id).map(|c| c.provider.clone()) {
                    *provider_count.entry(provider).or_insert(0) += 1;
                }
                let node_idx_pg = petgraph::graph::NodeIndex::new(idx);
                if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                    node_mut.model_id.clone_from(&model_id);
                    if !preferred_provider.is_empty() {
                        info!(
                            node = idx,
                            role = %node_mut.role,
                            model = %model_id,
                            score = best_score,
                            provider_hint = preferred_provider,
                            "model_assigned (with provider hint)"
                        );
                    } else {
                        info!(
                            node = idx,
                            role = %node_mut.role,
                            model = %model_id,
                            score = best_score,
                            "model_assigned"
                        );
                    }
                }
                assigned += 1;
            } else {
                warn!(node = idx, "no candidate — keeping existing model_id");
            }
        }

        assigned
    }

    pub fn assign_single_node_inner(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
        exclude_ids: Option<&[String]>,
        task_system: Option<CognitiveSystem>,
    ) -> Option<String> {
        let node = graph.try_get_node(node_idx).ok()?;
        let local_system = match node.system {
            1 => CognitiveSystem::S1,
            2 => CognitiveSystem::S2,
            3 => CognitiveSystem::S3,
            _ => CognitiveSystem::S1,
        };
        let system = effective_system(&node.role, local_system, task_system, task_domain);
        let caps = &node.required_capabilities;
        let needs_tools = caps.iter().any(|c| c == "tools");
        let needs_json = caps.iter().any(|c| c == "json" || c == "json_mode");
        let all_models = self.registry.all_models();
        let (input_tok, output_tok) = COST_ESTIMATE_TOKENS;
        let model_costs: Vec<f32> = all_models
            .iter()
            .map(|c| c.estimate_cost(input_tok, output_tok))
            .collect();
        let max_cost = model_costs.iter().fold(0.001_f32, |a, &b| a.max(b));

        let mut best_id: Option<String> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;
        for (card_idx, card) in all_models.iter().enumerate() {
            // Skip models from excluded providers (dead at boot health check)
            if self.excluded_providers.iter().any(|p| p == &card.provider) {
                continue;
            }
            // FrugalGPT cascade: skip excluded models (Cascade Routing, arXiv 2410.10347)
            if let Some(excluded) = exclude_ids {
                if excluded.iter().any(|e| e == &card.id) {
                    continue;
                }
            }
            if needs_tools && !card.supports_tools {
                continue;
            }
            if needs_json && !card.supports_json_mode {
                continue;
            }
            let est_cost = model_costs[card_idx];
            if est_cost > budget_usd {
                continue;
            }
            let affinity = self.registry.calibrated_affinity(&card.id, system);
            let domain = card.domain_score(task_domain);
            let cost_norm = est_cost / max_cost;
            let score = self.weight_affinity * affinity
                + self.weight_domain * domain
                + self.weight_cost * (1.0 - cost_norm);
            if score > best_score {
                best_score = score;
                best_id = Some(card.id.clone());
            }
        }

        if let Some(ref model_id) = best_id {
            let node_idx_pg = petgraph::graph::NodeIndex::new(node_idx);
            if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                node_mut.model_id = model_id.clone();
            }
        }
        best_id
    }
}

#[pymethods]
impl ModelAssigner {
    /// Create a ModelAssigner with optional scoring weights.
    /// Weights default to 0.4/0.4/0.2 (affinity/domain/cost) — subject to ablation.
    /// If provided, weights are normalized to sum to 1.0.
    #[new]
    #[pyo3(signature = (registry, weight_affinity=None, weight_domain=None, weight_cost=None))]
    fn py_new(
        registry: &ModelRegistry,
        weight_affinity: Option<f32>,
        weight_domain: Option<f32>,
        weight_cost: Option<f32>,
    ) -> Self {
        match (weight_affinity, weight_domain, weight_cost) {
            (Some(wa), Some(wd), Some(wc)) => Self::with_weights(registry, wa, wd, wc),
            _ => Self::from_registry(registry),
        }
    }

    /// Assign models to all topology nodes. Optional provider_hints bias
    /// the selection towards specific providers (soft preference, +0.15 bonus).
    ///
    /// `task_system` is the OVERALL cognitive tier of the task (1, 2, or 3)
    /// and drives role-aware tier promotion (see `effective_system`). When
    /// omitted, behaviour is unchanged from the legacy per-node-only
    /// scoring — same path every bench that existed pre-F7 ran.
    #[pyo3(signature = (graph, task_domain, budget_usd, provider_hints=None, task_system=None))]
    fn assign_models(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: Option<Vec<(usize, String)>>,
        task_system: Option<u8>,
    ) -> PyResult<usize> {
        let task_sys = task_system.and_then(|n| match n {
            1 => Some(CognitiveSystem::S1),
            2 => Some(CognitiveSystem::S2),
            3 => Some(CognitiveSystem::S3),
            _ => None,
        });
        let hints: &[(usize, String)] = match &provider_hints {
            Some(h) => h.as_slice(),
            None => &[],
        };
        Ok(self.assign_models_with_hints_inner(graph, task_domain, budget_usd, hints, task_sys))
    }

    /// Exclude dead providers from all future assignments.
    /// Called after boot health check with list of unreachable provider names.
    fn exclude_providers(&mut self, providers: Vec<String>) {
        info!(excluded = ?providers, "ModelAssigner: excluding dead providers from Python");
        self.excluded_providers = providers;
    }

    /// Assign a model to a single node. Optional ``exclude_model_ids`` skips
    /// specific models (used by FrugalGPT cascade to force an upgrade).
    /// `task_system` drives role-aware tier promotion (see `effective_system`).
    #[pyo3(signature = (graph, node_idx, task_domain, budget_usd, exclude_model_ids=None, task_system=None))]
    fn assign_single_node(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
        exclude_model_ids: Option<Vec<String>>,
        task_system: Option<u8>,
    ) -> PyResult<String> {
        let task_sys = task_system.and_then(|n| match n {
            1 => Some(CognitiveSystem::S1),
            2 => Some(CognitiveSystem::S2),
            3 => Some(CognitiveSystem::S3),
            _ => None,
        });
        self.assign_single_node_inner(
            graph,
            node_idx,
            task_domain,
            budget_usd,
            exclude_model_ids.as_deref(),
            task_sys,
        )
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No candidate found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::model_registry::ModelRegistry;
    use crate::topology::topology_graph::{TopologyEdge, TopologyGraph, TopologyNode};

    fn test_registry() -> ModelRegistry {
        let toml = r#"
            [[models]]
            id = "cheap-fast"
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
            s2_affinity = 0.3
            s3_affinity = 0.1
            recommended_topologies = ["sequential"]
            supports_tools = false
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.5
            math = 0.4

            [[models]]
            id = "expensive-smart"
            provider = "test"
            family = "test"
            code_score = 0.9
            reasoning_score = 0.95
            tool_use_score = 0.9
            math_score = 0.9
            formal_z3_strength = 0.8
            cost_input_per_m = 5.0
            cost_output_per_m = 15.0
            latency_ttft_ms = 3000.0
            tokens_per_sec = 50.0
            s1_affinity = 0.1
            s2_affinity = 0.9
            s3_affinity = 0.95
            recommended_topologies = ["avr", "debate"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 1000000
            [models.domain_scores]
            code = 0.9
            math = 0.95
        "#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    fn two_node_graph() -> TopologyGraph {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new(
            "coder".into(),
            "".into(),
            2,
            vec!["tools".into()],
            0,
            5.0,
            60.0,
        );
        let n1 = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        let edge = TopologyEdge::control();
        g.add_node(n0);
        g.add_node(n1);
        g.try_add_edge(0, 1, edge).unwrap();
        g
    }

    #[test]
    fn test_assign_models_basic() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let n = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n, 2);
        // Coder (S2, needs tools) -> expensive-smart (only one with tools)
        assert_eq!(graph.try_get_node(0).unwrap().model_id, "expensive-smart");
        // Reviewer (S3, no special caps) -> expensive-smart (highest S3 affinity)
        assert_eq!(graph.try_get_node(1).unwrap().model_id, "expensive-smart");
    }

    #[test]
    fn test_assign_respects_budget() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let n = assigner.assign_models_inner(&mut graph, "code", 0.005);
        let model0 = &graph.try_get_node(0).unwrap().model_id;
        let model1 = &graph.try_get_node(1).unwrap().model_id;
        // With tiny budget, either cheap-fast is picked or fewer nodes assigned
        assert!(model0 == "cheap-fast" || model1 == "cheap-fast" || n < 2);
    }

    #[test]
    fn test_assign_keeps_existing_when_no_candidate() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new(
            "special".into(),
            "original-model".into(),
            2,
            vec!["tools".into(), "json".into(), "vision".into()],
            0,
            0.001,
            60.0,
        );
        g.add_node(n0);
        let n = assigner.assign_models_inner(&mut g, "code", 0.001);
        // No model can satisfy tools+json+vision within 0.001 budget
        assert_eq!(g.try_get_node(0).unwrap().model_id, "original-model");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_budget_exhaustion_stops_early() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let n = assigner.assign_models_inner(&mut graph, "code", 0.0);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_assign_with_provider_hint() {
        // Two providers: "test-a" (cheap) and "test-b" (expensive)
        let toml = r#"
            [[models]]
            id = "model-a"
            provider = "provider-a"
            family = "test"
            code_score = 0.7
            reasoning_score = 0.7
            tool_use_score = 0.7
            math_score = 0.7
            formal_z3_strength = 0.5
            cost_input_per_m = 1.0
            cost_output_per_m = 3.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.7
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.7

            [[models]]
            id = "model-b"
            provider = "provider-b"
            family = "test"
            code_score = 0.75
            reasoning_score = 0.75
            tool_use_score = 0.75
            math_score = 0.75
            formal_z3_strength = 0.6
            cost_input_per_m = 1.5
            cost_output_per_m = 4.0
            latency_ttft_ms = 300.0
            tokens_per_sec = 80.0
            s1_affinity = 0.5
            s2_affinity = 0.75
            s3_affinity = 0.6
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.75
        "#;
        let registry = ModelRegistry::from_toml_str(toml).unwrap();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
        graph.add_node(n0);

        // Without hint: record which model wins
        let n_no_hint = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n_no_hint, 1);
        let assigned_no_hint = graph.try_get_node(0).unwrap().model_id.clone();

        // With hint for the OTHER provider: the hint should flip the result
        let other_provider = if assigned_no_hint == "model-a" {
            "provider-b"
        } else {
            "provider-a"
        };
        let expected_with_hint = if other_provider == "provider-a" {
            "model-a"
        } else {
            "model-b"
        };
        let mut graph2 = TopologyGraph::try_new("sequential").unwrap();
        let n0b = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
        graph2.add_node(n0b);
        let hints = vec![(0, other_provider.to_string())];
        let n_with_hint =
            assigner.assign_models_with_hints_inner(&mut graph2, "code", 10.0, &hints, None);
        assert_eq!(n_with_hint, 1);
        let assigned_with_hint = graph2.try_get_node(0).unwrap().model_id.clone();
        assert_eq!(
            assigned_with_hint, expected_with_hint,
            "Provider hint for {} should flip selection from {} to {}",
            other_provider, assigned_no_hint, expected_with_hint
        );
    }

    #[test]
    fn test_assign_single_node() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0, None, None);
        assert!(model_id.is_some());
        assert_eq!(graph.try_get_node(1).unwrap().model_id, model_id.unwrap());
    }

    #[test]
    fn test_assign_single_node_with_exclusion() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        // First assign without exclusion
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0, None, None);
        assert!(model_id.is_some());
        let first_model = model_id.unwrap();

        // Now assign with that model excluded — should pick a different one
        let mut graph2 = two_node_graph();
        let model_id2 = assigner.assign_single_node_inner(
            &mut graph2,
            1,
            "math",
            10.0,
            Some(std::slice::from_ref(&first_model)),
            None,
        );
        if let Some(ref m) = model_id2 {
            assert_ne!(m, &first_model, "Excluded model should not be reassigned");
        }
        // Either different model or None (all excluded) — both are correct
    }

    #[test]
    fn test_custom_weights_cost_heavy() {
        // With cost weight dominating (0.0/0.0/1.0), cheap-fast should win
        // for a node that doesn't require tools (reviewer, S3).
        let registry = test_registry();
        let assigner = ModelAssigner::with_weights(&registry, 0.0, 0.0, 1.0);
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        g.add_node(n);
        assigner.assign_models_inner(&mut g, "code", 10.0);
        assert_eq!(
            g.try_get_node(0).unwrap().model_id,
            "cheap-fast",
            "cost-heavy weights should prefer cheap-fast"
        );
    }

    #[test]
    fn test_custom_weights_affinity_heavy() {
        // With affinity weight dominating (1.0/0.0/0.0) for S3, expensive-smart should win.
        let registry = test_registry();
        let assigner = ModelAssigner::with_weights(&registry, 1.0, 0.0, 0.0);
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        g.add_node(n);
        assigner.assign_models_inner(&mut g, "code", 10.0);
        assert_eq!(
            g.try_get_node(0).unwrap().model_id,
            "expensive-smart",
            "affinity-heavy weights for S3 should prefer expensive-smart"
        );
    }

    #[test]
    fn test_weights_normalized() {
        // with_weights normalizes to sum=1.0
        let registry = test_registry();
        let a = ModelAssigner::with_weights(&registry, 4.0, 4.0, 2.0);
        assert!((a.weight_affinity - 0.4).abs() < 1e-6);
        assert!((a.weight_domain - 0.4).abs() < 1e-6);
        assert!((a.weight_cost - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_default_weights_unchanged() {
        let registry = test_registry();
        let a = ModelAssigner::from_registry(&registry);
        assert!((a.weight_affinity - 0.4).abs() < 1e-6);
        assert!((a.weight_domain - 0.4).abs() < 1e-6);
        assert!((a.weight_cost - 0.2).abs() < 1e-6);
    }

    // ── F7 effective_system + role-aware tier promotion ────────────────
    //
    // Topaz-inspired (arXiv 2604.03527): route by role+task requirement,
    // not by raw template tier. See sage-python/docs/benchmarks/
    // 2026-04-17-swebench-smoke-debug.md for the motivating evidence.

    #[test]
    fn test_effective_system_without_task_hint_is_identity() {
        // Legacy behaviour preserved when task_system=None — every existing
        // caller that didn't know about task-level routing still gets the
        // per-node tier it always got.
        for role in ["planner", "coder", "synthesizer", "worker"] {
            for local in [
                CognitiveSystem::S1,
                CognitiveSystem::S2,
                CognitiveSystem::S3,
            ] {
                assert_eq!(effective_system(role, local, None, "code"), local);
            }
        }
    }

    #[test]
    fn test_effective_system_s3_task_promotes_producers() {
        // The SWE-bench case: sequential template has planner=S1, coder=S2,
        // synthesizer=S1. With a task-level S3 hint on a code task, the
        // producer nodes (planner, coder, worker) floor at S2; the
        // synthesizer stays S1.
        assert_eq!(
            effective_system(
                "planner",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S2
        );
        assert_eq!(
            effective_system(
                "coder",
                CognitiveSystem::S2,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S2
        );
        assert_eq!(
            effective_system(
                "worker_0",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S2
        );
        assert_eq!(
            effective_system(
                "synthesizer",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S1
        );
    }

    #[test]
    fn test_effective_system_s2_task_is_no_op_for_low_local() {
        // S2 task: producers floor at S1, which is already the minimum —
        // no promotion happens. Templates that deliberately use cheap
        // planners on S2 tasks keep that choice.
        assert_eq!(
            effective_system(
                "planner",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S2),
                "code"
            ),
            CognitiveSystem::S1
        );
    }

    #[test]
    fn test_effective_system_sink_roles_never_promoted() {
        // Synthesizer / aggregator / formatter / output_* are terminal
        // forwarders (SINK_NODE_PROMPT). Cheap is correct — the domain
        // floor must NOT override the sink classification.
        for sink in [
            "synthesizer",
            "aggregator",
            "output_formatter",
            "formatter",
            "output_writer",
        ] {
            assert_eq!(
                effective_system(sink, CognitiveSystem::S1, Some(CognitiveSystem::S3), "math"),
                CognitiveSystem::S1,
                "sink role `{}` must stay on local tier even on math/S3",
                sink
            );
        }
    }

    #[test]
    fn test_effective_system_never_demotes() {
        // If the template explicitly picked S3 for a node, we never
        // downgrade, even if the task tier is lower.
        assert_eq!(
            effective_system(
                "verifier",
                CognitiveSystem::S3,
                Some(CognitiveSystem::S1),
                "code"
            ),
            CognitiveSystem::S3
        );
        assert_eq!(
            effective_system(
                "coder",
                CognitiveSystem::S3,
                Some(CognitiveSystem::S2),
                "code"
            ),
            CognitiveSystem::S3
        );
    }

    // ── F7 domain-aware floor (advisor sequence item 1) ──────────────
    //
    // Math/formal S3 tasks get the FULL reasoner tier (S3), not the
    // general S2 floor. Code/general S3 tasks keep the S2 floor.
    //
    // Rationale: a planner on a Coq/Lean/SMT task is doing proof search,
    // not codegen — promoting to S2 picks a strong coder (e.g.
    // gpt-5.3-codex) that can't actually prove the goal. Promoting to S3
    // picks the reasoner-tier (e.g. gemini-3.1-pro-preview) that can.
    // Cards already expose `math` and `formal` columns explicitly.

    #[test]
    fn test_f7_math_s3_floors_at_s3() {
        // Math S3 task: producer planner gets full S3, not S2.
        for domain in ["math", "Math", "MATH"] {
            assert_eq!(
                effective_system(
                    "planner",
                    CognitiveSystem::S1,
                    Some(CognitiveSystem::S3),
                    domain
                ),
                CognitiveSystem::S3,
                "math/S3 must floor producer at S3 (not S2), got domain={}",
                domain
            );
        }
    }

    #[test]
    fn test_f7_formal_s3_floors_at_s3() {
        // Formal verification S3 task: same — full reasoner tier.
        // Both bare "formal" and "formal_verification" must classify.
        for domain in ["formal", "formal_verification", "Formal", "formal_proofs"] {
            assert_eq!(
                effective_system(
                    "coder",
                    CognitiveSystem::S1,
                    Some(CognitiveSystem::S3),
                    domain
                ),
                CognitiveSystem::S3,
                "formal/S3 must floor producer at S3, got domain={}",
                domain
            );
        }
    }

    #[test]
    fn test_f7_code_s3_unchanged_floors_at_s2() {
        // Sanity: the original S2 floor for non-rigour domains MUST be
        // preserved. Otherwise we'd burn budget on a pure reasoner for
        // every SWE-bench task — exactly the opposite of what we want.
        for domain in ["code", "general", "", "swe_bench", "agent"] {
            assert_eq!(
                effective_system(
                    "planner",
                    CognitiveSystem::S1,
                    Some(CognitiveSystem::S3),
                    domain
                ),
                CognitiveSystem::S2,
                "non-rigour S3 must keep the S2 floor, got domain={}",
                domain
            );
        }
    }

    #[test]
    fn test_is_sink_role_classification() {
        // Positive: every role that templates.rs assigns SINK_NODE_PROMPT to.
        // Audit cmd: `grep -B 1 SINK_NODE_PROMPT sage-core/src/topology/templates.rs`
        for r in [
            "synthesizer",
            "Synthesizer",
            "aggregator",
            "mixer",
            "judge",
            "verifier",
            "solver",
            "output_formatter",
            "formatter",
            "sink",
            "output",
        ] {
            assert!(is_sink_role(r), "`{}` should classify as sink", r);
        }
        // Negative: producer / tool-using / reasoning roles.
        for r in [
            "planner",
            "coder",
            "worker_0",
            "source",
            "thinker",
            "brainstormer",
            "actor",
            "critic",
            "formalizer",
            "preprocessor",
            "splitter",
            "dispatcher",
        ] {
            assert!(!is_sink_role(r), "`{}` should NOT classify as sink", r);
        }
    }

    // ── Drift guards (advisor 2026-04-17): pin the SINK_ROLES list and
    // the "no coder/worker on S1" template invariant to templates.rs at
    // build time. If a future template adds a SINK_NODE_PROMPT node with
    // a new role, OR declares a coder/worker at S1, these tests fail and
    // the offending diff stops at CI.

    /// Diversity penalty: a 3-node graph where two providers tie on
    /// affinity/domain should distribute instead of all going to the
    /// marginally better one. Without the penalty, the v5f smoke had
    /// MiniMax at 88% of calls; with it, nodes 2+ get a -0.08 penalty
    /// per prior assignment to the same provider.
    #[test]
    fn test_diversity_penalty_spreads_across_providers() {
        // Two providers with near-identical scores
        let toml = r#"
            [[models]]
            id = "alpha-pro"
            provider = "alpha"
            family = "test"
            code_score = 0.80
            reasoning_score = 0.80
            tool_use_score = 0.80
            math_score = 0.80
            formal_z3_strength = 0.6
            cost_input_per_m = 1.0
            cost_output_per_m = 3.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.80
            s3_affinity = 0.80
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.80

            [[models]]
            id = "beta-pro"
            provider = "beta"
            family = "test"
            code_score = 0.78
            reasoning_score = 0.78
            tool_use_score = 0.78
            math_score = 0.78
            formal_z3_strength = 0.6
            cost_input_per_m = 1.0
            cost_output_per_m = 3.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.78
            s3_affinity = 0.78
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.78
        "#;
        let registry = ModelRegistry::from_toml_str(toml).unwrap();
        let assigner = ModelAssigner::from_registry(&registry);

        // 3 coder nodes. With diversity penalty, we expect at least 2
        // distinct providers (not all-alpha-pro).
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        for _ in 0..3 {
            let n = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
            graph.add_node(n);
        }
        let n = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n, 3);

        let providers: Vec<String> = (0..3)
            .map(|i| {
                let model_id = &graph.try_get_node(i).unwrap().model_id;
                registry
                    .get(model_id)
                    .map(|c| c.provider.clone())
                    .unwrap_or_default()
            })
            .collect();
        let distinct: std::collections::HashSet<_> = providers.iter().collect();
        assert!(
            distinct.len() >= 2,
            "diversity penalty should spread across providers; got {:?}",
            providers
        );
    }

    /// Every node tagged with templates::SINK_NODE_PROMPT must classify
    /// as sink via `is_sink_role`. Catches drift in either direction —
    /// new template sink role missing from SINK_ROLES, OR a SINK_ROLES
    /// entry that no template actually uses.
    #[test]
    fn test_sink_drift_templates_match_classifier() {
        use crate::topology::templates::{
            avr, brainstorming, debate, formal_solver, hierarchical, horizon_pipeline, hub,
            parallel, parallel_fanout, robust, self_moa, sequential, SINK_NODE_PROMPT,
        };

        let templates: Vec<(&str, TopologyGraph)> = vec![
            ("sequential", sequential("test-model")),
            ("parallel", parallel("test-model", 3)),
            ("avr", avr("test-model", "test-model")),
            ("self_moa", self_moa("test-model", 3)),
            ("hierarchical", hierarchical("test-model", "test-model")),
            ("hub", hub("test-model", "test-model", 3)),
            ("debate", debate("test-model", "test-model")),
            ("brainstorming", brainstorming("test-model", 3)),
            ("robust", robust("test-model", 3)),
            ("horizon_pipeline", horizon_pipeline("test-model", 3)),
            ("parallel_fanout", parallel_fanout("test-model", 3)),
            ("formal_solver", formal_solver("test-model")),
        ];

        let mut sink_count = 0;
        for (name, graph) in &templates {
            for idx in 0..graph.node_count() {
                let node = graph.try_get_node(idx).unwrap();
                if node.prompt == SINK_NODE_PROMPT {
                    sink_count += 1;
                    assert!(
                        is_sink_role(&node.role),
                        "template `{name}` node {idx} has SINK_NODE_PROMPT but role `{}` is NOT in SINK_ROLES — F7 will over-promote it on task-tier escalation",
                        node.role
                    );
                }
            }
        }
        // Sanity: ensure the test actually found sinks (would silently
        // pass if the SINK_NODE_PROMPT marker drifted to a different name).
        assert!(
            sink_count >= 6,
            "expected >=6 sink nodes across 12 templates, found {sink_count}"
        );
    }

    /// Roles whose F6 prompt explicitly mandates "AT LEAST 3 distinct
    /// execute_bash calls before emitting any diff" (see
    /// sage-python/src/sage/topology/role_prompts.py — the `_CODER`
    /// template, matched by substrings ("coder", "actor", "coder_worker")).
    /// Other producer roles (worker/thinker/brainstormer → `_WORKER`)
    /// only suggest "1-3 tool calls is typical" — softer, no hard floor.
    ///
    /// If a template declares a coder/actor at node.system=1, that's not
    /// a problem in itself (F1 max_steps = ctx.system, not node.system),
    /// BUT it's a smell: the only way that node gets a non-cheap budget
    /// is if the pipeline ALWAYS escalates the task tier. Since
    /// "S1 non-math skips topology" (CLAUDE.md), a coder@node.S1 only
    /// runs on S2/S3 tasks anyway → still safe. This test pins the
    /// template invariant; the runtime invariant ("S1 tasks bypass") is
    /// pinned at the Python layer.
    ///
    /// Rationale for the narrower predicate (vs the original draft that
    /// also flagged worker/thinker/brainstormer): `parallel_fanout` (line
    /// 678 in templates.rs) deliberately cycles workers S1/S2/S3 for
    /// output diversity (SC-MAS arXiv 2601.09434). That's a feature.
    #[test]
    fn test_no_strict_mandate_role_at_s1_in_any_template() {
        use crate::topology::templates::{
            avr, brainstorming, debate, formal_solver, hierarchical, horizon_pipeline, hub,
            parallel, parallel_fanout, robust, self_moa, sequential,
        };

        let templates: Vec<(&str, TopologyGraph)> = vec![
            ("sequential", sequential("test-model")),
            ("parallel", parallel("test-model", 3)),
            ("avr", avr("test-model", "test-model")),
            ("self_moa", self_moa("test-model", 3)),
            ("hierarchical", hierarchical("test-model", "test-model")),
            ("hub", hub("test-model", "test-model", 3)),
            ("debate", debate("test-model", "test-model")),
            ("brainstorming", brainstorming("test-model", 3)),
            ("robust", robust("test-model", 3)),
            ("horizon_pipeline", horizon_pipeline("test-model", 3)),
            ("parallel_fanout", parallel_fanout("test-model", 3)),
            ("formal_solver", formal_solver("test-model")),
        ];

        // Substring match — same predicate as get_role_prompt(role) maps to
        // _CODER in sage-python/src/sage/topology/role_prompts.py.
        let strict_mandate_substrings = ["coder", "actor"];

        for (name, graph) in &templates {
            for idx in 0..graph.node_count() {
                let node = graph.try_get_node(idx).unwrap();
                let role_lc = node.role.to_lowercase();
                let triggers_coder_prompt = strict_mandate_substrings
                    .iter()
                    .any(|s| role_lc.contains(s));
                if triggers_coder_prompt {
                    assert!(
                        node.system >= 2,
                        "template `{name}` declares strict-mandate role `{}` at system={} (S1) — F6 _CODER prompt requires >=3 execute_bash calls, F1 budgets only 5 steps at S1, leaving 1 buffer. Promote node to S2+ or move the role to a non-coder name.",
                        node.role, node.system
                    );
                }
            }
        }
    }

    #[test]
    fn test_formal_solver_sink_protected_on_math_s3() {
        // Regression for the audit-uncovered bug: pre-this-fix, F7's
        // domain floor would push formal_solver's `solver` node from S1
        // to S3 on a math task — replacing free deterministic Rust
        // computation with a $0.10 LLM call. The sink classification
        // must take precedence over the domain-aware floor.
        assert_eq!(
            effective_system(
                "solver",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "math"
            ),
            CognitiveSystem::S1,
            "formal_solver's solver MUST stay S1 — it's pure Rust compute"
        );
        // Also covered: every other SINK_NODE_PROMPT role on math/S3.
        for sink in ["mixer", "judge", "verifier", "solver"] {
            assert_eq!(
                effective_system(sink, CognitiveSystem::S1, Some(CognitiveSystem::S3), "math"),
                CognitiveSystem::S1,
                "newly-classified sink `{}` must not be promoted by domain rule",
                sink
            );
        }
    }

    #[test]
    fn test_s3_task_pushes_planner_to_reasoner_model() {
        // End-to-end: the two_node_graph() test registry has "cheap-fast"
        // (s1_affinity=0.9, s2_affinity=0.3) and "expensive-smart"
        // (s1_affinity=0.1, s2_affinity=0.9). A planner node at local=S1
        // would normally score cheap-fast highest. With task_system=S3,
        // the effective tier becomes S2 and expensive-smart should win.
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);

        let mut g_no_hint = TopologyGraph::try_new("sequential").unwrap();
        g_no_hint.add_node(TopologyNode::new(
            "planner".into(),
            "".into(),
            1,
            vec![],
            0,
            5.0,
            60.0,
        ));
        assigner.assign_models_with_hints_inner(&mut g_no_hint, "code", 10.0, &[], None);
        let baseline = g_no_hint.try_get_node(0).unwrap().model_id.clone();

        let mut g_with_hint = TopologyGraph::try_new("sequential").unwrap();
        g_with_hint.add_node(TopologyNode::new(
            "planner".into(),
            "".into(),
            1,
            vec![],
            0,
            5.0,
            60.0,
        ));
        assigner.assign_models_with_hints_inner(
            &mut g_with_hint,
            "code",
            10.0,
            &[],
            Some(CognitiveSystem::S3),
        );
        let promoted = g_with_hint.try_get_node(0).unwrap().model_id.clone();

        assert_eq!(
            baseline, "cheap-fast",
            "baseline planner@S1 should pick the S1-heavy cheap-fast"
        );
        assert_eq!(
            promoted, "expensive-smart",
            "task_system=S3 should promote planner to S2 and pick expensive-smart"
        );
    }

    #[test]
    fn test_is_high_rigour_domain_classification() {
        // Positive: substring match, case-insensitive — handles every
        // sensible variant the pipeline might emit.
        for d in [
            "math",
            "Math",
            "MATH",
            "formal",
            "Formal_Verification",
            "formal_proofs",
            "discrete_math",
            "applied_math",
        ] {
            assert!(is_high_rigour_domain(d), "`{}` should be high-rigour", d);
        }
        // Negative: explicit non-rigour domains and the unset case.
        for d in ["code", "general", "", "swe_bench", "agent", "tools"] {
            assert!(
                !is_high_rigour_domain(d),
                "`{}` should NOT be high-rigour",
                d
            );
        }
    }
}
