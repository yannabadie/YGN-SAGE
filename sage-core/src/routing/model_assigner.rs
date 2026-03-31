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
}

impl ModelAssigner {
    pub fn from_registry(registry: &ModelRegistry) -> Self {
        Self {
            registry: registry.clone(),
            weight_affinity: DEFAULT_WEIGHT_AFFINITY,
            weight_domain: DEFAULT_WEIGHT_DOMAIN,
            weight_cost: DEFAULT_WEIGHT_COST,
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
        }
    }

    pub fn assign_models_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> usize {
        self.assign_models_with_hints_inner(graph, task_domain, budget_usd, &[])
    }

    /// Assign models with optional per-node provider hints.
    ///
    /// `provider_hints` is a slice of `(node_idx, provider_name)` pairs.
    /// When a hint is present for a node, candidates from that provider get
    /// a +0.15 scoring bonus (soft preference, not hard filter — if no
    /// model from the hinted provider qualifies, the best alternative wins).
    pub fn assign_models_with_hints_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: &[(usize, String)],
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

            let system = match node.system {
                1 => CognitiveSystem::S1,
                2 => CognitiveSystem::S2,
                3 => CognitiveSystem::S3,
                _ => CognitiveSystem::S1,
            };

            let caps = &node.required_capabilities;
            let needs_tools = caps.iter().any(|c| c == "tools");
            let needs_json = caps.iter().any(|c| c == "json");
            let node_budget = node.max_cost_usd.min(remaining_budget);
            let preferred_provider = hint_map.get(&idx).copied().unwrap_or("");

            let mut best_id: Option<String> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;

            for (card_idx, card) in all_models.iter().enumerate() {
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
    ) -> Option<String> {
        let node = graph.try_get_node(node_idx).ok()?;
        let system = match node.system {
            1 => CognitiveSystem::S1,
            2 => CognitiveSystem::S2,
            3 => CognitiveSystem::S3,
            _ => CognitiveSystem::S1,
        };
        let caps = &node.required_capabilities;
        let needs_tools = caps.iter().any(|c| c == "tools");
        let needs_json = caps.iter().any(|c| c == "json");
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
    #[pyo3(signature = (graph, task_domain, budget_usd, provider_hints=None))]
    fn assign_models(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: Option<Vec<(usize, String)>>,
    ) -> PyResult<usize> {
        match provider_hints {
            Some(hints) => Ok(self.assign_models_with_hints_inner(
                graph,
                task_domain,
                budget_usd,
                &hints,
            )),
            None => Ok(self.assign_models_inner(graph, task_domain, budget_usd)),
        }
    }

    /// Assign a model to a single node. Optional ``exclude_model_ids`` skips
    /// specific models (used by FrugalGPT cascade to force an upgrade).
    #[pyo3(signature = (graph, node_idx, task_domain, budget_usd, exclude_model_ids=None))]
    fn assign_single_node(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
        exclude_model_ids: Option<Vec<String>>,
    ) -> PyResult<String> {
        self.assign_single_node_inner(
            graph, node_idx, task_domain, budget_usd,
            exclude_model_ids.as_deref(),
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
        let n1 = TopologyNode::new(
            "reviewer".into(),
            "".into(),
            3,
            vec![],
            0,
            5.0,
            60.0,
        );
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
        assert_eq!(
            graph.try_get_node(0).unwrap().model_id,
            "expensive-smart"
        );
        // Reviewer (S3, no special caps) -> expensive-smart (highest S3 affinity)
        assert_eq!(
            graph.try_get_node(1).unwrap().model_id,
            "expensive-smart"
        );
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
        let other_provider = if assigned_no_hint == "model-a" { "provider-b" } else { "provider-a" };
        let expected_with_hint = if other_provider == "provider-a" { "model-a" } else { "model-b" };
        let mut graph2 = TopologyGraph::try_new("sequential").unwrap();
        let n0b = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
        graph2.add_node(n0b);
        let hints = vec![(0, other_provider.to_string())];
        let n_with_hint = assigner.assign_models_with_hints_inner(&mut graph2, "code", 10.0, &hints);
        assert_eq!(n_with_hint, 1);
        let assigned_with_hint = graph2.try_get_node(0).unwrap().model_id.clone();
        assert_eq!(assigned_with_hint, expected_with_hint,
            "Provider hint for {} should flip selection from {} to {}",
            other_provider, assigned_no_hint, expected_with_hint);
    }

    #[test]
    fn test_assign_single_node() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0, None);
        assert!(model_id.is_some());
        assert_eq!(
            graph.try_get_node(1).unwrap().model_id,
            model_id.unwrap()
        );
    }

    #[test]
    fn test_assign_single_node_with_exclusion() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        // First assign without exclusion
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0, None);
        assert!(model_id.is_some());
        let first_model = model_id.unwrap();

        // Now assign with that model excluded — should pick a different one
        let mut graph2 = two_node_graph();
        let model_id2 = assigner.assign_single_node_inner(
            &mut graph2, 1, "math", 10.0, Some(&[first_model.clone()]),
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
            g.try_get_node(0).unwrap().model_id, "cheap-fast",
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
            g.try_get_node(0).unwrap().model_id, "expensive-smart",
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
}
