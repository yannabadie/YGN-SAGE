//! ModelAssigner — per-node model assignment using ModelCard scoring.

use pyo3::prelude::*;
use tracing::{info, warn};

use super::model_card::CognitiveSystem;
use super::model_registry::ModelRegistry;
use crate::topology::topology_graph::TopologyGraph;

const WEIGHT_AFFINITY: f32 = 0.4;
const WEIGHT_DOMAIN: f32 = 0.4;
const WEIGHT_COST: f32 = 0.2;
const BUDGET_EPSILON: f32 = 0.01;

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelAssigner {
    registry: ModelRegistry,
}

impl ModelAssigner {
    pub fn from_registry(registry: &ModelRegistry) -> Self {
        Self {
            registry: registry.clone(),
        }
    }

    pub fn assign_models_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> usize {
        let node_count = graph.node_count();
        let mut remaining_budget = budget_usd;
        let mut assigned = 0usize;

        let all_models = self.registry.all_models();
        if all_models.is_empty() {
            warn!("ModelAssigner: no models in registry, skipping assignment");
            return 0;
        }

        let max_cost = all_models
            .iter()
            .map(|c| c.estimate_cost(1000, 500))
            .fold(0.001_f32, f32::max);

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

            let mut best_id: Option<String> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;

            for card in &all_models {
                if needs_tools && !card.supports_tools {
                    continue;
                }
                if needs_json && !card.supports_json_mode {
                    continue;
                }
                let est_cost = card.estimate_cost(1000, 500);
                if est_cost > node_budget {
                    continue;
                }

                let affinity = self.registry.calibrated_affinity(&card.id, system);
                let domain = card.domain_score(task_domain);
                let cost_norm = est_cost / max_cost;
                let score = WEIGHT_AFFINITY * affinity
                    + WEIGHT_DOMAIN * domain
                    + WEIGHT_COST * (1.0 - cost_norm);

                if score > best_score {
                    best_score = score;
                    best_id = Some(card.id.clone());
                }
            }

            if let Some(model_id) = best_id {
                if let Some(card) = self.registry.get(&model_id) {
                    remaining_budget -= card.estimate_cost(1000, 500);
                }
                let node_idx_pg = petgraph::graph::NodeIndex::new(idx);
                if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                    node_mut.model_id = model_id.clone();
                    info!(
                        node = idx,
                        role = %node_mut.role,
                        model = %model_id,
                        score = best_score,
                        "model_assigned"
                    );
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
        let max_cost = all_models
            .iter()
            .map(|c| c.estimate_cost(1000, 500))
            .fold(0.001_f32, f32::max);

        let mut best_id: Option<String> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;
        for card in &all_models {
            if needs_tools && !card.supports_tools {
                continue;
            }
            if needs_json && !card.supports_json_mode {
                continue;
            }
            if card.estimate_cost(1000, 500) > budget_usd {
                continue;
            }
            let affinity = self.registry.calibrated_affinity(&card.id, system);
            let domain = card.domain_score(task_domain);
            let cost_norm = card.estimate_cost(1000, 500) / max_cost;
            let score =
                WEIGHT_AFFINITY * affinity + WEIGHT_DOMAIN * domain + WEIGHT_COST * (1.0 - cost_norm);
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
    #[new]
    fn py_new(registry: &ModelRegistry) -> Self {
        Self::from_registry(registry)
    }

    fn assign_models(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> PyResult<usize> {
        Ok(self.assign_models_inner(graph, task_domain, budget_usd))
    }

    fn assign_single_node(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
    ) -> PyResult<String> {
        self.assign_single_node_inner(graph, node_idx, task_domain, budget_usd)
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
    fn test_assign_single_node() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0);
        assert!(model_id.is_some());
        assert_eq!(
            graph.try_get_node(1).unwrap().model_id,
            model_id.unwrap()
        );
    }
}
