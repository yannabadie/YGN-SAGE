//! Template catalogue: 8 factory functions that produce pre-wired TopologyGraph instances.
//!
//! Each factory builds a complete topology with nodes, control edges, message edges,
//! and (where appropriate) state edges, gates, and conditions.

use super::topology_graph::*;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Sequential: A -> B -> C
// ---------------------------------------------------------------------------

/// Build a sequential pipeline: input_processor -> worker -> output_formatter.
pub fn sequential(model_id: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    let n0 = TopologyNode::new(
        "input_processor".into(),
        model_id.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let n1 = TopologyNode::new(
        "worker".into(),
        model_id.into(),
        2,
        vec!["reasoning".into()],
        0,
        2.0,
        120.0,
    );
    let n2 = TopologyNode::new(
        "output_formatter".into(),
        model_id.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );

    let i0 = g.add_node(n0);
    let i1 = g.add_node(n1);
    let i2 = g.add_node(n2);

    g.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();
    g.try_add_edge(i1, i2, TopologyEdge::control()).unwrap();

    g
}

// ---------------------------------------------------------------------------
// 2. Parallel: Source -> [W1..WN] -> Aggregator
// ---------------------------------------------------------------------------

/// Build a parallel fan-out/fan-in topology.
///
/// `worker_count` workers run concurrently; each sends a message edge to the aggregator
/// with field_mapping `{"result" -> "input_N"}`.
pub fn parallel(model_id: &str, worker_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("parallel").unwrap();

    let source = TopologyNode::new(
        "source".into(),
        model_id.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let si = g.add_node(source);

    let mut worker_indices = Vec::with_capacity(worker_count);
    for i in 0..worker_count {
        let w = TopologyNode::new(
            format!("worker_{}", i),
            model_id.into(),
            2,
            vec!["reasoning".into()],
            0,
            2.0,
            120.0,
        );
        worker_indices.push(g.add_node(w));
    }

    let agg = TopologyNode::new(
        "aggregator".into(),
        model_id.into(),
        1,
        vec!["aggregation".into()],
        0,
        1.0,
        60.0,
    );
    let ai = g.add_node(agg);

    for (i, &wi) in worker_indices.iter().enumerate() {
        // Control: source -> worker
        g.try_add_edge(si, wi, TopologyEdge::control()).unwrap();

        // Message: worker -> aggregator with field_mapping
        let mut mapping = HashMap::new();
        mapping.insert("result".to_string(), format!("input_{}", i));
        g.try_add_edge(wi, ai, TopologyEdge::message(Some(mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 3. AVR: Act <-> Verify (with gated back-edge for repair)
// ---------------------------------------------------------------------------

/// Build an Act-Verify-Repair topology.
///
/// Forward path: actor -> verifier -> output (control edges).
/// Back-edge: verifier -> actor (control, gate=Closed) for repair.
/// Message edge: actor -> verifier with {"code" -> "review_input"}.
pub fn avr(actor_model: &str, reviewer_model: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("avr").unwrap();

    let actor = TopologyNode::new(
        "actor".into(),
        actor_model.into(),
        2,
        vec!["code_generation".into()],
        0,
        3.0,
        120.0,
    );
    let verifier = TopologyNode::new(
        "verifier".into(),
        reviewer_model.into(),
        2,
        vec!["code_review".into()],
        0,
        2.0,
        60.0,
    );
    let output = TopologyNode::new(
        "output".into(),
        actor_model.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );

    let ai = g.add_node(actor);
    let vi = g.add_node(verifier);
    let oi = g.add_node(output);

    // Forward control path
    g.try_add_edge(ai, vi, TopologyEdge::control()).unwrap();
    g.try_add_edge(vi, oi, TopologyEdge::control()).unwrap();

    // Back-edge: verifier -> actor (repair path, initially closed)
    g.try_add_edge(vi, ai, TopologyEdge::control().with_gate(Gate::Closed))
        .unwrap();

    // Message: actor -> verifier
    let mut mapping = HashMap::new();
    mapping.insert("code".to_string(), "review_input".to_string());
    g.try_add_edge(ai, vi, TopologyEdge::message(Some(mapping)))
        .unwrap();

    g
}

// ---------------------------------------------------------------------------
// 4. SelfMoA: Multiple agents + mixture aggregation
// ---------------------------------------------------------------------------

/// Build a Self-Mixture-of-Agents topology.
///
/// All agents receive the same input and work in parallel.
/// An aggregator performs weighted mixture blending.
pub fn self_moa(model_id: &str, agent_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("selfmoa").unwrap();

    let dispatcher = TopologyNode::new(
        "dispatcher".into(),
        model_id.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let di = g.add_node(dispatcher);

    let mut agent_indices = Vec::with_capacity(agent_count);
    for i in 0..agent_count {
        let agent = TopologyNode::new(
            format!("agent_{}", i),
            model_id.into(),
            2,
            vec!["reasoning".into()],
            0,
            2.0,
            120.0,
        );
        agent_indices.push(g.add_node(agent));
    }

    let mixer = TopologyNode::new(
        "mixer".into(),
        model_id.into(),
        2,
        vec!["aggregation".into()],
        0,
        2.0,
        60.0,
    );
    let mi = g.add_node(mixer);

    for (i, &ai) in agent_indices.iter().enumerate() {
        g.try_add_edge(di, ai, TopologyEdge::control()).unwrap();

        let mut mapping = HashMap::new();
        mapping.insert("response".to_string(), format!("input_{}", i));
        g.try_add_edge(ai, mi, TopologyEdge::message(Some(mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 5. Hierarchical: Parent -> [Children] -> Parent collects
// ---------------------------------------------------------------------------

/// Build a hierarchical delegation topology.
///
/// Parent delegates via control edges, children report back via state edges.
pub fn hierarchical(parent_model: &str, child_model: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("hierarchical").unwrap();

    let parent = TopologyNode::new(
        "parent".into(),
        parent_model.into(),
        2,
        vec!["planning".into()],
        1,
        3.0,
        120.0,
    );
    let child1 = TopologyNode::new(
        "child_0".into(),
        child_model.into(),
        1,
        vec!["reasoning".into()],
        1, // same label as parent (children inherit parent's security context)
        2.0,
        60.0,
    );
    let child2 = TopologyNode::new(
        "child_1".into(),
        child_model.into(),
        1,
        vec!["reasoning".into()],
        1, // same label as parent
        2.0,
        60.0,
    );

    let pi = g.add_node(parent);
    let c1i = g.add_node(child1);
    let c2i = g.add_node(child2);

    // Parent delegates
    g.try_add_edge(pi, c1i, TopologyEdge::control()).unwrap();
    g.try_add_edge(pi, c2i, TopologyEdge::control()).unwrap();

    // Children report back via state edges
    g.try_add_edge(c1i, pi, TopologyEdge::state()).unwrap();
    g.try_add_edge(c2i, pi, TopologyEdge::state()).unwrap();

    g
}

// ---------------------------------------------------------------------------
// 6. Hub: Central coordinator + spoke delegation
// ---------------------------------------------------------------------------

/// Build a hub-and-spoke topology.
///
/// Hub node is connected to N spoke nodes via control + message edges.
/// Spokes report back via state edges. Switch conditions on hub->spoke edges.
pub fn hub(coordinator_model: &str, spoke_model: &str, spoke_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("hub").unwrap();

    let coordinator = TopologyNode::new(
        "coordinator".into(),
        coordinator_model.into(),
        2,
        vec!["planning".into(), "delegation".into()],
        1,
        3.0,
        120.0,
    );
    let ci = g.add_node(coordinator);

    for i in 0..spoke_count {
        let spoke = TopologyNode::new(
            format!("spoke_{}", i),
            spoke_model.into(),
            1,
            vec!["reasoning".into()],
            1, // same label as coordinator (spokes inherit hub's security context)
            2.0,
            60.0,
        );
        let si = g.add_node(spoke);

        // Hub -> spoke: control with switch condition + message
        g.try_add_edge(
            ci,
            si,
            TopologyEdge::control().with_condition(format!("task_type == 'type_{}'", i)),
        )
        .unwrap();

        let mut mapping = HashMap::new();
        mapping.insert("task".to_string(), "input".to_string());
        g.try_add_edge(ci, si, TopologyEdge::message(Some(mapping)))
            .unwrap();

        // Spoke -> hub: state (reporting)
        g.try_add_edge(si, ci, TopologyEdge::state()).unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 7. Debate: Agent A vs Agent B, Judge C
// ---------------------------------------------------------------------------

/// Build a debate topology.
///
/// topic_setter fans out to debater_a and debater_b in parallel,
/// both send their arguments to a judge node.
pub fn debate(debater_model: &str, judge_model: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("debate").unwrap();

    let topic = TopologyNode::new(
        "topic_setter".into(),
        debater_model.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let debater_a = TopologyNode::new(
        "debater_a".into(),
        debater_model.into(),
        2,
        vec!["reasoning".into()],
        0,
        2.0,
        120.0,
    );
    let debater_b = TopologyNode::new(
        "debater_b".into(),
        debater_model.into(),
        2,
        vec!["reasoning".into()],
        0,
        2.0,
        120.0,
    );
    let judge = TopologyNode::new(
        "judge".into(),
        judge_model.into(),
        2,
        vec!["evaluation".into()],
        0,
        2.0,
        60.0,
    );

    let ti = g.add_node(topic);
    let dai = g.add_node(debater_a);
    let dbi = g.add_node(debater_b);
    let ji = g.add_node(judge);

    // topic -> A, topic -> B (parallel start)
    g.try_add_edge(ti, dai, TopologyEdge::control()).unwrap();
    g.try_add_edge(ti, dbi, TopologyEdge::control()).unwrap();

    // A -> judge, B -> judge (message with "argument" field mapping)
    let mut mapping_a = HashMap::new();
    mapping_a.insert("argument".to_string(), "argument_a".to_string());
    g.try_add_edge(dai, ji, TopologyEdge::message(Some(mapping_a)))
        .unwrap();

    let mut mapping_b = HashMap::new();
    mapping_b.insert("argument".to_string(), "argument_b".to_string());
    g.try_add_edge(dbi, ji, TopologyEdge::message(Some(mapping_b)))
        .unwrap();

    g
}

// ---------------------------------------------------------------------------
// 8. Brainstorming: N thinkers diverge, then converge
// ---------------------------------------------------------------------------

/// Build a brainstorming topology.
///
/// N thinker agents each get the full task, then a synthesizer converges all ideas.
pub fn brainstorming(model_id: &str, thinker_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("brainstorming").unwrap();

    let prompt = TopologyNode::new(
        "prompt".into(),
        model_id.into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let pi = g.add_node(prompt);

    let mut thinker_indices = Vec::with_capacity(thinker_count);
    for i in 0..thinker_count {
        let thinker = TopologyNode::new(
            format!("thinker_{}", i),
            model_id.into(),
            2,
            vec!["reasoning".into(), "creativity".into()],
            0,
            2.0,
            120.0,
        );
        thinker_indices.push(g.add_node(thinker));
    }

    let synthesizer = TopologyNode::new(
        "synthesizer".into(),
        model_id.into(),
        2,
        vec!["aggregation".into(), "synthesis".into()],
        0,
        2.0,
        60.0,
    );
    let si = g.add_node(synthesizer);

    for (i, &ti) in thinker_indices.iter().enumerate() {
        g.try_add_edge(pi, ti, TopologyEdge::control()).unwrap();

        let mut mapping = HashMap::new();
        mapping.insert("idea".to_string(), format!("input_{}", i));
        g.try_add_edge(ti, si, TopologyEdge::message(Some(mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// TemplateStore
// ---------------------------------------------------------------------------

/// Registry that creates topologies from template names.
pub struct TemplateStore;

impl TemplateStore {
    /// Create a topology from a template name and default model.
    pub fn create(template_name: &str, model_id: &str) -> Result<TopologyGraph, String> {
        match template_name.to_lowercase().as_str() {
            "sequential" => Ok(sequential(model_id)),
            "parallel" => Ok(parallel(model_id, 3)),
            "avr" => Ok(avr(model_id, model_id)),
            "selfmoa" | "self_moa" | "self-moa" => Ok(self_moa(model_id, 3)),
            "hierarchical" => Ok(hierarchical(model_id, model_id)),
            "hub" => Ok(hub(model_id, model_id, 3)),
            "debate" => Ok(debate(model_id, model_id)),
            "brainstorming" => Ok(brainstorming(model_id, 3)),
            _ => Err(format!("Unknown template: {}", template_name)),
        }
    }

    /// List all available template names.
    pub fn available() -> Vec<&'static str> {
        vec![
            "sequential",
            "parallel",
            "avr",
            "selfmoa",
            "hierarchical",
            "hub",
            "debate",
            "brainstorming",
        ]
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// PyO3-exposed template store for creating topologies from template names.
#[pyclass]
pub struct PyTemplateStore;

#[pymethods]
impl PyTemplateStore {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Create a topology from a template name and default model ID.
    pub fn create(&self, template_name: &str, model_id: &str) -> PyResult<TopologyGraph> {
        TemplateStore::create(template_name, model_id)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// List all available template names.
    pub fn available(&self) -> Vec<String> {
        TemplateStore::available()
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TemplateStore(templates={})",
            TemplateStore::available().len()
        )
    }
}

impl Default for PyTemplateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_structure() {
        let g = sequential("model-a");
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.is_acyclic());
        assert_eq!(g.entry_nodes().len(), 1);
        assert_eq!(g.exit_nodes().len(), 1);
    }

    #[test]
    fn test_parallel_structure() {
        let g = parallel("model-a", 4);
        // 1 source + 4 workers + 1 aggregator = 6 nodes
        assert_eq!(g.node_count(), 6);
        // 4 control (source->worker) + 4 message (worker->agg) = 8
        assert_eq!(g.edge_count(), 8);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_avr_structure() {
        let g = avr("actor-model", "reviewer-model");
        assert_eq!(g.node_count(), 3);
        // 2 control forward + 1 control back-edge + 1 message = 4
        assert_eq!(g.edge_count(), 4);
        // Has cycle due to back-edge (even though gated)
        assert!(!g.is_acyclic());
    }

    #[test]
    fn test_self_moa_structure() {
        let g = self_moa("model-a", 5);
        // 1 dispatcher + 5 agents + 1 mixer = 7
        assert_eq!(g.node_count(), 7);
        // 5 control + 5 message = 10
        assert_eq!(g.edge_count(), 10);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_hierarchical_structure() {
        let g = hierarchical("parent-model", "child-model");
        assert_eq!(g.node_count(), 3);
        // 2 control (parent->child) + 2 state (child->parent) = 4
        assert_eq!(g.edge_count(), 4);
        // Has cycle (state edges form cycle)
        assert!(!g.is_acyclic());
    }

    #[test]
    fn test_hub_structure() {
        let g = hub("coord-model", "spoke-model", 3);
        // 1 coordinator + 3 spokes = 4
        assert_eq!(g.node_count(), 4);
        // Per spoke: 1 control + 1 message + 1 state = 3 edges x 3 spokes = 9
        assert_eq!(g.edge_count(), 9);
    }

    #[test]
    fn test_debate_structure() {
        let g = debate("debater-model", "judge-model");
        assert_eq!(g.node_count(), 4);
        // 2 control (topic->debaters) + 2 message (debaters->judge) = 4
        assert_eq!(g.edge_count(), 4);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_brainstorming_structure() {
        let g = brainstorming("model-a", 3);
        // 1 prompt + 3 thinkers + 1 synthesizer = 5
        assert_eq!(g.node_count(), 5);
        // 3 control + 3 message = 6
        assert_eq!(g.edge_count(), 6);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_template_store_create_all() {
        for name in TemplateStore::available() {
            let result = TemplateStore::create(name, "test-model");
            assert!(result.is_ok(), "Failed to create template '{}'", name);
            assert!(result.unwrap().node_count() > 0);
        }
    }

    #[test]
    fn test_template_store_unknown() {
        let result = TemplateStore::create("nonexistent", "model");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Unknown template"));
    }

    #[test]
    fn test_template_store_available() {
        let names = TemplateStore::available();
        assert_eq!(names.len(), 8);
    }
}
