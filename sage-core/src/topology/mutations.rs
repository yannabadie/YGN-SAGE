//! 7 topology mutation operators for evolutionary topology search.
//!
//! Each operator takes a `TopologyGraph` by value, applies a structural mutation,
//! validates via `HybridVerifier`, and returns a `MutationResult`. Invalid mutations
//! return `MutationResult::Invalid` and are NOT retried.
//!
//! Operator selection uses Thompson sampling on per-operator Beta posteriors
//! (Fix 1, based on AgentConductor arXiv 2602.17100 and AgentDropout arXiv 2503.18891).
//! Operators that produce valid, high-quality mutations get sampled more often.

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::topology::topology_graph::*;
use crate::topology::verifier::HybridVerifier;

// ---------------------------------------------------------------------------
// MutationStats — Thompson sampling per mutation operator
// ---------------------------------------------------------------------------

/// Names of the 7 mutation operators (indexed 0..6).
pub const OPERATOR_NAMES: [&str; 7] = [
    "add_node",
    "remove_node",
    "swap_model",
    "rewire_edge",
    "split_node",
    "merge_nodes",
    "mutate_prompt",
];

/// Per-operator Beta posteriors for Thompson sampling.
///
/// Each operator has a Beta(alpha, beta) distribution tracking its success rate.
/// `alpha` = successes + 1 (prior), `beta` = failures + 1 (prior).
/// Thompson sampling draws from each Beta and picks the operator with the
/// highest draw — naturally balancing exploration (try undersampled operators)
/// and exploitation (prefer operators that succeed more).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationStats {
    /// Beta posteriors: (alpha, beta) per operator.
    alphas: [f64; 7],
    betas: [f64; 7],
}

impl Default for MutationStats {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationStats {
    /// Uniform prior: Beta(1, 1) for all operators (no preference).
    pub fn new() -> Self {
        Self {
            alphas: [1.0; 7],
            betas: [1.0; 7],
        }
    }

    /// Thompson-sample the best operator index.
    ///
    /// Draws from Beta(alpha_i, beta_i) for each operator, returns the index
    /// with the highest draw. Uses Box-Muller Gaussian approximation
    /// (same as ContextualBandit — no rand_distr dependency).
    pub fn sample_operator<R: Rng>(&self, rng: &mut R) -> u32 {
        let mut best_idx = 0u32;
        let mut best_sample = f64::NEG_INFINITY;

        for i in 0..7 {
            let alpha = self.alphas[i];
            let beta = self.betas[i];
            let mean = alpha / (alpha + beta);
            let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
            let std = variance.sqrt();

            // Box-Muller transform for Gaussian approximation of Beta
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let sample = (mean + std * z).clamp(0.0, 1.0);

            if sample > best_sample {
                best_sample = sample;
                best_idx = i as u32;
            }
        }

        best_idx
    }

    /// Record a mutation outcome.
    ///
    /// `operator_idx`: which operator was used (0..6)
    /// `success`: true if MutationResult::Success (and quality > 0)
    pub fn record(&mut self, operator_idx: usize, success: bool) {
        if operator_idx >= 7 {
            return;
        }
        if success {
            self.alphas[operator_idx] += 1.0;
        } else {
            self.betas[operator_idx] += 1.0;
        }
    }

    /// Get success rate (mean) for an operator.
    pub fn success_rate(&self, operator_idx: usize) -> f64 {
        if operator_idx >= 7 {
            return 0.5;
        }
        self.alphas[operator_idx] / (self.alphas[operator_idx] + self.betas[operator_idx])
    }

    /// Raw alpha (Beta posterior successes + 1) for an operator. Returns `1.0`
    /// for out-of-range indices to mirror the uninformed Beta(1,1) prior.
    pub fn alpha(&self, operator_idx: usize) -> f64 {
        if operator_idx >= 7 {
            1.0
        } else {
            self.alphas[operator_idx]
        }
    }

    /// Raw beta (Beta posterior failures + 1) for an operator. Returns `1.0`
    /// for out-of-range indices to mirror the uninformed Beta(1,1) prior.
    pub fn beta(&self, operator_idx: usize) -> f64 {
        if operator_idx >= 7 {
            1.0
        } else {
            self.betas[operator_idx]
        }
    }

    /// Attempts = alpha + beta - 2 (drop the Beta(1,1) prior mass).
    pub fn attempts(&self, operator_idx: usize) -> u32 {
        if operator_idx >= 7 {
            return 0;
        }
        let raw = (self.alphas[operator_idx] + self.betas[operator_idx] - 2.0).max(0.0);
        raw.round() as u32
    }

    /// Successes = alpha - 1 (drop the Beta(1,1) prior mass).
    pub fn successes(&self, operator_idx: usize) -> u32 {
        if operator_idx >= 7 {
            return 0;
        }
        let raw = (self.alphas[operator_idx] - 1.0).max(0.0);
        raw.round() as u32
    }

    /// Summary string for logging.
    pub fn summary(&self) -> String {
        OPERATOR_NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{}={:.0}%", name, self.success_rate(i) * 100.0,))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ---------------------------------------------------------------------------
// MutationResult
// ---------------------------------------------------------------------------

/// Result of applying a mutation operator to a topology graph.
#[derive(Debug, Clone)]
pub enum MutationResult {
    /// Mutation produced a valid topology.
    Success(TopologyGraph),
    /// Mutation produced an invalid topology (verifier rejected).
    Invalid(String),
}

impl MutationResult {
    /// Returns `true` if the mutation was successful.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Returns `true` if the mutation was invalid.
    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid(_))
    }

    /// Unwrap the successful topology, panicking if invalid.
    pub fn unwrap(self) -> TopologyGraph {
        match self {
            Self::Success(g) => g,
            Self::Invalid(msg) => panic!("called unwrap on Invalid: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

/// Validate a graph via HybridVerifier. Returns MutationResult.
fn validate(graph: TopologyGraph) -> MutationResult {
    let verifier = HybridVerifier::new();
    let result = verifier.verify(&graph);
    if result.valid {
        MutationResult::Success(graph)
    } else {
        let msg = result.errors.join("; ");
        debug!(errors = %msg, "mutation_rejected_by_verifier");
        MutationResult::Invalid(msg)
    }
}

// ---------------------------------------------------------------------------
// 1. add_node
// ---------------------------------------------------------------------------

/// Insert a new agent node with the given role/model/system.
/// Connects it with a control edge from a randomly selected exit node.
/// Pass `exit_hint` to deterministically select which exit node to connect from,
/// or `None` to use the first exit node.
pub fn add_node(graph: TopologyGraph, role: &str, model_id: &str, system: u8) -> MutationResult {
    add_node_at(graph, role, model_id, system, None)
}

/// Insert a new agent node, connecting from the specified exit node index.
pub fn add_node_at(
    mut graph: TopologyGraph,
    role: &str,
    model_id: &str,
    system: u8,
    exit_hint: Option<usize>,
) -> MutationResult {
    let node = TopologyNode::new(
        role.to_string(),
        model_id.to_string(),
        system,
        Vec::new(),
        0,
        1.0,
        60.0,
    );

    let exit_nodes = graph.exit_nodes();
    let new_idx = graph.add_node(node);

    // Connect from selected exit node (if any exist).
    if !exit_nodes.is_empty() {
        let from_idx = exit_hint.unwrap_or(0).min(exit_nodes.len() - 1);
        let from = exit_nodes[from_idx];
        if let Err(e) = graph.try_add_edge(from, new_idx, TopologyEdge::control()) {
            return MutationResult::Invalid(format!("Failed to add edge: {}", e));
        }
    }

    info!(
        new_node_idx = new_idx,
        role = role,
        model_id = model_id,
        "mutation_add_node"
    );

    validate(graph)
}

// ---------------------------------------------------------------------------
// 2. remove_node
// ---------------------------------------------------------------------------

/// Remove a node (if graph has > 2 nodes, otherwise Invalid).
/// Rewires: all predecessors connect to all successors via control edges.
pub fn remove_node(graph: TopologyGraph, node_index: usize) -> MutationResult {
    let inner = graph.inner_graph();
    let node_count = inner.node_count();

    if node_count <= 2 {
        return MutationResult::Invalid(format!(
            "Cannot remove node from graph with only {} nodes (minimum 2)",
            node_count
        ));
    }

    let target = NodeIndex::new(node_index);
    if inner.node_weight(target).is_none() {
        return MutationResult::Invalid(format!(
            "Node index {} out of range (graph has {} nodes)",
            node_index, node_count
        ));
    }

    // Collect predecessors and successors before mutation.
    let predecessors: Vec<usize> = inner
        .edges_directed(target, petgraph::Direction::Incoming)
        .map(|e| e.source().index())
        .collect();

    let successors: Vec<usize> = inner
        .edges_directed(target, petgraph::Direction::Outgoing)
        .map(|e| e.target().index())
        .collect();

    // Collect all node weights and edges (excluding the target node).
    let mut nodes: Vec<TopologyNode> = Vec::new();
    let mut old_to_new: Vec<Option<usize>> = vec![None; node_count];
    let mut new_idx = 0usize;

    for idx in inner.node_indices() {
        if idx == target {
            continue;
        }
        nodes.push(inner[idx].clone());
        old_to_new[idx.index()] = Some(new_idx);
        new_idx += 1;
    }

    let mut edges: Vec<(usize, usize, TopologyEdge)> = Vec::new();
    for edge_ref in inner.edge_references() {
        let src = edge_ref.source().index();
        let tgt = edge_ref.target().index();
        // Skip edges involving the removed node.
        if src == node_index || tgt == node_index {
            continue;
        }
        if let (Some(new_src), Some(new_tgt)) = (old_to_new[src], old_to_new[tgt]) {
            edges.push((new_src, new_tgt, edge_ref.weight().clone()));
        }
    }

    // Add rewiring edges: predecessors -> successors.
    for &pred in &predecessors {
        for &succ in &successors {
            if pred == node_index || succ == node_index {
                continue;
            }
            if pred == succ {
                continue; // avoid self-loops
            }
            if let (Some(new_pred), Some(new_succ)) = (old_to_new[pred], old_to_new[succ]) {
                // Check if this edge already exists.
                let exists = edges
                    .iter()
                    .any(|(s, t, _)| *s == new_pred && *t == new_succ);
                if !exists {
                    edges.push((new_pred, new_succ, TopologyEdge::control()));
                }
            }
        }
    }

    // Rebuild graph.
    let mut new_graph = match TopologyGraph::try_new(&graph.template_type) {
        Ok(g) => g,
        Err(e) => return MutationResult::Invalid(format!("Failed to create graph: {}", e)),
    };
    for node in nodes {
        new_graph.add_node(node);
    }
    for (from, to, edge) in edges {
        if let Err(e) = new_graph.try_add_edge(from, to, edge) {
            return MutationResult::Invalid(format!("Failed to rewire: {}", e));
        }
    }

    info!(
        removed_node = node_index,
        new_node_count = new_graph.node_count(),
        "mutation_remove_node"
    );

    validate(new_graph)
}

// ---------------------------------------------------------------------------
// 3. swap_model
// ---------------------------------------------------------------------------

/// Change the model_id of a specific node.
pub fn swap_model(
    mut graph: TopologyGraph,
    node_index: usize,
    new_model_id: &str,
) -> MutationResult {
    let target = NodeIndex::new(node_index);
    match graph.inner_graph_mut().node_weight_mut(target) {
        Some(node) => {
            info!(
                node_index = node_index,
                old_model = %node.model_id,
                new_model = new_model_id,
                "mutation_swap_model"
            );
            node.model_id = new_model_id.to_string();
        }
        None => {
            return MutationResult::Invalid(format!("Node index {} out of range", node_index));
        }
    }

    validate(graph)
}

// ---------------------------------------------------------------------------
// 4. rewire_edge
// ---------------------------------------------------------------------------

/// Add a new control edge between `from` and `to` nodes.
/// If edge already exists or from == to (self-loop), returns Invalid.
pub fn rewire_edge(mut graph: TopologyGraph, from: usize, to: usize) -> MutationResult {
    // Reject self-loops early.
    if from == to {
        return MutationResult::Invalid(format!("Self-loop not allowed: from == to == {}", from));
    }

    // Check for duplicate edge.
    let inner = graph.inner_graph();
    let from_idx = NodeIndex::new(from);
    let to_idx = NodeIndex::new(to);

    if inner.node_weight(from_idx).is_none() {
        return MutationResult::Invalid(format!("Source node index {} out of range", from));
    }
    if inner.node_weight(to_idx).is_none() {
        return MutationResult::Invalid(format!("Target node index {} out of range", to));
    }

    // Check if a control edge already exists between these nodes.
    let already_exists = inner
        .edges_directed(from_idx, petgraph::Direction::Outgoing)
        .any(|e| e.target() == to_idx && e.weight().typed_edge_type() == EdgeType::Control);

    if already_exists {
        return MutationResult::Invalid(format!(
            "Control edge from {} to {} already exists",
            from, to
        ));
    }

    if let Err(e) = graph.try_add_edge(from, to, TopologyEdge::control()) {
        return MutationResult::Invalid(format!("Failed to add edge: {}", e));
    }

    info!(from = from, to = to, "mutation_rewire_edge");

    validate(graph)
}

// ---------------------------------------------------------------------------
// 5. split_node
// ---------------------------------------------------------------------------

/// Replace one node with two specialized nodes connected in sequence.
/// First new node inherits all incoming edges, second gets all outgoing edges.
/// Control edge from first to second.
pub fn split_node(
    graph: TopologyGraph,
    node_index: usize,
    role_a: &str,
    model_a: &str,
    role_b: &str,
    model_b: &str,
) -> MutationResult {
    let inner = graph.inner_graph();
    let node_count = inner.node_count();
    let target = NodeIndex::new(node_index);

    if inner.node_weight(target).is_none() {
        return MutationResult::Invalid(format!(
            "Node index {} out of range (graph has {} nodes)",
            node_index, node_count
        ));
    }

    let original = inner[target].clone();

    // Collect incoming and outgoing edges.
    let incoming: Vec<(usize, TopologyEdge)> = inner
        .edges_directed(target, petgraph::Direction::Incoming)
        .map(|e| (e.source().index(), e.weight().clone()))
        .collect();

    let outgoing: Vec<(usize, TopologyEdge)> = inner
        .edges_directed(target, petgraph::Direction::Outgoing)
        .map(|e| (e.target().index(), e.weight().clone()))
        .collect();

    // Collect all nodes and edges (excluding the target).
    let mut nodes: Vec<TopologyNode> = Vec::new();
    let mut old_to_new: Vec<Option<usize>> = vec![None; node_count];
    let mut new_idx = 0usize;

    for idx in inner.node_indices() {
        if idx == target {
            continue;
        }
        nodes.push(inner[idx].clone());
        old_to_new[idx.index()] = Some(new_idx);
        new_idx += 1;
    }

    // Create two new nodes.
    let node_a = TopologyNode::new(
        role_a.to_string(),
        model_a.to_string(),
        original.system,
        original.required_capabilities.clone(),
        original.security_label,
        original.max_cost_usd / 2.0,
        original.max_wall_time_s,
    );
    let node_b = TopologyNode::new(
        role_b.to_string(),
        model_b.to_string(),
        original.system,
        original.required_capabilities.clone(),
        original.security_label,
        original.max_cost_usd / 2.0,
        original.max_wall_time_s,
    );

    let idx_a = nodes.len();
    nodes.push(node_a);
    let idx_b = nodes.len();
    nodes.push(node_b);

    // Collect existing edges (excluding target).
    let mut edges: Vec<(usize, usize, TopologyEdge)> = Vec::new();
    for edge_ref in inner.edge_references() {
        let src = edge_ref.source().index();
        let tgt = edge_ref.target().index();
        if src == node_index || tgt == node_index {
            continue;
        }
        if let (Some(new_src), Some(new_tgt)) = (old_to_new[src], old_to_new[tgt]) {
            edges.push((new_src, new_tgt, edge_ref.weight().clone()));
        }
    }

    // Redirect incoming edges to node_a.
    for (src, edge) in &incoming {
        if *src == node_index {
            continue; // skip self-loops
        }
        if let Some(new_src) = old_to_new[*src] {
            edges.push((new_src, idx_a, edge.clone()));
        }
    }

    // Control edge: node_a -> node_b.
    edges.push((idx_a, idx_b, TopologyEdge::control()));

    // Redirect outgoing edges from node_b.
    for (tgt, edge) in &outgoing {
        if *tgt == node_index {
            continue; // skip self-loops
        }
        if let Some(new_tgt) = old_to_new[*tgt] {
            edges.push((idx_b, new_tgt, edge.clone()));
        }
    }

    // Rebuild graph.
    let mut new_graph = match TopologyGraph::try_new(&graph.template_type) {
        Ok(g) => g,
        Err(e) => return MutationResult::Invalid(format!("Failed to create graph: {}", e)),
    };
    for node in nodes {
        new_graph.add_node(node);
    }
    for (from, to, edge) in edges {
        if let Err(e) = new_graph.try_add_edge(from, to, edge) {
            return MutationResult::Invalid(format!("Failed to rebuild after split: {}", e));
        }
    }

    info!(
        split_node = node_index,
        role_a = role_a,
        role_b = role_b,
        new_node_count = new_graph.node_count(),
        "mutation_split_node"
    );

    validate(new_graph)
}

// ---------------------------------------------------------------------------
// 6. merge_nodes
// ---------------------------------------------------------------------------

/// Merge two adjacent nodes into one generalist node.
/// New node gets union of incoming edges (from both) and union of outgoing edges (from both).
/// If a and b are not adjacent, returns Invalid.
pub fn merge_nodes(
    graph: TopologyGraph,
    node_a: usize,
    node_b: usize,
    merged_role: &str,
    merged_model: &str,
) -> MutationResult {
    let inner = graph.inner_graph();
    let node_count = inner.node_count();
    let idx_a = NodeIndex::new(node_a);
    let idx_b = NodeIndex::new(node_b);

    if inner.node_weight(idx_a).is_none() {
        return MutationResult::Invalid(format!(
            "Node A index {} out of range (graph has {} nodes)",
            node_a, node_count
        ));
    }
    if inner.node_weight(idx_b).is_none() {
        return MutationResult::Invalid(format!(
            "Node B index {} out of range (graph has {} nodes)",
            node_b, node_count
        ));
    }
    if node_a == node_b {
        return MutationResult::Invalid("Cannot merge a node with itself".to_string());
    }

    // Check adjacency: there must be an edge from a->b or b->a.
    let a_to_b = inner
        .edges_directed(idx_a, petgraph::Direction::Outgoing)
        .any(|e| e.target() == idx_b);
    let b_to_a = inner
        .edges_directed(idx_b, petgraph::Direction::Outgoing)
        .any(|e| e.target() == idx_a);

    if !a_to_b && !b_to_a {
        return MutationResult::Invalid(format!(
            "Nodes {} and {} are not adjacent",
            node_a, node_b
        ));
    }

    let orig_a = &inner[idx_a];
    let orig_b = &inner[idx_b];

    // Create merged node.
    let merged_node = TopologyNode::new(
        merged_role.to_string(),
        merged_model.to_string(),
        orig_a.system.max(orig_b.system),
        {
            let mut caps: Vec<String> = orig_a.required_capabilities.clone();
            for cap in &orig_b.required_capabilities {
                if !caps.contains(cap) {
                    caps.push(cap.clone());
                }
            }
            caps
        },
        orig_a.security_label.max(orig_b.security_label),
        orig_a.max_cost_usd + orig_b.max_cost_usd,
        orig_a.max_wall_time_s.max(orig_b.max_wall_time_s),
    );

    // Build new index mapping.
    let mut nodes: Vec<TopologyNode> = Vec::new();
    let mut old_to_new: Vec<Option<usize>> = vec![None; node_count];
    let mut new_idx = 0usize;

    for idx in inner.node_indices() {
        if idx == idx_a || idx == idx_b {
            continue;
        }
        nodes.push(inner[idx].clone());
        old_to_new[idx.index()] = Some(new_idx);
        new_idx += 1;
    }

    // Merged node index.
    let merged_idx = nodes.len();
    nodes.push(merged_node);
    old_to_new[node_a] = Some(merged_idx);
    old_to_new[node_b] = Some(merged_idx);

    // Collect edges, remapping a and b to merged_idx.
    let mut edges: Vec<(usize, usize, TopologyEdge)> = Vec::new();
    let mut seen_edges: Vec<(usize, usize)> = Vec::new();

    for edge_ref in inner.edge_references() {
        let src = edge_ref.source().index();
        let tgt = edge_ref.target().index();

        // Skip edges between a and b (they are being merged).
        if (src == node_a && tgt == node_b) || (src == node_b && tgt == node_a) {
            continue;
        }

        if let (Some(new_src), Some(new_tgt)) = (old_to_new[src], old_to_new[tgt]) {
            // Skip self-loops created by the merge.
            if new_src == new_tgt {
                continue;
            }
            // Deduplicate edges.
            let key = (new_src, new_tgt);
            if !seen_edges.contains(&key) {
                edges.push((new_src, new_tgt, edge_ref.weight().clone()));
                seen_edges.push(key);
            }
        }
    }

    // Rebuild graph.
    let mut new_graph = match TopologyGraph::try_new(&graph.template_type) {
        Ok(g) => g,
        Err(e) => return MutationResult::Invalid(format!("Failed to create graph: {}", e)),
    };
    for node in nodes {
        new_graph.add_node(node);
    }
    for (from, to, edge) in edges {
        if let Err(e) = new_graph.try_add_edge(from, to, edge) {
            return MutationResult::Invalid(format!("Failed to rebuild after merge: {}", e));
        }
    }

    info!(
        node_a = node_a,
        node_b = node_b,
        merged_role = merged_role,
        new_node_count = new_graph.node_count(),
        "mutation_merge_nodes"
    );

    validate(new_graph)
}

// ---------------------------------------------------------------------------
// 7. mutate_prompt
// ---------------------------------------------------------------------------

/// Change the role string of a node (simulates prompt mutation).
pub fn mutate_prompt(
    mut graph: TopologyGraph,
    node_index: usize,
    new_role: &str,
) -> MutationResult {
    let target = NodeIndex::new(node_index);
    match graph.inner_graph_mut().node_weight_mut(target) {
        Some(node) => {
            info!(
                node_index = node_index,
                old_role = %node.role,
                new_role = new_role,
                "mutation_mutate_prompt"
            );
            node.role = new_role.to_string();
        }
        None => {
            return MutationResult::Invalid(format!("Node index {} out of range", node_index));
        }
    }

    validate(graph)
}

// ---------------------------------------------------------------------------
// apply_random_mutation
// ---------------------------------------------------------------------------

/// Role precedence tiers (not a strict total order — roles in the same tier can coexist).
/// Based on AgentConductor (arXiv 2602.17100) topological execution order.
///
/// Tier 0: Input/planning roles (always first)
/// Tier 1: Processing/working roles (middle)
/// Tier 2: Evaluation/verification roles (late)
/// Tier 3: Output/synthesis roles (always last)
///
/// Unknown roles default to tier 1 (permissive — they can go anywhere in the middle).
pub(crate) fn role_tier(role: &str) -> u8 {
    // Strip numeric suffixes (worker_0, stage_1, etc.)
    let base = role.split('_').next().unwrap_or(role);
    match base {
        // Tier 0: Input/decomposition. "parent" / "coordinator" delegate
        // out-going work in `hierarchical` and `hub` templates respectively
        // — semantically the entry point, not a sink. Classifying them as
        // tier 3 (sink) makes the role-ordering verifier reject every
        // parent/coordinator -> child/spoke control edge.
        "planner" | "preprocessor" | "splitter" | "dispatcher" | "topic" | "source" | "input"
        | "parent" | "coordinator" => 0,
        // Tier 1: Processing/working (most roles)
        "analyst" | "coder" | "actor" | "worker" | "debater" | "thinker" | "stage" | "spoke"
        | "child" | "agent" | "reasoner" => 1,
        // Tier 2: Evaluation/verification
        "reviewer" | "verifier" | "judge" | "output" => 2,
        // Tier 3: Final synthesis/formatting
        "synthesizer" | "aggregator" | "formatter" | "mixer" => 3,
        // Unknown roles: tier 1 (permissive — middle of pipeline)
        _ => 1,
    }
}

/// Get precedence index for backward compatibility. Returns tier * 10.
pub(crate) fn role_index(role: &str) -> usize {
    role_tier(role) as usize * 10
}

/// Roles available for mutation, grouped by tier.
///
/// These per-tier slices document the role taxonomy that `role_tier()` (~L775)
/// matches against. The mutation machinery itself uses `ALL_ROLES` directly
/// (sampled from in `add_node` and `mutate_role`), but keeping the per-tier
/// view alongside is load-bearing for future ablation work — when we vet the
/// `role_tier` match arms, this is the canonical "what's in tier N" reference.
/// `#[allow(dead_code)]` because cargo `--no-default-features` clippy doesn't
/// see the consumers (e.g. cognitive feature) and would deny on -D warnings.
#[allow(dead_code)]
const ROLES_TIER0: &[&str] = &["planner", "preprocessor", "splitter", "dispatcher"];
#[allow(dead_code)]
const ROLES_TIER1: &[&str] = &["analyst", "coder", "worker", "reasoner"];
#[allow(dead_code)]
const ROLES_TIER2: &[&str] = &["reviewer", "verifier", "judge"];
#[allow(dead_code)]
const ROLES_TIER3: &[&str] = &["synthesizer", "aggregator", "formatter", "mixer"];
const ALL_ROLES: &[&str] = &[
    "planner",
    "preprocessor",
    "splitter",
    "dispatcher",
    "analyst",
    "coder",
    "worker",
    "reasoner",
    "reviewer",
    "verifier",
    "judge",
    "synthesizer",
    "aggregator",
    "formatter",
    "mixer",
];

/// Pick a role from tier >= min_tier.
fn pick_role_from_tier<R: Rng>(rng: &mut R, min_tier: u8) -> &'static str {
    let candidates: Vec<&str> = ALL_ROLES
        .iter()
        .copied()
        .filter(|r| role_tier(r) >= min_tier)
        .collect();
    if candidates.is_empty() {
        ALL_ROLES[ALL_ROLES.len() - 1]
    } else {
        candidates[rng.random_range(0..candidates.len())]
    }
}

/// Pick a role from tier <= max_tier.
fn pick_role_up_to_tier<R: Rng>(rng: &mut R, max_tier: u8) -> &'static str {
    let candidates: Vec<&str> = ALL_ROLES
        .iter()
        .copied()
        .filter(|r| role_tier(r) <= max_tier)
        .collect();
    if candidates.is_empty() {
        ALL_ROLES[0]
    } else {
        candidates[rng.random_range(0..candidates.len())]
    }
}

/// Pick one of the 7 mutations at random and apply it with random parameters.
///
/// Uses Thompson sampling on `MutationStats` to pick the operator most likely
/// to succeed (Fix 1). Operators that produce valid topologies get sampled more.
/// Uses empty model_id ("") so ModelAssigner picks from cards.toml at Stage 3.
/// Role selection respects tier ordering for semantic coherence (Fix 2).
pub fn apply_random_mutation<R: Rng>(graph: TopologyGraph, rng: &mut R) -> MutationResult {
    apply_mutation_with_stats(graph, rng, &MutationStats::new())
}

/// Apply a random mutation using Thompson sampling from `stats`.
/// Returns the operator index used (for recording outcome) and the result.
pub fn apply_mutation_tracked<R: Rng>(
    graph: TopologyGraph,
    rng: &mut R,
    stats: &MutationStats,
) -> (u32, MutationResult) {
    let node_count = graph.node_count();

    if node_count == 0 {
        let role = ALL_ROLES[rng.random_range(0..ALL_ROLES.len())];
        let system: u8 = rng.random_range(1..=3);
        return (0, add_node_at(graph, role, "", system, None));
    }

    let mutation_idx = stats.sample_operator(rng);
    let result = apply_mutation_with_stats(graph, rng, stats);
    (mutation_idx, result)
}

fn apply_mutation_with_stats<R: Rng>(
    graph: TopologyGraph,
    rng: &mut R,
    stats: &MutationStats,
) -> MutationResult {
    let node_count = graph.node_count();

    if node_count == 0 {
        let role = ALL_ROLES[rng.random_range(0..ALL_ROLES.len())];
        let system: u8 = rng.random_range(1..=3);
        return add_node_at(graph, role, "", system, None);
    }

    // Thompson-sample the best operator
    let mutation_idx = stats.sample_operator(rng);

    match mutation_idx {
        0 => {
            // add_node — role must be >= last exit node's role (precedence order)
            let exit_nodes = graph.exit_nodes();
            let exit_hint = if !exit_nodes.is_empty() {
                Some(rng.random_range(0..exit_nodes.len()))
            } else {
                None
            };
            // Pick a role from tier >= exit node's tier (respects ordering)
            let min_tier = if let Some(hint) = exit_hint {
                let exit_node = graph
                    .inner_graph()
                    .node_weight(petgraph::graph::NodeIndex::new(exit_nodes[hint]));
                exit_node.map(|n| role_tier(&n.role)).unwrap_or(0)
            } else {
                0
            };
            let role = pick_role_from_tier(rng, min_tier);
            let system: u8 = rng.random_range(1..=3);
            debug!(mutation = "add_node", role = role, "apply_random_mutation");
            add_node_at(graph, role, "", system, exit_hint)
        }
        1 => {
            // remove_node
            let idx = rng.random_range(0..node_count);
            debug!(
                mutation = "remove_node",
                node_index = idx,
                "apply_random_mutation"
            );
            remove_node(graph, idx)
        }
        2 => {
            // swap_model — use empty model_id, ModelAssigner picks at Stage 3
            let idx = rng.random_range(0..node_count);
            debug!(
                mutation = "swap_model",
                node_index = idx,
                "apply_random_mutation"
            );
            swap_model(graph, idx, "")
        }
        3 => {
            // rewire_edge — only if respects role ordering
            let from = rng.random_range(0..node_count);
            let to = rng.random_range(0..node_count);
            debug!(
                mutation = "rewire_edge",
                from = from,
                to = to,
                "apply_random_mutation"
            );
            rewire_edge(graph, from, to)
        }
        4 => {
            // split_node — role_a <= original role <= role_b
            let idx = rng.random_range(0..node_count);
            let orig_tier = {
                let inner = graph.inner_graph();
                let target = petgraph::graph::NodeIndex::new(idx);
                inner
                    .node_weight(target)
                    .map(|n| role_tier(&n.role))
                    .unwrap_or(1)
            };
            // role_a: same or earlier tier, role_b: same or later tier
            let role_a = pick_role_up_to_tier(rng, orig_tier);
            let role_b = pick_role_from_tier(rng, orig_tier);
            debug!(
                mutation = "split_node",
                node_index = idx,
                "apply_random_mutation"
            );
            split_node(graph, idx, role_a, "", role_b, "")
        }
        5 => {
            // merge_nodes
            if node_count < 2 {
                return MutationResult::Invalid("Cannot merge with fewer than 2 nodes".to_string());
            }
            let a = rng.random_range(0..node_count);
            let mut b = rng.random_range(0..node_count);
            if b == a {
                b = (a + 1) % node_count;
            }
            // Merged role: pick from the later tier of the two (preserves ordering)
            let tier_a = {
                let inner = graph.inner_graph();
                inner
                    .node_weight(petgraph::graph::NodeIndex::new(a))
                    .map(|n| role_tier(&n.role))
                    .unwrap_or(1)
            };
            let tier_b = {
                let inner = graph.inner_graph();
                inner
                    .node_weight(petgraph::graph::NodeIndex::new(b))
                    .map(|n| role_tier(&n.role))
                    .unwrap_or(1)
            };
            let merged_role = pick_role_from_tier(rng, tier_a.max(tier_b));
            debug!(
                mutation = "merge_nodes",
                node_a = a,
                node_b = b,
                "apply_random_mutation"
            );
            merge_nodes(graph, a, b, merged_role, "")
        }
        6 => {
            // mutate_prompt — pick a role from same tier as current
            let idx = rng.random_range(0..node_count);
            let current_tier = {
                let inner = graph.inner_graph();
                inner
                    .node_weight(petgraph::graph::NodeIndex::new(idx))
                    .map(|n| role_tier(&n.role))
                    .unwrap_or(1)
            };
            let role = pick_role_from_tier(rng, current_tier);
            debug!(
                mutation = "mutate_prompt",
                node_index = idx,
                new_role = role,
                "apply_random_mutation"
            );
            mutate_prompt(graph, idx, role)
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;

    fn make_sequential() -> TopologyGraph {
        templates::sequential("gemini-2.5-flash")
    }

    #[test]
    fn test_add_node_increases_count() {
        let graph = make_sequential();
        let original_count = graph.node_count();
        // Use a tier-3 role (aggregator) since exit node is synthesizer (tier 3)
        // model_id "test" to pass capability check
        let result = add_node(graph, "aggregator", "test", 1);
        assert!(result.is_success(), "Expected Success, got: {:?}", result);
        let new_graph = result.unwrap();
        assert_eq!(new_graph.node_count(), original_count + 1);
    }

    #[test]
    fn test_remove_node_decreases_count() {
        let graph = make_sequential(); // 3 nodes
        assert_eq!(graph.node_count(), 3);
        let result = remove_node(graph, 1); // remove middle node
        assert!(result.is_success(), "Expected Success, got: {:?}", result);
        let new_graph = result.unwrap();
        assert_eq!(new_graph.node_count(), 2);
    }

    #[test]
    fn test_swap_model_changes_model_id() {
        let graph = make_sequential();
        let result = swap_model(graph, 0, "gpt-5.3-codex");
        assert!(result.is_success(), "Expected Success, got: {:?}", result);
        let new_graph = result.unwrap();
        let node = new_graph.try_get_node(0).unwrap();
        assert_eq!(node.model_id, "gpt-5.3-codex");
    }

    #[test]
    fn test_mutate_prompt_changes_role() {
        let graph = make_sequential();
        let result = mutate_prompt(graph, 0, "super_coder");
        assert!(result.is_success(), "Expected Success, got: {:?}", result);
        let new_graph = result.unwrap();
        let node = new_graph.try_get_node(0).unwrap();
        assert_eq!(node.role, "super_coder");
    }

    #[test]
    fn test_apply_random_produces_result() {
        let graph = make_sequential();
        let mut rng = rand::rng();
        let result = apply_random_mutation(graph, &mut rng);
        // Either Success or Invalid — should not panic.
        assert!(result.is_success() || result.is_invalid());
    }
}
