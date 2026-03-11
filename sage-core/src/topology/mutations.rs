//! 7 topology mutation operators for evolutionary topology search.
//!
//! Each operator takes a `TopologyGraph` by value, applies a structural mutation,
//! validates via `HybridVerifier`, and returns a `MutationResult`. Invalid mutations
//! return `MutationResult::Invalid` and are NOT retried.

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rand::Rng;
use tracing::{debug, info};

use crate::topology::topology_graph::*;
use crate::topology::verifier::HybridVerifier;

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
    let mut new_graph = TopologyGraph::try_new(&graph.template_type).unwrap();
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
    let mut new_graph = TopologyGraph::try_new(&graph.template_type).unwrap();
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
    let mut new_graph = TopologyGraph::try_new(&graph.template_type).unwrap();
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

/// Fixed set of model IDs for random mutation.
const MODEL_IDS: &[&str] = &["gemini-2.5-flash", "gemini-3.1-pro", "gpt-5.3-codex"];

/// Fixed set of role names for random mutation.
const ROLES: &[&str] = &[
    "coder",
    "reviewer",
    "planner",
    "formatter",
    "analyst",
    "reasoner",
];

/// Pick one of the 7 mutations at random and apply it with random parameters.
pub fn apply_random_mutation<R: Rng>(graph: TopologyGraph, rng: &mut R) -> MutationResult {
    let node_count = graph.node_count();

    // If graph is empty or has only 1 node, limit to add_node.
    if node_count == 0 {
        let model = MODEL_IDS[rng.random_range(0..MODEL_IDS.len())];
        let role = ROLES[rng.random_range(0..ROLES.len())];
        let system: u8 = rng.random_range(1..=3);
        return add_node_at(graph, role, model, system, None);
    }

    // Choose a mutation (0-6).
    let mutation_idx = rng.random_range(0u32..7);

    match mutation_idx {
        0 => {
            // add_node — random exit node for mutation diversity
            let exit_count = graph.exit_nodes().len();
            let exit_hint = if exit_count > 0 {
                Some(rng.random_range(0..exit_count))
            } else {
                None
            };
            let model = MODEL_IDS[rng.random_range(0..MODEL_IDS.len())];
            let role = ROLES[rng.random_range(0..ROLES.len())];
            let system: u8 = rng.random_range(1..=3);
            debug!(
                mutation = "add_node",
                role = role,
                model = model,
                "apply_random_mutation"
            );
            add_node_at(graph, role, model, system, exit_hint)
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
            // swap_model
            let idx = rng.random_range(0..node_count);
            let model = MODEL_IDS[rng.random_range(0..MODEL_IDS.len())];
            debug!(
                mutation = "swap_model",
                node_index = idx,
                model = model,
                "apply_random_mutation"
            );
            swap_model(graph, idx, model)
        }
        3 => {
            // rewire_edge
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
            // split_node
            let idx = rng.random_range(0..node_count);
            let role_a = ROLES[rng.random_range(0..ROLES.len())];
            let model_a = MODEL_IDS[rng.random_range(0..MODEL_IDS.len())];
            let role_b = ROLES[rng.random_range(0..ROLES.len())];
            let model_b = MODEL_IDS[rng.random_range(0..MODEL_IDS.len())];
            debug!(
                mutation = "split_node",
                node_index = idx,
                "apply_random_mutation"
            );
            split_node(graph, idx, role_a, model_a, role_b, model_b)
        }
        5 => {
            // merge_nodes (need at least 2 nodes)
            if node_count < 2 {
                return MutationResult::Invalid("Cannot merge with fewer than 2 nodes".to_string());
            }
            let a = rng.random_range(0..node_count);
            let mut b = rng.random_range(0..node_count);
            // Ensure b != a.
            if b == a {
                b = (a + 1) % node_count;
            }
            let role = ROLES[rng.random_range(0..ROLES.len())];
            let model = MODEL_IDS[rng.random_range(0..MODEL_IDS.len())];
            debug!(
                mutation = "merge_nodes",
                node_a = a,
                node_b = b,
                "apply_random_mutation"
            );
            merge_nodes(graph, a, b, role, model)
        }
        6 => {
            // mutate_prompt
            let idx = rng.random_range(0..node_count);
            let role = ROLES[rng.random_range(0..ROLES.len())];
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
        let result = add_node(graph, "analyst", "gemini-2.5-flash", 1);
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
