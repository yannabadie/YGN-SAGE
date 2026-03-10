use sage_core::topology::mutations::*;
use sage_core::topology::templates;
use sage_core::topology::topology_graph::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_sequential() -> TopologyGraph {
    templates::sequential("gemini-2.5-flash")
}

fn make_two_node_graph() -> TopologyGraph {
    let mut g = TopologyGraph::try_new("sequential").unwrap();
    let n0 = TopologyNode::new(
        "coder".into(),
        "gemini-2.5-flash".into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let n1 = TopologyNode::new(
        "reviewer".into(),
        "gemini-2.5-flash".into(),
        2,
        vec!["reasoning".into()],
        0,
        1.0,
        60.0,
    );
    let i0 = g.add_node(n0);
    let i1 = g.add_node(n1);
    g.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();
    g
}

fn make_single_node_graph() -> TopologyGraph {
    let mut g = TopologyGraph::try_new("sequential").unwrap();
    let n0 = TopologyNode::new(
        "solo".into(),
        "gemini-2.5-flash".into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    g.add_node(n0);
    g
}

// ---------------------------------------------------------------------------
// Test 1: add_node increases node count
// ---------------------------------------------------------------------------

#[test]
fn test_add_node_increases_count() {
    let graph = make_sequential(); // 3 nodes
    assert_eq!(graph.node_count(), 3);

    let result = add_node(graph, "analyst", "gemini-2.5-flash", 1);
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    assert_eq!(new_graph.node_count(), 4);
}

// ---------------------------------------------------------------------------
// Test 2: add_node on single-node graph still produces valid topology
// ---------------------------------------------------------------------------

#[test]
fn test_add_node_on_single_node_graph() {
    let graph = make_single_node_graph();
    assert_eq!(graph.node_count(), 1);

    let result = add_node(graph, "helper", "gemini-2.5-flash", 1);
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    assert_eq!(new_graph.node_count(), 2);
    // The new node should be connected to the exit (originally the single node).
    assert!(new_graph.edge_count() >= 1);
}

// ---------------------------------------------------------------------------
// Test 3: remove_node decreases count (on 3+ node graph)
// ---------------------------------------------------------------------------

#[test]
fn test_remove_node_decreases_count() {
    let graph = make_sequential(); // 3 nodes: 0->1->2
    assert_eq!(graph.node_count(), 3);

    let result = remove_node(graph, 1); // remove middle node
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    assert_eq!(new_graph.node_count(), 2);
    // Rewiring: predecessor of 1 (node 0) should connect to successor of 1 (node 2).
    assert!(new_graph.edge_count() >= 1);
}

// ---------------------------------------------------------------------------
// Test 4: remove_node on 2-node graph returns Invalid
// ---------------------------------------------------------------------------

#[test]
fn test_remove_node_on_two_node_graph_invalid() {
    let graph = make_two_node_graph(); // 2 nodes
    assert_eq!(graph.node_count(), 2);

    let result = remove_node(graph, 0);
    assert!(result.is_invalid(), "Expected Invalid for 2-node graph removal");
}

// ---------------------------------------------------------------------------
// Test 5: swap_model changes model_id
// ---------------------------------------------------------------------------

#[test]
fn test_swap_model_changes_model_id() {
    let graph = make_sequential();
    let original = graph.try_get_node(0).unwrap();
    assert_eq!(original.model_id, "gemini-2.5-flash");

    let result = swap_model(graph, 0, "gpt-5.3-codex");
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    let node = new_graph.try_get_node(0).unwrap();
    assert_eq!(node.model_id, "gpt-5.3-codex");
}

// ---------------------------------------------------------------------------
// Test 6: rewire_edge adds new edge
// ---------------------------------------------------------------------------

#[test]
fn test_rewire_edge_adds_edge() {
    let graph = make_sequential(); // 0->1->2
    let original_edges = graph.edge_count();

    // Add edge from 0 directly to 2 (skipping 1).
    let result = rewire_edge(graph, 0, 2);
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    assert_eq!(new_graph.edge_count(), original_edges + 1);
}

// ---------------------------------------------------------------------------
// Test 7: rewire_edge duplicate returns Invalid
// ---------------------------------------------------------------------------

#[test]
fn test_rewire_edge_duplicate_returns_invalid() {
    let graph = make_sequential(); // 0->1->2, has control edge 0->1

    // Try to add duplicate control edge 0->1.
    let result = rewire_edge(graph, 0, 1);
    assert!(
        result.is_invalid(),
        "Expected Invalid for duplicate edge, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 8: split_node increases count by 1
// ---------------------------------------------------------------------------

#[test]
fn test_split_node_increases_count_by_one() {
    let graph = make_sequential(); // 3 nodes
    assert_eq!(graph.node_count(), 3);

    let result = split_node(graph, 1, "analyzer", "gemini-2.5-flash", "synthesizer", "gemini-2.5-flash");
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    // Original 3 - 1 removed + 2 new = 4.
    assert_eq!(new_graph.node_count(), 4);
}

// ---------------------------------------------------------------------------
// Test 9: merge_nodes decreases count by 1
// ---------------------------------------------------------------------------

#[test]
fn test_merge_nodes_decreases_count_by_one() {
    let graph = make_sequential(); // 3 nodes: 0->1->2
    assert_eq!(graph.node_count(), 3);

    // Merge adjacent nodes 0 and 1.
    let result = merge_nodes(graph, 0, 1, "merged_worker", "gemini-2.5-flash");
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    // Original 3 - 2 merged + 1 new = 2.
    assert_eq!(new_graph.node_count(), 2);
}

// ---------------------------------------------------------------------------
// Test 10: mutate_prompt changes role
// ---------------------------------------------------------------------------

#[test]
fn test_mutate_prompt_changes_role() {
    let graph = make_sequential();
    let original = graph.try_get_node(0).unwrap();
    assert_eq!(original.role, "input_processor");

    let result = mutate_prompt(graph, 0, "super_coder");
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    let node = new_graph.try_get_node(0).unwrap();
    assert_eq!(node.role, "super_coder");
}

// ---------------------------------------------------------------------------
// Test 11: apply_random_mutation produces a result
// ---------------------------------------------------------------------------

#[test]
fn test_apply_random_mutation_produces_result() {
    let mut rng = rand::rng();

    // Run 20 random mutations; all should complete without panicking.
    for _ in 0..20 {
        let g = make_sequential();
        let result = apply_random_mutation(g, &mut rng);
        assert!(
            result.is_success() || result.is_invalid(),
            "Unexpected variant"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 12: Invalid mutation rejected by verifier (security label violation)
// ---------------------------------------------------------------------------

#[test]
fn test_security_violation_after_swap_rejected() {
    // Build a graph with a high-security node flowing to a low-security node.
    // The verifier should reject this.
    let mut graph = TopologyGraph::try_new("sequential").unwrap();
    let high = TopologyNode::new(
        "high_sec".into(),
        "gemini-2.5-flash".into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let low = TopologyNode::new(
        "low_sec".into(),
        "gemini-2.5-flash".into(),
        1,
        vec!["text_processing".into()],
        0,
        1.0,
        60.0,
    );
    let hi = graph.add_node(high);
    let li = graph.add_node(low);
    graph.try_add_edge(hi, li, TopologyEdge::control()).unwrap();

    // Now change the first node's security label to 2 (confidential)
    // while the second stays at 0 (public) -- this violates label lattice.
    graph
        .inner_graph_mut()
        .node_weight_mut(petgraph::graph::NodeIndex::new(0))
        .unwrap()
        .security_label = 2;

    // swapping model should still keep the security violation
    let result = swap_model(graph, 0, "gpt-5.3-codex");
    assert!(
        result.is_invalid(),
        "Expected Invalid due to security label violation, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 13: merge_nodes on non-adjacent returns Invalid
// ---------------------------------------------------------------------------

#[test]
fn test_merge_non_adjacent_returns_invalid() {
    let graph = make_sequential(); // 0->1->2

    // Nodes 0 and 2 are not directly adjacent.
    let result = merge_nodes(graph, 0, 2, "merged", "gemini-2.5-flash");
    assert!(
        result.is_invalid(),
        "Expected Invalid for non-adjacent merge, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 14: remove_node out of range returns Invalid
// ---------------------------------------------------------------------------

#[test]
fn test_remove_node_out_of_range() {
    let graph = make_sequential(); // 3 nodes
    let result = remove_node(graph, 99);
    assert!(
        result.is_invalid(),
        "Expected Invalid for out-of-range index, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 15: swap_model out of range returns Invalid
// ---------------------------------------------------------------------------

#[test]
fn test_swap_model_out_of_range() {
    let graph = make_sequential();
    let result = swap_model(graph, 99, "gpt-5.3-codex");
    assert!(
        result.is_invalid(),
        "Expected Invalid for out-of-range index, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 16: split_node preserves connectivity
// ---------------------------------------------------------------------------

#[test]
fn test_split_node_preserves_connectivity() {
    let graph = make_sequential(); // 0->1->2

    // Split node 1 (the middle "worker") into two nodes.
    let result = split_node(graph, 1, "pre_worker", "gemini-2.5-flash", "post_worker", "gemini-2.5-flash");
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();

    // Should be acyclic (sequential was acyclic and split maintains DAG structure).
    assert!(new_graph.is_acyclic(), "Split graph should remain acyclic");

    // Should have entry and exit nodes.
    assert!(!new_graph.entry_nodes().is_empty(), "Should have entry nodes");
    assert!(!new_graph.exit_nodes().is_empty(), "Should have exit nodes");
}

// ---------------------------------------------------------------------------
// Test 17: add_node on empty graph
// ---------------------------------------------------------------------------

#[test]
fn test_add_node_on_empty_graph() {
    let graph = TopologyGraph::try_new("sequential").unwrap();
    assert_eq!(graph.node_count(), 0);

    let result = add_node(graph, "first", "gemini-2.5-flash", 1);
    assert!(result.is_success(), "Expected Success, got: {:?}", result);
    let new_graph = result.unwrap();
    assert_eq!(new_graph.node_count(), 1);
}
