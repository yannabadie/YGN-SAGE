use sage_core::topology::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Sequential topology
// ---------------------------------------------------------------------------

#[test]
fn test_create_sequential_topology() {
    let mut topo = TopologyGraph::try_new("sequential").unwrap();
    assert_eq!(topo.template_type, "sequential");
    assert_eq!(topo.template(), TopologyTemplate::Sequential);

    let coder = TopologyNode::with_id("n1".into(), "coder".into(), "gemini-2.5-flash".into());
    let reviewer = TopologyNode::with_id("n2".into(), "reviewer".into(), "gemini-2.5-flash".into());

    let i0 = topo.add_node(coder);
    let i1 = topo.add_node(reviewer);
    topo.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();

    assert_eq!(topo.node_count(), 2);
    assert_eq!(topo.edge_count(), 1);
    assert!(topo.is_acyclic());
    assert!(!topo.has_cycles());
}

// ---------------------------------------------------------------------------
// 2. Parallel topology
// ---------------------------------------------------------------------------

#[test]
fn test_create_parallel_topology() {
    let mut topo = TopologyGraph::try_new("parallel").unwrap();

    // source -> 3 workers -> aggregator
    let source = TopologyNode::with_id("src".into(), "source".into(), "model".into());
    let w1 = TopologyNode::with_id("w1".into(), "worker".into(), "model".into());
    let w2 = TopologyNode::with_id("w2".into(), "worker".into(), "model".into());
    let w3 = TopologyNode::with_id("w3".into(), "worker".into(), "model".into());
    let agg = TopologyNode::with_id("agg".into(), "aggregator".into(), "model".into());

    let si = topo.add_node(source);
    let i1 = topo.add_node(w1);
    let i2 = topo.add_node(w2);
    let i3 = topo.add_node(w3);
    let ai = topo.add_node(agg);

    topo.try_add_edge(si, i1, TopologyEdge::control()).unwrap();
    topo.try_add_edge(si, i2, TopologyEdge::control()).unwrap();
    topo.try_add_edge(si, i3, TopologyEdge::control()).unwrap();
    topo.try_add_edge(i1, ai, TopologyEdge::message(None))
        .unwrap();
    topo.try_add_edge(i2, ai, TopologyEdge::message(None))
        .unwrap();
    topo.try_add_edge(i3, ai, TopologyEdge::message(None))
        .unwrap();

    assert_eq!(topo.node_count(), 5);
    assert_eq!(topo.edge_count(), 6);
    assert!(topo.is_acyclic());
}

// ---------------------------------------------------------------------------
// 3. AVR topology has loop
// ---------------------------------------------------------------------------

#[test]
fn test_avr_topology_has_loop() {
    let mut topo = TopologyGraph::try_new("avr").unwrap();

    let act = TopologyNode::with_id("act".into(), "actor".into(), "model".into());
    let verify = TopologyNode::with_id("ver".into(), "verifier".into(), "model".into());
    let refine = TopologyNode::with_id("ref".into(), "refiner".into(), "model".into());

    let ai = topo.add_node(act);
    let vi = topo.add_node(verify);
    let ri = topo.add_node(refine);

    // Forward: act -> verify -> refine
    topo.try_add_edge(ai, vi, TopologyEdge::control()).unwrap();
    topo.try_add_edge(vi, ri, TopologyEdge::control()).unwrap();
    // Back-edge: refine -> act (the AVR loop)
    topo.try_add_edge(ri, ai, TopologyEdge::control()).unwrap();

    // Full graph is cyclic
    assert!(!topo.is_acyclic());
    assert!(topo.has_cycles());

    // But we can still analyze control edges
    let control_edges = topo.edges_of_type(EdgeType::Control);
    assert_eq!(control_edges.len(), 3);
}

// ---------------------------------------------------------------------------
// 4. Three-flow edges
// ---------------------------------------------------------------------------

#[test]
fn test_three_flow_edges() {
    let mut topo = TopologyGraph::try_new("hub").unwrap();

    let n0 = TopologyNode::with_id("a".into(), "hub".into(), "model".into());
    let n1 = TopologyNode::with_id("b".into(), "spoke".into(), "model".into());

    let i0 = topo.add_node(n0);
    let i1 = topo.add_node(n1);

    topo.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();
    topo.try_add_edge(i0, i1, TopologyEdge::message(None))
        .unwrap();
    topo.try_add_edge(i0, i1, TopologyEdge::state()).unwrap();

    assert_eq!(topo.edge_count(), 3);

    let ctrl = topo.edges_of_type(EdgeType::Control);
    let msg = topo.edges_of_type(EdgeType::Message);
    let st = topo.edges_of_type(EdgeType::State);

    assert_eq!(ctrl.len(), 1);
    assert_eq!(msg.len(), 1);
    assert_eq!(st.len(), 1);

    assert_eq!(ctrl[0].2.typed_edge_type(), EdgeType::Control);
    assert_eq!(msg[0].2.typed_edge_type(), EdgeType::Message);
    assert_eq!(st[0].2.typed_edge_type(), EdgeType::State);
}

// ---------------------------------------------------------------------------
// 5. Field mapping on message edge
// ---------------------------------------------------------------------------

#[test]
fn test_field_mapping_on_message_edge() {
    let mut mapping = HashMap::new();
    mapping.insert("code".to_string(), "input_code".to_string());
    mapping.insert("context".to_string(), "review_context".to_string());

    let edge = TopologyEdge::message(Some(mapping));
    assert_eq!(edge.typed_edge_type(), EdgeType::Message);

    let fm = edge.field_mapping.as_ref().unwrap();
    assert_eq!(fm.get("code"), Some(&"input_code".to_string()));
    assert_eq!(fm.get("context"), Some(&"review_context".to_string()));
    assert_eq!(fm.len(), 2);

    // Round-trip through a graph
    let mut topo = TopologyGraph::try_new("sequential").unwrap();
    let n0 = TopologyNode::with_id("a".into(), "coder".into(), "model".into());
    let n1 = TopologyNode::with_id("b".into(), "reviewer".into(), "model".into());
    let i0 = topo.add_node(n0);
    let i1 = topo.add_node(n1);
    topo.try_add_edge(i0, i1, edge).unwrap();

    let msg_edges = topo.edges_of_type(EdgeType::Message);
    assert_eq!(msg_edges.len(), 1);
    let retrieved_fm = msg_edges[0].2.field_mapping.as_ref().unwrap();
    assert_eq!(retrieved_fm.get("code"), Some(&"input_code".to_string()));
}

// ---------------------------------------------------------------------------
// 6. Gate open/closed
// ---------------------------------------------------------------------------

#[test]
fn test_gate_open_closed() {
    let open_edge = TopologyEdge::control();
    assert_eq!(open_edge.gate, "open");
    assert_eq!(open_edge.typed_gate(), Gate::Open);

    let closed_edge = TopologyEdge::control().with_gate(Gate::Closed);
    assert_eq!(closed_edge.gate, "closed");
    assert_eq!(closed_edge.typed_gate(), Gate::Closed);

    let cond_edge = TopologyEdge::control().with_condition("score > 0.8".to_string());
    assert_eq!(cond_edge.condition, Some("score > 0.8".to_string()));
}

// ---------------------------------------------------------------------------
// 7. Topological sort — sequential
// ---------------------------------------------------------------------------

#[test]
fn test_topological_sort_sequential() {
    let mut topo = TopologyGraph::try_new("sequential").unwrap();

    let a = TopologyNode::with_id("A".into(), "first".into(), "model".into());
    let b = TopologyNode::with_id("B".into(), "second".into(), "model".into());
    let c = TopologyNode::with_id("C".into(), "third".into(), "model".into());

    let ia = topo.add_node(a);
    let ib = topo.add_node(b);
    let ic = topo.add_node(c);

    topo.try_add_edge(ia, ib, TopologyEdge::control()).unwrap();
    topo.try_add_edge(ib, ic, TopologyEdge::control()).unwrap();

    let order = topo.try_topological_sort().unwrap();
    assert_eq!(order, vec![0, 1, 2]);
}

// ---------------------------------------------------------------------------
// 8. Topological sort fails on cycle
// ---------------------------------------------------------------------------

#[test]
fn test_topological_sort_fails_on_cycle() {
    let mut topo = TopologyGraph::try_new("avr").unwrap();

    let a = TopologyNode::with_id("A".into(), "act".into(), "model".into());
    let b = TopologyNode::with_id("B".into(), "verify".into(), "model".into());

    let ia = topo.add_node(a);
    let ib = topo.add_node(b);

    topo.try_add_edge(ia, ib, TopologyEdge::control()).unwrap();
    topo.try_add_edge(ib, ia, TopologyEdge::control()).unwrap();

    let result = topo.try_topological_sort();
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("cycle"),
        "Expected 'cycle' in: {}",
        err_msg
    );
}

// ---------------------------------------------------------------------------
// 9. Entry and exit nodes
// ---------------------------------------------------------------------------

#[test]
fn test_entry_exit_nodes() {
    let mut topo = TopologyGraph::try_new("sequential").unwrap();

    let a = TopologyNode::with_id("A".into(), "entry".into(), "model".into());
    let b = TopologyNode::with_id("B".into(), "middle".into(), "model".into());
    let c = TopologyNode::with_id("C".into(), "exit".into(), "model".into());

    let ia = topo.add_node(a);
    let ib = topo.add_node(b);
    let ic = topo.add_node(c);

    topo.try_add_edge(ia, ib, TopologyEdge::control()).unwrap();
    topo.try_add_edge(ib, ic, TopologyEdge::control()).unwrap();

    let entries = topo.entry_nodes();
    assert_eq!(entries, vec![0]); // A has no incoming

    let exits = topo.exit_nodes();
    assert_eq!(exits, vec![2]); // C has no outgoing

    // Parallel fork: source has no incoming, two workers have no outgoing
    let mut topo2 = TopologyGraph::try_new("parallel").unwrap();
    let src = TopologyNode::with_id("src".into(), "source".into(), "model".into());
    let w1 = TopologyNode::with_id("w1".into(), "worker".into(), "model".into());
    let w2 = TopologyNode::with_id("w2".into(), "worker".into(), "model".into());

    let si = topo2.add_node(src);
    let wi1 = topo2.add_node(w1);
    let wi2 = topo2.add_node(w2);

    topo2
        .try_add_edge(si, wi1, TopologyEdge::control())
        .unwrap();
    topo2
        .try_add_edge(si, wi2, TopologyEdge::control())
        .unwrap();

    assert_eq!(topo2.entry_nodes(), vec![0]);
    assert_eq!(topo2.exit_nodes(), vec![1, 2]);
}

// ---------------------------------------------------------------------------
// 10. Topology template parsing
// ---------------------------------------------------------------------------

#[test]
fn test_topology_template_parsing() {
    // All 8 templates
    let valid = [
        ("sequential", TopologyTemplate::Sequential),
        ("parallel", TopologyTemplate::Parallel),
        ("avr", TopologyTemplate::AVR),
        ("selfmoa", TopologyTemplate::SelfMoA),
        ("self_moa", TopologyTemplate::SelfMoA),
        ("self-moa", TopologyTemplate::SelfMoA),
        ("hierarchical", TopologyTemplate::Hierarchical),
        ("hub", TopologyTemplate::Hub),
        ("debate", TopologyTemplate::Debate),
        ("brainstorming", TopologyTemplate::Brainstorming),
    ];

    for (name, expected) in valid {
        let result = TopologyGraph::parse_template(name);
        assert_eq!(result, Some(expected), "Failed to parse '{}'", name);
    }

    // Case insensitive
    assert_eq!(
        TopologyGraph::parse_template("Sequential"),
        Some(TopologyTemplate::Sequential)
    );
    assert_eq!(
        TopologyGraph::parse_template("PARALLEL"),
        Some(TopologyTemplate::Parallel)
    );

    // Unknown template
    assert_eq!(TopologyGraph::parse_template("unknown"), None);
    assert_eq!(TopologyGraph::parse_template(""), None);

    // Constructor rejects unknown
    let result = TopologyGraph::try_new("nonexistent");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 11. Node repr (via Display)
// ---------------------------------------------------------------------------

#[test]
fn test_node_repr() {
    let node = TopologyNode::with_id("n1".into(), "coder".into(), "gemini-2.5-flash".into());
    let repr = node.to_string();
    assert!(repr.contains("coder"));
    assert!(repr.contains("gemini-2.5-flash"));
    assert!(repr.contains("S1"));
}

// ---------------------------------------------------------------------------
// 12. Empty topology
// ---------------------------------------------------------------------------

#[test]
fn test_empty_topology() {
    let topo = TopologyGraph::try_new("debate").unwrap();
    assert_eq!(topo.node_count(), 0);
    assert_eq!(topo.edge_count(), 0);
    assert!(topo.is_acyclic());
    assert!(!topo.has_cycles());
    assert!(topo.node_ids().is_empty());
    assert!(topo.entry_nodes().is_empty());
    assert!(topo.exit_nodes().is_empty());

    // Topological sort of empty graph succeeds with empty vec
    let order = topo.try_topological_sort().unwrap();
    assert!(order.is_empty());

    // ID is a ULID (26 chars)
    assert_eq!(topo.id.len(), 26);
}

// ---------------------------------------------------------------------------
// Extra: edge index bounds checking
// ---------------------------------------------------------------------------

#[test]
fn test_add_edge_out_of_bounds() {
    let mut topo = TopologyGraph::try_new("sequential").unwrap();
    let n = TopologyNode::with_id("n".into(), "role".into(), "model".into());
    topo.add_node(n);

    // from=0 is valid, to=5 is out of bounds
    let result = topo.try_add_edge(0, 5, TopologyEdge::control());
    assert!(result.is_err());

    // from=5 is out of bounds
    let result = topo.try_add_edge(5, 0, TopologyEdge::control());
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Extra: get_node out of bounds
// ---------------------------------------------------------------------------

#[test]
fn test_get_node_out_of_bounds() {
    let topo = TopologyGraph::try_new("hub").unwrap();
    let result = topo.try_get_node(0);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Extra: TopologyGraph repr (via Display)
// ---------------------------------------------------------------------------

#[test]
fn test_graph_repr() {
    let mut topo = TopologyGraph::try_new("brainstorming").unwrap();
    let n = TopologyNode::with_id("n1".into(), "thinker".into(), "model".into());
    topo.add_node(n);

    let repr = topo.to_string();
    assert!(repr.contains("brainstorming"));
    assert!(repr.contains("nodes=1"));
    assert!(repr.contains("edges=0"));
}

// ---------------------------------------------------------------------------
// Extra: edge weight
// ---------------------------------------------------------------------------

#[test]
fn test_edge_weight() {
    let edge = TopologyEdge::control();
    assert!((edge.weight - 1.0).abs() < f32::EPSILON);
}
