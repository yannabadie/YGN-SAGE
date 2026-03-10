use sage_core::topology::templates::{self, TemplateStore};
use sage_core::topology::verifier::HybridVerifier;
use sage_core::topology::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn verifier() -> HybridVerifier {
    HybridVerifier::new()
}

// ---------------------------------------------------------------------------
// 1. Sequential template valid
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_template_valid() {
    let g = templates::sequential("gemini-2.5-flash");
    let r = verifier().verify(&g);
    assert!(r.valid, "Sequential should pass: errors={:?}", r.errors);
    assert!(r.errors.is_empty());
    // 3 nodes, 2 edges
    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 2);
}

// ---------------------------------------------------------------------------
// 2. Parallel template valid
// ---------------------------------------------------------------------------

#[test]
fn test_parallel_template_valid() {
    let g = templates::parallel("gemini-2.5-flash", 3);
    let r = verifier().verify(&g);
    assert!(r.valid, "Parallel should pass: errors={:?}", r.errors);
    assert!(r.errors.is_empty());
    // 1 source + 3 workers + 1 aggregator = 5
    assert_eq!(g.node_count(), 5);
}

// ---------------------------------------------------------------------------
// 3. AVR template valid (closed back-edge is OK)
// ---------------------------------------------------------------------------

#[test]
fn test_avr_template_valid() {
    let g = templates::avr("actor-model", "reviewer-model");
    let r = verifier().verify(&g);
    assert!(r.valid, "AVR should pass: errors={:?}", r.errors);
    assert!(r.errors.is_empty());
    // Graph is cyclic (back-edge present) but the verifier allows closed gates
    assert!(!g.is_acyclic());
}

// ---------------------------------------------------------------------------
// 4. All 8 templates pass verification
// ---------------------------------------------------------------------------

#[test]
fn test_all_templates_pass_verification() {
    let v = verifier();
    let templates_to_test = [
        ("sequential", templates::sequential("m")),
        ("parallel", templates::parallel("m", 3)),
        ("avr", templates::avr("m", "m")),
        ("selfmoa", templates::self_moa("m", 3)),
        ("hierarchical", templates::hierarchical("m", "m")),
        ("hub", templates::hub("m", "m", 3)),
        ("debate", templates::debate("m", "m")),
        ("brainstorming", templates::brainstorming("m", 3)),
    ];

    for (name, g) in &templates_to_test {
        let r = v.verify(g);
        assert!(
            r.valid,
            "Template '{}' should pass verification: errors={:?}",
            name, r.errors
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Security label violation
// ---------------------------------------------------------------------------

#[test]
fn test_security_label_violation() {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    // High-security node -> low-security node = violation
    let high = TopologyNode::new(
        "sender".into(),
        "model".into(),
        1,
        vec!["text".into()],
        2, // confidential
        1.0,
        60.0,
    );
    let low = TopologyNode::new(
        "receiver".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0, // public
        1.0,
        60.0,
    );

    let hi = g.add_node(high);
    let li = g.add_node(low);
    g.try_add_edge(hi, li, TopologyEdge::control()).unwrap();

    let r = verifier().verify(&g);
    assert!(!r.valid, "Should fail: high->low security flow");
    assert!(
        r.errors.iter().any(|e| e.contains("Security label")),
        "Expected security label error in: {:?}",
        r.errors
    );
}

// ---------------------------------------------------------------------------
// 6. Fan-out exceeded
// ---------------------------------------------------------------------------

#[test]
fn test_fan_out_exceeded() {
    let mut v = HybridVerifier::new();
    v.max_fan_out = 3;

    let mut g = TopologyGraph::try_new("parallel").unwrap();
    let hub = TopologyNode::new(
        "hub".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let hi = g.add_node(hub);

    // Add 5 targets => fan-out = 5 > limit 3
    for i in 0..5 {
        let spoke = TopologyNode::new(
            format!("spoke_{}", i),
            "model".into(),
            1,
            vec!["text".into()],
            0,
            1.0,
            60.0,
        );
        let si = g.add_node(spoke);
        g.try_add_edge(hi, si, TopologyEdge::control()).unwrap();
    }

    let r = v.verify(&g);
    assert!(!r.valid, "Should fail: fan-out exceeded");
    assert!(
        r.errors.iter().any(|e| e.contains("fan-out")),
        "Expected fan-out error in: {:?}",
        r.errors
    );
}

// ---------------------------------------------------------------------------
// 7. No entry node
// ---------------------------------------------------------------------------

#[test]
fn test_no_entry_node() {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    // Create a cycle with only control edges, no entry point
    let a = TopologyNode::new(
        "a".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let b = TopologyNode::new(
        "b".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );

    let ai = g.add_node(a);
    let bi = g.add_node(b);
    g.try_add_edge(ai, bi, TopologyEdge::control()).unwrap();
    g.try_add_edge(bi, ai, TopologyEdge::control()).unwrap();

    let r = verifier().verify(&g);
    assert!(!r.valid, "Should fail: no entry node");
    assert!(
        r.errors.iter().any(|e| e.contains("entry node") || e.contains("cycle")),
        "Expected no-entry or cycle error in: {:?}",
        r.errors
    );
}

// ---------------------------------------------------------------------------
// 8. Budget exceeds limit
// ---------------------------------------------------------------------------

#[test]
fn test_budget_exceeds_limit() {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    let expensive = TopologyNode::new(
        "rich".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        5000.0, // $5000
        60.0,
    );
    let also_expensive = TopologyNode::new(
        "richer".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        5001.0, // $5001, total = $10001 > $10000
        60.0,
    );

    let ei = g.add_node(expensive);
    let aei = g.add_node(also_expensive);
    g.try_add_edge(ei, aei, TopologyEdge::control()).unwrap();

    let r = verifier().verify(&g);
    assert!(!r.valid, "Should fail: budget > $10000");
    assert!(
        r.errors.iter().any(|e| e.contains("budget") && e.contains("10000")),
        "Expected budget error in: {:?}",
        r.errors
    );
}

// ---------------------------------------------------------------------------
// 9. Role coherence warning (reviewer with S1)
// ---------------------------------------------------------------------------

#[test]
fn test_role_coherence_warning() {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    let input = TopologyNode::new(
        "input".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    // Reviewer at S1 should produce a warning
    let reviewer = TopologyNode::new(
        "reviewer".into(),
        "model".into(),
        1, // S1 -- too low for reviewer role
        vec!["review".into()],
        0,
        1.0,
        60.0,
    );

    let ii = g.add_node(input);
    let ri = g.add_node(reviewer);
    g.try_add_edge(ii, ri, TopologyEdge::control()).unwrap();

    let r = verifier().verify(&g);
    assert!(r.valid, "Should still be valid (just a warning)");
    assert!(
        r.warnings.iter().any(|w| w.contains("Role coherence")),
        "Expected role coherence warning in: {:?}",
        r.warnings
    );
}

// ---------------------------------------------------------------------------
// 10. Field mapping empty key
// ---------------------------------------------------------------------------

#[test]
fn test_field_mapping_empty_key() {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    let a = TopologyNode::new(
        "a".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let b = TopologyNode::new(
        "b".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );

    let ai = g.add_node(a);
    let bi = g.add_node(b);

    // Also add a control edge so b is reachable and a is entry
    g.try_add_edge(ai, bi, TopologyEdge::control()).unwrap();

    // Message edge with empty key in field mapping
    let mut mapping = HashMap::new();
    mapping.insert("".to_string(), "target_field".to_string());
    g.try_add_edge(ai, bi, TopologyEdge::message(Some(mapping)))
        .unwrap();

    let r = verifier().verify(&g);
    assert!(r.valid, "Empty key is a warning, not an error");
    assert!(
        r.warnings.iter().any(|w| w.contains("empty key")),
        "Expected empty key warning in: {:?}",
        r.warnings
    );
}

// ---------------------------------------------------------------------------
// 11. Loop termination: closed-gate back-edge to node with timeout=0
// ---------------------------------------------------------------------------

#[test]
fn test_loop_termination_no_timeout() {
    let mut g = TopologyGraph::try_new("avr").unwrap();

    // Actor with max_wall_time=0 (no timeout)
    let actor = TopologyNode::new(
        "actor".into(),
        "model".into(),
        2,
        vec!["code".into()],
        0,
        1.0,
        0.0, // No timeout!
    );
    let verifier_node = TopologyNode::new(
        "verifier".into(),
        "model".into(),
        2,
        vec!["review".into()],
        0,
        1.0,
        60.0,
    );
    let output = TopologyNode::new(
        "output".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );

    let ai = g.add_node(actor);
    let vi = g.add_node(verifier_node);
    let oi = g.add_node(output);

    g.try_add_edge(ai, vi, TopologyEdge::control()).unwrap();
    g.try_add_edge(vi, oi, TopologyEdge::control()).unwrap();
    // Closed-gate back-edge targets actor (which has 0 timeout)
    g.try_add_edge(
        vi,
        ai,
        TopologyEdge::control().with_gate(Gate::Closed),
    )
    .unwrap();

    let r = verifier().verify(&g);
    assert!(r.valid, "Loop termination is a warning, not an error");
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("Loop termination")),
        "Expected loop termination warning in: {:?}",
        r.warnings
    );
}

// ---------------------------------------------------------------------------
// 12. TemplateStore::available() returns 8 items
// ---------------------------------------------------------------------------

#[test]
fn test_template_store_all_names() {
    let names = TemplateStore::available();
    assert_eq!(names.len(), 8);
    assert!(names.contains(&"sequential"));
    assert!(names.contains(&"parallel"));
    assert!(names.contains(&"avr"));
    assert!(names.contains(&"selfmoa"));
    assert!(names.contains(&"hierarchical"));
    assert!(names.contains(&"hub"));
    assert!(names.contains(&"debate"));
    assert!(names.contains(&"brainstorming"));
}

// ---------------------------------------------------------------------------
// 13. TemplateStore::create("unknown") returns error
// ---------------------------------------------------------------------------

#[test]
fn test_template_store_unknown() {
    let result = TemplateStore::create("unknown", "model");
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.contains("Unknown template"),
        "Expected 'Unknown template' in: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// Extra: Fan-in exceeded
// ---------------------------------------------------------------------------

#[test]
fn test_fan_in_exceeded() {
    let mut v = HybridVerifier::new();
    v.max_fan_in = 2;

    let mut g = TopologyGraph::try_new("parallel").unwrap();

    let target = TopologyNode::new(
        "sink".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let ti = g.add_node(target);

    // 4 source nodes all pointing to the same target => fan-in = 4 > limit 2
    for i in 0..4 {
        let src = TopologyNode::new(
            format!("src_{}", i),
            "model".into(),
            1,
            vec!["text".into()],
            0,
            1.0,
            60.0,
        );
        let si = g.add_node(src);
        g.try_add_edge(si, ti, TopologyEdge::control()).unwrap();
    }

    let r = v.verify(&g);
    assert!(!r.valid, "Should fail: fan-in exceeded");
    assert!(
        r.errors.iter().any(|e| e.contains("fan-in")),
        "Expected fan-in error in: {:?}",
        r.errors
    );
}

// ---------------------------------------------------------------------------
// Extra: TemplateStore aliases (self_moa, self-moa)
// ---------------------------------------------------------------------------

#[test]
fn test_template_store_aliases() {
    for alias in &["selfmoa", "self_moa", "self-moa"] {
        let result = TemplateStore::create(alias, "model");
        assert!(
            result.is_ok(),
            "Alias '{}' should work: {:?}",
            alias,
            result.err()
        );
    }
}

// ---------------------------------------------------------------------------
// Extra: Switch completeness warning
// ---------------------------------------------------------------------------

#[test]
fn test_switch_completeness_warning() {
    let mut g = TopologyGraph::try_new("hub").unwrap();

    let hub_node = TopologyNode::new(
        "hub".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let spoke_a = TopologyNode::new(
        "spoke_a".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let spoke_b = TopologyNode::new(
        "spoke_b".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );

    let hi = g.add_node(hub_node);
    let sai = g.add_node(spoke_a);
    let sbi = g.add_node(spoke_b);

    // One edge with condition, one without => missing default branch
    g.try_add_edge(
        hi,
        sai,
        TopologyEdge::control().with_condition("x > 0".into()),
    )
    .unwrap();
    g.try_add_edge(hi, sbi, TopologyEdge::control()).unwrap();

    let r = verifier().verify(&g);
    assert!(
        r.warnings.iter().any(|w| w.contains("Switch completeness")),
        "Expected switch completeness warning in: {:?}",
        r.warnings
    );
}

// ---------------------------------------------------------------------------
// Extra: Verification result display
// ---------------------------------------------------------------------------

#[test]
fn test_verification_result_display() {
    let g = templates::sequential("m");
    let r = verifier().verify(&g);
    let display = format!("{}", r);
    assert!(display.contains("VALID"));
}

// ---------------------------------------------------------------------------
// Extra: Control-flow open-gate cycle is rejected
// ---------------------------------------------------------------------------

#[test]
fn test_open_gate_cycle_rejected() {
    let mut g = TopologyGraph::try_new("avr").unwrap();

    let a = TopologyNode::new(
        "a".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let b = TopologyNode::new(
        "b".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );

    let ai = g.add_node(a);
    let bi = g.add_node(b);

    // Open-gate cycle in control flow
    g.try_add_edge(ai, bi, TopologyEdge::control()).unwrap();
    g.try_add_edge(bi, ai, TopologyEdge::control()).unwrap();

    let r = verifier().verify(&g);
    assert!(!r.valid, "Open-gate cycle should be rejected");
    assert!(
        r.errors
            .iter()
            .any(|e| e.contains("cycle")),
        "Expected cycle error in: {:?}",
        r.errors
    );
}

// ---------------------------------------------------------------------------
// Extra: Field mapping with empty value
// ---------------------------------------------------------------------------

#[test]
fn test_field_mapping_empty_value() {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    let a = TopologyNode::new(
        "a".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let b = TopologyNode::new(
        "b".into(),
        "model".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );

    let ai = g.add_node(a);
    let bi = g.add_node(b);

    g.try_add_edge(ai, bi, TopologyEdge::control()).unwrap();

    let mut mapping = HashMap::new();
    mapping.insert("valid_key".to_string(), "".to_string());
    g.try_add_edge(ai, bi, TopologyEdge::message(Some(mapping)))
        .unwrap();

    let r = verifier().verify(&g);
    assert!(
        r.warnings.iter().any(|w| w.contains("empty value")),
        "Expected empty value warning in: {:?}",
        r.warnings
    );
}
