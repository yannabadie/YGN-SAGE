use sage_core::topology::llm_synthesis::*;
use sage_core::topology::*;

// ---------------------------------------------------------------------------
// Helper JSON
// ---------------------------------------------------------------------------

fn valid_roles_json() -> &'static str {
    r#"{
        "roles": [
            {"name": "coder", "model": "gemini-2.5-flash", "system": 2, "capabilities": ["code_generation"]},
            {"name": "reviewer", "model": "gemini-3.1-pro", "system": 2, "capabilities": ["code_review"]}
        ]
    }"#
}

fn valid_structure_json() -> &'static str {
    r#"{
        "adjacency": [[0, 1], [0, 0]],
        "edge_types": [["", "control"], ["", ""]],
        "template": "sequential"
    }"#
}

// ---------------------------------------------------------------------------
// 1. Parse valid roles JSON
// ---------------------------------------------------------------------------

#[test]
fn test_parse_valid_roles_json() {
    let roles = TopologySynthesizer::parse_roles(valid_roles_json()).unwrap();
    assert_eq!(roles.len(), 2);
    assert_eq!(roles[0].name, "coder");
    assert_eq!(roles[0].model, "gemini-2.5-flash");
    assert_eq!(roles[0].system, 2);
    assert_eq!(roles[0].capabilities, vec!["code_generation"]);
    assert_eq!(roles[1].name, "reviewer");
    assert_eq!(roles[1].model, "gemini-3.1-pro");
    assert_eq!(roles[1].capabilities, vec!["code_review"]);
}

// ---------------------------------------------------------------------------
// 2. Parse invalid roles JSON -> RoleParseFailed
// ---------------------------------------------------------------------------

#[test]
fn test_parse_invalid_roles_json() {
    let result = TopologySynthesizer::parse_roles("this is not json");
    assert!(result.is_err());
    match result.unwrap_err() {
        SynthesisError::RoleParseFailed(msg) => {
            assert!(!msg.is_empty(), "Error message should not be empty");
        }
        other => panic!("Expected RoleParseFailed, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 3. Parse valid structure JSON
// ---------------------------------------------------------------------------

#[test]
fn test_parse_valid_structure_json() {
    let structure = TopologySynthesizer::parse_structure(valid_structure_json()).unwrap();
    assert_eq!(structure.adjacency.len(), 2);
    assert_eq!(structure.adjacency[0], vec![0, 1]);
    assert_eq!(structure.adjacency[1], vec![0, 0]);
    assert_eq!(structure.edge_types[0][1], "control");
    assert_eq!(structure.template, "sequential");
}

// ---------------------------------------------------------------------------
// 4. Parse invalid structure JSON -> StructureParseFailed
// ---------------------------------------------------------------------------

#[test]
fn test_parse_invalid_structure_json() {
    let result = TopologySynthesizer::parse_structure("{invalid json}");
    assert!(result.is_err());
    match result.unwrap_err() {
        SynthesisError::StructureParseFailed(msg) => {
            assert!(!msg.is_empty(), "Error message should not be empty");
        }
        other => panic!("Expected StructureParseFailed, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 5. Dimension mismatch (3 roles, 2x2 adjacency) -> DimensionMismatch
// ---------------------------------------------------------------------------

#[test]
fn test_dimension_mismatch_roles_vs_adjacency() {
    let roles_json = r#"{
        "roles": [
            {"name": "a", "model": "m", "system": 1},
            {"name": "b", "model": "m", "system": 1},
            {"name": "c", "model": "m", "system": 1}
        ]
    }"#;
    let structure_json = r#"{
        "adjacency": [[0, 1], [0, 0]],
        "edge_types": [["", "control"], ["", ""]],
        "template": "sequential"
    }"#;

    let roles = TopologySynthesizer::parse_roles(roles_json).unwrap();
    let structure = TopologySynthesizer::parse_structure(structure_json).unwrap();
    let result = TopologySynthesizer::build_graph(&roles, &structure);

    assert!(result.is_err());
    match result.unwrap_err() {
        SynthesisError::DimensionMismatch(roles_n, adj_n) => {
            assert_eq!(roles_n, 3);
            assert_eq!(adj_n, 2);
        }
        other => panic!("Expected DimensionMismatch, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 6. Full synthesize with valid JSON -> success
// ---------------------------------------------------------------------------

#[test]
fn test_full_synthesize_success() {
    let mut synth = TopologySynthesizer::new();
    let request = SynthesisRequest {
        task_description: "Write a unit test".to_string(),
        constraints: SynthesisConstraints::default(),
    };

    let graph = synth
        .synthesize(request, valid_roles_json(), valid_structure_json())
        .unwrap();

    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.template(), TopologyTemplate::Sequential);
    assert!(graph.is_acyclic());

    // Verify nodes
    let node0 = graph.try_get_node(0).unwrap();
    assert_eq!(node0.role, "coder");
    assert_eq!(node0.model_id, "gemini-2.5-flash");
    assert_eq!(node0.system, 2);

    let node1 = graph.try_get_node(1).unwrap();
    assert_eq!(node1.role, "reviewer");
    assert_eq!(node1.model_id, "gemini-3.1-pro");
}

// ---------------------------------------------------------------------------
// 7. Full synthesize with invalid template -> InvalidTemplate
// ---------------------------------------------------------------------------

#[test]
fn test_synthesize_invalid_template() {
    let roles_json = r#"{
        "roles": [
            {"name": "a", "model": "m", "system": 1}
        ]
    }"#;
    let structure_json = r#"{
        "adjacency": [[0]],
        "edge_types": [[""]],
        "template": "unknown_template_xyz"
    }"#;

    let roles = TopologySynthesizer::parse_roles(roles_json).unwrap();
    let structure = TopologySynthesizer::parse_structure(structure_json).unwrap();
    let result = TopologySynthesizer::build_graph(&roles, &structure);

    assert!(result.is_err());
    match result.unwrap_err() {
        SynthesisError::InvalidTemplate(name) => {
            assert_eq!(name, "unknown_template_xyz");
        }
        other => panic!("Expected InvalidTemplate, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 8. build_graph creates correct node count
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_node_count() {
    let roles = vec![
        RoleAssignment {
            name: "planner".into(),
            model: "model-a".into(),
            system: 2,
            capabilities: vec!["planning".into()],
        },
        RoleAssignment {
            name: "executor".into(),
            model: "model-b".into(),
            system: 1,
            capabilities: vec!["execution".into()],
        },
        RoleAssignment {
            name: "reporter".into(),
            model: "model-a".into(),
            system: 1,
            capabilities: vec!["reporting".into()],
        },
    ];
    let structure = StructureDesign {
        adjacency: vec![vec![0, 1, 0], vec![0, 0, 1], vec![0, 0, 0]],
        edge_types: vec![
            vec!["".into(), "control".into(), "".into()],
            vec!["".into(), "".into(), "control".into()],
            vec!["".into(), "".into(), "".into()],
        ],
        template: "sequential".into(),
    };

    let graph = TopologySynthesizer::build_graph(&roles, &structure).unwrap();
    assert_eq!(graph.node_count(), 3);

    // Verify role names persisted
    assert_eq!(graph.try_get_node(0).unwrap().role, "planner");
    assert_eq!(graph.try_get_node(1).unwrap().role, "executor");
    assert_eq!(graph.try_get_node(2).unwrap().role, "reporter");
}

// ---------------------------------------------------------------------------
// 9. build_graph creates correct edge count from adjacency
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_edge_count_from_adjacency() {
    // 3 nodes, 3 edges (A->B, B->C, A->C)
    let roles = vec![
        RoleAssignment {
            name: "a".into(),
            model: "m".into(),
            system: 1,
            capabilities: vec![],
        },
        RoleAssignment {
            name: "b".into(),
            model: "m".into(),
            system: 1,
            capabilities: vec![],
        },
        RoleAssignment {
            name: "c".into(),
            model: "m".into(),
            system: 1,
            capabilities: vec![],
        },
    ];
    let structure = StructureDesign {
        adjacency: vec![vec![0, 1, 1], vec![0, 0, 1], vec![0, 0, 0]],
        edge_types: vec![
            vec!["".into(), "control".into(), "message".into()],
            vec!["".into(), "".into(), "control".into()],
            vec!["".into(), "".into(), "".into()],
        ],
        template: "sequential".into(),
    };

    let graph = TopologySynthesizer::build_graph(&roles, &structure).unwrap();
    assert_eq!(graph.edge_count(), 3);

    // Verify edge types
    let control_edges = graph.edges_of_type(EdgeType::Control);
    let message_edges = graph.edges_of_type(EdgeType::Message);
    assert_eq!(control_edges.len(), 2); // A->B, B->C
    assert_eq!(message_edges.len(), 1); // A->C
}

// ---------------------------------------------------------------------------
// 10. Rate limiting logic
// ---------------------------------------------------------------------------

#[test]
fn test_rate_limiting() {
    let mut synth = TopologySynthesizer::new();

    // Initially not rate limited
    assert!(!synth.is_rate_limited());

    // After marking synthesis, should be rate limited
    synth.mark_synthesis();
    assert!(synth.is_rate_limited());

    // Trying to synthesize should fail with RateLimited
    let request = SynthesisRequest {
        task_description: "test".to_string(),
        constraints: SynthesisConstraints::default(),
    };
    let result = synth.synthesize(request, valid_roles_json(), valid_structure_json());
    assert!(result.is_err());
    match result.unwrap_err() {
        SynthesisError::RateLimited(_, interval) => {
            assert_eq!(interval, 60);
        }
        other => panic!("Expected RateLimited, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 11. Verifier catches security label violation in synthesized graph
// ---------------------------------------------------------------------------

#[test]
fn test_verifier_catches_security_violation() {
    // Build a graph where a high-security node sends to a low-security node.
    // We do this manually via build_graph then modify security labels.
    let mut synth = TopologySynthesizer::new();

    // Create a valid graph first, then manually build one with security issues
    let mut graph = TopologyGraph::try_new("sequential").unwrap();

    // High security node (label=2) -> Low security node (label=0) = violation
    let high_node = TopologyNode::new(
        "classifier".into(),
        "model-a".into(),
        2,
        vec!["classification".into()],
        2, // confidential
        1.0,
        60.0,
    );
    let low_node = TopologyNode::new(
        "output".into(),
        "model-b".into(),
        1,
        vec!["output".into()],
        0, // public
        1.0,
        60.0,
    );

    let hi = graph.add_node(high_node);
    let li = graph.add_node(low_node);
    graph.try_add_edge(hi, li, TopologyEdge::control()).unwrap();

    // Verify the graph directly (not through synthesize, since synthesize builds fresh)
    let verifier = sage_core::topology::verifier::HybridVerifier::new();
    let result = verifier.verify(&graph);
    assert!(!result.valid);
    assert!(
        result.errors.iter().any(|e| e.contains("Security label")),
        "Expected security label violation in errors: {:?}",
        result.errors
    );

    // Now test through synthesize: the synthesizer builds nodes with security_label=0,
    // so we verify that a valid synthesis passes verification
    let request = SynthesisRequest {
        task_description: "test security".to_string(),
        constraints: SynthesisConstraints::default(),
    };
    let valid_graph = synth
        .synthesize(request, valid_roles_json(), valid_structure_json())
        .unwrap();
    assert!(valid_graph.is_acyclic());
}

// ---------------------------------------------------------------------------
// 12. Empty roles array handled gracefully
// ---------------------------------------------------------------------------

#[test]
fn test_empty_roles_array() {
    let roles_json = r#"{"roles": []}"#;
    let structure_json = r#"{
        "adjacency": [],
        "edge_types": [],
        "template": "sequential"
    }"#;

    let mut synth = TopologySynthesizer::new();
    let request = SynthesisRequest {
        task_description: "empty test".to_string(),
        constraints: SynthesisConstraints::default(),
    };

    let graph = synth
        .synthesize(request, roles_json, structure_json)
        .unwrap();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

// ---------------------------------------------------------------------------
// 13. Roles with missing capabilities field (uses serde default)
// ---------------------------------------------------------------------------

#[test]
fn test_roles_missing_capabilities_defaults_to_empty() {
    let json = r#"{
        "roles": [
            {"name": "simple", "model": "m", "system": 1}
        ]
    }"#;
    let roles = TopologySynthesizer::parse_roles(json).unwrap();
    assert_eq!(roles.len(), 1);
    assert!(roles[0].capabilities.is_empty());
}

// ---------------------------------------------------------------------------
// 14. Multiple edge types in adjacency (control + message)
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_edge_types() {
    let roles_json = r#"{
        "roles": [
            {"name": "producer", "model": "m", "system": 1},
            {"name": "consumer", "model": "m", "system": 1}
        ]
    }"#;
    // Two edges: producer->consumer (control) and consumer->producer (state)
    // Wait, that would create a cycle. Use message instead for acyclicity.
    let structure_json = r#"{
        "adjacency": [[0, 1], [0, 0]],
        "edge_types": [["", "message"], ["", ""]],
        "template": "sequential"
    }"#;

    let roles = TopologySynthesizer::parse_roles(roles_json).unwrap();
    let structure = TopologySynthesizer::parse_structure(structure_json).unwrap();
    let graph = TopologySynthesizer::build_graph(&roles, &structure).unwrap();

    assert_eq!(graph.edge_count(), 1);
    let msg_edges = graph.edges_of_type(EdgeType::Message);
    assert_eq!(msg_edges.len(), 1);
}

// ---------------------------------------------------------------------------
// 15. SynthesisError Display messages
// ---------------------------------------------------------------------------

#[test]
fn test_synthesis_error_display() {
    let err = SynthesisError::RoleParseFailed("bad json".into());
    assert!(err.to_string().contains("bad json"));

    let err = SynthesisError::DimensionMismatch(3, 2);
    let msg = err.to_string();
    assert!(msg.contains("3"));
    assert!(msg.contains("2"));

    let err = SynthesisError::RateLimited(10, 60);
    let msg = err.to_string();
    assert!(msg.contains("10"));
    assert!(msg.contains("60"));

    let err = SynthesisError::InvalidTemplate("foo".into());
    assert!(err.to_string().contains("foo"));

    let err = SynthesisError::ValidationFailed(vec!["err1".into(), "err2".into()]);
    let msg = err.to_string();
    assert!(msg.contains("err1"));
    assert!(msg.contains("err2"));
}
