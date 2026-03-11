//! LLM-synthesized topology generation: 3-stage pipeline.
//!
//! Stage 1: Parse role assignments from JSON (LLM output)
//! Stage 2: Parse structure design (adjacency + edge types) from JSON
//! Stage 3: Build TopologyGraph, validate via HybridVerifier
//!
//! The Rust core handles parsing and validation; actual LLM calls are
//! delegated to Python via callbacks. For pure Rust tests, use mock JSON.

use super::topology_graph::{TopologyEdge, TopologyGraph, TopologyNode, TopologyTemplate};
use super::verifier::HybridVerifier;
use serde::Deserialize;
use std::time::Instant;
use tracing::{info, info_span, warn};

// ---------------------------------------------------------------------------
// SynthesisError
// ---------------------------------------------------------------------------

/// Errors that can occur during topology synthesis.
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error("Role assignment parse error: {0}")]
    RoleParseFailed(String),
    #[error("Structure design parse error: {0}")]
    StructureParseFailed(String),
    #[error("Dimension mismatch: {0} roles but adjacency is {1}x{1}")]
    DimensionMismatch(usize, usize),
    #[error("Validation failed: {0:?}")]
    ValidationFailed(Vec<String>),
    #[error("Rate limited: last synthesis was {0}s ago (minimum {1}s)")]
    RateLimited(u64, u64),
    #[error("Invalid template: {0}")]
    InvalidTemplate(String),
}

// ---------------------------------------------------------------------------
// SynthesisStage (for tracing/debugging)
// ---------------------------------------------------------------------------

/// Pipeline stage identifier for tracing spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisStage {
    RoleAssignment,
    StructureDesign,
    Validation,
}

impl SynthesisStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RoleAssignment => "role_assignment",
            Self::StructureDesign => "structure_design",
            Self::Validation => "validation",
        }
    }
}

// ---------------------------------------------------------------------------
// SynthesisRequest / SynthesisConstraints
// ---------------------------------------------------------------------------

/// Constraints for the synthesis pipeline.
#[derive(Debug, Clone)]
pub struct SynthesisConstraints {
    pub max_agents: u32,
    pub max_cost_usd: f32,
    pub max_latency_ms: f32,
    pub available_models: Vec<String>,
}

impl Default for SynthesisConstraints {
    fn default() -> Self {
        Self {
            max_agents: 6,
            max_cost_usd: 1.0,
            max_latency_ms: 30_000.0,
            available_models: Vec::new(),
        }
    }
}

/// A request to synthesize a new topology via LLM.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub task_description: String,
    pub constraints: SynthesisConstraints,
}

// ---------------------------------------------------------------------------
// Parsed data types
// ---------------------------------------------------------------------------

/// A single role assignment parsed from Stage 1 JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct RoleAssignment {
    pub name: String,
    pub model: String,
    pub system: u8,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// The structure design parsed from Stage 2 JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct StructureDesign {
    pub adjacency: Vec<Vec<u8>>,
    pub edge_types: Vec<Vec<String>>,
    pub template: String,
}

/// Wrapper for Stage 1 JSON envelope.
#[derive(Debug, Deserialize)]
struct RolesEnvelope {
    roles: Vec<RoleAssignment>,
}

// ---------------------------------------------------------------------------
// TopologySynthesizer
// ---------------------------------------------------------------------------

/// 3-stage pipeline for LLM-based topology creation.
///
/// Parses pre-computed LLM JSON outputs (roles + structure), builds a
/// `TopologyGraph`, and validates it via `HybridVerifier`.
pub struct TopologySynthesizer {
    verifier: HybridVerifier,
    rate_limit_interval_secs: u64,
    last_synthesis_time: Option<Instant>,
}

impl Default for TopologySynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologySynthesizer {
    /// Create a new synthesizer with default rate limit (60s).
    pub fn new() -> Self {
        Self {
            verifier: HybridVerifier::new(),
            rate_limit_interval_secs: 60,
            last_synthesis_time: None,
        }
    }

    /// Check whether the synthesizer is currently rate-limited.
    pub fn is_rate_limited(&self) -> bool {
        if let Some(last) = self.last_synthesis_time {
            let elapsed = last.elapsed().as_secs();
            elapsed < self.rate_limit_interval_secs
        } else {
            false
        }
    }

    /// Record that a synthesis just occurred (updates the rate-limit clock).
    pub fn mark_synthesis(&mut self) {
        self.last_synthesis_time = Some(Instant::now());
    }

    // -------------------------------------------------------------------
    // Stage 1: Parse role assignments
    // -------------------------------------------------------------------

    /// Parse Stage 1 JSON into a list of role assignments.
    pub fn parse_roles(json: &str) -> Result<Vec<RoleAssignment>, SynthesisError> {
        let _span = info_span!(
            "synthesis_stage",
            stage = SynthesisStage::RoleAssignment.as_str()
        )
        .entered();

        let envelope: RolesEnvelope = serde_json::from_str(json)
            .map_err(|e| SynthesisError::RoleParseFailed(e.to_string()))?;

        info!(role_count = envelope.roles.len(), "parsed role assignments");
        Ok(envelope.roles)
    }

    // -------------------------------------------------------------------
    // Stage 2: Parse structure design
    // -------------------------------------------------------------------

    /// Parse Stage 2 JSON into a structure design.
    pub fn parse_structure(json: &str) -> Result<StructureDesign, SynthesisError> {
        let _span = info_span!(
            "synthesis_stage",
            stage = SynthesisStage::StructureDesign.as_str()
        )
        .entered();

        let design: StructureDesign = serde_json::from_str(json)
            .map_err(|e| SynthesisError::StructureParseFailed(e.to_string()))?;

        info!(
            adjacency_size = design.adjacency.len(),
            template = %design.template,
            "parsed structure design"
        );
        Ok(design)
    }

    // -------------------------------------------------------------------
    // Stage 3: Build graph from parsed data
    // -------------------------------------------------------------------

    /// Build a `TopologyGraph` from parsed roles and structure.
    pub fn build_graph(
        roles: &[RoleAssignment],
        structure: &StructureDesign,
    ) -> Result<TopologyGraph, SynthesisError> {
        let _span = info_span!(
            "synthesis_stage",
            stage = SynthesisStage::Validation.as_str()
        )
        .entered();

        let n = roles.len();
        let adj_n = structure.adjacency.len();

        // Dimension check: adjacency must be NxN where N = role count
        if n != adj_n {
            return Err(SynthesisError::DimensionMismatch(n, adj_n));
        }

        // Validate each row of adjacency is also length N
        for (i, row) in structure.adjacency.iter().enumerate() {
            if row.len() != n {
                return Err(SynthesisError::DimensionMismatch(n, row.len()));
            }
            // Same check for edge_types rows
            if i < structure.edge_types.len() && structure.edge_types[i].len() != n {
                return Err(SynthesisError::DimensionMismatch(
                    n,
                    structure.edge_types[i].len(),
                ));
            }
        }

        // Validate edge_types outer dimension
        if structure.edge_types.len() != n {
            return Err(SynthesisError::DimensionMismatch(
                n,
                structure.edge_types.len(),
            ));
        }

        // Validate template name
        if TopologyTemplate::parse(&structure.template).is_none() {
            return Err(SynthesisError::InvalidTemplate(structure.template.clone()));
        }

        let mut graph =
            TopologyGraph::try_new(&structure.template).map_err(SynthesisError::InvalidTemplate)?;

        // Add nodes from role assignments
        for role in roles {
            let node = TopologyNode::new(
                role.name.clone(),
                role.model.clone(),
                role.system,
                role.capabilities.clone(),
                0, // default security label
                1.0,
                60.0,
            );
            graph.add_node(node);
        }

        // Add edges from adjacency matrix + edge_types
        for i in 0..n {
            for j in 0..n {
                if structure.adjacency[i][j] == 1 {
                    let edge_type_str = &structure.edge_types[i][j];
                    let edge = if edge_type_str.is_empty() {
                        // Default to control edge if type is empty but adjacency is 1
                        TopologyEdge::control()
                    } else {
                        TopologyEdge::try_new(
                            edge_type_str.clone(),
                            None,
                            "open".to_string(),
                            None,
                            1.0,
                        )
                        .map_err(|e| {
                            SynthesisError::StructureParseFailed(format!(
                                "Invalid edge type at [{i}][{j}]: {e}"
                            ))
                        })?
                    };

                    graph.try_add_edge(i, j, edge).map_err(|e| {
                        SynthesisError::StructureParseFailed(format!(
                            "Failed to add edge [{i}]->[{j}]: {e}"
                        ))
                    })?;
                }
            }
        }

        info!(
            nodes = graph.node_count(),
            edges = graph.edge_count(),
            template = %structure.template,
            "built topology graph"
        );

        Ok(graph)
    }

    // -------------------------------------------------------------------
    // Full pipeline: parse + build + verify
    // -------------------------------------------------------------------

    /// Run the full 3-stage synthesis pipeline.
    ///
    /// 1. Parse `stage1_json` into role assignments
    /// 2. Parse `stage2_json` into structure design
    /// 3. Build + validate the TopologyGraph via HybridVerifier
    pub fn synthesize(
        &mut self,
        _request: SynthesisRequest,
        stage1_json: &str,
        stage2_json: &str,
    ) -> Result<TopologyGraph, SynthesisError> {
        // Rate-limit check
        if self.is_rate_limited() {
            let elapsed = self
                .last_synthesis_time
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0);
            return Err(SynthesisError::RateLimited(
                elapsed,
                self.rate_limit_interval_secs,
            ));
        }

        let _span = info_span!("topology_synthesis").entered();

        // Stage 1: parse roles
        let roles = Self::parse_roles(stage1_json)?;

        // Stage 2: parse structure
        let structure = Self::parse_structure(stage2_json)?;

        // Stage 3: build graph + validate
        let graph = Self::build_graph(&roles, &structure)?;

        // Verify the built graph
        let result = self.verifier.verify(&graph);
        if !result.valid {
            warn!(errors = ?result.errors, "synthesized topology failed verification");
            return Err(SynthesisError::ValidationFailed(result.errors));
        }

        if !result.warnings.is_empty() {
            warn!(warnings = ?result.warnings, "synthesized topology has warnings");
        }

        // Mark synthesis time for rate limiting
        self.mark_synthesis();

        info!(
            graph_id = %graph.id,
            nodes = graph.node_count(),
            edges = graph.edge_count(),
            "synthesis complete"
        );

        Ok(graph)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_parse_valid_roles() {
        let roles = TopologySynthesizer::parse_roles(valid_roles_json()).unwrap();
        assert_eq!(roles.len(), 2);
        assert_eq!(roles[0].name, "coder");
        assert_eq!(roles[0].model, "gemini-2.5-flash");
        assert_eq!(roles[0].system, 2);
        assert_eq!(roles[0].capabilities, vec!["code_generation"]);
        assert_eq!(roles[1].name, "reviewer");
    }

    #[test]
    fn test_parse_invalid_roles_json() {
        let result = TopologySynthesizer::parse_roles("not json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SynthesisError::RoleParseFailed(_)),
            "Expected RoleParseFailed, got: {err}"
        );
    }

    #[test]
    fn test_parse_valid_structure() {
        let structure = TopologySynthesizer::parse_structure(valid_structure_json()).unwrap();
        assert_eq!(structure.adjacency.len(), 2);
        assert_eq!(structure.adjacency[0], vec![0, 1]);
        assert_eq!(structure.edge_types[0][1], "control");
        assert_eq!(structure.template, "sequential");
    }

    #[test]
    fn test_parse_invalid_structure_json() {
        let result = TopologySynthesizer::parse_structure("{bad}");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SynthesisError::StructureParseFailed(_)),
            "Expected StructureParseFailed, got: {err}"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        // 3 roles but 2x2 adjacency
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
        let err = result.unwrap_err();
        assert!(
            matches!(err, SynthesisError::DimensionMismatch(3, 2)),
            "Expected DimensionMismatch(3, 2), got: {err}"
        );
    }

    #[test]
    fn test_synthesize_success() {
        let mut synth = TopologySynthesizer::new();
        let request = SynthesisRequest {
            task_description: "Write a function".to_string(),
            constraints: SynthesisConstraints::default(),
        };
        let graph = synth
            .synthesize(request, valid_roles_json(), valid_structure_json())
            .unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.template(), TopologyTemplate::Sequential);
    }

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
            "template": "nonexistent_template"
        }"#;

        let roles = TopologySynthesizer::parse_roles(roles_json).unwrap();
        let structure = TopologySynthesizer::parse_structure(structure_json).unwrap();
        let result = TopologySynthesizer::build_graph(&roles, &structure);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SynthesisError::InvalidTemplate(_)),
            "Expected InvalidTemplate, got: {err}"
        );
    }

    #[test]
    fn test_build_graph_correct_node_count() {
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
                system: 2,
                capabilities: vec!["code".into()],
            },
            RoleAssignment {
                name: "c".into(),
                model: "m".into(),
                system: 1,
                capabilities: vec![],
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
    }

    #[test]
    fn test_build_graph_correct_edge_count() {
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
        ];
        let structure = StructureDesign {
            adjacency: vec![vec![0, 1], vec![0, 0]],
            edge_types: vec![
                vec!["".into(), "control".into()],
                vec!["".into(), "".into()],
            ],
            template: "sequential".into(),
        };
        let graph = TopologySynthesizer::build_graph(&roles, &structure).unwrap();
        // Adjacency has exactly one 1, so one edge
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_rate_limiting() {
        let mut synth = TopologySynthesizer::new();
        assert!(!synth.is_rate_limited());

        // Mark a synthesis
        synth.mark_synthesis();
        assert!(synth.is_rate_limited());

        // Trying to synthesize again should fail
        let request = SynthesisRequest {
            task_description: "test".to_string(),
            constraints: SynthesisConstraints::default(),
        };
        let result = synth.synthesize(request, valid_roles_json(), valid_structure_json());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SynthesisError::RateLimited(_, 60)),
            "Expected RateLimited, got: {err}"
        );
    }

    #[test]
    fn test_empty_roles_array() {
        let json = r#"{"roles": []}"#;
        let roles = TopologySynthesizer::parse_roles(json).unwrap();
        assert!(roles.is_empty());

        // Building a graph with 0 roles and 0x0 adjacency should succeed
        let structure = StructureDesign {
            adjacency: vec![],
            edge_types: vec![],
            template: "sequential".into(),
        };
        let graph = TopologySynthesizer::build_graph(&roles, &structure).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_default_constraints() {
        let c = SynthesisConstraints::default();
        assert_eq!(c.max_agents, 6);
        assert!((c.max_cost_usd - 1.0).abs() < f32::EPSILON);
        assert!((c.max_latency_ms - 30_000.0).abs() < f32::EPSILON);
        assert!(c.available_models.is_empty());
    }

    #[test]
    fn test_synthesis_stage_as_str() {
        assert_eq!(SynthesisStage::RoleAssignment.as_str(), "role_assignment");
        assert_eq!(SynthesisStage::StructureDesign.as_str(), "structure_design");
        assert_eq!(SynthesisStage::Validation.as_str(), "validation");
    }
}
