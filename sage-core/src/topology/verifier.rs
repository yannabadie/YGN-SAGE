//! HybridVerifier: fast structural + semantic verification of TopologyGraph instances.
//!
//! All checks are O(V+E) graph algorithms — no Z3/SMT involved.
//! Provides both hard errors (invalid topology) and soft warnings (semantic issues).

use super::topology_graph::*;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::{Bfs, EdgeRef};
use pyo3::prelude::*;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Result of a hybrid verification pass.
#[pyclass]
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the topology is structurally valid (no errors).
    #[pyo3(get)]
    pub valid: bool,
    /// Hard errors that make the topology invalid.
    #[pyo3(get)]
    pub errors: Vec<String>,
    /// Soft warnings about potential semantic issues.
    #[pyo3(get)]
    pub warnings: Vec<String>,
}

impl VerificationResult {
    fn new() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_errors(&mut self, errs: Vec<String>) {
        if !errs.is_empty() {
            self.valid = false;
            self.errors.extend(errs);
        }
    }

    fn add_warnings(&mut self, warns: Vec<String>) {
        self.warnings.extend(warns);
    }
}

#[pymethods]
impl VerificationResult {
    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.valid {
            write!(f, "VerificationResult: VALID")?;
        } else {
            write!(f, "VerificationResult: INVALID ({} errors)", self.errors.len())?;
        }
        if !self.warnings.is_empty() {
            write!(f, ", {} warnings", self.warnings.len())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HybridVerifier
// ---------------------------------------------------------------------------

/// Fast topology verifier using graph algorithms (no SMT solver).
pub struct HybridVerifier {
    /// Maximum allowed incoming edges per node.
    pub max_fan_in: usize,
    /// Maximum allowed outgoing edges per node.
    pub max_fan_out: usize,
}

impl Default for HybridVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridVerifier {
    /// Create a new verifier with default limits (fan-in=10, fan-out=10).
    pub fn new() -> Self {
        Self {
            max_fan_in: 10,
            max_fan_out: 10,
        }
    }

    /// Run all verification checks on a topology.
    pub fn verify(&self, graph: &TopologyGraph) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Structural checks
        let (errs, warns) = self.check_dag_validity(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_entry_exit_reachability(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_capability_coverage(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_budget_feasibility(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_fan_limits(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_security_labels(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        // Semantic diagnosis checks
        let (errs, warns) = self.check_role_coherence(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_switch_completeness(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_loop_termination(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        let (errs, warns) = self.check_field_mapping_consistency(graph);
        result.add_errors(errs);
        result.add_warnings(warns);

        result
    }

    // -----------------------------------------------------------------------
    // Check 1: DAG validity (control-flow only)
    // -----------------------------------------------------------------------

    /// Control-flow subgraph must be acyclic, UNLESS cycles involve only Closed-gate edges.
    fn check_dag_validity(&self, graph: &TopologyGraph) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        // Build a filtered control-only subgraph with only Open-gate edges
        let mut control_graph: DiGraph<(), ()> = DiGraph::new();

        // Add the same number of nodes
        for _ in inner.node_indices() {
            control_graph.add_node(());
        }

        // Add only control edges that are open
        for edge_ref in inner.edge_references() {
            let edge = edge_ref.weight();
            if edge.typed_edge_type() == EdgeType::Control && edge.typed_gate() == Gate::Open {
                control_graph.add_edge(edge_ref.source(), edge_ref.target(), ());
            }
        }

        if petgraph::algo::is_cyclic_directed(&control_graph) {
            errors.push(
                "Control-flow subgraph (open gates only) contains a cycle".to_string(),
            );
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Check 2: Entry/Exit reachability
    // -----------------------------------------------------------------------

    /// Must have at least 1 entry node (no incoming open-gate control edges) and 1 exit
    /// node (no outgoing open-gate control edges). All nodes must be reachable from an
    /// entry via BFS on open-gate control edges.
    ///
    /// Closed-gate back-edges are excluded: they represent dormant repair paths and do
    /// not affect the normal execution flow.
    fn check_entry_exit_reachability(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        if inner.node_count() == 0 {
            return (errors, warnings);
        }

        // Build control-only subgraph using only OPEN-gate control edges.
        // Closed-gate back-edges are excluded so they don't create false
        // incoming edges on entry nodes (e.g., the AVR repair path).
        let mut control_graph: DiGraph<(), ()> = DiGraph::new();
        for _ in inner.node_indices() {
            control_graph.add_node(());
        }
        for edge_ref in inner.edge_references() {
            let edge = edge_ref.weight();
            if edge.typed_edge_type() == EdgeType::Control && edge.typed_gate() == Gate::Open {
                control_graph.add_edge(edge_ref.source(), edge_ref.target(), ());
            }
        }

        // Find entry nodes (no incoming open-gate control edges)
        let entry_nodes: Vec<NodeIndex> = control_graph
            .node_indices()
            .filter(|&idx| {
                control_graph
                    .neighbors_directed(idx, petgraph::Direction::Incoming)
                    .next()
                    .is_none()
            })
            .collect();

        if entry_nodes.is_empty() {
            errors.push("No entry node found (every node has incoming control edges)".to_string());
            return (errors, warnings);
        }

        // Find exit nodes (no outgoing open-gate control edges)
        let exit_nodes: Vec<NodeIndex> = control_graph
            .node_indices()
            .filter(|&idx| {
                control_graph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .collect();

        if exit_nodes.is_empty() {
            errors.push(
                "No exit node found (every node has outgoing control edges)".to_string(),
            );
        }

        // BFS reachability: all nodes must be reachable from at least one entry
        let mut reachable = HashSet::new();
        for &entry in &entry_nodes {
            let mut bfs = Bfs::new(&control_graph, entry);
            while let Some(node) = bfs.next(&control_graph) {
                reachable.insert(node);
            }
        }

        for idx in inner.node_indices() {
            if !reachable.contains(&idx) {
                let node = &inner[idx];
                errors.push(format!(
                    "Node '{}' (role='{}') is unreachable from any entry node via control edges",
                    node.node_id, node.role
                ));
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Check 3: Capability coverage
    // -----------------------------------------------------------------------

    /// Each node must have either a non-empty model_id or non-empty required_capabilities.
    fn check_capability_coverage(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        for idx in inner.node_indices() {
            let node = &inner[idx];
            if node.model_id.is_empty() && node.required_capabilities.is_empty() {
                errors.push(format!(
                    "Node '{}' (role='{}') has no model_id and no required_capabilities",
                    node.node_id, node.role
                ));
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Check 4: Budget feasibility
    // -----------------------------------------------------------------------

    /// Sum of all node max_cost_usd must be > 0 and < 10000.
    fn check_budget_feasibility(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        if inner.node_count() == 0 {
            return (errors, warnings);
        }

        let total_budget: f32 = inner
            .node_weights()
            .map(|n| n.max_cost_usd)
            .sum();

        if total_budget <= 0.0 {
            errors.push(format!(
                "Total budget is non-positive: ${:.2}",
                total_budget
            ));
        }

        if total_budget >= 10_000.0 {
            errors.push(format!(
                "Total budget exceeds limit: ${:.2} >= $10000.00",
                total_budget
            ));
        }

        if !total_budget.is_finite() {
            errors.push("Total budget is not finite (NaN or Infinity)".to_string());
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Check 5: Fan-in / Fan-out limits
    // -----------------------------------------------------------------------

    /// No node should exceed max_fan_in incoming edges or max_fan_out outgoing edges.
    fn check_fan_limits(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        for idx in inner.node_indices() {
            let node = &inner[idx];

            let fan_in = inner
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .count();
            let fan_out = inner
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .count();

            if fan_in > self.max_fan_in {
                errors.push(format!(
                    "Node '{}' (role='{}') fan-in {} exceeds limit {}",
                    node.node_id, node.role, fan_in, self.max_fan_in
                ));
            }
            if fan_out > self.max_fan_out {
                errors.push(format!(
                    "Node '{}' (role='{}') fan-out {} exceeds limit {}",
                    node.node_id, node.role, fan_out, self.max_fan_out
                ));
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Check 6: Security label lattice
    // -----------------------------------------------------------------------

    /// Information can only flow from lower to higher labels (or same).
    /// An edge from security_label=2 to security_label=0 is a violation.
    fn check_security_labels(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        for edge_ref in inner.edge_references() {
            let src = &inner[edge_ref.source()];
            let tgt = &inner[edge_ref.target()];

            if src.security_label > tgt.security_label {
                errors.push(format!(
                    "Security label violation: edge from '{}' (label={}) to '{}' (label={})",
                    src.node_id, src.security_label, tgt.node_id, tgt.security_label
                ));
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Semantic check 7: Role coherence
    // -----------------------------------------------------------------------

    /// Warn if a node has role containing "reviewer"/"verifier"/"judge" but system tier is S1.
    fn check_role_coherence(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let errors = Vec::new();
        let mut warnings = Vec::new();

        let complex_roles = ["reviewer", "verifier", "judge", "evaluator"];

        for idx in inner.node_indices() {
            let node = &inner[idx];
            let role_lower = node.role.to_lowercase();

            if node.system == 1
                && complex_roles
                    .iter()
                    .any(|r| role_lower.contains(r))
            {
                warnings.push(format!(
                    "Role coherence: node '{}' has role '{}' (requires complex reasoning) but system tier is S1",
                    node.node_id, node.role
                ));
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Semantic check 8: Switch condition completeness
    // -----------------------------------------------------------------------

    /// If any outgoing edge from a node has a condition, warn if not ALL outgoing edges
    /// from that node have conditions (missing default branch).
    fn check_switch_completeness(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let errors = Vec::new();
        let mut warnings = Vec::new();

        for idx in inner.node_indices() {
            let outgoing: Vec<_> = inner
                .edges_directed(idx, petgraph::Direction::Outgoing)
                .collect();

            if outgoing.is_empty() {
                continue;
            }

            let with_condition = outgoing
                .iter()
                .filter(|e| e.weight().condition.is_some())
                .count();
            let without_condition = outgoing.len() - with_condition;

            if with_condition > 0 && without_condition > 0 {
                let node = &inner[idx];
                warnings.push(format!(
                    "Switch completeness: node '{}' (role='{}') has {} edges with conditions and {} without (missing default branch?)",
                    node.node_id, node.role, with_condition, without_condition
                ));
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Semantic check 9: Loop termination guarantee
    // -----------------------------------------------------------------------

    /// If graph has any closed-gate back-edge, check that the back-edge target
    /// has max_wall_time_s > 0 (timeout ensures termination).
    fn check_loop_termination(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let errors = Vec::new();
        let mut warnings = Vec::new();

        for edge_ref in inner.edge_references() {
            let edge = edge_ref.weight();
            if edge.typed_gate() == Gate::Closed && edge.typed_edge_type() == EdgeType::Control {
                let target = &inner[edge_ref.target()];
                if target.max_wall_time_s <= 0.0 {
                    warnings.push(format!(
                        "Loop termination: closed-gate back-edge targets node '{}' (role='{}') \
                         which has max_wall_time_s={:.1} (no timeout guarantee)",
                        target.node_id, target.role, target.max_wall_time_s
                    ));
                }
            }
        }

        (errors, warnings)
    }

    // -----------------------------------------------------------------------
    // Semantic check 10: Field mapping consistency
    // -----------------------------------------------------------------------

    /// For message edges with field_mapping, check that all keys and values are non-empty.
    fn check_field_mapping_consistency(
        &self,
        graph: &TopologyGraph,
    ) -> (Vec<String>, Vec<String>) {
        let inner = graph.inner_graph();
        let errors = Vec::new();
        let mut warnings = Vec::new();

        for edge_ref in inner.edge_references() {
            let edge = edge_ref.weight();
            if edge.typed_edge_type() == EdgeType::Message {
                if let Some(ref mapping) = edge.field_mapping {
                    for (key, value) in mapping {
                        if key.is_empty() {
                            let src = &inner[edge_ref.source()];
                            let tgt = &inner[edge_ref.target()];
                            warnings.push(format!(
                                "Field mapping: empty key in message edge from '{}' to '{}'",
                                src.node_id, tgt.node_id
                            ));
                        }
                        if value.is_empty() {
                            let src = &inner[edge_ref.source()];
                            let tgt = &inner[edge_ref.target()];
                            warnings.push(format!(
                                "Field mapping: empty value for key '{}' in message edge from '{}' to '{}'",
                                key, src.node_id, tgt.node_id
                            ));
                        }
                    }
                }
            }
        }

        (errors, warnings)
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// PyO3-exposed hybrid verifier for validating topology graphs from Python.
#[pyclass]
pub struct PyHybridVerifier {
    inner: HybridVerifier,
}

#[pymethods]
impl PyHybridVerifier {
    #[new]
    #[pyo3(signature = (max_fan_in=10, max_fan_out=10))]
    pub fn new(max_fan_in: usize, max_fan_out: usize) -> Self {
        Self {
            inner: HybridVerifier {
                max_fan_in,
                max_fan_out,
            },
        }
    }

    /// Verify a topology graph.
    pub fn verify(&self, graph: &TopologyGraph) -> VerificationResult {
        self.inner.verify(graph)
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridVerifier(fan_in={}, fan_out={})",
            self.inner.max_fan_in, self.inner.max_fan_out
        )
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;

    fn verifier() -> HybridVerifier {
        HybridVerifier::new()
    }

    #[test]
    fn test_defaults() {
        let v = verifier();
        assert_eq!(v.max_fan_in, 10);
        assert_eq!(v.max_fan_out, 10);
    }

    #[test]
    fn test_sequential_passes() {
        let g = templates::sequential("m");
        let r = verifier().verify(&g);
        assert!(r.valid, "errors: {:?}", r.errors);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn test_avr_closed_gate_ok() {
        let g = templates::avr("actor", "reviewer");
        let r = verifier().verify(&g);
        // AVR has a closed-gate back-edge which is OK
        assert!(r.valid, "errors: {:?}", r.errors);
    }

    #[test]
    fn test_empty_graph_passes() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        let r = verifier().verify(&g);
        assert!(r.valid, "errors: {:?}", r.errors);
    }

    #[test]
    fn test_security_violation() {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let high = TopologyNode::new("high".into(), "m".into(), 1, vec![], 2, 1.0, 60.0);
        let low = TopologyNode::new("low".into(), "m".into(), 1, vec![], 0, 1.0, 60.0);
        let hi = g.add_node(high);
        let li = g.add_node(low);
        g.try_add_edge(hi, li, TopologyEdge::control()).unwrap();

        let r = verifier().verify(&g);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("Security label")));
    }

    #[test]
    fn test_result_display() {
        let mut r = VerificationResult::new();
        assert!(r.to_string().contains("VALID"));

        r.add_errors(vec!["err".into()]);
        assert!(r.to_string().contains("INVALID"));
    }
}
