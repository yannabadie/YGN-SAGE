//! LTL (Linear Temporal Logic) model checking for TopologyGraph.
//!
//! Verifies temporal properties on multi-agent topologies:
//! - **Reachability**: can agent A reach agent B?
//! - **Safety**: no high-to-low security label information flow
//! - **Liveness**: every entry node can reach at least one exit node
//! - **Bounded liveness**: all entry-to-exit paths stay within a depth limit
//!
//! All checks use petgraph BFS/DFS — O(V+E), no SMT solver needed.

use crate::topology::topology_graph::TopologyGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, EdgeRef};
use pyo3::prelude::*;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// LtlResult
// ---------------------------------------------------------------------------

/// Result of an LTL verification check.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LtlResult {
    /// Whether the property holds (no violations).
    #[pyo3(get)]
    pub passed: bool,
    /// Human-readable descriptions of each violation found.
    #[pyo3(get)]
    pub violations: Vec<String>,
}

#[pymethods]
impl LtlResult {
    fn __repr__(&self) -> String {
        if self.passed {
            "LtlResult(passed=True)".to_string()
        } else {
            format!(
                "LtlResult(passed=False, violations={})",
                self.violations.len()
            )
        }
    }
}

impl LtlResult {
    fn ok() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
        }
    }

    fn fail(violations: Vec<String>) -> Self {
        Self {
            passed: false,
            violations,
        }
    }
}

// ---------------------------------------------------------------------------
// LtlVerifier
// ---------------------------------------------------------------------------

/// LTL model checker for TopologyGraph instances.
///
/// Checks temporal properties (reachability, safety, liveness, bounded liveness)
/// using graph algorithms on the underlying petgraph DiGraph.
#[pyclass]
#[derive(Default)]
pub struct LtlVerifier;

#[pymethods]
impl LtlVerifier {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Check if node `to_idx` is reachable from node `from_idx` via BFS.
    ///
    /// Considers all edge types (control, message, state).
    #[pyo3(name = "check_reachability")]
    pub fn py_check_reachability(
        &self,
        graph: &TopologyGraph,
        from_idx: usize,
        to_idx: usize,
    ) -> PyResult<bool> {
        let nc = graph.node_count();
        if from_idx >= nc {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "from_idx {} out of range (graph has {} nodes)",
                from_idx, nc
            )));
        }
        if to_idx >= nc {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "to_idx {} out of range (graph has {} nodes)",
                to_idx, nc
            )));
        }
        Ok(Self::check_reachability(graph, from_idx, to_idx))
    }

    /// Check safety: no edge flows from a higher security label to a lower one.
    #[pyo3(name = "check_safety")]
    pub fn py_check_safety(&self, graph: &TopologyGraph) -> LtlResult {
        Self::check_safety(graph)
    }

    /// Check liveness: every entry node can reach at least one exit node.
    #[pyo3(name = "check_liveness")]
    pub fn py_check_liveness(&self, graph: &TopologyGraph) -> LtlResult {
        Self::check_liveness(graph)
    }

    /// Check bounded liveness: all entry-to-exit paths have length <= depth_limit.
    #[pyo3(name = "check_bounded_liveness")]
    pub fn py_check_bounded_liveness(
        &self,
        graph: &TopologyGraph,
        depth_limit: usize,
    ) -> LtlResult {
        Self::check_bounded_liveness(graph, depth_limit)
    }

    fn __repr__(&self) -> String {
        "LtlVerifier()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Pure Rust implementation (no Python dependency)
// ---------------------------------------------------------------------------

impl LtlVerifier {
    /// BFS reachability: can we reach `to_idx` starting from `from_idx`?
    ///
    /// Traverses all edge types (control, message, state).
    pub fn check_reachability(graph: &TopologyGraph, from_idx: usize, to_idx: usize) -> bool {
        let inner = graph.inner_graph();
        let from = NodeIndex::new(from_idx);
        let to = NodeIndex::new(to_idx);

        let mut bfs = Bfs::new(inner, from);
        while let Some(node) = bfs.next(inner) {
            if node == to {
                return true;
            }
        }
        false
    }

    /// Safety check: information must not flow from higher to lower security labels.
    ///
    /// For every edge, if source.security_label > target.security_label, it is a
    /// violation (HIGH -> LOW info flow).
    pub fn check_safety(graph: &TopologyGraph) -> LtlResult {
        let inner = graph.inner_graph();
        let mut violations = Vec::new();

        for edge_ref in inner.edge_references() {
            let src = &inner[edge_ref.source()];
            let tgt = &inner[edge_ref.target()];

            if src.security_label > tgt.security_label {
                violations.push(format!(
                    "Safety violation: edge from '{}' (role='{}', label={}) to '{}' (role='{}', label={})",
                    src.node_id, src.role, src.security_label,
                    tgt.node_id, tgt.role, tgt.security_label
                ));
            }
        }

        if violations.is_empty() {
            LtlResult::ok()
        } else {
            LtlResult::fail(violations)
        }
    }

    /// Liveness check: every entry node must be able to reach at least one exit node.
    ///
    /// Entry nodes have no incoming edges; exit nodes have no outgoing edges.
    /// Uses BFS from each entry node, checking if any exit node is discovered.
    pub fn check_liveness(graph: &TopologyGraph) -> LtlResult {
        let inner = graph.inner_graph();

        if inner.node_count() == 0 {
            return LtlResult::ok();
        }

        let entry_nodes = graph.entry_nodes();
        let exit_nodes: HashSet<usize> = graph.exit_nodes().into_iter().collect();

        if entry_nodes.is_empty() {
            return LtlResult::fail(vec![
                "Liveness violation: no entry nodes found (every node has incoming edges)"
                    .to_string(),
            ]);
        }

        if exit_nodes.is_empty() {
            return LtlResult::fail(vec![
                "Liveness violation: no exit nodes found (every node has outgoing edges)"
                    .to_string(),
            ]);
        }

        let mut violations = Vec::new();

        for &entry_idx in &entry_nodes {
            let entry_ni = NodeIndex::new(entry_idx);
            let mut bfs = Bfs::new(inner, entry_ni);
            let mut reaches_exit = false;

            while let Some(node) = bfs.next(inner) {
                if exit_nodes.contains(&node.index()) {
                    reaches_exit = true;
                    break;
                }
            }

            if !reaches_exit {
                let entry_node = &inner[entry_ni];
                violations.push(format!(
                    "Liveness violation: entry node '{}' (role='{}') cannot reach any exit node",
                    entry_node.node_id, entry_node.role
                ));
            }
        }

        if violations.is_empty() {
            LtlResult::ok()
        } else {
            LtlResult::fail(violations)
        }
    }

    /// Bounded liveness: every path from an entry node to an exit node must
    /// have length <= `depth_limit`.
    ///
    /// Uses DFS with depth tracking from each entry node. Reports any exit
    /// node reachable only via a path exceeding the depth limit.
    pub fn check_bounded_liveness(graph: &TopologyGraph, depth_limit: usize) -> LtlResult {
        let inner = graph.inner_graph();

        if inner.node_count() == 0 {
            return LtlResult::ok();
        }

        let entry_nodes = graph.entry_nodes();
        let exit_nodes: HashSet<usize> = graph.exit_nodes().into_iter().collect();

        if entry_nodes.is_empty() || exit_nodes.is_empty() {
            return LtlResult::ok();
        }

        let mut violations = Vec::new();

        for &entry_idx in &entry_nodes {
            // DFS with depth tracking: find all paths to exit nodes
            let mut stack: Vec<(NodeIndex, usize)> = vec![(NodeIndex::new(entry_idx), 0)];
            // Track (node, depth) pairs we've seen to avoid infinite loops in cyclic graphs
            let mut visited_at_depth: HashSet<(usize, usize)> = HashSet::new();

            while let Some((node, depth)) = stack.pop() {
                if !visited_at_depth.insert((node.index(), depth)) {
                    continue;
                }

                if exit_nodes.contains(&node.index()) && depth > depth_limit {
                    let entry_node = &inner[NodeIndex::new(entry_idx)];
                    let exit_node = &inner[node];
                    violations.push(format!(
                        "Bounded liveness violation: path from entry '{}' (role='{}') to exit '{}' (role='{}') has depth {} > limit {}",
                        entry_node.node_id, entry_node.role,
                        exit_node.node_id, exit_node.role,
                        depth, depth_limit
                    ));
                }

                // Only continue DFS if we haven't exceeded the limit by too much
                // (cap at depth_limit + 1 to detect violations without unbounded exploration)
                if depth <= depth_limit {
                    for neighbor in inner.neighbors(node) {
                        stack.push((neighbor, depth + 1));
                    }
                }
            }
        }

        if violations.is_empty() {
            LtlResult::ok()
        } else {
            LtlResult::fail(violations)
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;
    use crate::topology::topology_graph::{TopologyEdge, TopologyGraph, TopologyNode};

    // -- Helpers --

    /// Build a simple sequential graph: A -> B -> C (all label=0).
    fn make_sequential() -> TopologyGraph {
        templates::sequential("m")
    }

    /// Build a parallel graph: source -> [w0, w1, w2] -> aggregator.
    fn make_parallel() -> TopologyGraph {
        templates::parallel("m", 3)
    }

    // -- Reachability tests --

    #[test]
    fn test_reachability_sequential() {
        let g = make_sequential();
        // A(0) -> B(1) -> C(2): A reaches C
        assert!(LtlVerifier::check_reachability(&g, 0, 2));
        // C(2) does NOT reach A(0) in a DAG
        assert!(!LtlVerifier::check_reachability(&g, 2, 0));
        // A reaches itself (BFS includes start node)
        assert!(LtlVerifier::check_reachability(&g, 0, 0));
    }

    #[test]
    fn test_reachability_parallel() {
        let g = make_parallel();
        // source(0), w0(1), w1(2), w2(3), aggregator(4)
        // Workers don't reach each other (no w0->w1 path)
        assert!(!LtlVerifier::check_reachability(&g, 1, 2));
        assert!(!LtlVerifier::check_reachability(&g, 2, 1));
        assert!(!LtlVerifier::check_reachability(&g, 1, 3));
        // Source reaches aggregator via workers
        assert!(LtlVerifier::check_reachability(&g, 0, 4));
        // Aggregator doesn't reach source
        assert!(!LtlVerifier::check_reachability(&g, 4, 0));
    }

    // -- Safety tests --

    #[test]
    fn test_safety_violation() {
        // HIGH(2) -> LOW(0) is a violation
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let high = TopologyNode::new("high".into(), "m".into(), 1, vec![], 2, 1.0, 60.0);
        let low = TopologyNode::new("low".into(), "m".into(), 1, vec![], 0, 1.0, 60.0);
        let hi = g.add_node(high);
        let li = g.add_node(low);
        g.try_add_edge(hi, li, TopologyEdge::control()).unwrap();

        let result = LtlVerifier::check_safety(&g);
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert!(result.violations[0].contains("Safety violation"));
        assert!(result.violations[0].contains("label=2"));
        assert!(result.violations[0].contains("label=0"));
    }

    #[test]
    fn test_safety_same_level() {
        // Same labels (1 -> 1) is safe
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let a = TopologyNode::new("a".into(), "m".into(), 1, vec![], 1, 1.0, 60.0);
        let b = TopologyNode::new("b".into(), "m".into(), 1, vec![], 1, 1.0, 60.0);
        let ai = g.add_node(a);
        let bi = g.add_node(b);
        g.try_add_edge(ai, bi, TopologyEdge::control()).unwrap();

        let result = LtlVerifier::check_safety(&g);
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_safety_low_to_high() {
        // LOW(0) -> HIGH(2) is safe (information flows up)
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let low = TopologyNode::new("low".into(), "m".into(), 1, vec![], 0, 1.0, 60.0);
        let high = TopologyNode::new("high".into(), "m".into(), 1, vec![], 2, 1.0, 60.0);
        let li = g.add_node(low);
        let hi = g.add_node(high);
        g.try_add_edge(li, hi, TopologyEdge::control()).unwrap();

        let result = LtlVerifier::check_safety(&g);
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    // -- Liveness tests --

    #[test]
    fn test_liveness_sequential() {
        let g = make_sequential();
        let result = LtlVerifier::check_liveness(&g);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_liveness_parallel() {
        let g = make_parallel();
        let result = LtlVerifier::check_liveness(&g);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_liveness_disconnected_entry() {
        // Entry node with no path to any exit node
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let entry = TopologyNode::new("entry".into(), "m".into(), 1, vec![], 0, 1.0, 60.0);
        let island = TopologyNode::new("island".into(), "m".into(), 1, vec![], 0, 1.0, 60.0);
        let exit = TopologyNode::new("exit".into(), "m".into(), 1, vec![], 0, 1.0, 60.0);
        let ei = g.add_node(entry);
        let _ii = g.add_node(island); // disconnected node — also an entry AND exit
        let xi = g.add_node(exit);
        g.try_add_edge(ei, xi, TopologyEdge::control()).unwrap();

        // "island" is both entry (no incoming) and exit (no outgoing), so it reaches itself.
        // "entry" can reach "exit". All entries reach at least one exit.
        let result = LtlVerifier::check_liveness(&g);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_liveness_empty_graph() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        let result = LtlVerifier::check_liveness(&g);
        assert!(result.passed);
    }

    // -- Bounded liveness tests --

    #[test]
    fn test_bounded_liveness_ok() {
        // Sequential: A->B->C has path length 2, limit 5 should pass
        let g = make_sequential();
        let result = LtlVerifier::check_bounded_liveness(&g, 5);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_bounded_liveness_exact() {
        // Sequential: A->B->C has path length 2, limit 2 should pass
        let g = make_sequential();
        let result = LtlVerifier::check_bounded_liveness(&g, 2);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_bounded_liveness_exceeded() {
        // Sequential: A->B->C has path length 2, limit 1 should fail
        let g = make_sequential();
        let result = LtlVerifier::check_bounded_liveness(&g, 1);
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
        assert!(result.violations[0].contains("Bounded liveness violation"));
        assert!(result.violations[0].contains("depth 2"));
        assert!(result.violations[0].contains("limit 1"));
    }

    #[test]
    fn test_bounded_liveness_parallel_ok() {
        // Parallel: source -> workers -> aggregator, depth 2, limit 5
        let g = make_parallel();
        let result = LtlVerifier::check_bounded_liveness(&g, 5);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_bounded_liveness_parallel_tight() {
        // Parallel: source -> worker -> aggregator is depth 2
        let g = make_parallel();
        let result = LtlVerifier::check_bounded_liveness(&g, 2);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_bounded_liveness_parallel_exceeded() {
        // Parallel: source -> worker -> aggregator is depth 2, limit 1 should fail
        let g = make_parallel();
        let result = LtlVerifier::check_bounded_liveness(&g, 1);
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_bounded_liveness_empty_graph() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        let result = LtlVerifier::check_bounded_liveness(&g, 0);
        assert!(result.passed);
    }

    // -- Integration: templates pass all LTL checks --

    #[test]
    fn test_all_templates_safe() {
        // All 8 templates use label=0 (public), so safety should pass
        for name in &[
            "sequential",
            "parallel",
            "avr",
            "selfmoa",
            "debate",
            "brainstorming",
        ] {
            let g = crate::topology::templates::TemplateStore::create(name, "m").unwrap();
            let result = LtlVerifier::check_safety(&g);
            assert!(result.passed, "Template '{}' safety failed: {:?}", name, result.violations);
        }
        // hierarchical and hub use label=1 for all nodes, still safe (same level)
        let g = crate::topology::templates::TemplateStore::create("hierarchical", "m").unwrap();
        assert!(LtlVerifier::check_safety(&g).passed);
        let g = crate::topology::templates::TemplateStore::create("hub", "m").unwrap();
        assert!(LtlVerifier::check_safety(&g).passed);
    }
}
