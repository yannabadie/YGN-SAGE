//! Topology density function (S_complex) from AgentConductor (arXiv 2602.17100).
//!
//! Provides a differentiable metric correlating topology structure to token cost.
//! S_complex combines three signals:
//! - **S_node**: penalizes excess nodes relative to difficulty-aware N_max
//! - **S_edge**: penalizes dense connectivity (edge ratio vs complete graph)
//! - **S_depth**: measures parallelism (1.0 = fully parallel, 0.0 = fully sequential)
//!
//! N_max bounds are derived from AgentConductor's statistical analysis:
//! - S1 (simple/factual): 4 nodes
//! - S2 (moderate reasoning/code): 7 nodes
//! - S3 (complex formal/multi-step): 10 nodes

use super::topology_graph::TopologyGraph;
use petgraph::algo::toposort;
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use tracing::instrument;

/// Maximum node count per cognitive system (difficulty-aware bounds).
/// Derived from AgentConductor's statistical analysis (arXiv 2602.17100).
const N_MAX_S1: usize = 4;
const N_MAX_S2: usize = 7;
const N_MAX_S3: usize = 10;

// ---------------------------------------------------------------------------
// DensityScore
// ---------------------------------------------------------------------------

/// Result of topology density computation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DensityScore {
    /// Combined density score (0.0-1.0, lower = sparser/cheaper).
    #[pyo3(get)]
    pub s_complex: f32,
    /// Node count penalty: exp(-|V| / N_max). Higher = sparser.
    #[pyo3(get)]
    pub s_node: f32,
    /// Edge density penalty: exp(-|E| / max_edges). Higher = sparser.
    #[pyo3(get)]
    pub s_edge: f32,
    /// Parallelism measure: 1 - (longest_path / |V|). 1.0 = fully parallel.
    #[pyo3(get)]
    pub s_depth: f32,
    /// Max allowed nodes for this cognitive system level.
    #[pyo3(get)]
    pub n_max: usize,
    /// True if node_count > n_max (topology exceeds difficulty budget).
    #[pyo3(get)]
    pub over_budget: bool,
}

#[pymethods]
impl DensityScore {
    fn __repr__(&self) -> String {
        format!(
            "DensityScore(s_complex={:.4}, s_node={:.4}, s_edge={:.4}, s_depth={:.4}, n_max={}, over_budget={})",
            self.s_complex, self.s_node, self.s_edge, self.s_depth, self.n_max, self.over_budget
        )
    }
}

// ---------------------------------------------------------------------------
// TopologyDensity
// ---------------------------------------------------------------------------

/// Computes S_complex density scores for TopologyGraph instances.
///
/// S_complex = sigmoid(S_node + 2 * S_edge + S_depth), normalized to [0, 1].
/// Lower scores indicate sparser (cheaper) topologies.
#[pyclass]
pub struct TopologyDensity;

#[pymethods]
impl TopologyDensity {
    #[new]
    pub fn new() -> Self {
        TopologyDensity
    }

    /// Compute S_complex density score for a topology graph.
    ///
    /// Arguments:
    /// - `graph`: the TopologyGraph to analyze
    /// - `system`: cognitive system level (1=S1, 2=S2, 3=S3)
    ///
    /// Returns a DensityScore with all component signals.
    #[instrument(skip(self, graph))]
    pub fn compute(&self, graph: &TopologyGraph, system: u8) -> DensityScore {
        let v = graph.node_count() as f32;
        let e = graph.edge_count() as f32;

        let n_max = match system {
            1 => N_MAX_S1,
            2 => N_MAX_S2,
            _ => N_MAX_S3,
        };
        let n_max_f = n_max as f32;

        // S_node: exp(-|V| / N_max). Penalizes excess nodes relative to difficulty.
        let s_node = (-v / n_max_f).exp();

        // S_edge: exp(-|E| / max_edges). Penalizes dense connectivity.
        // max_edges = |V| * (|V| - 1) / 2 for undirected complete graph.
        let max_edges = v * (v - 1.0) / 2.0;
        let s_edge = if max_edges > 0.0 {
            (-e / max_edges).exp()
        } else {
            1.0 // No edges possible with 0 or 1 nodes.
        };

        // S_depth: 1 - (longest_path / |V|). Measures parallelism.
        // Longest path computed via DAG DP on topological order.
        let depth = longest_path_length(graph);
        let s_depth = if v > 0.0 {
            1.0 - (depth as f32 / v)
        } else {
            1.0
        };

        // S_complex: combined score via sigmoid normalization.
        // AgentConductor Theorem 1: S_complex = exp(S_node + 2 * S_edge + S_depth)
        // We apply sigmoid (1 / (1 + exp(-x))) to normalize to [0, 1].
        let raw = s_node + 2.0 * s_edge + s_depth;
        let s_complex = 1.0 / (1.0 + (-raw).exp());

        DensityScore {
            s_complex,
            s_node,
            s_edge,
            s_depth,
            n_max,
            over_budget: graph.node_count() > n_max,
        }
    }

    /// Get N_max for a cognitive system level.
    ///
    /// Arguments:
    /// - `system`: 1=S1, 2=S2, 3+=S3
    #[staticmethod]
    pub fn n_max_for_system(system: u8) -> usize {
        match system {
            1 => N_MAX_S1,
            2 => N_MAX_S2,
            _ => N_MAX_S3,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: longest path computation
// ---------------------------------------------------------------------------

/// Compute the longest path length in edges for a DAG using DP on topological order.
///
/// For cyclic graphs, falls back to node_count - 1 (conservative upper bound).
/// Returns 0 for empty graphs.
fn longest_path_length(graph: &TopologyGraph) -> usize {
    let inner = graph.inner_graph();
    let n = inner.node_count();
    if n == 0 {
        return 0;
    }

    // Topological sort; if cyclic, use conservative upper bound.
    let topo = match toposort(inner, None) {
        Ok(order) => order,
        Err(_) => return n.saturating_sub(1),
    };

    // DP: dist[node] = longest path ending at node (in edge count).
    let mut dist = vec![0usize; n];
    let mut max_dist = 0usize;

    for &node in &topo {
        for edge in inner.edges_directed(node, petgraph::Direction::Outgoing) {
            let target = edge.target().index();
            let candidate = dist[node.index()] + 1;
            if candidate > dist[target] {
                dist[target] = candidate;
            }
            if dist[target] > max_dist {
                max_dist = dist[target];
            }
        }
    }

    max_dist
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::topology_graph::{TopologyEdge, TopologyNode};

    /// Helper: build a graph with `n` nodes (sequential template) and no edges.
    fn graph_with_nodes(n: usize) -> TopologyGraph {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        for i in 0..n {
            g.add_node(TopologyNode::with_id(
                format!("node_{}", i),
                "worker".into(),
                "test-model".into(),
            ));
        }
        g
    }

    /// Helper: build a fully sequential chain: 0->1->2->...->n-1.
    fn sequential_chain(n: usize) -> TopologyGraph {
        let mut g = graph_with_nodes(n);
        for i in 0..n.saturating_sub(1) {
            g.try_add_edge(i, i + 1, TopologyEdge::control()).unwrap();
        }
        g
    }

    /// Helper: build a parallel fan-out: 0->{1,2,...,n-1}.
    fn parallel_fan(n: usize) -> TopologyGraph {
        let mut g = graph_with_nodes(n);
        for i in 1..n {
            g.try_add_edge(0, i, TopologyEdge::control()).unwrap();
        }
        g
    }

    #[test]
    fn test_empty_graph() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        let density = TopologyDensity::new();
        let score = density.compute(&g, 2);

        // Empty graph: no nodes, no edges.
        assert_eq!(score.s_node, 1.0, "exp(0) = 1.0 for 0 nodes");
        assert_eq!(score.s_edge, 1.0, "no edges possible");
        assert_eq!(score.s_depth, 1.0, "fully parallel (vacuously)");
        assert!(!score.over_budget, "0 nodes never over budget");
        assert_eq!(score.n_max, 7, "S2 n_max = 7");
    }

    #[test]
    fn test_single_node_density() {
        let g = graph_with_nodes(1);
        let density = TopologyDensity::new();
        let score = density.compute(&g, 1);

        // Single node: s_node = exp(-1/4) ~ 0.778, s_edge = 1.0, s_depth = 1 - 0/1 = 1.0.
        let expected_s_node = (-1.0_f32 / 4.0).exp();
        assert!((score.s_node - expected_s_node).abs() < 1e-5);
        assert_eq!(score.s_edge, 1.0, "no edges possible with 1 node");
        // longest_path for 1 node with no edges = 0, so s_depth = 1 - 0/1 = 1.0.
        assert!((score.s_depth - 1.0).abs() < 1e-5);
        assert!(!score.over_budget, "1 node < N_max=4 for S1");
        assert_eq!(score.n_max, 4);
    }

    #[test]
    fn test_dense_graph_penalty() {
        // 4 nodes, fully connected (6 edges for complete graph of 4).
        let mut g = graph_with_nodes(4);
        // Add all 6 directed edges (only forward direction for DAG).
        g.try_add_edge(0, 1, TopologyEdge::control()).unwrap();
        g.try_add_edge(0, 2, TopologyEdge::control()).unwrap();
        g.try_add_edge(0, 3, TopologyEdge::control()).unwrap();
        g.try_add_edge(1, 2, TopologyEdge::control()).unwrap();
        g.try_add_edge(1, 3, TopologyEdge::control()).unwrap();
        g.try_add_edge(2, 3, TopologyEdge::control()).unwrap();

        let density = TopologyDensity::new();
        let score = density.compute(&g, 2);

        // max_edges = 4*3/2 = 6. s_edge = exp(-6/6) = exp(-1) ~ 0.368.
        let expected_s_edge = (-1.0_f32).exp();
        assert!(
            (score.s_edge - expected_s_edge).abs() < 1e-5,
            "dense graph should have s_edge ~ exp(-1), got {}",
            score.s_edge
        );

        // Sparse graph for comparison.
        let sparse = graph_with_nodes(4); // No edges.
        let sparse_score = density.compute(&sparse, 2);
        assert!(
            sparse_score.s_edge > score.s_edge,
            "sparse graph should have higher s_edge (less penalty)"
        );
    }

    #[test]
    fn test_n_max_s1_s2_s3() {
        assert_eq!(TopologyDensity::n_max_for_system(1), 4, "S1 = 4");
        assert_eq!(TopologyDensity::n_max_for_system(2), 7, "S2 = 7");
        assert_eq!(TopologyDensity::n_max_for_system(3), 10, "S3 = 10");
        // Unknown systems default to S3.
        assert_eq!(TopologyDensity::n_max_for_system(0), 10, "unknown -> S3");
        assert_eq!(TopologyDensity::n_max_for_system(255), 10, "unknown -> S3");
    }

    #[test]
    fn test_over_budget_detected() {
        // 5 nodes exceeds S1 N_max=4.
        let g = graph_with_nodes(5);
        let density = TopologyDensity::new();
        let score = density.compute(&g, 1);
        assert!(score.over_budget, "5 nodes > N_max=4 for S1");
        assert_eq!(score.n_max, 4);

        // 4 nodes is exactly at the limit — not over budget.
        let g2 = graph_with_nodes(4);
        let score2 = density.compute(&g2, 1);
        assert!(!score2.over_budget, "4 nodes == N_max=4 for S1, not over");

        // 8 nodes exceeds S2 N_max=7.
        let g3 = graph_with_nodes(8);
        let score3 = density.compute(&g3, 2);
        assert!(score3.over_budget, "8 nodes > N_max=7 for S2");
    }

    #[test]
    fn test_s_complex_normalized() {
        // S_complex should always be in [0, 1] (sigmoid output).
        let density = TopologyDensity::new();

        for n in [0, 1, 3, 7, 15] {
            let g = sequential_chain(n);
            for system in [1, 2, 3] {
                let score = density.compute(&g, system);
                assert!(
                    score.s_complex >= 0.0 && score.s_complex <= 1.0,
                    "s_complex out of range for {} nodes, S{}: {}",
                    n,
                    system,
                    score.s_complex
                );
            }
        }
    }

    #[test]
    fn test_sequential_vs_parallel_depth() {
        let density = TopologyDensity::new();

        // Sequential chain: longest path = n-1, s_depth = 1 - (n-1)/n.
        let seq = sequential_chain(4);
        let seq_score = density.compute(&seq, 2);

        // Parallel fan: longest path = 1 (source->any worker), s_depth = 1 - 1/n.
        let par = parallel_fan(4);
        let par_score = density.compute(&par, 2);

        assert!(
            par_score.s_depth > seq_score.s_depth,
            "parallel ({}) should have higher s_depth than sequential ({})",
            par_score.s_depth,
            seq_score.s_depth
        );
    }

    #[test]
    fn test_longest_path_sequential() {
        // Chain: 0->1->2->3. Longest path = 3 edges.
        let g = sequential_chain(4);
        assert_eq!(longest_path_length(&g), 3);
    }

    #[test]
    fn test_longest_path_parallel() {
        // Fan: 0->{1,2,3}. Longest path = 1 edge.
        let g = parallel_fan(4);
        assert_eq!(longest_path_length(&g), 1);
    }

    #[test]
    fn test_longest_path_empty() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        assert_eq!(longest_path_length(&g), 0);
    }

    #[test]
    fn test_longest_path_disconnected() {
        // 4 nodes, no edges. Longest path = 0.
        let g = graph_with_nodes(4);
        assert_eq!(longest_path_length(&g), 0);
    }
}
