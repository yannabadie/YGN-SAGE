//! TopologyExecutor — dual-mode execution engine for topology graphs.
//!
//! Two execution modes:
//! - **Static**: Deterministic O(V+E) topological ordering for acyclic topologies
//!   (Sequential, Parallel, Hierarchical, Brainstorming).
//! - **Dynamic**: Gate-based readiness polling with loop support for cyclic topologies
//!   (AVR, Hub, Debate, SelfMoA).

use super::topology_graph::*;
use petgraph::visit::EdgeRef;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// ExecutionMode
// ---------------------------------------------------------------------------

/// Selects between static (DAG) and dynamic (gate-based) scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Deterministic O(V+E) topological ordering.
    /// Used for Sequential, Parallel, Hierarchical, Brainstorming.
    Static,
    /// Gate-based readiness polling with loop support.
    /// Used for AVR, Hub, Debate, SelfMoA.
    Dynamic,
}

// ---------------------------------------------------------------------------
// NodeStatus
// ---------------------------------------------------------------------------

/// Per-node execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Skipped,
}

// ---------------------------------------------------------------------------
// TopologyExecutor
// ---------------------------------------------------------------------------

/// Dual-mode topology executor.
///
/// Manages node scheduling for both static (DAG-based) and dynamic (gate-based)
/// execution of topology graphs.
pub struct TopologyExecutor {
    mode: ExecutionMode,
    node_status: Vec<NodeStatus>,
    iteration_count: u32,
    max_iterations: u32,
}

impl TopologyExecutor {
    /// Default safety limit for iteration count.
    const DEFAULT_MAX_ITERATIONS: u32 = 1000;

    /// Create a new executor for the given graph.
    ///
    /// Auto-selects execution mode based on the graph's template type.
    pub fn new(graph: &TopologyGraph) -> Self {
        let mode = Self::mode_for(graph.template());
        let node_count = graph.node_count();
        info!(
            mode = ?mode,
            node_count = node_count,
            template = graph.template().as_str(),
            "TopologyExecutor created"
        );
        Self {
            mode,
            node_status: vec![NodeStatus::Pending; node_count],
            iteration_count: 0,
            max_iterations: Self::DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Determine the execution mode for a given topology template.
    pub fn mode_for(template: TopologyTemplate) -> ExecutionMode {
        match template {
            TopologyTemplate::Sequential
            | TopologyTemplate::Parallel
            | TopologyTemplate::Hierarchical
            | TopologyTemplate::Brainstorming => ExecutionMode::Static,
            TopologyTemplate::AVR
            | TopologyTemplate::Hub
            | TopologyTemplate::Debate
            | TopologyTemplate::SelfMoA => ExecutionMode::Dynamic,
        }
    }

    /// Return the executor's current mode.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Return the next wave of ready nodes.
    ///
    /// - **Static mode**: Returns nodes whose ALL predecessors are `Completed`,
    ///   using petgraph `toposort()`. First call returns entry nodes.
    /// - **Dynamic mode**: Returns nodes where ALL incoming **open-gate** control
    ///   edges come from `Completed` nodes. Closed-gate edges are ignored.
    ///
    /// Returns an empty `Vec` when done or when `max_iterations` is exceeded.
    pub fn next_ready(&mut self, graph: &TopologyGraph) -> Vec<usize> {
        if self.is_done() {
            debug!("All nodes completed or skipped — no more ready nodes");
            return Vec::new();
        }
        if self.is_max_iterations_exceeded() {
            warn!(
                iteration_count = self.iteration_count,
                max_iterations = self.max_iterations,
                "Max iterations exceeded — halting"
            );
            return Vec::new();
        }

        self.iteration_count += 1;

        let ready = match self.mode {
            ExecutionMode::Static => self.next_ready_static(graph),
            ExecutionMode::Dynamic => self.next_ready_dynamic(graph),
        };

        // Mark discovered nodes as Ready
        for &idx in &ready {
            if self.node_status[idx] == NodeStatus::Pending {
                self.node_status[idx] = NodeStatus::Ready;
            }
        }

        debug!(mode = ?self.mode, ready_count = ready.len(), iteration = self.iteration_count, "next_ready");
        ready
    }

    /// Static mode: use topological order to find the next wave of executable nodes.
    fn next_ready_static(&self, graph: &TopologyGraph) -> Vec<usize> {
        let inner = graph.inner_graph();
        let mut ready = Vec::new();

        for node_idx in inner.node_indices() {
            let idx = node_idx.index();
            // Only consider Pending nodes
            if self.node_status[idx] != NodeStatus::Pending {
                continue;
            }
            // Check that ALL predecessors (across all edge types) are Completed
            let all_preds_done = inner
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .all(|edge| {
                    let src = edge.source().index();
                    matches!(
                        self.node_status[src],
                        NodeStatus::Completed | NodeStatus::Skipped
                    )
                });
            if all_preds_done {
                ready.push(idx);
            }
        }

        ready
    }

    /// Dynamic mode: gate-aware readiness — only open-gate control edges matter.
    fn next_ready_dynamic(&self, graph: &TopologyGraph) -> Vec<usize> {
        let inner = graph.inner_graph();
        let mut ready = Vec::new();

        for node_idx in inner.node_indices() {
            let idx = node_idx.index();
            // Only consider Pending nodes
            if self.node_status[idx] != NodeStatus::Pending {
                continue;
            }

            // Collect all incoming control edges that have an open gate
            let open_control_edges: Vec<_> = inner
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .filter(|edge| {
                    let ew = edge.weight();
                    ew.typed_edge_type() == EdgeType::Control && ew.typed_gate() == Gate::Open
                })
                .collect();

            // If there are no open control edges at all, this node is an entry
            // node (no control dependencies) — it is ready if it has no incoming
            // edges of any kind, OR if all non-control/closed-gate deps are met.
            // For simplicity: a node with zero open-gate control predecessors is
            // ready only if it truly has no incoming edges (entry node) or all
            // incoming open control edges are satisfied.
            if open_control_edges.is_empty() {
                // Entry node or all incoming control edges are closed-gate.
                // Check if the node has ANY incoming edges at all.
                let has_incoming = inner
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .next()
                    .is_some();
                if !has_incoming {
                    // True entry node — always ready
                    ready.push(idx);
                } else {
                    // All incoming control edges are closed-gate (or non-control).
                    // The node is ready because its active control deps are empty.
                    ready.push(idx);
                }
            } else {
                // Has open-gate control predecessors — all must be Completed
                let all_open_done = open_control_edges.iter().all(|edge| {
                    let src = edge.source().index();
                    matches!(
                        self.node_status[src],
                        NodeStatus::Completed | NodeStatus::Skipped
                    )
                });
                if all_open_done {
                    ready.push(idx);
                }
            }
        }

        ready
    }

    /// Mark a node as Completed.
    pub fn mark_completed(&mut self, node_index: usize) {
        if node_index < self.node_status.len() {
            debug!(node_index, "Marking node Completed");
            self.node_status[node_index] = NodeStatus::Completed;
        } else {
            warn!(node_index, len = self.node_status.len(), "Node index out of range");
        }
    }

    /// Mark a node as Running.
    pub fn mark_running(&mut self, node_index: usize) {
        if node_index < self.node_status.len() {
            debug!(node_index, "Marking node Running");
            self.node_status[node_index] = NodeStatus::Running;
        } else {
            warn!(node_index, len = self.node_status.len(), "Node index out of range");
        }
    }

    /// Mark a node as Skipped (for closed-gate branches).
    pub fn mark_skipped(&mut self, node_index: usize) {
        if node_index < self.node_status.len() {
            debug!(node_index, "Marking node Skipped");
            self.node_status[node_index] = NodeStatus::Skipped;
        } else {
            warn!(node_index, len = self.node_status.len(), "Node index out of range");
        }
    }

    /// Open a gate on a control edge between two nodes.
    ///
    /// This enables the target node to become ready once the source completes.
    /// Used for loop re-entry in AVR pattern (reviewer -> actor back-edge).
    pub fn open_gate(&self, graph: &mut TopologyGraph, from: usize, to: usize) {
        let inner = graph.inner_graph_mut();
        for edge_idx in inner.edge_indices() {
            if let Some((src, tgt)) = inner.edge_endpoints(edge_idx) {
                if src.index() == from && tgt.index() == to {
                    let edge = inner.edge_weight_mut(edge_idx).unwrap();
                    if edge.typed_edge_type() == EdgeType::Control {
                        debug!(from, to, "Opening gate on control edge");
                        edge.set_gate(Gate::Open);
                        return;
                    }
                }
            }
        }
        warn!(from, to, "No control edge found between nodes for open_gate");
    }

    /// Close a gate on a control edge between two nodes.
    ///
    /// This prevents the target node from requiring the source's completion.
    pub fn close_gate(&self, graph: &mut TopologyGraph, from: usize, to: usize) {
        let inner = graph.inner_graph_mut();
        for edge_idx in inner.edge_indices() {
            if let Some((src, tgt)) = inner.edge_endpoints(edge_idx) {
                if src.index() == from && tgt.index() == to {
                    let edge = inner.edge_weight_mut(edge_idx).unwrap();
                    if edge.typed_edge_type() == EdgeType::Control {
                        debug!(from, to, "Closing gate on control edge");
                        edge.set_gate(Gate::Closed);
                        return;
                    }
                }
            }
        }
        warn!(from, to, "No control edge found between nodes for close_gate");
    }

    /// True if all nodes are either Completed or Skipped.
    pub fn is_done(&self) -> bool {
        self.node_status
            .iter()
            .all(|s| matches!(s, NodeStatus::Completed | NodeStatus::Skipped))
    }

    /// True if `iteration_count >= max_iterations`.
    pub fn is_max_iterations_exceeded(&self) -> bool {
        self.iteration_count >= self.max_iterations
    }

    /// Get the status of a specific node.
    pub fn node_status(&self, index: usize) -> Option<NodeStatus> {
        self.node_status.get(index).copied()
    }

    /// Reset all node statuses to Pending and iteration count to 0.
    pub fn reset(&mut self) {
        info!("Resetting TopologyExecutor");
        for status in &mut self.node_status {
            *status = NodeStatus::Pending;
        }
        self.iteration_count = 0;
    }

    /// Get the current iteration count.
    pub fn iteration_count(&self) -> u32 {
        self.iteration_count
    }

    /// Set the maximum iterations safety limit.
    pub fn set_max_iterations(&mut self, max: u32) {
        self.max_iterations = max;
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;

    #[test]
    fn test_mode_for_static_templates() {
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::Sequential),
            ExecutionMode::Static
        );
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::Parallel),
            ExecutionMode::Static
        );
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::Hierarchical),
            ExecutionMode::Static
        );
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::Brainstorming),
            ExecutionMode::Static
        );
    }

    #[test]
    fn test_mode_for_dynamic_templates() {
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::AVR),
            ExecutionMode::Dynamic
        );
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::Hub),
            ExecutionMode::Dynamic
        );
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::Debate),
            ExecutionMode::Dynamic
        );
        assert_eq!(
            TopologyExecutor::mode_for(TopologyTemplate::SelfMoA),
            ExecutionMode::Dynamic
        );
    }

    #[test]
    fn test_new_initializes_all_pending() {
        let graph = templates::sequential("model");
        let exec = TopologyExecutor::new(&graph);
        assert_eq!(exec.mode(), ExecutionMode::Static);
        for i in 0..graph.node_count() {
            assert_eq!(exec.node_status(i), Some(NodeStatus::Pending));
        }
    }

    #[test]
    fn test_static_sequential_order() {
        // Sequential: A(0) -> B(1) -> C(2)
        let graph = templates::sequential("model");
        let mut exec = TopologyExecutor::new(&graph);

        // First wave: only entry node (0)
        let wave1 = exec.next_ready(&graph);
        assert_eq!(wave1, vec![0]);

        exec.mark_running(0);
        exec.mark_completed(0);

        // Second wave: node 1
        let wave2 = exec.next_ready(&graph);
        assert_eq!(wave2, vec![1]);

        exec.mark_running(1);
        exec.mark_completed(1);

        // Third wave: node 2
        let wave3 = exec.next_ready(&graph);
        assert_eq!(wave3, vec![2]);

        exec.mark_running(2);
        exec.mark_completed(2);

        // Done
        assert!(exec.is_done());
        let wave4 = exec.next_ready(&graph);
        assert!(wave4.is_empty());
    }

    #[test]
    fn test_static_parallel_workers_simultaneous() {
        // Parallel: source(0) -> [w0(1), w1(2), w2(3)] -> aggregator(4)
        let graph = templates::parallel("model", 3);
        let mut exec = TopologyExecutor::new(&graph);

        // First wave: source
        let wave1 = exec.next_ready(&graph);
        assert_eq!(wave1, vec![0]);
        exec.mark_completed(0);

        // Second wave: all 3 workers ready simultaneously
        let mut wave2 = exec.next_ready(&graph);
        wave2.sort();
        assert_eq!(wave2, vec![1, 2, 3]);

        for &w in &wave2 {
            exec.mark_completed(w);
        }

        // Third wave: aggregator
        let wave3 = exec.next_ready(&graph);
        assert_eq!(wave3, vec![4]);
        exec.mark_completed(4);

        assert!(exec.is_done());
    }
}
