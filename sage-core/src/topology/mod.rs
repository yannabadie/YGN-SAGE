//! TopologyGraph — unified IR for multi-agent topologies.
//!
//! Wraps `petgraph::DiGraph` with typed nodes (roles, capabilities, budgets)
//! and three-flow edges (Control, Message, State).

pub mod topology_graph;
pub use topology_graph::*;
