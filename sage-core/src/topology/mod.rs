//! TopologyGraph — unified IR for multi-agent topologies.
//!
//! Wraps `petgraph::DiGraph` with typed nodes (roles, capabilities, budgets)
//! and three-flow edges (Control, Message, State).

pub mod llm_synthesis;
pub mod map_elites;
pub mod mutations;
pub mod smmu_bridge;
pub mod templates;
pub mod topology_graph;
pub mod verifier;
pub use topology_graph::*;
