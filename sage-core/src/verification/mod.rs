//! Verification modules for topology and contract checking.
//!
//! - **ltl**: graph property checking for TopologyGraph (always available, uses petgraph)
//! - **smt**: OxiZ-backed SMT verification (behind `smt` feature flag)

// Graph property checking — always available (uses petgraph, no OxiZ dependency)
pub mod ltl;
pub use ltl::{GraphPropertyChecker, LtlResult};
#[allow(deprecated)]
pub use ltl::LtlVerifier;

// OxiZ SMT verification — behind `smt` feature flag
#[cfg(feature = "smt")]
mod smt;

#[cfg(feature = "smt")]
pub use smt::*;

// Quality labeler — requires both SMT (OxiZ) and tool-executor (tree-sitter)
#[cfg(all(feature = "smt", feature = "tool-executor"))]
pub mod quality_labeler;
