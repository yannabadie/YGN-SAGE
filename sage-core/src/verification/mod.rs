//! Verification modules for topology and contract checking.
//!
//! - **ltl**: LTL model checking for TopologyGraph (always available, uses petgraph)
//! - **smt**: OxiZ-backed SMT verification (behind `smt` feature flag)

// LTL model checking — always available (uses petgraph, no OxiZ dependency)
pub mod ltl;

// OxiZ SMT verification — behind `smt` feature flag
#[cfg(feature = "smt")]
mod smt;

#[cfg(feature = "smt")]
pub use smt::*;
