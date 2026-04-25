//! Rust OTel bridge for B1.b. Conditionally compiles `init` (real) or
//! `stub` (no-op) based on the `otel` feature flag.

#[cfg(feature = "otel")]
mod init;
#[cfg(feature = "otel")]
pub use init::*;

#[cfg(not(feature = "otel"))]
mod stub;
#[cfg(not(feature = "otel"))]
pub use stub::*;
