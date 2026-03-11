// ebpf: disabled until solana_rbpf is re-added (Phase 2)
// Its build script cross-compiles for BPF target, breaking CI on Ubuntu.
// #[cfg(feature = "sandbox")]
// pub mod ebpf;
#[cfg(feature = "sandbox")]
pub mod wasm;
// z3_validator: implemented in Python (sage.sandbox.z3_validator) using z3-solver package.
// Rust implementation removed — z3 crate was never added to Cargo.toml dependencies.
#[cfg(feature = "tool-executor")]
pub mod subprocess;
#[cfg(feature = "tool-executor")]
pub mod tool_executor;
#[cfg(feature = "tool-executor")]
pub mod validator;
