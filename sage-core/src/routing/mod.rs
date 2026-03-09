//! Adaptive Router — learned S1/S2/S3 routing pipeline.
//!
//! Stage 0: `features.rs` — structural feature extraction (keyword-based complexity scoring).
//! Stage 1: `router.rs` — ONNX classifier for S1/S2/S3 (behind `onnx` feature).
//! Stage 2-3: Python-side dynamic routing with feedback.

pub mod features;

#[cfg(feature = "onnx")]
mod router;
#[cfg(feature = "onnx")]
pub use router::AdaptiveRouter;
#[cfg(feature = "onnx")]
pub use router::RoutingResult;
