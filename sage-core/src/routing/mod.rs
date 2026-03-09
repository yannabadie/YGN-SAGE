//! Adaptive Router — learned S1/S2/S3 routing pipeline.
//!
//! Stage 0: `features.rs` — structural feature extraction (keyword-based complexity scoring).
//! Stage 1: (future) `router.rs` — ONNX logistic model for S1/S2/S3 classification.
//! Stage 2-3: Python-side dynamic routing with feedback.

pub mod features;
