//! Adaptive Router — learned S1/S2/S3 routing pipeline.
//!
//! Stage 0: `features.rs` — structural feature extraction (keyword-based complexity scoring).
//! Stage 2-3: Python-side dynamic routing with feedback.

pub mod bandit;
pub mod features;
pub mod knn;
pub mod model_assigner;
pub mod model_card;
pub mod model_registry;
pub mod quality;
pub mod system_router;

#[cfg(feature = "cognitive")]
pub mod persistence;
