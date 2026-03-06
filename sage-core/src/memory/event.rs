use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

/// A single event in working memory
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub event_type: String,
    #[pyo3(get)]
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[pyo3(get)]
    pub is_summary: bool,
}

#[pymethods]
impl MemoryEvent {
    #[new]
    pub fn py_new(event_type: &str, content: &str) -> Self {
        Self::new(event_type, content)
    }

    #[getter]
    pub fn timestamp_str(&self) -> String {
        self.timestamp.to_rfc3339()
    }

    #[getter]
    pub fn timestamp_ns(&self) -> i64 {
        self.timestamp.timestamp_nanos_opt().unwrap_or(0)
    }
}

impl MemoryEvent {
    pub fn new(event_type: &str, content: &str) -> Self {
        Self {
            id: Ulid::new().to_string(),
            event_type: event_type.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            is_summary: false,
        }
    }

    pub fn summary(content: &str) -> Self {
        Self {
            id: Ulid::new().to_string(),
            event_type: "summary".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            is_summary: true,
        }
    }
}
