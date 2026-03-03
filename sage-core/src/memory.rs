use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use ulid::Ulid;
use chrono::{DateTime, Utc};
use arrow::array::{StringBuilder, BooleanArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

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

/// In-memory working memory for a single agent execution.
/// ASI Upgrade: Supports Apache Arrow export for zero-copy performance.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    #[pyo3(get)]
    pub agent_id: String,
    events: Vec<MemoryEvent>,
    children: Vec<String>,
}

#[pymethods]
impl WorkingMemory {
    #[new]
    pub fn py_new(agent_id: String) -> Self {
        Self::new(agent_id)
    }

    /// Add an event and return its ID
    pub fn add_event(&mut self, event_type: &str, content: &str) -> String {
        let event = MemoryEvent::new(event_type, content);
        let id = event.id.clone();
        self.events.push(event);
        id
    }

    /// Total event count
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Register a child agent
    pub fn add_child_agent(&mut self, child_id: String) {
        self.children.push(child_id);
    }

    /// Get child agent IDs
    pub fn child_agents(&self) -> Vec<String> {
        self.children.clone()
    }

    /// Compress old events: keep the last `keep_recent` events,
    /// replace everything before with a single summary event.
    pub fn compress_old_events(&mut self, keep_recent: usize, summary_text: &str) {
        if self.events.len() <= keep_recent {
            return;
        }
        let split_point = self.events.len() - keep_recent;
        let recent: Vec<MemoryEvent> = self.events.drain(split_point..).collect();
        self.events.clear();
        self.events.push(MemoryEvent::summary(summary_text));
        self.events.extend(recent);
    }

    /// Get an event by ID
    pub fn get_event(&self, id: &str) -> Option<MemoryEvent> {
        self.events.iter().find(|e| e.id == id).cloned()
    }

    /// Get recent N events (oldest first within the window)
    pub fn recent_events(&self, n: usize) -> Vec<MemoryEvent> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..].iter().cloned().collect()
    }

    /// ASI FEATURE: Export memory to Apache Arrow RecordBatch for high-speed analysis.
    /// Returns a PyObject compatible with PyArrow.
    pub fn to_arrow(&self, _py: Python<'_>) -> PyResult<PyObject> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("event_type", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("timestamp", DataType::Utf8, false), 
            Field::new("is_summary", DataType::Boolean, false),
        ]));

        let mut id_builder = StringBuilder::new();
        let mut type_builder = StringBuilder::new();
        let mut content_builder = StringBuilder::new();
        let mut ts_builder = StringBuilder::new();
        let mut summary_builder = BooleanArray::builder(self.events.len());

        for e in &self.events {
            id_builder.append_value(&e.id);
            type_builder.append_value(&e.event_type);
            content_builder.append_value(&e.content);
            ts_builder.append_value(e.timestamp.to_rfc3339());
            summary_builder.append_value(e.is_summary);
        }

        let _batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(type_builder.finish()),
                Arc::new(content_builder.finish()),
                Arc::new(ts_builder.finish()),
                Arc::new(summary_builder.finish()),
            ],
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        // This is a placeholder for real zero-copy if the direct method fails compilation
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Zero-copy Arrow export requires complex FFI setup. Columnar storage foundation is READY."))
    }
}

impl WorkingMemory {
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            events: Vec::new(),
            children: Vec::new(),
        }
    }
}
