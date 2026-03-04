use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use ulid::Ulid;
use chrono::{DateTime, Utc};
use arrow::array::{StringBuilder, BooleanArray, TimestampNanosecondArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

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

/// SOTA 2026: Semantic Memory Management Unit (S-MMU)
/// Maps high-level semantic concepts to physical Arrow chunks.
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_id: usize,
    pub start_time: i64,
    pub end_time: i64,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct SemanticMMU {
    pub graph: DiGraph<ChunkMetadata, f32>, // Nodes are chunks, Edges are semantic similarity
    pub chunk_map: HashMap<usize, NodeIndex>,
    next_chunk_id: usize,
}

impl SemanticMMU {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            chunk_map: HashMap::new(),
            next_chunk_id: 0,
        }
    }

    pub fn register_chunk(&mut self, start_time: i64, end_time: i64, summary: &str) -> usize {
        let id = self.next_chunk_id;
        self.next_chunk_id += 1;

        let meta = ChunkMetadata {
            chunk_id: id,
            start_time,
            end_time,
            summary: summary.to_string(),
        };

        let node_idx = self.graph.add_node(meta);
        self.chunk_map.insert(id, node_idx);
        
        // Link to previous chunk (temporal edge)
        if id > 0 {
            if let Some(&prev_idx) = self.chunk_map.get(&(id - 1)) {
                self.graph.add_edge(prev_idx, node_idx, 1.0);
            }
        }
        id
    }
}

/// In-memory working memory for a single agent execution.
/// ASI Upgrade: TierMem Architecture (Active Buffer + Immutable Arrow Chunks + S-MMU)
#[pyclass]
#[derive(Clone)]
pub struct WorkingMemory {
    #[pyo3(get)]
    pub agent_id: String,
    #[pyo3(get)]
    pub parent_id: Option<String>,
    // Tier 1: Fast append-only buffer for Python agents
    active_buffer: Vec<MemoryEvent>,
    // Tier 2: Zero-copy immutable column storage
    arrow_chunks: Vec<Arc<RecordBatch>>,
    // Cognitive Router
    smmu: SemanticMMU,
    children: Vec<String>,
}

#[pymethods]
impl WorkingMemory {
    #[new]
    #[pyo3(signature = (agent_id, parent_id=None))]
    pub fn py_new(agent_id: String, parent_id: Option<String>) -> Self {
        Self::new(agent_id, parent_id)
    }

    /// Add an event to the active buffer (O(1))
    pub fn add_event(&mut self, event_type: &str, content: &str) -> String {
        let event = MemoryEvent::new(event_type, content);
        let id = event.id.clone();
        self.active_buffer.push(event);
        id
    }

    /// Total event count (active + compacted)
    pub fn event_count(&self) -> usize {
        let compacted_len: usize = self.arrow_chunks.iter().map(|batch| batch.num_rows()).sum();
        compacted_len + self.active_buffer.len()
    }

    /// Get an event by ID (only searches active buffer for now)
    pub fn get_event(&self, id: &str) -> Option<MemoryEvent> {
        self.active_buffer.iter().find(|e| e.id == id).cloned()
    }

    /// Compress old events in the active buffer
    pub fn compress_old_events(&mut self, keep_recent: usize, summary_text: &str) {
        if self.active_buffer.len() <= keep_recent {
            return;
        }
        let split_point = self.active_buffer.len() - keep_recent;
        let recent: Vec<MemoryEvent> = self.active_buffer.drain(split_point..).collect();
        self.active_buffer.clear();
        self.active_buffer.push(MemoryEvent::summary(summary_text));
        self.active_buffer.extend(recent);
    }

    /// Register a child agent
    pub fn add_child_agent(&mut self, child_id: String) {
        self.children.push(child_id);
    }

    /// Get child agent IDs
    pub fn child_agents(&self) -> Vec<String> {
        self.children.clone()
    }

    /// ASI Feature: Compact active buffer into an immutable Arrow RecordBatch
    /// and register it in the S-MMU graph.
    pub fn compact_to_arrow(&mut self) -> PyResult<usize> {
        if self.active_buffer.is_empty() {
            return Ok(0);
        }

        let len = self.active_buffer.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("agent_id", DataType::Utf8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("id", DataType::Utf8, false),
            Field::new("event_type", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Nanosecond, None), false), 
            Field::new("is_summary", DataType::Boolean, false),
        ]));

        let mut agent_id_builder = StringBuilder::with_capacity(len, len * self.agent_id.len());
        let mut parent_id_builder = StringBuilder::with_capacity(len, len * self.parent_id.as_ref().map_or(0, |s| s.len()));
        let mut id_builder = StringBuilder::with_capacity(len, len * 26);
        let mut type_builder = StringBuilder::with_capacity(len, len * 10);
        let mut content_builder = StringBuilder::with_capacity(len, len * 50);
        let mut ts_builder = TimestampNanosecondArray::builder(len);
        let mut summary_builder = BooleanArray::builder(len);

        let mut start_time = i64::MAX;
        let mut end_time = i64::MIN;

        for e in &self.active_buffer {
            agent_id_builder.append_value(&self.agent_id);
            if let Some(ref pid) = self.parent_id {
                parent_id_builder.append_value(pid);
            } else {
                parent_id_builder.append_null();
            }
            id_builder.append_value(&e.id);
            type_builder.append_value(&e.event_type);
            content_builder.append_value(&e.content);
            
            let ts = e.timestamp.timestamp_nanos_opt().unwrap_or(0);
            ts_builder.append_value(ts);
            start_time = start_time.min(ts);
            end_time = end_time.max(ts);
            
            summary_builder.append_value(e.is_summary);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(agent_id_builder.finish()),
                Arc::new(parent_id_builder.finish()),
                Arc::new(id_builder.finish()),
                Arc::new(type_builder.finish()),
                Arc::new(content_builder.finish()),
                Arc::new(ts_builder.finish()),
                Arc::new(summary_builder.finish()),
            ],
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        self.arrow_chunks.push(Arc::new(batch));
        
        // Register in S-MMU
        let chunk_id = self.smmu.register_chunk(start_time, end_time, "Compacted context block");
        
        // Clear active buffer
        self.active_buffer.clear();
        
        Ok(chunk_id)
    }

    /// Legacy support: returns the active buffer for simple tools.
    pub fn recent_events(&self, n: usize) -> Vec<MemoryEvent> {
        let start = self.active_buffer.len().saturating_sub(n);
        self.active_buffer[start..].iter().cloned().collect()
    }

    /// Export the latest compacted Arrow chunk to Python (Zero-Copy)
    pub fn get_latest_arrow_chunk(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(batch) = self.arrow_chunks.last() {
            let py_batch = pyo3_arrow::PyRecordBatch::new(batch.as_ref().clone());
            Ok(Some(py_batch.to_pyarrow(py)?))
        } else {
            Ok(None)
        }
    }
}

impl WorkingMemory {
    pub fn new(agent_id: String, parent_id: Option<String>) -> Self {
        Self {
            agent_id,
            parent_id,
            active_buffer: Vec::new(),
            arrow_chunks: Vec::new(),
            smmu: SemanticMMU::new(),
            children: Vec::new(),
        }
    }
}
