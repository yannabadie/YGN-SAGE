//! Memory Module — Multi-tier agent memory architecture.
//!
//! - `event.rs`      — MemoryEvent (immutable event record)
//! - `arrow_tier.rs`  — Arrow compaction (Tier 2 storage)
//! - `smmu.rs`        — Multi-View S-MMU with 4 orthogonal graphs
//! - `paging.rs`      — Semantic paging / eviction policy

pub mod arrow_tier;
pub mod event;
pub mod paging;
pub mod rag_cache;
pub mod smmu;

// Re-export public types so that `memory::MemoryEvent` and `memory::WorkingMemory`
// continue to work without changing import paths.
pub use event::MemoryEvent;

use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use std::sync::Arc;

use smmu::MultiViewMMU;

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
    smmu: MultiViewMMU,
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
    /// and register it in the S-MMU graph (backward-compatible, no metadata).
    pub fn compact_to_arrow(&mut self) -> PyResult<usize> {
        self.compact_to_arrow_with_meta(vec![], None, None, None)
    }

    /// Compact with full metadata: keywords, embedding, optional parent chunk, and summary.
    #[pyo3(signature = (keywords, embedding=None, parent_chunk_id=None, summary=None))]
    pub fn compact_to_arrow_with_meta(
        &mut self,
        keywords: Vec<String>,
        embedding: Option<Vec<f32>>,
        parent_chunk_id: Option<usize>,
        summary: Option<String>,
    ) -> PyResult<usize> {
        if self.active_buffer.is_empty() {
            return Ok(0);
        }
        let chunk_id = arrow_tier::compact_buffer_to_arrow(
            &self.agent_id,
            &self.parent_id,
            &self.active_buffer,
            &mut self.arrow_chunks,
            &mut self.smmu,
            keywords,
            embedding,
            parent_chunk_id,
            summary,
        )?;
        self.active_buffer.clear();
        Ok(chunk_id)
    }

    /// Retrieve relevant chunks for the given active chunk, walking up to
    /// `max_hops` in the multi-view graph.
    ///
    /// `weights`: optional `[temporal, semantic, causal, entity]` factors.
    /// Returns list of `(chunk_id, score)` tuples, descending by score.
    #[pyo3(signature = (active_chunk_id, max_hops, weights=None))]
    pub fn retrieve_relevant_chunks(
        &self,
        active_chunk_id: usize,
        max_hops: usize,
        weights: Option<[f32; 4]>,
    ) -> Vec<(usize, f32)> {
        let w = weights.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        self.smmu.retrieve_relevant(active_chunk_id, max_hops, w)
    }

    /// Get page-out (eviction) candidates — chunks least relevant to the
    /// active chunk.
    #[pyo3(signature = (active_chunk_id, max_hops, budget))]
    pub fn get_page_out_candidates(
        &self,
        active_chunk_id: usize,
        max_hops: usize,
        budget: usize,
    ) -> Vec<usize> {
        paging::page_out_candidates(&self.smmu, active_chunk_id, max_hops, budget)
    }

    /// Number of chunks registered in the S-MMU.
    pub fn smmu_chunk_count(&self) -> usize {
        self.smmu.chunk_count()
    }

    /// Legacy support: returns the active buffer for simple tools.
    pub fn recent_events(&self, n: usize) -> Vec<MemoryEvent> {
        let start = self.active_buffer.len().saturating_sub(n);
        self.active_buffer[start..].to_vec()
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
            smmu: MultiViewMMU::new(),
            children: Vec::new(),
        }
    }
}
