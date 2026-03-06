//! Arrow Tier: Zero-copy immutable column storage for compacted memory events.
//!
//! Converts the active buffer (Vec<MemoryEvent>) into an Apache Arrow RecordBatch
//! and registers the resulting chunk in the S-MMU.

use std::sync::Arc;

use arrow::array::{BooleanArray, StringBuilder, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;

use super::event::MemoryEvent;
use super::smmu::MultiViewMMU;

/// Build the canonical Arrow schema for memory event batches.
fn event_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("id", DataType::Utf8, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("is_summary", DataType::Boolean, false),
    ]))
}

/// Compact the `active_buffer` into an immutable Arrow RecordBatch, register it
/// in the S-MMU, and return the assigned chunk ID.
///
/// The caller must clear the active buffer after a successful call.
#[allow(clippy::too_many_arguments)]
pub fn compact_buffer_to_arrow(
    agent_id: &str,
    parent_id: &Option<String>,
    active_buffer: &[MemoryEvent],
    arrow_chunks: &mut Vec<Arc<RecordBatch>>,
    smmu: &mut MultiViewMMU,
    keywords: Vec<String>,
    embedding: Option<Vec<f32>>,
    parent_chunk_id: Option<usize>,
) -> PyResult<usize> {
    if active_buffer.is_empty() {
        return Ok(0);
    }

    let len = active_buffer.len();
    let schema = event_schema();

    let mut agent_id_builder = StringBuilder::with_capacity(len, len * agent_id.len());
    let mut parent_id_builder =
        StringBuilder::with_capacity(len, len * parent_id.as_ref().map_or(0, |s| s.len()));
    let mut id_builder = StringBuilder::with_capacity(len, len * 26);
    let mut type_builder = StringBuilder::with_capacity(len, len * 10);
    let mut content_builder = StringBuilder::with_capacity(len, len * 50);
    let mut ts_builder = TimestampNanosecondArray::builder(len);
    let mut summary_builder = BooleanArray::builder(len);

    let mut start_time = i64::MAX;
    let mut end_time = i64::MIN;

    for e in active_buffer {
        agent_id_builder.append_value(agent_id);
        if let Some(ref pid) = parent_id {
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
        schema,
        vec![
            Arc::new(agent_id_builder.finish()),
            Arc::new(parent_id_builder.finish()),
            Arc::new(id_builder.finish()),
            Arc::new(type_builder.finish()),
            Arc::new(content_builder.finish()),
            Arc::new(ts_builder.finish()),
            Arc::new(summary_builder.finish()),
        ],
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e))
    })?;

    arrow_chunks.push(Arc::new(batch));

    // Register in S-MMU
    let chunk_id = smmu.register_chunk(
        start_time,
        end_time,
        "Compacted context block",
        keywords,
        embedding,
        parent_chunk_id,
    );

    Ok(chunk_id)
}
