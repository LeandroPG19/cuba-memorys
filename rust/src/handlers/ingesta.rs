//! Handler: cuba_ingesta — Bulk knowledge ingestion.
//!
//! Accepts structured arrays or long text for batch observation creation.
//! Internally uses the same dedup and embedding pipeline as cronica.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "ingest" => ingest(pool, &args).await,
        "parse" => parse(pool, &args).await,
        _ => anyhow::bail!("Invalid action: {action}. Use ingest/parse"),
    }
}

/// Ingest an array of structured items.
///
/// Each item: { entity_name, content, observation_type? }
/// Internally validates, ensures entities, and batch-inserts with dedup.
async fn ingest(pool: &PgPool, args: &Value) -> Result<Value> {
    let items = args
        .get("items")
        .and_then(|v| v.as_array())
        .context("'items' array is required for ingest action")?;

    if items.is_empty() {
        anyhow::bail!("items array is empty");
    }
    if items.len() > 200 {
        anyhow::bail!(
            "ingest limit is 200 items per call (got {})",
            items.len()
        );
    }

    // Build batch_add-compatible observations array
    let observations: Vec<Value> = items
        .iter()
        .filter_map(|item| {
            let entity_name = item.get("entity_name").and_then(|v| v.as_str())?;
            let content = item.get("content").and_then(|v| v.as_str())?;
            if entity_name.is_empty() || content.is_empty() {
                return None;
            }
            let obs_type = item
                .get("observation_type")
                .and_then(|v| v.as_str())
                .unwrap_or("fact");
            let source = item
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("agent");
            Some(serde_json::json!({
                "entity_name": entity_name,
                "content": content,
                "observation_type": obs_type,
                "source": source
            }))
        })
        .collect();

    let skipped = items.len() - observations.len();

    // Delegate to cronica::handle with batch_add action
    let batch_args = serde_json::json!({
        "action": "batch_add",
        "observations": observations
    });

    let result = super::cronica::handle(pool, batch_args).await?;

    // Augment response with ingesta metadata
    let mut response = result;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("action".to_string(), serde_json::json!("ingest"));
        obj.insert("skipped_invalid".to_string(), serde_json::json!(skipped));
        obj.insert(
            "total_items".to_string(),
            serde_json::json!(items.len()),
        );
    }

    Ok(response)
}

/// Parse long text into observations by splitting on double-newlines.
///
/// Each paragraph becomes an observation attached to the specified entity.
/// Auto-classifies paragraphs by keyword heuristics.
async fn parse(pool: &PgPool, args: &Value) -> Result<Value> {
    let entity_name = args
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required for parse action");
    }

    let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
    if text.is_empty() {
        anyhow::bail!("text is required for parse action");
    }

    // Split on double-newline (paragraph boundaries)
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| p.len() > 10) // Skip very short fragments
        .collect();

    if paragraphs.is_empty() {
        return Ok(serde_json::json!({
            "action": "parse",
            "entity_name": entity_name,
            "parsed_count": 0,
            "note": "No substantial paragraphs found (min 10 chars after split on double-newline)"
        }));
    }

    // Build items with auto-classification
    let items: Vec<Value> = paragraphs
        .iter()
        .map(|p| {
            let obs_type = classify_paragraph(p);
            serde_json::json!({
                "entity_name": entity_name,
                "content": p,
                "observation_type": obs_type,
                "source": "agent"
            })
        })
        .collect();

    let parsed_count = items.len();

    // Delegate to ingest
    let ingest_args = serde_json::json!({
        "action": "ingest",
        "items": items
    });

    let result = ingest(pool, &ingest_args).await?;

    let mut response = result;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("action".to_string(), serde_json::json!("parse"));
        obj.insert("parsed_count".to_string(), serde_json::json!(parsed_count));
    }

    Ok(response)
}

/// Auto-classify a paragraph by keyword heuristics.
fn classify_paragraph(text: &str) -> &'static str {
    let lower = text.to_lowercase();
    if lower.contains("decided") || lower.contains("decision") || lower.contains("chose") {
        "decision"
    } else if lower.contains("learned") || lower.contains("lesson") || lower.contains("takeaway") {
        "lesson"
    } else if lower.contains("error") || lower.contains("bug") || lower.contains("failed") {
        "error"
    } else if lower.contains("fix") || lower.contains("solution") || lower.contains("resolved") {
        "solution"
    } else if lower.contains("prefer") || lower.contains("preference") || lower.contains("always use") {
        "preference"
    } else {
        "fact"
    }
}
