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
        "auto_extract" => auto_extract(pool, &args).await,
        _ => anyhow::bail!("Invalid action: {action}. Use ingest/parse/auto_extract"),
    }
}

/// v0.11: LLM-powered fact extraction (closes the extraction gap; mem0/Zep have
/// it, cuba did not). Instead of the heuristic paragraph split of `parse`, it
/// asks the *calling client's* LLM — via MCP Sampling, so $0 and no API key — to
/// distill the salient, durable facts from a turn/conversation as structured
/// items, then feeds them through the same dedup + PE-gating + embedding
/// pipeline as `ingest`. Additive only: it never deletes.
///
/// Degrades honestly: if the client did not advertise `sampling`, it says so and
/// points at `parse` (the heuristic fallback) rather than silently doing nothing.
async fn auto_extract(pool: &PgPool, args: &Value) -> Result<Value> {
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if text.is_empty() {
        anyhow::bail!("text is required for auto_extract action");
    }

    if !crate::protocol::client_supports_sampling() {
        return Ok(serde_json::json!({
            "action": "auto_extract",
            "extracted": 0,
            "added": 0,
            "degraded": true,
            "note": "client did not advertise MCP sampling capability — cannot call an LLM \
                     to extract. Use action='parse' (heuristic paragraph split) instead, or \
                     connect from a sampling-capable client."
        }));
    }

    let hint = args
        .get("entity_hint")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let prompt = build_extraction_prompt(text, hint);

    // Larger reply budget than the judge — a turn can yield several facts.
    let reply = crate::protocol::request_sampling_max(&prompt, 1024)
        .await
        .context("MCP sampling for auto_extract failed")?;

    let items = parse_extracted_items(&reply);
    let extracted = items.len();
    if extracted == 0 {
        return Ok(serde_json::json!({
            "action": "auto_extract",
            "extracted": 0,
            "added": 0,
            "note": "the model returned no durable facts worth remembering from this text"
        }));
    }

    // Phase 4 (opt-in): knowledge-update. Before ingesting, ask the judge whether
    // each new fact SUPERSEDES an existing, related-but-not-duplicate observation;
    // if so, mark the old one superseded (never deleted). This is exactly the
    // "invalidate on update" that makes Zep/Graphiti win LongMemEval, built on
    // cuba's bitemporal supersede. Default off — pure ADD otherwise.
    let supersede_conflicts = args
        .get("supersede_conflicts")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Phase 1.2: the judge classifies each candidate vs its most similar existing
    // fact into an explicit ADD/UPDATE/DELETE/NOOP operation (the LLM decides, not
    // a cosine threshold). UPDATE supersedes the old row; the new fact is ingested
    // by the pipeline below regardless (Add/Update both keep the new fact).
    let ops = if supersede_conflicts {
        resolve_conflicts(pool, &items).await
    } else {
        crate::cognitive::memory_op::OpBreakdown::default()
    };

    // Reuse the ingest path: validation + dedup + PE-gating + embedding.
    let ingest_args = serde_json::json!({ "action": "ingest", "items": items });
    let result = ingest(pool, &ingest_args).await?;

    let mut response = result;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("action".to_string(), serde_json::json!("auto_extract"));
        obj.insert("extracted".to_string(), serde_json::json!(extracted));
        if supersede_conflicts {
            // Back-compat: `superseded` is the UPDATE count; `operations` is the
            // full ADD/UPDATE/DELETE/NOOP breakdown of the judge's decisions.
            obj.insert("superseded".to_string(), serde_json::json!(ops.update));
            obj.insert("operations".to_string(), ops.to_json());
        }
    }
    Ok(response)
}

/// Phase 1.2 (builds on Phase 4): for each candidate fact, find the most similar
/// existing observation of the same entity in the "related but not a duplicate"
/// band, ask the LLM judge to classify the relationship, and map it to an
/// explicit ADD/UPDATE/DELETE/NOOP operation ([`crate::cognitive::memory_op`]).
/// An UPDATE supersedes the old row (data-safe: only ever sets
/// observation_type='superseded' — the row is retained, never deleted).
///
/// The candidate is always ingested afterwards by the pipeline, so ADD and UPDATE
/// both keep the new fact; the difference is whether the old one is superseded.
/// Best-effort per item. Returns the operation breakdown.
async fn resolve_conflicts(pool: &PgPool, items: &[Value]) -> crate::cognitive::memory_op::OpBreakdown {
    use crate::cognitive::memory_op::{MemoryOp, OpBreakdown};

    // Below the dup threshold (0.85, already blocked) but related enough to
    // possibly conflict. Unrelated facts never reach the judge.
    const REL_LO: f64 = 0.30;
    const REL_HI: f64 = 0.85;
    // Minimum judge confidence to act on (supersede). Below this → Noop.
    const CONF_FLOOR: f64 = 0.5;

    let judge = crate::cognitive::judge::resolve_judge();
    let mut ops = OpBreakdown::default();

    for item in items {
        let (Some(entity_name), Some(content)) = (
            item.get("entity_name").and_then(|v| v.as_str()),
            item.get("content").and_then(|v| v.as_str()),
        ) else {
            continue;
        };

        // Most similar existing, non-superseded observation of this entity.
        let candidate: Option<(uuid::Uuid, String, f64)> = sqlx::query_as(
            "SELECT o.id, o.content, similarity(o.content, $2)::float8 AS sim
             FROM brain_observations o
             JOIN brain_entities e ON e.id = o.entity_id
             WHERE e.name = $1 AND o.observation_type != 'superseded'
             ORDER BY sim DESC
             LIMIT 1",
        )
        .bind(entity_name)
        .bind(content)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();

        // No related prior fact (or out of band) → this is a free ADD.
        let Some((old_id, old_content, sim)) = candidate else {
            ops.record(MemoryOp::Add);
            continue;
        };
        if !(REL_LO..REL_HI).contains(&sim) {
            ops.record(MemoryOp::Add);
            continue; // unrelated, or a near-duplicate the dedup gate handles
        }

        // The LLM judge — not a cosine threshold — decides the operation.
        let op = match judge.judge(content, &old_content).await {
            Ok(j) => MemoryOp::from_judgment(&j.verdict, j.confidence, CONF_FLOOR),
            Err(e) => {
                tracing::warn!(error = %e, "auto_extract: judge failed — treating as NOOP");
                MemoryOp::Noop
            }
        };

        if op.supersedes_old() {
            let done = sqlx::query(
                "UPDATE brain_observations SET observation_type = 'superseded', updated_at = NOW()
                 WHERE id = $1 AND observation_type != 'superseded'",
            )
            .bind(old_id)
            .execute(pool)
            .await;
            match done {
                Ok(r) if r.rows_affected() > 0 => {
                    tracing::info!(old_id = %old_id, op = op.as_str(), "auto_extract superseded a stale observation");
                }
                // Row already superseded by a concurrent op → downgrade to Noop.
                Ok(_) => {
                    ops.record(MemoryOp::Noop);
                    continue;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "auto_extract: supersede failed — treating as NOOP");
                    ops.record(MemoryOp::Noop);
                    continue;
                }
            }
        }
        ops.record(op);
    }
    ops
}

/// Build the extraction instruction for the client LLM. Asks for STRICT JSON so
/// `parse_extracted_items` can recover it even if the model adds prose/fences.
fn build_extraction_prompt(text: &str, hint: &str) -> String {
    let hint_line = if hint.is_empty() {
        String::new()
    } else {
        format!("\nThe main subject is likely: \"{hint}\". Prefer it as entity_name when it fits.")
    };
    format!(
        "You extract durable, reusable memories from an AI coding agent's work log.\n\
         From the text below, extract only facts worth remembering across sessions — \
         decisions made, lessons learned, errors and their fixes, stable preferences, \
         key technical facts. Ignore chit-chat, transient state, and anything that will \
         not matter next week.{hint_line}\n\n\
         Return STRICT JSON: an array of objects, each\n\
         {{\"entity_name\": <the project/technology/concept the fact is about>, \
         \"content\": <one self-contained sentence>, \
         \"observation_type\": one of [fact, decision, lesson, preference, error, solution, context, tool_usage]}}\n\
         Return [] if nothing is worth remembering. No prose, no markdown — just the JSON array.\n\n\
         TEXT:\n{text}"
    )
}

/// Recover the JSON array from an LLM reply that may wrap it in ```json fences
/// or surrounding prose. Returns ingest-ready items (invalid rows dropped).
fn parse_extracted_items(reply: &str) -> Vec<Value> {
    // Isolate the outermost [...] span so stray prose/fences don't break parsing.
    let slice = match (reply.find('['), reply.rfind(']')) {
        (Some(a), Some(b)) if b > a => &reply[a..=b],
        _ => return Vec::new(),
    };
    let parsed: Value = match serde_json::from_str(slice) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(arr) = parsed.as_array() else {
        return Vec::new();
    };

    const VALID_TYPES: &[&str] = &[
        "fact",
        "decision",
        "lesson",
        "preference",
        "error",
        "solution",
        "context",
        "tool_usage",
    ];

    arr.iter()
        .filter_map(|item| {
            let entity_name = item.get("entity_name").and_then(|v| v.as_str())?.trim();
            let content = item.get("content").and_then(|v| v.as_str())?.trim();
            if entity_name.is_empty() || content.is_empty() {
                return None;
            }
            let obs_type = item
                .get("observation_type")
                .and_then(|v| v.as_str())
                .filter(|t| VALID_TYPES.contains(t))
                .unwrap_or("fact");
            Some(serde_json::json!({
                "entity_name": entity_name,
                "content": content,
                "observation_type": obs_type,
                "source": "inference"
            }))
        })
        .collect()
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
        anyhow::bail!("ingest limit is 200 items per call (got {})", items.len());
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
        obj.insert("total_items".to_string(), serde_json::json!(items.len()));
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
    } else if lower.contains("prefer")
        || lower.contains("preference")
        || lower.contains("always use")
    {
        "preference"
    } else {
        "fact"
    }
}

#[cfg(test)]
mod tests {
    use super::parse_extracted_items;

    #[test]
    fn parses_clean_json_array() {
        let reply = r#"[{"entity_name":"cuba-memorys","content":"uses pgvector","observation_type":"fact"}]"#;
        let items = parse_extracted_items(reply);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["entity_name"], "cuba-memorys");
        assert_eq!(items[0]["observation_type"], "fact");
        assert_eq!(items[0]["source"], "inference");
    }

    #[test]
    fn recovers_json_from_markdown_fences_and_prose() {
        let reply = "Sure! Here are the facts:\n```json\n[{\"entity_name\":\"X\",\"content\":\"did Y\",\"observation_type\":\"decision\"}]\n```\nHope that helps.";
        let items = parse_extracted_items(reply);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["observation_type"], "decision");
    }

    #[test]
    fn drops_invalid_rows_and_normalizes_bad_type() {
        let reply = r#"[
            {"entity_name":"","content":"no entity","observation_type":"fact"},
            {"entity_name":"A","content":"","observation_type":"fact"},
            {"entity_name":"B","content":"good","observation_type":"nonsense"}
        ]"#;
        let items = parse_extracted_items(reply);
        assert_eq!(items.len(), 1, "only the one with entity+content survives");
        assert_eq!(items[0]["entity_name"], "B");
        assert_eq!(
            items[0]["observation_type"], "fact",
            "unknown type falls back to fact"
        );
    }

    #[test]
    fn empty_or_no_array_yields_nothing() {
        assert!(parse_extracted_items("[]").is_empty());
        assert!(parse_extracted_items("I couldn't find any facts.").is_empty());
        assert!(parse_extracted_items("").is_empty());
    }
}
