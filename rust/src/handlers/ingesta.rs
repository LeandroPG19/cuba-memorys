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

    let supersede_conflicts = args
        .get("supersede_conflicts")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let ops = if supersede_conflicts {
        resolve_conflicts(pool, &items).await
    } else {
        crate::cognitive::memory_op::OpBreakdown::default()
    };

    let ingest_args = serde_json::json!({ "action": "ingest", "items": items });
    let result = ingest(pool, &ingest_args).await?;

    let mut response = result;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("action".to_string(), serde_json::json!("auto_extract"));
        obj.insert("extracted".to_string(), serde_json::json!(extracted));
        if supersede_conflicts {
            obj.insert("superseded".to_string(), serde_json::json!(ops.update));
            obj.insert("operations".to_string(), ops.to_json());
        }
    }
    Ok(response)
}

async fn resolve_conflicts(
    pool: &PgPool,
    items: &[Value],
) -> crate::cognitive::memory_op::OpBreakdown {
    use crate::cognitive::memory_op::{MemoryOp, OpBreakdown};

    const REL_LO: f64 = 0.30;
    const REL_HI: f64 = 0.85;
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

        let Some((old_id, old_content, sim)) = candidate else {
            ops.record(MemoryOp::Add);
            continue;
        };
        if !(REL_LO..REL_HI).contains(&sim) {
            ops.record(MemoryOp::Add);
            continue;
        }

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

fn parse_extracted_items(reply: &str) -> Vec<Value> {
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

    let batch_args = serde_json::json!({
        "action": "batch_add",
        "observations": observations
    });

    let result = super::cronica::handle(pool, batch_args).await?;

    let mut response = result;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("action".to_string(), serde_json::json!("ingest"));
        obj.insert("skipped_invalid".to_string(), serde_json::json!(skipped));
        obj.insert("total_items".to_string(), serde_json::json!(items.len()));
    }

    Ok(response)
}

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

    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| p.len() > 10)
        .collect();

    if paragraphs.is_empty() {
        return Ok(serde_json::json!({
            "action": "parse",
            "entity_name": entity_name,
            "parsed_count": 0,
            "note": "No substantial paragraphs found (min 10 chars after split on double-newline)"
        }));
    }

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
