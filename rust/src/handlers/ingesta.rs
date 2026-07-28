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

    let hint = args
        .get("entity_hint")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let prompt = build_extraction_prompt(text, hint);

    let Some((reply, backend)) = extraction_reply(&prompt).await else {
        return Ok(serde_json::json!({
            "action": "auto_extract",
            "extracted": 0,
            "added": 0,
            "degraded": true,
            "note": "no LLM reachable: the client advertises no MCP sampling capability and no \
                     local CLI was found on PATH. Install the Claude Code CLI (or set \
                     CUBA_JUEZ_CLI), or use action='parse' for a heuristic paragraph split."
        }));
    };

    let (items, relations) = parse_extraction_reply(&reply);
    let extracted = items.len();
    if extracted == 0 && relations.is_empty() {
        return Ok(serde_json::json!({
            "action": "auto_extract",
            "extracted": 0,
            "added": 0,
            "relations_linked": 0,
            "backend": backend,
            "note": "the model returned no durable facts worth remembering from this text"
        }));
    }

    let relations_linked = link_entities(pool, &relations).await;

    if extracted == 0 {
        return Ok(serde_json::json!({
            "action": "auto_extract",
            "extracted": 0,
            "added": 0,
            "relations_linked": relations_linked,
            "backend": backend,
            "note": "no durable facts, but the text did support graph relations"
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

    let untrusted = args
        .get("untrusted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let items = if untrusted {
        items
            .into_iter()
            .map(|mut i| {
                i["trust"] = serde_json::json!(crate::core::trust::QUARANTINED);
                i
            })
            .collect()
    } else {
        items
    };

    let ingest_args = serde_json::json!({ "action": "ingest", "items": items });
    let result = ingest(pool, &ingest_args).await?;

    let mut response = result;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("action".to_string(), serde_json::json!("auto_extract"));
        obj.insert("extracted".to_string(), serde_json::json!(extracted));
        obj.insert(
            "relations_linked".to_string(),
            serde_json::json!(relations_linked),
        );
        obj.insert("backend".to_string(), serde_json::json!(backend));
        if supersede_conflicts {
            obj.insert("superseded".to_string(), serde_json::json!(ops.update));
            obj.insert("operations".to_string(), ops.to_json());
        }
    }
    Ok(response)
}

const EXTRACTION_MAX_TOKENS: u32 = 1024;
const EXTRACTION_BUDGET_RATIO: f32 = 0.6;

pub fn extraction_budget() -> std::time::Duration {
    crate::protocol::handler_timeout().mul_f32(EXTRACTION_BUDGET_RATIO)
}

async fn extraction_reply(prompt: &str) -> Option<(String, &'static str)> {
    if crate::protocol::client_supports_sampling() {
        match crate::protocol::request_sampling_max(prompt, EXTRACTION_MAX_TOKENS).await {
            Ok(reply) => return Some((reply, "mcp_sampling")),
            Err(why) => {
                tracing::warn!(error = %why, "MCP sampling failed, falling back to a local LLM CLI")
            }
        }
    }

    let backend = crate::cognitive::judge::resolve_offline_llm()?;
    let name = backend.backend_name();
    match tokio::time::timeout(extraction_budget(), backend.run_prompt(prompt)).await {
        Ok(Ok(raw)) => Some((crate::cognitive::judge::unwrap_cli_reply(&raw), name)),
        Ok(Err(why)) => {
            tracing::warn!(error = %why, backend = name, "LLM extraction failed");
            None
        }
        Err(_) => {
            tracing::warn!(
                backend = name,
                budget_secs = extraction_budget().as_secs(),
                "LLM extraction exceeded its share of the handler timeout"
            );
            None
        }
    }
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
    let rel_types = crate::constants::VALID_RELATION_TYPES.join(", ");
    format!(
        "You extract durable, reusable memories from an AI coding agent's work log.\n\
         From the text below, extract two things:\n\n\
         1. FACTS worth remembering across sessions — decisions made, lessons learned, \
         errors and their fixes, stable preferences, key technical facts. Ignore chit-chat, \
         transient state, and anything that will not matter next week.\n\
         2. RELATIONS between the entities those facts are about — how they connect. \
         Only state a relation the text actually supports; do not invent plausible-sounding \
         links. Both endpoints should be things the text genuinely talks about.{hint_line}\n\n\
         Return STRICT JSON, exactly this shape:\n\
         {{\"facts\": [{{\"entity_name\": <the project/technology/concept the fact is about>, \
         \"content\": <one self-contained sentence>, \
         \"observation_type\": one of [fact, decision, lesson, preference, error, solution, context, tool_usage]}}],\n\
         \x20\"relations\": [{{\"from\": <entity name>, \"to\": <entity name>, \
         \"relation_type\": one of [{rel_types}]}}]}}\n\n\
         Use [] for either list when there is nothing worth recording. \
         No prose, no markdown — just the JSON object.\n\n\
         TEXT:\n{text}"
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedRelation {
    pub from: String,
    pub to: String,
    pub relation_type: String,
}

fn parse_extraction_reply(reply: &str) -> (Vec<Value>, Vec<ExtractedRelation>) {
    if let Some(obj) = extract_json_object(reply) {
        let facts = obj.get("facts").map(items_from_array).unwrap_or_default();
        let relations = obj
            .get("relations")
            .map(relations_from_array)
            .unwrap_or_default();
        return (facts, relations);
    }
    (parse_extracted_items(reply), Vec::new())
}

fn extract_json_object(reply: &str) -> Option<Value> {
    let open = reply.find('{')?;
    if reply.find('[').is_some_and(|bracket| bracket < open) {
        return None;
    }
    let close = reply.rfind('}')?;
    if close <= open {
        return None;
    }
    let parsed: Value = serde_json::from_str(&reply[open..=close]).ok()?;
    parsed.is_object().then_some(parsed)
}

fn relations_from_array(value: &Value) -> Vec<ExtractedRelation> {
    let Some(arr) = value.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|item| {
            let from = item.get("from").and_then(|v| v.as_str())?.trim();
            let to = item.get("to").and_then(|v| v.as_str())?.trim();
            let relation_type = item
                .get("relation_type")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|t| crate::constants::VALID_RELATION_TYPES.contains(t))
                .unwrap_or("related_to");
            if from.is_empty() || to.is_empty() || from.eq_ignore_ascii_case(to) {
                return None;
            }
            Some(ExtractedRelation {
                from: from.to_string(),
                to: to.to_string(),
                relation_type: relation_type.to_string(),
            })
        })
        .collect()
}

pub async fn link_relations_from_reply(pool: &PgPool, reply: &str) -> Result<u32> {
    let (_, relations) = parse_extraction_reply(reply);
    Ok(link_entities(pool, &relations).await)
}

async fn link_entities(pool: &PgPool, relations: &[ExtractedRelation]) -> u32 {
    let project_id = crate::project::current_project_id(pool)
        .await
        .ok()
        .flatten();
    let mut created = 0u32;

    for rel in relations {
        let from_id = upsert_entity(pool, &rel.from, project_id).await;
        let to_id = upsert_entity(pool, &rel.to, project_id).await;
        let (Some(from_id), Some(to_id)) = (from_id, to_id) else {
            continue;
        };

        let done = sqlx::query(
            "INSERT INTO brain_relations
                (from_entity, to_entity, relation_type, project_id, provenance)
             VALUES ($1, $2, $3, $4, 'inferred')
             ON CONFLICT (from_entity, to_entity, relation_type)
             DO UPDATE SET strength = LEAST(brain_relations.strength + 0.1, 1.0),
                           last_traversed = NOW()",
        )
        .bind(from_id)
        .bind(to_id)
        .bind(&rel.relation_type)
        .bind(project_id)
        .execute(pool)
        .await;

        match done {
            Ok(r) if r.rows_affected() > 0 => created += 1,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, from = %rel.from, to = %rel.to, "auto_extract: could not write relation")
            }
        }
    }
    created
}

async fn upsert_entity(
    pool: &PgPool,
    name: &str,
    project_id: Option<uuid::Uuid>,
) -> Option<uuid::Uuid> {
    let row: Result<(uuid::Uuid,), _> = sqlx::query_as(
        "INSERT INTO brain_entities (name, entity_type, project_id)
         VALUES ($1, 'concept', $2)
         ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
         RETURNING id",
    )
    .bind(name)
    .bind(project_id)
    .fetch_one(pool)
    .await;
    match row {
        Ok((id,)) => Some(id),
        Err(e) => {
            tracing::warn!(error = %e, entity = %name, "auto_extract: could not upsert entity");
            None
        }
    }
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
    items_from_array(&parsed)
}

fn items_from_array(value: &Value) -> Vec<Value> {
    let Some(arr) = value.as_array() else {
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
            let mut row = serde_json::json!({
                "entity_name": entity_name,
                "content": content,
                "observation_type": obs_type,
                "source": source
            });
            if let Some(trust) = item.get("trust").and_then(|v| v.as_str()) {
                row["trust"] = serde_json::json!(trust);
            }
            Some(row)
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

    use super::parse_extraction_reply;

    #[test]
    fn parses_facts_and_relations_from_the_object_shape() {
        let reply = r#"{
            "facts": [{"entity_name":"cuba-memorys","content":"uses pgvector","observation_type":"fact"}],
            "relations": [{"from":"cuba-memorys","to":"PostgreSQL","relation_type":"depends_on"}]
        }"#;
        let (facts, relations) = parse_extraction_reply(reply);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0]["entity_name"], "cuba-memorys");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].from, "cuba-memorys");
        assert_eq!(relations[0].to, "PostgreSQL");
        assert_eq!(relations[0].relation_type, "depends_on");
    }

    #[test]
    fn a_bare_array_still_yields_its_facts() {
        let reply = r#"[{"entity_name":"X","content":"did Y","observation_type":"decision"}]"#;
        let (facts, relations) = parse_extraction_reply(reply);
        assert_eq!(facts.len(), 1, "old-format facts must still land");
        assert!(relations.is_empty());
    }

    #[test]
    fn an_invalid_relation_type_falls_back_instead_of_dropping_the_edge() {
        let reply =
            r#"{"facts":[],"relations":[{"from":"A","to":"B","relation_type":"invented_type"}]}"#;
        let (_, relations) = parse_extraction_reply(reply);
        assert_eq!(relations.len(), 1);
        assert_eq!(
            relations[0].relation_type, "related_to",
            "an unknown type must degrade to the generic one, not violate the DB CHECK"
        );
    }

    #[test]
    fn self_loops_and_empty_endpoints_are_dropped() {
        let reply = r#"{"facts":[],"relations":[
            {"from":"A","to":"A","relation_type":"uses"},
            {"from":"a","to":"A","relation_type":"uses"},
            {"from":"","to":"B","relation_type":"uses"},
            {"from":"C","to":"","relation_type":"uses"},
            {"from":"D","to":"E","relation_type":"uses"}
        ]}"#;
        let (_, relations) = parse_extraction_reply(reply);
        assert_eq!(relations.len(), 1, "only D->E survives");
        assert_eq!(relations[0].from, "D");
    }

    #[test]
    fn missing_relations_key_is_not_an_error() {
        let reply = r#"{"facts":[{"entity_name":"X","content":"y","observation_type":"fact"}]}"#;
        let (facts, relations) = parse_extraction_reply(reply);
        assert_eq!(facts.len(), 1);
        assert!(relations.is_empty());
    }

    #[test]
    fn prose_and_fences_around_the_object_are_tolerated() {
        let reply = "Here you go:\n```json\n{\"facts\":[],\"relations\":[{\"from\":\"A\",\"to\":\"B\",\"relation_type\":\"causes\"}]}\n```\nDone.";
        let (_, relations) = parse_extraction_reply(reply);
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "causes");
    }

    #[test]
    fn every_relation_type_the_prompt_offers_is_accepted() {
        for t in crate::constants::VALID_RELATION_TYPES {
            let reply = format!(
                r#"{{"facts":[],"relations":[{{"from":"A","to":"B","relation_type":"{t}"}}]}}"#
            );
            let (_, relations) = parse_extraction_reply(&reply);
            assert_eq!(relations.len(), 1, "type {t} was rejected");
            assert_eq!(relations[0].relation_type, *t);
        }
    }
}
