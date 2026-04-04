//! Handler: cuba_decreto — Architecture/design decisions.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let entity_name = "architecture_decisions";

    match action {
        "record" => {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("Untitled");
            let context = args.get("context").and_then(|v| v.as_str()).unwrap_or("");
            let alternatives = args
                .get("alternatives")
                .cloned()
                .unwrap_or(Value::Array(vec![]));
            let chosen = args.get("chosen").and_then(|v| v.as_str()).unwrap_or("");
            let rationale = args.get("rationale").and_then(|v| v.as_str()).unwrap_or("");

            let decision_content = format!(
                "**{}**\nContext: {}\nAlternatives: {}\nChosen: {}\nRationale: {}",
                title, context, alternatives, chosen, rationale
            );

            // Ensure entity exists
            sqlx::query("INSERT INTO brain_entities (name, entity_type) VALUES ($1, 'concept') ON CONFLICT (name) DO NOTHING")
                .bind(entity_name).execute(pool).await?;
            let entity_id: (uuid::Uuid,) =
                sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
                    .bind(entity_name)
                    .fetch_one(pool)
                    .await?;

            let row: (uuid::Uuid,) = sqlx::query_as(
                "INSERT INTO brain_observations (entity_id, content, observation_type, source) VALUES ($1, $2, 'decision', 'agent') RETURNING id"
            ).bind(entity_id.0).bind(&decision_content).fetch_one(pool).await?;

            Ok(serde_json::json!({"action": "record", "id": row.0.to_string(), "title": title}))
        }
        "query" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let decisions: Vec<(uuid::Uuid, String, f64)> = sqlx::query_as(
                "SELECT id, content, similarity(content, $1)::float8 AS sim FROM brain_observations
                 WHERE observation_type = 'decision' AND similarity(content, $1) > 0.2
                 ORDER BY sim DESC LIMIT 10",
            )
            .bind(query)
            .fetch_all(pool)
            .await?;
            let results: Vec<Value> = decisions.iter().map(|(id, content, sim)| {
                serde_json::json!({"id": id.to_string(), "content": content, "similarity": sim})
            }).collect();
            Ok(serde_json::json!({"action": "query", "results": results, "count": results.len()}))
        }
        "list" => {
            let decisions: Vec<(uuid::Uuid, String)> = sqlx::query_as(
                "SELECT id, content FROM brain_observations WHERE observation_type = 'decision' ORDER BY created_at DESC LIMIT 20"
            ).fetch_all(pool).await?;
            let list: Vec<Value> = decisions
                .iter()
                .map(|(id, c)| serde_json::json!({"id": id.to_string(), "content": c}))
                .collect();
            Ok(serde_json::json!({"action": "list", "decisions": list, "count": list.len()}))
        }
        _ => anyhow::bail!("Invalid action: {action}"),
    }
}
