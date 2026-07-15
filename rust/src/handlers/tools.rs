use anyhow::Result;
use serde_json::{Value, json};
use sqlx::PgPool;

use crate::constants::tool_definitions;

const NOT_PROXYABLE: [&str; 2] = ["cuba_call", "cuba_tools"];

pub async fn handle_tools(_pool: &PgPool, args: Value) -> Result<Value> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    let detail = args
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("summary");

    let all = tool_definitions();
    let matched: Vec<&Value> = all
        .iter()
        .filter(|t| {
            if query.is_empty() {
                return true;
            }
            let name = t.get("name").and_then(Value::as_str).unwrap_or("");
            let desc = t.get("description").and_then(Value::as_str).unwrap_or("");
            name.to_lowercase().contains(&query) || desc.to_lowercase().contains(&query)
        })
        .collect();

    let tools: Vec<Value> = match detail {
        "names" => matched
            .iter()
            .filter_map(|t| t.get("name").cloned())
            .collect(),
        "full" => matched.iter().map(|t| (*t).clone()).collect(),
        _ => matched
            .iter()
            .map(|t| {
                json!({
                    "name": t.get("name"),
                    "description": t.get("description"),
                })
            })
            .collect(),
    };

    Ok(json!({
        "count": tools.len(),
        "total_tools": all.len(),
        "detail": detail,
        "tools": tools,
        "hint": "Invocá cualquiera con cuba_call {\"tool\": \"<nombre>\", \"args\": {...}}. \
                 Pedí detail=\"full\" para ver el esquema exacto de argumentos antes de llamarla.",
    }))
}

pub async fn handle_call(pool: &PgPool, args: Value) -> Result<Value> {
    let tool = args
        .get("tool")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("falta 'tool': el nombre de la herramienta a invocar"))?;

    if NOT_PROXYABLE.contains(&tool) {
        anyhow::bail!("{tool} no se puede invocar a través de cuba_call");
    }

    let known = tool_definitions()
        .iter()
        .any(|t| t.get("name").and_then(Value::as_str) == Some(tool));
    if !known {
        anyhow::bail!(
            "herramienta desconocida: {tool}. Buscá la correcta con \
             cuba_tools {{\"query\": \"...\"}}"
        );
    }

    let inner = args
        .get("args")
        .cloned()
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

    Box::pin(super::dispatch(pool, tool, inner)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn every_advertised_tool_is_reachable_through_the_proxy() {
        let names: Vec<String> = tool_definitions()
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str).map(String::from))
            .collect();
        for n in &names {
            if NOT_PROXYABLE.contains(&n.as_str()) {
                continue;
            }
            assert!(
                crate::handlers::is_known_tool(n),
                "{n} está anunciada pero el dispatcher no la conoce"
            );
        }
    }

    #[test]
    fn the_proxy_cannot_call_itself() {
        assert!(NOT_PROXYABLE.contains(&"cuba_call"));
        assert!(NOT_PROXYABLE.contains(&"cuba_tools"));
    }
}
