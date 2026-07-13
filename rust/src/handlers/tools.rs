//! `cuba_tools` and `cuba_call` — progressive disclosure of the tool surface.
//!
//! All 25 tool schemas ride in the agent's context on every single session:
//! 23,563 characters of JSON, most of it describing maintenance surfaces (REM
//! cycles, the audit log, GDPR erasure, Bayesian calibration) that an agent never
//! reaches for mid-task. It pays that on every conversation, whether it searches
//! memory once or not at all.
//!
//! Anthropic measured the general form of this problem and the fix: presenting
//! tools on demand rather than up-front took one workflow from 150,000 tokens to
//! 2,000 — a 98.7% reduction ("Code execution with MCP", 2026). Claude Code
//! already does it to its own tool list; that is what "deferred tools" are.
//!
//! Two tools implement it here:
//!
//! - `cuba_tools` — search the catalogue. Returns names, or names plus
//!   descriptions, or the full JSON Schema, so the agent pulls in exactly the
//!   detail it needs and no more.
//! - `cuba_call` — invoke any tool by name, through the same dispatcher the MCP
//!   protocol uses.
//!
//! With `CUBA_TOOL_PROFILE=lean` the server advertises the day-to-day core plus
//! these two, and everything else stays one `cuba_tools` call away. **No function
//! is removed** — the full 25 remain callable. Only their schemas stop being
//! pre-loaded.

use anyhow::Result;
use serde_json::{Value, json};
use sqlx::PgPool;

use crate::constants::tool_definitions;

/// Tools that must never be reachable through `cuba_call`.
///
/// `cuba_call` calling itself is an unbounded recursion with a database
/// connection at the bottom of it.
const NOT_PROXYABLE: [&str; 2] = ["cuba_call", "cuba_tools"];

/// `cuba_tools` — find tools and load their schemas on demand.
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
            // Substring over name and description: the agent is searching for a
            // capability ("audit", "decay", "contradiction"), not a exact tool id.
            name.to_lowercase().contains(&query) || desc.to_lowercase().contains(&query)
        })
        .collect();

    let tools: Vec<Value> = match detail {
        // Cheapest: just the names. Enough to decide whether to look closer.
        "names" => matched
            .iter()
            .filter_map(|t| t.get("name").cloned())
            .collect(),
        // The whole schema — what you need to actually call it correctly.
        "full" => matched.iter().map(|t| (*t).clone()).collect(),
        // Default: name + description. One line each.
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

/// `cuba_call` — invoke a tool by name.
pub async fn handle_call(pool: &PgPool, args: Value) -> Result<Value> {
    let tool = args
        .get("tool")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("falta 'tool': el nombre de la herramienta a invocar"))?;

    if NOT_PROXYABLE.contains(&tool) {
        anyhow::bail!("{tool} no se puede invocar a través de cuba_call");
    }

    // Reject unknown names here rather than letting the dispatcher's fallback
    // produce a vaguer error — and tell the agent how to find the right one.
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

    // The same dispatcher tools/call uses: a proxied call and a direct call take
    // the identical path, so they cannot drift into behaving differently.
    Box::pin(super::dispatch(pool, tool, inner)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn every_advertised_tool_is_reachable_through_the_proxy() {
        // The whole promise of the lean profile: nothing is removed, only
        // deferred. If a tool were unreachable through cuba_call, the profile
        // would be silently amputating the server.
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
