//! MCP JSON-RPC protocol handler — stdin/stdout transport.
//!
//! FIX B4: REM session protection uses exact entity_id match (not ILIKE).
//! FIX V2: tokio::time::timeout(30s) on every handler dispatch.

use crate::db;
use crate::handlers;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// MCP protocol timeout per handler request.
const HANDLER_TIMEOUT: Duration = Duration::from_secs(30);

/// REM sleep interval (4 hours).
const REM_INTERVAL: Duration = Duration::from_secs(4 * 3600);

// ── JSON-RPC Types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

// ── MCP Protocol Constants ───────────────────────────────────────

/// Server capabilities advertised during initialize.
fn server_info() -> Value {
    serde_json::json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": { "listChanged": false }
        },
        "serverInfo": {
            "name": "cuba-memorys",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

// ── Main Protocol Loop ──────────────────────────────────────────

/// Run the MCP protocol over stdin/stdout.
pub async fn run_mcp() -> Result<()> {
    // Connect to database (credentials from env, never hardcoded)
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set (e.g., postgresql://user:pass@host:port/db)");

    let pool = db::create_pool(&database_url).await?;

    // Spawn REM daemon
    let rem_pool = pool.clone();
    let _rem_handle = tokio::spawn(async move {
        rem_daemon(rem_pool).await;
    });

    // Read JSON-RPC from stdin, write to stdout
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    tracing::info!("MCP protocol ready on stdin/stdout");

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "invalid JSON-RPC request");
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: "Parse error".into(),
                        data: Some(Value::String(e.to_string())),
                    }),
                };
                let response_bytes = serde_json::to_vec(&error_response)?;
                stdout.write_all(&response_bytes).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        // Notifications have no id — must NOT send a response (MCP spec).
        let is_notification = request.id.is_none();
        let id = request.id.clone().unwrap_or(Value::Null);
        let response = handle_request(&pool, request).await;

        // Skip response for notifications (Python parity: returns None).
        if is_notification {
            if let Err(e) = &response {
                tracing::warn!(error = %e, "notification handler error (suppressed)");
            }
            continue;
        }

        let json_response = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: match &response {
                Ok(v) => Some(v.clone()),
                Err(_) => None,
            },
            error: match response {
                Ok(_) => None,
                Err(e) => Some(JsonRpcError {
                    code: -32603,
                    message: e.to_string(),
                    data: None,
                }),
            },
        };

        let response_bytes = serde_json::to_vec(&json_response)?;
        stdout.write_all(&response_bytes).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
}

/// Route a JSON-RPC request to the appropriate handler.
async fn handle_request(pool: &PgPool, request: JsonRpcRequest) -> Result<Value> {
    match request.method.as_str() {
        // MCP lifecycle
        "initialize" => Ok(server_info()),
        "initialized" | "notifications/initialized" => Ok(Value::Null),
        "notifications/cancelled" => Ok(Value::Null),
        "ping" => Ok(serde_json::json!({})),

        // Tool listing
        "tools/list" => Ok(serde_json::json!({
            "tools": crate::constants::tool_definitions()
        })),

        // Tool execution with timeout (V2)
        "tools/call" => {
            let params = request.params.unwrap_or(Value::Null);
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tracing::info!(tool = %tool_name, "executing tool");

            // V2: Timeout wrapper
            match tokio::time::timeout(
                HANDLER_TIMEOUT,
                handlers::dispatch(pool, &tool_name, arguments),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    tracing::error!(tool = %tool_name, "handler timed out after 30s");
                    anyhow::bail!("Handler timed out after 30 seconds")
                }
            }
        }

        _ => {
            tracing::warn!(method = %request.method, "unknown method");
            anyhow::bail!("Unknown method: {}", request.method)
        }
    }
}

// ── REM Sleep Daemon ────────────────────────────────────────────

/// Background daemon that runs memory consolidation every 4 hours.
///
/// FIX B4: Session protection uses exact entity associations, not ILIKE.
async fn rem_daemon(pool: PgPool) {
    let mut interval = tokio::time::interval(REM_INTERVAL);
    // Skip the first tick (fires immediately)
    interval.tick().await;

    loop {
        interval.tick().await;
        tracing::info!("REM sleep cycle starting");

        // Run consolidation in a spawned task (I/O-bound: DB queries + graph ops)
        let pool_clone = pool.clone();
        let result = tokio::spawn(async move { run_rem_consolidation(&pool_clone).await }).await;

        match result {
            Ok(Ok(())) => tracing::info!("REM sleep cycle completed"),
            Ok(Err(e)) => tracing::error!(error = %e, "REM consolidation error"),
            Err(e) => tracing::error!(error = %e, "REM task panicked"),
        }
    }
}

/// Execute REM sleep consolidation steps.
///
/// Steps: FSRS decay → PageRank → Neighbor Diffusion → TF-IDF rebuild.
async fn run_rem_consolidation(pool: &PgPool) -> Result<()> {
    // 1. Get active session goals (for session protection - FIX B4)
    let active_session: Option<(uuid::Uuid, Vec<String>)> = {
        let row: Option<(uuid::Uuid, serde_json::Value)> = sqlx::query_as(
            "SELECT id, goals FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
        )
        .fetch_optional(pool)
        .await?;

        row.map(|(id, goals)| {
            let goal_list: Vec<String> = serde_json::from_value(goals).unwrap_or_default();
            (id, goal_list)
        })
    };

    // 2. Get entities to protect from decay (FIX B4: exact entity_ids, not ILIKE)
    let protected_entity_ids: Vec<uuid::Uuid> = if let Some((_session_id, _)) = &active_session {
        // Protect entities accessed during active session (last 8h)
        sqlx::query_scalar(
            "SELECT DISTINCT entity_id FROM brain_observations
             WHERE created_at > NOW() - INTERVAL '8 hours'",
        )
        .fetch_all(pool)
        .await?
    } else {
        vec![]
    };

    // 3. FSRS decay (batch UPDATE — V1)
    let decayed = crate::cognitive::fsrs::batch_decay(pool, &protected_entity_ids).await?;
    tracing::info!(decayed_count = decayed, "FSRS decay applied");

    // 4. PageRank (batch — P1 fix)
    let ranked = crate::graph::pagerank::compute_and_store(pool).await?;
    tracing::info!(ranked_count = ranked, "PageRank updated");

    // 5. Neighbor Diffusion (§C — Collins & Loftus 1975)
    crate::cognitive::spreading::neighbor_diffusion(pool).await?;
    tracing::info!("neighbor diffusion completed");

    // 6. TF-IDF index: BM25 index is in-memory, rebuilt on demand per REM cycle
    // (no persistent index to rebuild — Bm25Index lives in handler state)
    tracing::info!("REM consolidation complete (TF-IDF: in-memory, on-demand)");

    Ok(())
}
