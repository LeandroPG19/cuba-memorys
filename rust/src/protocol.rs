use crate::db;
use crate::handlers;

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use sqlx::PgPool;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn handler_timeout() -> Duration {
    std::env::var("CUBA_HANDLER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30))
}

const REM_INTERVAL: Duration = Duration::from_secs(4 * 3600);

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

static CLIENT_SUPPORTS_SAMPLING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn client_supports_sampling() -> bool {
    CLIENT_SUPPORTS_SAMPLING.load(std::sync::atomic::Ordering::Relaxed)
}

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use tokio::sync::{Mutex, mpsc, oneshot};

static OUTBOUND: OnceLock<mpsc::UnboundedSender<Value>> = OnceLock::new();

static PENDING: OnceLock<Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>>> =
    OnceLock::new();

static NEXT_SERVER_ID: AtomicU64 = AtomicU64::new(1);

static CANCEL_TOKENS: OnceLock<Mutex<std::collections::HashMap<String, CancelToken>>> =
    OnceLock::new();

#[derive(Clone, Default)]
pub struct CancelToken {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl CancelToken {
    pub fn cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn cancel(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn outbound() -> &'static mpsc::UnboundedSender<Value> {
    OUTBOUND
        .get()
        .expect("OUTBOUND not initialized — run_mcp must be invoked first")
}

fn pending() -> &'static Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>> {
    PENDING.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn cancel_tokens() -> &'static Mutex<std::collections::HashMap<String, CancelToken>> {
    CANCEL_TOKENS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

pub async fn request_sampling(prompt: &str) -> anyhow::Result<String> {
    request_sampling_max(prompt, 256).await
}

pub async fn request_sampling_max(prompt: &str, max_tokens: u32) -> anyhow::Result<String> {
    if !client_supports_sampling() {
        anyhow::bail!(
            "client does not advertise capabilities.sampling — \
             set CUBA_JUDGE=claude_cli or rely on auto fallback"
        );
    }

    let id = NEXT_SERVER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (tx, rx) = oneshot::channel::<Value>();
    pending().lock().await.insert(id, tx);

    let req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": format!("srv_{id}"),
        "method": "sampling/createMessage",
        "params": {
            "messages": [
                {
                    "role": "user",
                    "content": { "type": "text", "text": prompt }
                }
            ],
            "maxTokens": max_tokens,
            "modelPreferences": {
                "intelligencePriority": 0.6,
                "speedPriority": 0.4
            }
        }
    });
    outbound()
        .send(req)
        .map_err(|_| anyhow::anyhow!("outbound channel closed"))?;

    let response = match tokio::time::timeout(handler_timeout(), rx).await {
        Ok(Ok(v)) => v,
        Ok(Err(_)) => anyhow::bail!("sampling response channel dropped"),
        Err(_) => {
            pending().lock().await.remove(&id);
            anyhow::bail!("sampling timed out after 30s");
        }
    };

    let text = response
        .get("result")
        .and_then(|r| r.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .or_else(|| {
            response
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
        })
        .ok_or_else(|| anyhow::anyhow!("malformed sampling response: {response}"))?;
    Ok(text.to_string())
}

pub fn notify_progress(token: &str, progress: f64, total: Option<f64>, message: Option<&str>) {
    let mut params = serde_json::json!({
        "progressToken": token,
        "progress": progress,
    });
    if let Some(t) = total {
        params["total"] = serde_json::json!(t);
    }
    if let Some(m) = message {
        params["message"] = serde_json::json!(m);
    }
    let notif = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": params,
    });
    if let Some(tx) = OUTBOUND.get() {
        let _ = tx.send(notif);
    }
}

pub async fn register_cancel_token(request_id: &Value) -> CancelToken {
    let token = CancelToken::default();
    let key = request_id.to_string();
    cancel_tokens().lock().await.insert(key, token.clone());
    token
}

pub async fn unregister_cancel_token(request_id: &Value) {
    let key = request_id.to_string();
    cancel_tokens().lock().await.remove(&key);
}

fn server_info() -> Value {
    serde_json::json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "listChanged": false, "subscribe": false }
        },
        "serverInfo": {
            "name": "cuba-memorys",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

pub async fn run_mcp() -> Result<()> {
    let database_url = crate::setup::resolve_database_url().await;

    // Dying here means never speaking the protocol: the client sees a process
    // that exited, not a server that cannot reach its database — no tool
    // list, no reason. Start anyway on a pool that has not connected yet.
    // `initialize` and `tools/list` touch no database, so the client still
    // gets the full tool list, and each call then fails with the actual error
    // instead of a corpse.
    let (pool, connected) = match db::create_pool(&database_url).await {
        Ok(pool) => (pool, true),
        Err(why) => {
            tracing::warn!(
                error = %format!("{why:#}"),
                "starting without PostgreSQL — tools will fail until it is reachable"
            );
            (db::create_lazy_pool(&database_url)?, false)
        }
    };

    // No database, nothing to consolidate: the REM cycle would just wake up
    // to fail on every tick.
    let rem_handle = connected.then(|| {
        let rem_pool = pool.clone();
        tokio::spawn(async move {
            rem_daemon(rem_pool).await;
        })
    });

    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    OUTBOUND
        .set(out_tx)
        .map_err(|_| anyhow::anyhow!("OUTBOUND already initialized"))?;

    let writer_handle = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(msg) = out_rx.recv().await {
            let mut bytes = match serde_json::to_vec(&msg) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!(error = %e, "failed to serialize outbound");
                    continue;
                }
            };
            bytes.push(b'\n');
            if let Err(e) = stdout.write_all(&bytes).await {
                tracing::error!(error = %e, "stdout write failed — terminating writer");
                break;
            }
            if let Err(e) = stdout.flush().await {
                tracing::error!(error = %e, "stdout flush failed");
                break;
            }
        }
    });

    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();
    let mut in_flight: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();

    tracing::info!("MCP protocol ready on stdin/stdout (V0.9.2 correlator)");

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let parsed: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "invalid JSON-RPC");
                let _ = outbound().send(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": "Parse error", "data": e.to_string() }
                }));
                continue;
            }
        };

        let has_method = parsed.get("method").and_then(|v| v.as_str()).is_some();
        let id_value = parsed.get("id").cloned();

        if !has_method
            && let Some(idv) = id_value.as_ref()
            && let Some(srv_id) = idv
                .as_str()
                .and_then(|s| s.strip_prefix("srv_"))
                .and_then(|s| s.parse::<u64>().ok())
        {
            if let Some(tx) = pending().lock().await.remove(&srv_id) {
                let _ = tx.send(parsed);
            } else {
                tracing::warn!(srv_id, "stale sampling response — no pending entry");
            }
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_value(parsed) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "invalid JSON-RPC envelope");
                let _ = outbound().send(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id_value.unwrap_or(Value::Null),
                    "error": { "code": -32600, "message": "Invalid request", "data": e.to_string() }
                }));
                continue;
            }
        };

        let is_notification = request.id.is_none();
        let req_id = request.id.clone().unwrap_or(Value::Null);
        let pool_clone = pool.clone();

        in_flight.spawn(async move {
            let response = handle_request(&pool_clone, request).await;
            if is_notification {
                if let Err(e) = &response {
                    tracing::warn!(error = %e, "notification handler error (suppressed)");
                }
                return;
            }
            let envelope = match response {
                Ok(v) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": v,
                }),
                Err(e) => {
                    let chain = format!("{e:#}");
                    tracing::error!(error = %chain, "handler failed");
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": { "code": -32603, "message": chain }
                    })
                }
            };
            let _ = outbound().send(envelope);
        });
    }

    tracing::info!(
        "stdin closed — draining {} in-flight handlers",
        in_flight.len()
    );
    let drain = async { while in_flight.join_next().await.is_some() {} };
    let _ = tokio::time::timeout(handler_timeout() * 2, drain).await;

    if let Some(tx) = OUTBOUND.get() {
        let _ = tx;
    }

    if let Some(handle) = rem_handle {
        handle.abort();
    }
    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), writer_handle).await;
    tracing::info!("REM daemon + writer drained, shutting down");

    Ok(())
}

async fn handle_request(pool: &PgPool, request: JsonRpcRequest) -> Result<Value> {
    match request.method.as_str() {
        "initialize" => {
            if let Some(params) = &request.params {
                let sampling_advertised = params
                    .get("capabilities")
                    .and_then(|c| c.get("sampling"))
                    .is_some();
                CLIENT_SUPPORTS_SAMPLING
                    .store(sampling_advertised, std::sync::atomic::Ordering::Relaxed);
                if sampling_advertised {
                    tracing::info!("client supports MCP sampling — judge auto-prefers it");
                }
            }
            Ok(server_info())
        }
        "initialized" | "notifications/initialized" => Ok(Value::Null),
        "notifications/cancelled" => {
            if let Some(params) = &request.params
                && let Some(req_id) = params.get("requestId")
            {
                let key = req_id.to_string();
                if let Some(token) = cancel_tokens().lock().await.get(&key) {
                    token.cancel();
                    tracing::info!(req_id = %key, "client requested cancellation");
                }
            }
            Ok(Value::Null)
        }
        "ping" => Ok(serde_json::json!({})),

        "tools/list" => Ok(serde_json::json!({
            "tools": crate::constants::tools_for_profile()
        })),

        "resources/list" => list_resources(pool).await,
        "resources/read" => {
            let params = request.params.unwrap_or(Value::Null);
            let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");
            read_resource(pool, uri).await
        }

        "tools/call" => {
            let req_id = request.id.clone().unwrap_or(Value::Null);
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

            let token = register_cancel_token(&req_id).await;
            let cancel_fut = async {
                while !token.cancelled() {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            };

            let outcome = tokio::select! {
                result = tokio::time::timeout(
                    handler_timeout(),
                    handlers::dispatch(pool, &tool_name, arguments),
                ) => match result {
                    Ok(r) => r,
                    Err(_) => {
                        tracing::error!(tool = %tool_name, "handler timed out after 30s");
                        Err(anyhow::anyhow!("Handler timed out after 30 seconds"))
                    }
                },
                _ = cancel_fut => {
                    tracing::warn!(tool = %tool_name, req_id = %req_id, "handler cancelled by client");
                    Err(anyhow::anyhow!("Handler cancelled by client"))
                }
            };

            unregister_cancel_token(&req_id).await;
            outcome
        }

        _ => {
            tracing::warn!(method = %request.method, "unknown method");
            anyhow::bail!("Unknown method: {}", request.method)
        }
    }
}

async fn rem_daemon(pool: PgPool) {
    let mut interval = tokio::time::interval(REM_INTERVAL);
    interval.tick().await;

    loop {
        interval.tick().await;
        tracing::info!("REM sleep cycle starting");

        let pool_clone = pool.clone();
        let result = tokio::spawn(async move { run_rem_consolidation(&pool_clone).await }).await;

        match result {
            Ok(Ok(())) => tracing::info!("REM sleep cycle completed"),
            Ok(Err(e)) => tracing::error!(error = %e, "REM consolidation error"),
            Err(e) => tracing::error!(error = %e, "REM task panicked"),
        }
    }
}

async fn run_rem_consolidation(pool: &PgPool) -> Result<()> {
    let active_session: Option<(uuid::Uuid, Vec<String>)> = match crate::session::session_id() {
        Some(sid) => {
            let row: Option<(uuid::Uuid, serde_json::Value)> = sqlx::query_as(
                "SELECT id, goals FROM brain_sessions WHERE id = $1 AND ended_at IS NULL",
            )
            .bind(sid)
            .fetch_optional(pool)
            .await?;

            row.map(|(id, goals)| {
                let goal_list: Vec<String> = serde_json::from_value(goals).unwrap_or_default();
                (id, goal_list)
            })
        }
        None => None,
    };

    let protected_entity_ids: Vec<uuid::Uuid> = if let Some((_session_id, _)) = &active_session {
        sqlx::query_scalar(
            "SELECT DISTINCT entity_id FROM brain_observations
             WHERE created_at > NOW() - INTERVAL '8 hours'",
        )
        .fetch_all(pool)
        .await?
    } else {
        vec![]
    };

    let stratified_decay_sql = "UPDATE brain_observations SET
        importance = GREATEST(
            importance * EXP(-0.693
                * EXTRACT(EPOCH FROM (NOW() - GREATEST(last_accessed, last_decayed_at))) / 86400.0
                / ((CASE observation_type
                        WHEN 'fact'       THEN 30.0
                        WHEN 'preference' THEN 30.0
                        WHEN 'error'      THEN 14.0
                        WHEN 'solution'   THEN 14.0
                        WHEN 'context'    THEN  7.0
                        WHEN 'tool_usage' THEN  7.0
                        ELSE 30.0
                    END) * (1.0 + LN(1.0 + access_count::float8)))
            ),
            0.01
        ),
        last_decayed_at = NOW(),
        updated_at = NOW()
     WHERE observation_type NOT IN ('decision', 'lesson', 'superseded')
       AND last_accessed < NOW() - INTERVAL '1 day'";

    let decayed = if protected_entity_ids.is_empty() {
        sqlx::query(stratified_decay_sql)
            .execute(pool)
            .await?
            .rows_affected()
    } else {
        let sql_with_protection = format!(
            "{} AND entity_id NOT IN (SELECT UNNEST($1::uuid[]))",
            stratified_decay_sql
        );
        sqlx::query(&sql_with_protection)
            .bind(&protected_entity_ids)
            .execute(pool)
            .await?
            .rows_affected()
    };
    tracing::info!(
        decayed_count = decayed,
        "stratified exponential decay applied"
    );

    let episode_decayed = sqlx::query(
        "UPDATE brain_episodes SET
            importance = GREATEST(
                0.5 / POWER(1.0 + 0.1 * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0, 0.5),
                0.01
            )
         WHERE created_at < NOW() - INTERVAL '1 hour'",
    )
    .execute(pool)
    .await
    .map(|r| r.rows_affected())
    .unwrap_or(0);
    tracing::info!(
        episode_decayed_count = episode_decayed,
        "episode power-law decay applied"
    );

    let ranked = crate::graph::pagerank::compute_and_store(pool).await?;
    tracing::info!(ranked_count = ranked, "PageRank updated");

    tracing::info!("REM consolidation complete");

    Ok(())
}

async fn list_resources(pool: &PgPool) -> Result<Value> {
    let mut resources: Vec<Value> = Vec::new();

    let entities: Vec<(String, String)> = sqlx::query_as(
        "SELECT name, entity_type FROM brain_entities
         ORDER BY access_count DESC NULLS LAST, updated_at DESC NULLS LAST
         LIMIT 50",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    for (name, etype) in &entities {
        resources.push(serde_json::json!({
            "uri": format!("cuba://entity/{name}"),
            "name": name,
            "description": format!("{etype} entity with observations and relations"),
            "mimeType": "application/json"
        }));
    }

    let projects: Vec<(String,)> =
        sqlx::query_as("SELECT name FROM brain_projects ORDER BY last_active_at DESC LIMIT 100")
            .fetch_all(pool)
            .await
            .unwrap_or_default();
    for (name,) in &projects {
        resources.push(serde_json::json!({
            "uri": format!("cuba://project/{name}"),
            "name": format!("project: {name}"),
            "description": "project metadata + per-table counts",
            "mimeType": "application/json"
        }));
    }

    let snapshots: Vec<(uuid::Uuid, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
        "SELECT id, created_at FROM brain_compaction_snapshots
         ORDER BY created_at DESC LIMIT 20",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    for (id, ts) in &snapshots {
        resources.push(serde_json::json!({
            "uri": format!("cuba://snapshot/{id}"),
            "name": format!("snapshot {}", &id.to_string()[..8]),
            "description": format!("compaction snapshot from {}", ts.to_rfc3339()),
            "mimeType": "text/markdown"
        }));
    }

    Ok(serde_json::json!({"resources": resources}))
}

async fn read_resource(pool: &PgPool, uri: &str) -> Result<Value> {
    let stripped = uri
        .strip_prefix("cuba://")
        .ok_or_else(|| anyhow::anyhow!("URI must start with cuba://"))?;

    if let Some(name) = stripped.strip_prefix("entity/") {
        let row: Option<(String, String, f64, i32)> = sqlx::query_as(
            "SELECT name, entity_type, importance::float8, access_count
             FROM brain_entities WHERE name = $1",
        )
        .bind(name)
        .fetch_optional(pool)
        .await?;
        let entity = row.ok_or_else(|| anyhow::anyhow!("entity not found: {name}"))?;
        let observations: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as(
            "SELECT o.id, o.content, o.observation_type, o.importance::float8
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE e.name = $1 AND o.observation_type != 'superseded'
             ORDER BY o.importance DESC, o.created_at DESC LIMIT 20",
        )
        .bind(name)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        let body = serde_json::json!({
            "name": entity.0,
            "entity_type": entity.1,
            "importance": entity.2,
            "access_count": entity.3,
            "observations": observations.iter().map(|(id, c, t, i)| serde_json::json!({
                "id": id.to_string(), "content": c, "type": t, "importance": i
            })).collect::<Vec<_>>(),
        });
        return Ok(serde_json::json!({
            "contents": [{"uri": uri, "mimeType": "application/json", "text": body.to_string()}]
        }));
    }

    if let Some(name) = stripped.strip_prefix("project/") {
        let pid: Option<(uuid::Uuid,)> =
            sqlx::query_as("SELECT id FROM brain_projects WHERE name = $1")
                .bind(name)
                .fetch_optional(pool)
                .await?;
        let pid = pid
            .map(|(id,)| id)
            .ok_or_else(|| anyhow::anyhow!("project not found: {name}"))?;
        let counts: (i64, i64, i64, i64) = sqlx::query_as(
            "SELECT
                (SELECT COUNT(*) FROM brain_entities WHERE project_id = $1),
                (SELECT COUNT(*) FROM brain_observations WHERE project_id = $1),
                (SELECT COUNT(*) FROM brain_episodes WHERE project_id = $1),
                (SELECT COUNT(*) FROM brain_relations WHERE project_id = $1)",
        )
        .bind(pid)
        .fetch_one(pool)
        .await?;
        let body = serde_json::json!({
            "name": name,
            "id": pid.to_string(),
            "entities": counts.0,
            "observations": counts.1,
            "episodes": counts.2,
            "relations": counts.3,
        });
        return Ok(serde_json::json!({
            "contents": [{"uri": uri, "mimeType": "application/json", "text": body.to_string()}]
        }));
    }

    if let Some(id_str) = stripped.strip_prefix("snapshot/") {
        let id: uuid::Uuid = id_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid snapshot UUID"))?;
        let row: Option<(String,)> =
            sqlx::query_as("SELECT summary_md FROM brain_compaction_snapshots WHERE id = $1")
                .bind(id)
                .fetch_optional(pool)
                .await?;
        let md = row
            .map(|(m,)| m)
            .ok_or_else(|| anyhow::anyhow!("snapshot not found: {id}"))?;
        return Ok(serde_json::json!({
            "contents": [{"uri": uri, "mimeType": "text/markdown", "text": md}]
        }));
    }

    anyhow::bail!("Unknown cuba:// URI scheme: {uri}")
}
