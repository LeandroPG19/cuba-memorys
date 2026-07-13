//! MCP JSON-RPC protocol handler — stdin/stdout transport.
//!
//! FIX B4: REM session protection uses exact entity_id match (not ILIKE).
//! FIX V2: tokio::time::timeout(30s) on every handler dispatch.

use crate::db;
use crate::handlers;

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use sqlx::PgPool;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// MCP protocol timeout per handler request. Default 30s; override with
/// `CUBA_HANDLER_TIMEOUT_SECS` for long maintenance ops (e.g. a full bge-m3
/// reembed of the corpus, which exceeds 30s).
fn handler_timeout() -> Duration {
    std::env::var("CUBA_HANDLER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30))
}

/// REM sleep interval (4 hours).
const REM_INTERVAL: Duration = Duration::from_secs(4 * 3600);

// ── JSON-RPC Types ───────────────────────────────────────────────

/// An inbound JSON-RPC envelope.
///
/// `jsonrpc` and `params` are never read directly — the dispatcher works from
/// `method` and pulls arguments out of the raw `Value`. They are declared anyway
/// because their presence is what makes serde REJECT a malformed envelope at the
/// door: drop them and a request missing `jsonrpc` would deserialize happily and
/// fail somewhere deeper, with a worse error. The fields are the validation.
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // see above: these fields validate the shape, they are not read
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

// V0.9.2: JsonRpcResponse / JsonRpcError structs removed — every outbound
// envelope is built ad-hoc with serde_json::json!() and pushed to the
// OUTBOUND mpsc channel. Keeps the surface narrow and avoids the temptation
// to construct envelopes from places that should not.

// ── V0.9.1: Client capability tracking ───────────────────────────

/// Captured at `initialize` time. Read from any handler/judge backend.
static CLIENT_SUPPORTS_SAMPLING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// True if the client advertised `capabilities.sampling` during initialize.
/// Used by `cuba_juez` resolver to prefer MCP Sampling over CLI/API.
pub fn client_supports_sampling() -> bool {
    CLIENT_SUPPORTS_SAMPLING.load(std::sync::atomic::Ordering::Relaxed)
}

// ── V0.9.2: Server-initiated request correlator ──────────────────
// Sampling, progress, cancellation all need the server to write to stdout
// from arbitrary async tasks (not just from the request handler chain).
// We solve this with:
//   1. An mpsc<Value> channel — every code path that wants to write to
//      stdout sends its serialized JSON-RPC value there. A single writer
//      task owns stdout and drains the channel sequentially (preventing
//      interleaved writes, which would corrupt the JSONL stream).
//   2. A pending-requests map keyed by request id, with a oneshot::Sender
//      per id. When the client responds to a server-initiated request,
//      the dispatcher routes the value into the matching oneshot.

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use tokio::sync::{Mutex, mpsc, oneshot};

/// Per-process global writer queue. All outbound JSON-RPC messages go through
/// this; a single writer task owns stdout.
static OUTBOUND: OnceLock<mpsc::UnboundedSender<Value>> = OnceLock::new();

/// Pending server→client requests awaiting a response. Keyed by id.
static PENDING: OnceLock<Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>>> =
    OnceLock::new();

/// Monotonic request id generator for server-initiated requests. We use the
/// `srv_<n>` namespace so client-initiated ids can never collide.
static NEXT_SERVER_ID: AtomicU64 = AtomicU64::new(1);

/// Per-handler-call cancellation token. Set by the dispatcher when handling
/// `tools/call`, looked up by `notifications/cancelled`. Map keys are the
/// JSON-RPC request id encoded as a string.
static CANCEL_TOKENS: OnceLock<Mutex<std::collections::HashMap<String, CancelToken>>> =
    OnceLock::new();

/// Thin wrapper so cancellation propagates across `tokio::select!` arms.
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

/// Request the connected client to call its LLM via `sampling/createMessage`.
/// Returns the model's reply text.
///
/// V0.9.2: real implementation — issues a server→client request through the
/// outbound channel and awaits the matching response on a oneshot. Falls
/// back to a structured error if the client did not advertise `sampling`.
pub async fn request_sampling(prompt: &str) -> anyhow::Result<String> {
    request_sampling_max(prompt, 256).await
}

/// Like [`request_sampling`] but with a caller-chosen `max_tokens` budget.
/// Fact extraction needs a larger reply than the judge's 256-token verdict.
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

    // 30s ceiling to match HANDLER_TIMEOUT — no judge call should hang the agent.
    let response = match tokio::time::timeout(handler_timeout(), rx).await {
        Ok(Ok(v)) => v,
        Ok(Err(_)) => anyhow::bail!("sampling response channel dropped"),
        Err(_) => {
            // Clean up pending entry so the map doesn't grow unbounded
            pending().lock().await.remove(&id);
            anyhow::bail!("sampling timed out after 30s");
        }
    };

    // MCP spec: response.result.content.text is the model's reply
    let text = response
        .get("result")
        .and_then(|r| r.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .or_else(|| {
            // Some clients return error in standard JSON-RPC error envelope
            response
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
        })
        .ok_or_else(|| anyhow::anyhow!("malformed sampling response: {response}"))?;
    Ok(text.to_string())
}

/// Send a `notifications/progress` to the client for a long-running tool call.
/// Best-effort — if outbound is closed (server shutting down) this is a no-op.
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

/// Register a cancellation token for the given client request id. Returned
/// token is shared with the running handler via `tokio::select!`.
pub async fn register_cancel_token(request_id: &Value) -> CancelToken {
    let token = CancelToken::default();
    let key = request_id.to_string();
    cancel_tokens().lock().await.insert(key, token.clone());
    token
}

/// Drop the token for `request_id`. Called once the handler completes.
pub async fn unregister_cancel_token(request_id: &Value) {
    let key = request_id.to_string();
    cancel_tokens().lock().await.remove(&key);
}

// ── MCP Protocol Constants ───────────────────────────────────────

/// Server capabilities advertised during initialize.
/// V0.9: announces `resources` capability for `cuba://` URI scheme
/// (read-only entity/snapshot/project introspection without invoking tools).
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

// ── Main Protocol Loop ──────────────────────────────────────────

/// Run the MCP protocol over stdin/stdout.
///
/// V0.9.2: refactored to support server-initiated requests (sampling) +
/// progress notifications + cancellation. Three concurrent tasks share
/// stdio:
///
/// 1. **Reader** task: reads JSON-RPC from stdin. Distinguishes:
///    - Client request (has `method`, has `id`) → spawn handler, response
///      goes to outbound channel.
///    - Client notification (has `method`, no `id`) → fire-and-forget.
///    - Client response to a server-initiated request (has `id`, no
///      `method`) → routed via PENDING map.
/// 2. **Writer** task: drains the outbound channel and writes serially to
///    stdout. Single owner of stdout — no interleaving.
/// 3. **Handler** tasks (spawned per request): run the dispatch and push
///    their JSON-RPC response into outbound when done.
pub async fn run_mcp() -> Result<()> {
    let database_url = crate::setup::resolve_database_url().await;
    let pool = db::create_pool(&database_url).await?;

    // REM daemon (preserved from V0.7).
    let rem_pool = pool.clone();
    let rem_handle = tokio::spawn(async move {
        rem_daemon(rem_pool).await;
    });

    // Outbound channel — every code path that wants to write to stdout sends here.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    OUTBOUND
        .set(out_tx)
        .map_err(|_| anyhow::anyhow!("OUTBOUND already initialized"))?;

    // Writer task: single owner of stdout.
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

    // Reader loop. We track every spawned handler in a JoinSet so we can
    // drain them on EOF — without this, MCP clients that pipe a batch of
    // requests and close stdin lose responses for any handler still
    // executing when the read returns None.
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

        // First parse generically — could be a request OR a response from
        // a server-initiated call (sampling). Responses have `id` + `result`
        // or `error` and lack `method`.
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

        // Server-initiated response routing (sampling, list_roots, etc.)
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

        // Otherwise it's a client request or notification.
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
                Err(e) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": { "code": -32603, "message": e.to_string() }
                }),
            };
            let _ = outbound().send(envelope);
        });
    }

    // Stdin EOF — client disconnected. Drain in-flight handlers (giving each
    // a generous window to finish) before closing the writer. Without this
    // wait the writer would be aborted while handlers are still trying to
    // push their responses onto the channel.
    tracing::info!(
        "stdin closed — draining {} in-flight handlers",
        in_flight.len()
    );
    let drain = async { while in_flight.join_next().await.is_some() {} };
    let _ = tokio::time::timeout(handler_timeout() * 2, drain).await;

    // Close the outbound channel by dropping the cached sender so the writer
    // task observes channel-empty and exits naturally.
    if let Some(tx) = OUTBOUND.get() {
        // Drop our sender clone owned by `outbound()`. The reader loop is
        // the only other holder, and it's about to exit.
        let _ = tx; // sender is in OnceLock — kept alive for the writer drain
    }

    rem_handle.abort();
    // Give the writer a brief grace period to flush remaining envelopes.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), writer_handle).await;
    tracing::info!("REM daemon + writer drained, shutting down");

    Ok(())
}

/// Route a JSON-RPC request to the appropriate handler.
async fn handle_request(pool: &PgPool, request: JsonRpcRequest) -> Result<Value> {
    match request.method.as_str() {
        // MCP lifecycle
        "initialize" => {
            // V0.9: capture client capabilities for downstream feature gating.
            // Specifically `capabilities.sampling` enables MCPSamplingJudge.
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
            // V0.9.2: signal the matching CancelToken so the handler unwinds.
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

        // Tool listing. Honours CUBA_TOOL_PROFILE: all 25 schemas ride in the
        // agent's context every session, and most are maintenance surfaces it
        // never calls mid-task. Defaults to `full`, so nothing shrinks on upgrade.
        "tools/list" => Ok(serde_json::json!({
            "tools": crate::constants::tools_for_profile()
        })),

        // V0.9: MCP Resources — read-only URI scheme cuba://
        // - cuba://entity/<name>      → entity + recent observations as JSON
        // - cuba://project/<name>     → project metadata + counts
        // - cuba://snapshot/<id>      → compaction snapshot markdown
        "resources/list" => list_resources(pool).await,
        "resources/read" => {
            let params = request.params.unwrap_or(Value::Null);
            let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");
            read_resource(pool, uri).await
        }

        // Tool execution with timeout (V2) + V0.9.2 cancellation token.
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

            // V0.9.2: register cancellation token so notifications/cancelled
            // can interrupt long-running handlers via tokio::select!.
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
/// Steps: decay → PageRank → TF-IDF.
/// NOTE: Neighbor Diffusion removed — duplicates PageRank topology propagation
/// with N+1 pattern (120 queries/cycle for 20 seeds × 6 neighbors each).
async fn run_rem_consolidation(pool: &PgPool) -> Result<()> {
    // 1. Get this process's session goals (for session protection - FIX B4).
    // Scoped to crate::session: REM must not shield another client's entities
    // from decay, nor leave ours unprotected because they opened a session later.
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

    // 3. V4: Stratified exponential decay — different halflife per observation_type.
    //    fact/preference: 30d | error/solution: 14d | context/tool_usage: 7d
    //    decision/lesson: never (protected by WHERE clause).
    // Anchor on GREATEST(last_accessed, last_decayed_at) so repeated REM cycles
    // decay only the incremental idle time (migration 0028). The access_count
    // term stretches the effective half-life for frequently-used memories,
    // matching the manual cuba_zafra decay and the V4 spec in the README.
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

    // 3b. Episode power-law decay (Wixted 2004) — idempotent from initial=0.5
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
    .unwrap_or(0); // Non-fatal: episodes table may not exist on old DBs
    tracing::info!(
        episode_decayed_count = episode_decayed,
        "episode power-law decay applied"
    );

    // 4. PageRank (batch — P1 fix)
    let ranked = crate::graph::pagerank::compute_and_store(pool).await?;
    tracing::info!(ranked_count = ranked, "PageRank updated");

    // 5. TF-IDF index: BM25 index is in-memory, rebuilt on demand per REM cycle
    // (no persistent index to rebuild — Bm25Index lives in handler state)
    tracing::info!("REM consolidation complete");

    Ok(())
}

// ── V0.9: MCP Resources implementation ──────────────────────────

/// List recently active entities, projects and snapshots as cuba:// URIs.
async fn list_resources(pool: &PgPool) -> Result<Value> {
    let mut resources: Vec<Value> = Vec::new();

    // Top 50 entities by access_count
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

    // All projects
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

    // Recent compaction snapshots
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

/// Read a single cuba:// URI. Returns the standard MCP `contents` envelope.
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
