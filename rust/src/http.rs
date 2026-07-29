//! The shared-daemon transport.
//!
//! stdio gives every client its own process, and every process its own copy of
//! the ONNX models — roughly 6 GB each. Three editor windows meant three copies.
//! Here one process holds the models and answers every client over loopback
//! HTTP, which is also the shape the 2026-07-28 MCP specification settled on:
//! no session handshake, every request self-describing.
//!
//! Clients identify themselves with the `Mcp-Client-Id` header so that
//! `jornada start` in one window does not become the active session of another.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::FutureExt;
use serde_json::Value;
use sqlx::PgPool;

use crate::protocol::{self, JsonRpcRequest};

pub const DEFAULT_ADDR: &str = "127.0.0.1:8787";

/// Observations can be long, and a rejected body reads to the client as a dead
/// server. 8 MB is far above anything a tool call carries.
const MAX_BODY: usize = 8 * 1024 * 1024;

/// A client that has not spoken in this long is assumed gone, and its session
/// row is dropped. The daemon is meant to run for weeks.
const CLIENT_TTL: Duration = Duration::from_secs(24 * 3600);
const REAP_INTERVAL: Duration = Duration::from_secs(3600);

#[derive(Clone)]
struct AppState {
    pool: PgPool,
    token: Option<Arc<String>>,
    started: Instant,
    served: Arc<AtomicU64>,
    seen: Arc<std::sync::RwLock<std::collections::HashMap<String, Instant>>>,
}

pub fn bind_addr() -> String {
    std::env::var("CUBA_HTTP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string())
}

fn auth_token() -> Option<String> {
    std::env::var("CUBA_HTTP_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
}

/// Rejects anything that is not loopback. The daemon answers without
/// authentication by default, so binding it to a routable address would hand
/// the whole knowledge graph to the network.
fn ensure_loopback(addr: &SocketAddr) -> Result<()> {
    if addr.ip().is_loopback() || auth_token().is_some() {
        return Ok(());
    }
    anyhow::bail!(
        "refusing to bind {addr}: the daemon serves the whole brain with no auth. \
         Use a 127.0.0.1 address, or set CUBA_HTTP_TOKEN to require a bearer token"
    )
}

pub async fn serve(addr: &str) -> Result<()> {
    let addr: SocketAddr = addr
        .parse()
        .with_context(|| format!("invalid listen address: {addr}"))?;
    ensure_loopback(&addr)?;

    // Every session read from now on resolves per client instead of per process.
    crate::session::enable_daemon_mode();

    let database_url = crate::setup::resolve_database_url().await;
    let (pool, connected) = match crate::db::create_pool(&database_url).await {
        Ok(pool) => {
            crate::db::assert_embedding_dim(&pool).await?;
            (pool, true)
        }
        Err(why) => {
            tracing::warn!(
                error = %format!("{why:#}"),
                "starting without PostgreSQL — tools will fail until it is reachable"
            );
            (crate::db::create_lazy_pool(&database_url), false)
        }
    };

    // One process now, so one consolidation cycle. Under stdio every window ran
    // its own REM against the same database.
    if connected {
        let rem_pool = pool.clone();
        tokio::spawn(async move { protocol::rem_daemon(rem_pool).await });
    }

    let state = AppState {
        pool,
        token: auth_token().map(Arc::new),
        started: Instant::now(),
        served: Arc::new(AtomicU64::new(0)),
        seen: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
    };

    let reaper_seen = state.seen.clone();
    tokio::spawn(async move { reap_idle_clients(reaper_seen).await });

    let app = Router::new()
        .route("/mcp", post(mcp_endpoint))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot bind {addr} — is another daemon already running?"))?;

    tracing::info!(
        %addr,
        auth = auth_token().is_some(),
        "cuba-memorys daemon listening — point clients at http://{addr}/mcp"
    );

    // Load the models now, in the background. The listener is already up, so a
    // client connecting during the load waits on its first search instead of
    // timing out the connection — which is exactly how the stdio server used to
    // strand 6 GB processes nobody was talking to.
    tokio::spawn(async {
        let started = Instant::now();
        warm_models().await;
        tracing::info!(secs = started.elapsed().as_secs_f32(), "models warm");
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("http server failed")?;

    tracing::info!("daemon shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!(error = %e, "cannot install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("SIGINT received"),
        _ = terminate => tracing::info!("SIGTERM received"),
    }
}

/// Pays the model load once, at startup, instead of on some client's first
/// search. Both calls are the ones the search path itself uses, so this only
/// moves the cost — it changes no behaviour and no model.
async fn warm_models() {
    if crate::search::rerank::is_configured() {
        if crate::search::rerank::warm_up().await {
            tracing::info!("reranker warm");
        } else {
            tracing::warn!("reranker configured but failed to warm up — identity fallback");
        }
    }
    match crate::embeddings::onnx::embed("warm up").await {
        Ok(_) => tracing::info!(
            model = %crate::embeddings::onnx::current_model(),
            loaded = crate::embeddings::onnx::is_model_loaded(),
            "embedding model warm"
        ),
        Err(e) => tracing::warn!(error = %format!("{e:#}"), "embedding warm-up failed"),
    }
}

async fn reap_idle_clients(
    seen: Arc<std::sync::RwLock<std::collections::HashMap<String, Instant>>>,
) {
    let mut ticker = tokio::time::interval(REAP_INTERVAL);
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let stale: Vec<String> = match seen.read() {
            Ok(guard) => guard
                .iter()
                .filter(|(_, last)| last.elapsed() > CLIENT_TTL)
                .map(|(k, _)| k.clone())
                .collect(),
            Err(_) => continue,
        };
        if stale.is_empty() {
            continue;
        }
        if let Ok(mut guard) = seen.write() {
            for key in &stale {
                guard.remove(key);
            }
        }
        for key in &stale {
            crate::session::forget_client(key);
        }
        tracing::info!(count = stale.len(), "reaped idle clients");
    }
}

/// Which client this request belongs to. The header is what clients configure;
/// `_meta` covers the 2026-07-28 style of carrying identity per request, and
/// `clientInfo` catches an `initialize` that set neither.
fn client_key(headers: &HeaderMap, payload: &Value) -> String {
    if let Some(id) = headers
        .get("mcp-client-id")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return id.to_string();
    }

    let first = payload
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(payload);
    let params = first.get("params");

    if let Some(id) = params
        .and_then(|p| p.get("_meta"))
        .and_then(|m| {
            m.get("io.modelcontextprotocol/client-id")
                .or_else(|| m.get("clientId"))
        })
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return id.to_string();
    }

    if let Some(name) = params
        .and_then(|p| p.get("clientInfo"))
        .and_then(|c| c.get("name"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return name.to_string();
    }

    "anonymous".to_string()
}

fn authorized(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(expected) = state.token.as_ref() else {
        return true;
    };
    let presented = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");

    // Constant-time compare: a length-dependent early exit is enough to walk a
    // token out of a local server one byte at a time.
    let a = presented.as_bytes();
    let b = expected.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn error_envelope(id: Value, code: i64, message: impl Into<String>) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() }
    })
}

async fn mcp_endpoint(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if !authorized(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(error_envelope(Value::Null, -32001, "invalid bearer token")),
        )
            .into_response();
    }

    let payload: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(error_envelope(
                    Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                )),
            )
                .into_response();
        }
    };

    let key = client_key(&headers, &payload);
    if let Ok(mut guard) = state.seen.write() {
        guard.insert(key.clone(), Instant::now());
    }

    let batch = payload.as_array().cloned();
    let items = batch.clone().unwrap_or_else(|| vec![payload]);

    let mut responses: Vec<Value> = Vec::with_capacity(items.len());
    for item in items {
        state.served.fetch_add(1, Ordering::Relaxed);
        if let Some(reply) = dispatch_one(&state, &key, item).await {
            responses.push(reply);
        }
    }

    // Nothing but notifications: JSON-RPC says answer with no body.
    if responses.is_empty() {
        return StatusCode::ACCEPTED.into_response();
    }

    if batch.is_some() {
        axum::Json(Value::Array(responses)).into_response()
    } else {
        axum::Json(responses.remove(0)).into_response()
    }
}

async fn dispatch_one(state: &AppState, key: &str, item: Value) -> Option<Value> {
    let id = item.get("id").cloned().unwrap_or(Value::Null);

    let request: JsonRpcRequest = match serde_json::from_value(item) {
        Ok(r) => r,
        Err(e) => {
            return Some(error_envelope(id, -32600, format!("Invalid request: {e}")));
        }
    };

    let is_notification = request.id.is_none();
    let req_id = request.id.clone().unwrap_or(Value::Null);
    let pool = state.pool.clone();
    let method = request.method.clone();

    // One process serves every window now, so a panic in one client's handler
    // would take down everyone else's server. Contain it here and answer that
    // one request with an error instead.
    let work = crate::session::with_client(key.to_string(), async move {
        protocol::handle_request(&pool, request).await
    });

    let outcome = match std::panic::AssertUnwindSafe(work).catch_unwind().await {
        Ok(result) => result,
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            tracing::error!(client = %key, method = %method, detail = %detail, "handler panicked");
            Err(anyhow::anyhow!("handler panicked: {detail}"))
        }
    };

    if is_notification {
        if let Err(e) = &outcome {
            tracing::warn!(error = %format!("{e:#}"), "notification handler error (suppressed)");
        }
        return None;
    }

    Some(match outcome {
        Ok(v) => serde_json::json!({ "jsonrpc": "2.0", "id": req_id, "result": v }),
        Err(e) => {
            let chain = format!("{e:#}");
            tracing::error!(client = %key, method = %method, error = %chain, "handler failed");
            error_envelope(req_id, -32603, chain)
        }
    })
}

async fn health(State(state): State<AppState>) -> Response {
    let db_ok = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .is_ok();

    let clients: Vec<String> = state
        .seen
        .read()
        .map(|g| g.keys().cloned().collect())
        .unwrap_or_default();

    let body = serde_json::json!({
        "status": if db_ok { "ok" } else { "degraded" },
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": state.started.elapsed().as_secs(),
        "requests_served": state.served.load(Ordering::Relaxed),
        "database": if db_ok { "up" } else { "unreachable" },
        "clients": clients,
    });

    let code = if db_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with(name: &'static str, value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(name, value.parse().unwrap());
        h
    }

    #[test]
    fn header_wins_over_payload() {
        let payload = serde_json::json!({
            "params": { "clientInfo": { "name": "claude-code" } }
        });
        let h = headers_with("mcp-client-id", "window-3");
        assert_eq!(client_key(&h, &payload), "window-3");
    }

    #[test]
    fn falls_back_to_meta_then_client_info() {
        let meta = serde_json::json!({
            "params": { "_meta": { "io.modelcontextprotocol/client-id": "from-meta" } }
        });
        assert_eq!(client_key(&HeaderMap::new(), &meta), "from-meta");

        let info = serde_json::json!({
            "params": { "clientInfo": { "name": "warp" } }
        });
        assert_eq!(client_key(&HeaderMap::new(), &info), "warp");

        let bare = serde_json::json!({ "method": "ping" });
        assert_eq!(client_key(&HeaderMap::new(), &bare), "anonymous");
    }

    #[test]
    fn batch_identity_comes_from_the_first_entry() {
        let batch = serde_json::json!([
            { "params": { "clientInfo": { "name": "first" } } },
            { "params": { "clientInfo": { "name": "second" } } },
        ]);
        assert_eq!(client_key(&HeaderMap::new(), &batch), "first");
    }

    #[test]
    fn blank_header_does_not_win() {
        let payload = serde_json::json!({
            "params": { "clientInfo": { "name": "real-client" } }
        });
        let h = headers_with("mcp-client-id", "   ");
        assert_eq!(client_key(&h, &payload), "real-client");
    }

    #[test]
    fn non_loopback_bind_is_refused_without_a_token() {
        let public: SocketAddr = "0.0.0.0:8787".parse().unwrap();
        assert!(ensure_loopback(&public).is_err());

        let local: SocketAddr = "127.0.0.1:8787".parse().unwrap();
        assert!(ensure_loopback(&local).is_ok());
    }

    // Tokio context: the lazy pool registers a background reaper on construction.
    #[tokio::test]
    async fn token_comparison_rejects_wrong_and_short_tokens() {
        let state = AppState {
            pool: crate::db::create_lazy_pool("postgres://unused/unused"),
            token: Some(Arc::new("s3cret".to_string())),
            started: Instant::now(),
            served: Arc::new(AtomicU64::new(0)),
            seen: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        };

        assert!(authorized(
            &state,
            &headers_with("authorization", "Bearer s3cret")
        ));
        assert!(!authorized(
            &state,
            &headers_with("authorization", "Bearer s3cre")
        ));
        assert!(!authorized(
            &state,
            &headers_with("authorization", "Bearer wrongg")
        ));
        assert!(!authorized(&state, &HeaderMap::new()));
    }
}
