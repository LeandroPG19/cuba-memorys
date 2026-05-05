//! Handler: cuba_archivo — Tamper-evident audit log (v0.9, CFR-21 inspired).
//!
//! Append-only log with SHA-256 hash chain. Every row's `current_hash`
//! commits to the previous row's `prev_hash`, the action, the canonical
//! payload and the ISO8601 timestamp. Verification recomputes the chain
//! sequentially and detects tampering by mismatch.
//!
//! Mutation is blocked at the PostgreSQL level via trigger
//! `brain_audit_block_mutation` — only `cuba_admin` role can bypass for
//! GDPR-driven rectification.
//!
//! Actions:
//! - `append {action, payload}` — adds an event to the chain.
//! - `verify {limit?}` — walks the chain and reports first break, if any.
//! - `tail {limit?}` — returns the most recent N events (read-only).

use anyhow::{Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action_kind = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    match action_kind {
        "append" => append(pool, &args).await,
        "verify" => verify(pool, &args).await,
        "tail" => tail(pool, &args).await,
        _ => anyhow::bail!("Invalid action: {action_kind}. Use append/verify/tail"),
    }
}

/// V0.9.3: deterministic ISO8601 with explicit microsecond precision and
/// `+00:00` UTC offset. PostgreSQL TIMESTAMPTZ has μs resolution; if we hash
/// chrono's default RFC3339 (which includes ns) we'd never recompute the
/// same digest after a DB round-trip. This format is what both `append` and
/// `verify` use so the chain is reproducible.
fn canonical_iso(t: chrono::DateTime<chrono::Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
}

/// Compute the canonical hash of a row.
///
/// `prev_hash` empty for the first row. Payload is serialized with
/// `serde_json::to_vec` (RFC 8259 — keys ordered as inserted by serde,
/// fixed for our INSERT path because we always pass `Value::Object`).
fn compute_hash(prev_hash: &[u8], action: &str, payload: &[u8], created_at_iso: &str) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(prev_hash);
    h.update(b"|");
    h.update(action.as_bytes());
    h.update(b"|");
    h.update(payload);
    h.update(b"|");
    h.update(created_at_iso.as_bytes());
    h.finalize().to_vec()
}

async fn append(pool: &PgPool, args: &Value) -> Result<Value> {
    let action = args
        .get("event_action")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("event_action is required"))?;
    let payload = args
        .get("payload")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    // V0.9.3: SERIALIZABLE retry loop. Two concurrent `append` calls
    // would otherwise hit Postgres' read/write dependency check
    // (SQLSTATE 40001). We retry up to N times with exponential backoff;
    // each retry recomputes prev_hash from the latest tail (which has
    // shifted thanks to the other writer's commit), preserving chain
    // integrity.
    const MAX_RETRIES: usize = 5;
    let mut attempt = 0;
    loop {
        let mut tx = pool.begin().await?;
        sqlx::query("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            .execute(&mut *tx)
            .await
            .ok();

        let prev: Option<(Vec<u8>,)> = sqlx::query_as(
            "SELECT current_hash FROM brain_audit_log ORDER BY id DESC LIMIT 1 FOR UPDATE",
        )
        .fetch_optional(&mut *tx)
        .await?;
        let prev_hash: Vec<u8> = prev.map(|(h,)| h).unwrap_or_default();

        // Truncate now() to microsecond precision so the string we hash
        // matches what PostgreSQL persists in TIMESTAMPTZ. Without this,
        // chrono's nanoseconds are silently dropped on the round-trip and
        // `verify` would fail to recompute the same hash. RFC3339 format
        // pinned to 6 fractional digits + `+00:00` for determinism.
        let now = chrono::DateTime::<chrono::Utc>::from_timestamp_micros(
            chrono::Utc::now().timestamp_micros(),
        )
        .expect("epoch in range");
        let now_iso = canonical_iso(now);
        let payload_bytes = serde_json::to_vec(&payload).context("serialize payload")?;
        let current = compute_hash(&prev_hash, action, &payload_bytes, &now_iso);

        let insert_res: Result<(i64,), sqlx::Error> = sqlx::query_as(
            "INSERT INTO brain_audit_log (prev_hash, action, payload, current_hash, created_at)
             VALUES ($1, $2, $3, $4, $5) RETURNING id",
        )
        .bind(if prev_hash.is_empty() {
            None
        } else {
            Some(&prev_hash)
        })
        .bind(action)
        .bind(&payload)
        .bind(&current)
        .bind(now)
        .fetch_one(&mut *tx)
        .await;

        match insert_res {
            Ok(row) => match tx.commit().await {
                Ok(()) => {
                    return Ok(serde_json::json!({
                        "action": "append",
                        "id": row.0,
                        "current_hash": hex::encode(&current),
                        "retries": attempt,
                    }));
                }
                Err(e) if is_serialization_failure(&e) && attempt < MAX_RETRIES => {
                    attempt += 1;
                    let backoff_ms = 5u64 * (1 << attempt);
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            },
            Err(e) if is_serialization_failure(&e) && attempt < MAX_RETRIES => {
                let _ = tx.rollback().await;
                attempt += 1;
                let backoff_ms = 5u64 * (1 << attempt);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                continue;
            }
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(anyhow::Error::from(e).context("insert audit row"));
            }
        }
    }
}

/// True if `e` is a Postgres SQLSTATE 40001 (serialization failure) — these
/// are expected under concurrent appenders and warrant a retry.
fn is_serialization_failure(e: &sqlx::Error) -> bool {
    if let sqlx::Error::Database(db_err) = e
        && let Some(code) = db_err.code()
    {
        return code == "40001";
    }
    false
}

async fn verify(pool: &PgPool, args: &Value) -> Result<Value> {
    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10_000)
        .clamp(1, 1_000_000);

    type Row = (i64, Option<Vec<u8>>, String, Value, Vec<u8>, chrono::DateTime<chrono::Utc>);
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, prev_hash, action, payload, current_hash, created_at
         FROM brain_audit_log ORDER BY id ASC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let mut last_hash: Vec<u8> = Vec::new();
    for (id, prev_hash, action, payload, stored_hash, created_at) in &rows {
        let prev_for_check = prev_hash.clone().unwrap_or_default();
        if prev_for_check != last_hash {
            return Ok(serde_json::json!({
                "action": "verify",
                "ok": false,
                "first_break_id": id,
                "reason": "prev_hash mismatch with previous row's current_hash"
            }));
        }
        let payload_bytes = serde_json::to_vec(payload)?;
        // V0.9.3: same canonical timestamp format used at append time so
        // the round-trip through Postgres TIMESTAMPTZ does not drop digits.
        let recomputed = compute_hash(
            &prev_for_check,
            action,
            &payload_bytes,
            &canonical_iso(*created_at),
        );
        if &recomputed != stored_hash {
            return Ok(serde_json::json!({
                "action": "verify",
                "ok": false,
                "first_break_id": id,
                "reason": "current_hash recomputation differs (row tampered)"
            }));
        }
        last_hash = stored_hash.clone();
    }

    Ok(serde_json::json!({
        "action": "verify",
        "ok": true,
        "rows_checked": rows.len(),
    }))
}

async fn tail(pool: &PgPool, args: &Value) -> Result<Value> {
    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .clamp(1, 200);
    type Row = (i64, String, Value, chrono::DateTime<chrono::Utc>);
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, action, payload, created_at FROM brain_audit_log
         ORDER BY id DESC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    let entries: Vec<Value> = rows
        .into_iter()
        .map(|(id, action, payload, created_at)| {
            serde_json::json!({
                "id": id,
                "action": action,
                "payload": payload,
                "created_at": created_at.to_rfc3339(),
            })
        })
        .collect();
    Ok(serde_json::json!({"action": "tail", "entries": entries.clone(), "count": entries.len()}))
}
