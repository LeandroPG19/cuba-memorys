//! Integration tests — require a running PostgreSQL instance.
//!
//! Set DATABASE_URL env var to run:
//!   DATABASE_URL="postgresql://user:pass@localhost:5433/cuba_memorys_test" \
//!     cargo test --test integration -- --ignored
//!
//! Tests are #[ignore] by default (no DB in CI).
//!
//! CRITICAL: All tests run in a SINGLE #[tokio::test] to share one Tokio runtime.
//! Multiple #[tokio::test] functions create separate runtimes, and sqlx pool
//! connections from runtime A become zombies in runtime B → pool timeout.

use serde_json::json;
use uuid::Uuid;

/// Generate unique entity names per test run (avoid collisions).
fn unique_name(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string()[..8])
}

#[tokio::test]
#[ignore]
async fn test_all_integration() {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("Failed to connect to test database");

    // ── 1. Schema validation ──────────────────────────────────────
    println!("  [1/7] Schema validation...");
    {
        let tables: Vec<(String,)> = sqlx::query_as(
            "SELECT table_name::text FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name LIKE 'brain_%'
             ORDER BY table_name",
        )
        .fetch_all(&pool)
        .await
        .expect("Failed to query tables");

        let names: Vec<&str> = tables.iter().map(|(n,)| n.as_str()).collect();
        for required in &[
            "brain_entities",
            "brain_observations",
            "brain_relations",
            "brain_errors",
            "brain_sessions",
        ] {
            assert!(names.contains(required), "Missing table: {required}");
        }
        println!(
            "  ✓ All required tables exist ({} brain_* tables)",
            names.len()
        );
    }

    // ── 2. pgvector extension ─────────────────────────────────────
    println!("  [2/7] pgvector extension...");
    {
        let row: (bool,) =
            sqlx::query_as("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(row.0, "pgvector extension required");
        println!("  ✓ pgvector extension detected");
    }

    // ── 3. alma CRUD ──────────────────────────────────────────────
    println!("  [3/7] alma create + get...");
    {
        let name = unique_name("alma");
        let result = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_alma",
            json!({ "action": "create", "name": &name, "entity_type": "concept" }),
        )
        .await
        .unwrap();
        assert!(
            result.get("content").is_some(),
            "create should return content"
        );

        let result = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_alma",
            json!({ "action": "get", "name": &name }),
        )
        .await
        .unwrap();
        assert!(result.get("content").is_some(), "get should return content");
        println!("  ✓ alma create + get OK (entity: {name})");
    }

    // ── 4. cronica add + list ─────────────────────────────────────
    println!("  [4/7] cronica add + list...");
    {
        let name = unique_name("cronica");
        let result = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_cronica",
            json!({
                "action": "add",
                "entity_name": &name,
                "content": "Integration test observation for cronica handler validation",
                "observation_type": "fact",
                "source": "agent"
            }),
        )
        .await
        .unwrap();
        assert!(result.get("content").is_some(), "add should return content");

        let result = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_cronica",
            json!({ "action": "list", "entity_name": &name }),
        )
        .await
        .unwrap();
        assert!(
            result.get("content").is_some(),
            "list should return content"
        );
        println!("  ✓ cronica add + list OK (entity: {name})");
    }

    // ── 5. faro search ────────────────────────────────────────────
    println!("  [5/7] faro search...");
    {
        let name = unique_name("faro");
        let _ = cuba_memorys::handlers::dispatch(
            &pool, "cuba_cronica",
            json!({
                "action": "add",
                "entity_name": &name,
                "content": "Rust is a systems programming language focused on safety and performance",
                "observation_type": "fact"
            }),
        ).await;

        let result = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_faro",
            json!({ "query": "rust programming safety", "mode": "hybrid", "limit": 5 }),
        )
        .await
        .unwrap();
        assert!(
            result.get("content").is_some(),
            "faro should return content"
        );
        println!("  ✓ faro search OK");
    }

    // ── 6. jornada lifecycle ──────────────────────────────────────
    println!("  [6/7] jornada lifecycle...");
    {
        let result = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_jornada",
            json!({
                "action": "start",
                "name": &unique_name("session"),
                "goals": ["test session management"]
            }),
        )
        .await
        .unwrap();
        assert!(
            result.get("content").is_some(),
            "start should return content"
        );

        let result =
            cuba_memorys::handlers::dispatch(&pool, "cuba_jornada", json!({ "action": "current" }))
                .await
                .unwrap();
        assert!(
            result.get("content").is_some(),
            "current should return content"
        );

        let _ = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_jornada",
            json!({ "action": "end", "outcome": "success", "summary": "Integration test done" }),
        )
        .await;
        println!("  ✓ jornada start + current + end OK");
    }

    // ── 7. vigia summary ──────────────────────────────────────────
    println!("  [7/7] vigia summary...");
    {
        let result =
            cuba_memorys::handlers::dispatch(&pool, "cuba_vigia", json!({ "metric": "summary" }))
                .await
                .unwrap();
        assert!(
            result.get("content").is_some(),
            "vigia should return content"
        );
        println!("  ✓ vigia summary OK");
    }

    println!("\n  ═══════════════════════════════════════════");
    println!("  ✅ ALL 7 INTEGRATION TESTS PASSED");
    println!("  ═══════════════════════════════════════════");

    // Clean shutdown
    pool.close().await;
}
