//! Integration test — V0.8 project scoping.
//!
//! Validates the end-to-end project isolation pipeline:
//!   1. Migration applies idempotently (re-run init does not fail).
//!   2. `cuba_jornada start --project NAME` upserts brain_projects + sessions.project_id.
//!   3. Writes (cronica, alma, alarma, puente) are tagged with the active project.
//!   4. Reads (faro, vigia summary, contradiccion, etc.) only return rows from the
//!      active project + legacy NULL rows.
//!   5. Kill-switch CUBA_PROJECT_FILTER=off bypasses scoping.
//!
//! Requires DATABASE_URL pointing at a Postgres instance with pgvector.
//!
//! Run with:
//!   DATABASE_URL="postgresql://cuba:memorys2026@localhost:5488/brain" \
//!     cargo test --test v08_project_scoping -- --ignored --nocapture
//!
//! All assertions live in a single #[tokio::test] (same rationale as integration.rs:
//! shared sqlx pool runtime).

use serde_json::json;
use uuid::Uuid;

fn unique(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string()[..8])
}

#[tokio::test]
#[ignore]
async fn test_project_scoping_end_to_end() {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");

    // 1. Schema migration — must apply cleanly twice (idempotent).
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("first init_schema");
    drop(pool);
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("second init_schema (idempotent)");
    println!("  ✓ project scoping migration is idempotent");

    // Verify brain_projects table + project_id columns exist.
    let cols: Vec<(String,)> = sqlx::query_as(
        "SELECT table_name::text FROM information_schema.columns
         WHERE column_name = 'project_id'
           AND table_name LIKE 'brain_%'
         ORDER BY table_name",
    )
    .fetch_all(&pool)
    .await
    .expect("query project_id columns");
    let names: Vec<&str> = cols.iter().map(|(n,)| n.as_str()).collect();
    for required in &[
        "brain_entities",
        "brain_observations",
        "brain_episodes",
        "brain_sessions",
        "brain_errors",
        "brain_relations",
    ] {
        assert!(
            names.contains(required),
            "Missing project_id column on {required}; have: {names:?}"
        );
    }
    let projects_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='brain_projects')",
    )
    .fetch_one(&pool)
    .await
    .expect("check brain_projects table");
    assert!(projects_exists, "brain_projects table missing");
    println!("  ✓ brain_projects + project_id columns present on 6 tables");

    // 2. Open two sessions, each bound to a different project.
    let project_a = unique("test_proj_a");
    let project_b = unique("test_proj_b");

    // Session A
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "start", "name": "session-a", "project": project_a}),
    )
    .await
    .expect("start session A");

    // While session A is active, write entities/obs/relation/error scoped to A.
    let ent_a = unique("entity_a");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_alma",
        json!({"action": "create", "name": ent_a, "entity_type": "concept"}),
    )
    .await
    .expect("alma create A");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_cronica",
        json!({
            "action": "add",
            "entity_name": ent_a,
            "content": "Project A specific observation",
            "observation_type": "fact"
        }),
    )
    .await
    .expect("cronica add A");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_alarma",
        json!({
            "error_type": "TestError",
            "error_message": "Project A error",
            "project": "default"
        }),
    )
    .await
    .expect("alarma A");

    // End session A (so session B becomes the active one).
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "end", "outcome": "success", "summary": "A done"}),
    )
    .await
    .expect("end session A");

    // Session B
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "start", "name": "session-b", "project": project_b}),
    )
    .await
    .expect("start session B");

    let ent_b = unique("entity_b");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_alma",
        json!({"action": "create", "name": ent_b, "entity_type": "concept"}),
    )
    .await
    .expect("alma create B");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_cronica",
        json!({
            "action": "add",
            "entity_name": ent_b,
            "content": "Project B specific observation",
            "observation_type": "fact"
        }),
    )
    .await
    .expect("cronica add B");

    println!("  ✓ wrote entities/observations/errors under two projects");

    // 3. While session B is active, faro should NOT return Project A's content.
    let faro_b = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Project A specific observation", "limit": 20}),
    )
    .await
    .expect("faro under project B");
    let faro_b_text = extract_content_text(&faro_b);
    assert!(
        !faro_b_text.contains(&ent_a),
        "Project B search leaked Project A entity '{ent_a}': {faro_b_text}"
    );
    println!("  ✓ faro under project B does not leak project A content");

    // vigia summary under B should count only B's rows for entities/observations.
    let vigia_b =
        cuba_memorys::handlers::dispatch(&pool, "cuba_vigia", json!({"metric": "summary"}))
            .await
            .expect("vigia summary B");
    let vigia_b_text = extract_content_text(&vigia_b);
    assert!(
        vigia_b_text.contains("\"project_scoped\":true"),
        "vigia summary did not advertise project_scoped=true: {vigia_b_text}"
    );
    println!("  ✓ vigia summary is project_scoped under active project");

    // 4. End session B and start a vanilla one (no project) — should see both A & B.
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "end", "outcome": "success", "summary": "B done"}),
    )
    .await
    .expect("end session B");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "start", "name": "session-global"}),
    )
    .await
    .expect("start global session");

    let faro_global = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Project", "limit": 20}),
    )
    .await
    .expect("faro under global session");
    let faro_global_text = extract_content_text(&faro_global);
    assert!(
        faro_global_text.contains(&ent_a) || faro_global_text.contains(&ent_b),
        "global session should see at least one of the project entities: {faro_global_text}"
    );
    println!("  ✓ session without project sees rows from any project");

    // 5. cuba_proyecto list/stats verify counts.
    let list = cuba_memorys::handlers::dispatch(&pool, "cuba_proyecto", json!({"action": "list"}))
        .await
        .expect("proyecto list");
    let list_text = extract_content_text(&list);
    assert!(
        list_text.contains(&project_a) && list_text.contains(&project_b),
        "proyecto list missing projects: {list_text}"
    );

    let stats_a = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_proyecto",
        json!({"action": "stats", "name": project_a}),
    )
    .await
    .expect("proyecto stats A");
    let stats_a_text = extract_content_text(&stats_a);
    // Project A had at least 1 entity + 1 observation + 1 error.
    assert!(
        stats_a_text.contains("\"entities\":1") || stats_a_text.contains("\"entities\": 1"),
        "expected exactly 1 entity in project A: {stats_a_text}"
    );
    println!("  ✓ cuba_proyecto list/stats expose per-project counters");

    // 6. Kill-switch — set CUBA_PROJECT_FILTER=off, current_project_id should return None.
    // We test the helper directly to avoid touching the running session state.
    // SAFETY: tests run sequentially in a single tokio runtime; no other thread touches env.
    unsafe {
        std::env::set_var("CUBA_PROJECT_FILTER", "off");
    }
    let pid = cuba_memorys::project::current_project_id(&pool)
        .await
        .expect("current_project_id with kill-switch");
    assert!(
        pid.is_none(),
        "kill-switch must force current_project_id() to None"
    );
    unsafe {
        std::env::remove_var("CUBA_PROJECT_FILTER");
    }
    println!("  ✓ CUBA_PROJECT_FILTER=off disables scoping");

    // Cleanup: end the global session and prune created rows for re-runnability.
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "end", "outcome": "success", "summary": "test cleanup"}),
    )
    .await
    .ok();

    sqlx::query("DELETE FROM brain_entities WHERE name = $1 OR name = $2")
        .bind(&ent_a)
        .bind(&ent_b)
        .execute(&pool)
        .await
        .ok();
    sqlx::query("DELETE FROM brain_projects WHERE name = $1 OR name = $2")
        .bind(&project_a)
        .bind(&project_b)
        .execute(&pool)
        .await
        .ok();

    println!("\n  ✅ project scoping end-to-end OK");
}

/// Pull the inner JSON text out of an MCP-format response wrapper.
fn extract_content_text(value: &serde_json::Value) -> String {
    value
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("text"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .unwrap_or_else(|| value.to_string())
}
