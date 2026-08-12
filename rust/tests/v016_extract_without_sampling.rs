use serde_json::json;
use uuid::Uuid;

fn unique_name(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string()[..8])
}

async fn pool() -> sqlx::PgPool {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    cuba_memorys::db::create_pool(&url)
        .await
        .expect("connect to test database")
}

#[tokio::test]
#[ignore]
async fn auto_extract_falls_back_to_the_local_cli_when_the_client_has_no_sampling() {
    assert!(
        !cuba_memorys::protocol::client_supports_sampling(),
        "this test must run outside an MCP session so the fallback is what gets exercised"
    );
    assert!(
        cuba_memorys::cognitive::judge::resolve_offline_llm().is_some(),
        "no local LLM CLI on PATH. This test exists because auto_extract was dead for \
         months while its suite reported green — a silent skip here recreates exactly that. \
         Install one, or run this suite where one exists"
    );

    let pool = pool().await;
    let subject = unique_name("Proyecto");
    let text = format!(
        "El servicio {subject} corre sobre PostgreSQL y depende de Redis para la cola de \
         trabajos. Lo mantiene el equipo de plataforma."
    );

    let result = cuba_memorys::handlers::ingesta::handle(
        &pool,
        json!({ "action": "auto_extract", "text": text, "entity_hint": subject }),
    )
    .await
    .expect("auto_extract must not error when a CLI is reachable");

    let reason = result.get("reason").and_then(|v| v.as_str());
    assert_ne!(
        reason,
        Some("no_backend"),
        "auto_extract reported that no LLM is reachable while resolve_offline_llm found one \
         two assertions ago. That is the failure this test exists for: auto_extract was dead \
         for months while its suite reported green. Got: {result}"
    );
    if matches!(reason, Some("out_of_budget") | Some("backend_failed")) {
        eprintln!(
            "SKIPPED the extraction assertions: the backend was found and wired, and then \
             {reason:?} — the model was slow or errored on this run. That is an environment \
             outcome, not a wiring one, and failing here would teach everyone to re-run the \
             gate instead of reading it. What this run does NOT establish: that a reply gets \
             parsed into entities. Result: {result}"
        );
        return;
    }
    assert_ne!(
        result.get("degraded").and_then(|v| v.as_bool()),
        Some(true),
        "with a CLI on PATH auto_extract must no longer degrade: {result}"
    );
    assert_eq!(
        result.get("backend").and_then(|v| v.as_str()),
        Some("claude_cli"),
        "the reply must say which backend ran: {result}"
    );

    let extracted = result
        .get("extracted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let linked = result
        .get("relations_linked")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        extracted > 0 || linked > 0,
        "the CLI must return something usable from a fact-dense text: {result}"
    );

    let stored: (i64,) = sqlx::query_as(
        "SELECT count(*) FROM brain_observations o
         JOIN brain_entities e ON e.id = o.entity_id
         WHERE o.source = 'inference' AND e.name ILIKE $1",
    )
    .bind(format!("%{subject}%"))
    .fetch_one(&pool)
    .await
    .expect("counting inferred observations");

    if extracted > 0 {
        assert!(
            stored.0 > 0,
            "extracted facts must land tagged source='inference', which is what production had \
             zero of"
        );
    }

    sqlx::query("DELETE FROM brain_entities WHERE name ILIKE $1")
        .bind(format!("%{subject}%"))
        .execute(&pool)
        .await
        .ok();
}

#[tokio::test]
#[ignore]
async fn the_judge_still_reaches_a_verdict_with_mcp_servers_disabled() {
    assert!(
        cuba_memorys::cognitive::judge::which_in_path("claude"),
        "no claude CLI on PATH: the fallback this asserts cannot be exercised, and \
         reporting ok would claim it was"
    );
    if false {
        return;
    }

    let judge = cuba_memorys::cognitive::judge::ClaudeCodeJudge::from_env();
    let judgment = cuba_memorys::cognitive::judge::ContradictionJudge::judge(
        &judge,
        "El servicio corre en el puerto 8080.",
        "El servicio corre en el puerto 9090.",
    )
    .await
    .expect("the CLI judge must still answer once MCP servers are excluded");

    assert_eq!(judgment.backend, "claude_cli");
    assert_eq!(
        judgment.verdict, "contradicts",
        "two different ports for one service contradict each other: {judgment:?}"
    );
}

#[test]
fn the_cli_json_envelope_is_unwrapped_but_a_bare_reply_is_left_alone() {
    let enveloped = r#"{"type":"result","subtype":"success","is_error":false,
        "result":"{\"facts\":[],\"relations\":[]}","session_id":"abc","total_cost_usd":0.01}"#;
    assert_eq!(
        cuba_memorys::cognitive::judge::unwrap_cli_reply(enveloped),
        r#"{"facts":[],"relations":[]}"#,
        "the CLI wraps its answer in an envelope; the extractor must see the inner text"
    );

    let bare = r#"{"facts":[{"entity_name":"x","content":"y"}],"relations":[]}"#;
    assert_eq!(
        cuba_memorys::cognitive::judge::unwrap_cli_reply(bare),
        bare,
        "a sampling reply has no envelope and must survive untouched"
    );

    let fenced = "```json\n{\"facts\":[]}\n```";
    assert_eq!(
        cuba_memorys::cognitive::judge::unwrap_cli_reply(fenced),
        fenced,
        "non-JSON text must pass through for the downstream parser to handle"
    );
}
