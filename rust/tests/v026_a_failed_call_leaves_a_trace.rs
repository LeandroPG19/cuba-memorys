use serde_json::json;

#[tokio::test]
#[ignore]
async fn a_handler_that_fails_is_recorded_and_one_that_works_is_not() {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("connect to test database");
    sqlx::query("DELETE FROM brain_handler_failures")
        .execute(&pool)
        .await
        .expect("start from an empty failure log");

    let failed = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_sync",
        json!({"action": "no-such-action-at-all"}),
    )
    .await;
    assert!(failed.is_err(), "that call was supposed to fail");

    let rows: Vec<(String, String)> =
        sqlx::query_as("SELECT tool, error FROM brain_handler_failures")
            .fetch_all(&pool)
            .await
            .expect("read the failure log");
    assert_eq!(
        rows.len(),
        1,
        "until now nothing recorded that a tool call failed: the line went to tracing and from \
         there to the journal, where it survives until rotation and where nobody looks unless \
         they already suspect something. Got: {rows:?}"
    );
    assert_eq!(rows[0].0, "cuba_sync");
    assert!(
        rows[0].1.contains("Invalid action"),
        "and the message has to be the one the handler gave, or the log says a call failed \
         without saying how: {:?}",
        rows[0].1
    );

    cuba_memorys::handlers::dispatch(&pool, "cuba_sync", json!({"action": "status"}))
        .await
        .expect("a good call");

    let after: i64 = sqlx::query_scalar("SELECT count(*) FROM brain_handler_failures")
        .fetch_one(&pool)
        .await
        .expect("count");
    assert_eq!(
        after, 1,
        "successful calls must NOT land here. cuba_faro is called constantly, and a row per \
         call would make the telemetry table grow faster than the memory it watches on a \
         corpus that takes four writes a day. That is the whole reason the recent calls live \
         in a ring in memory instead"
    );

    let ring = cuba_memorys::observability::recent_calls(10);
    assert!(
        ring.iter()
            .any(|c| c.tool == "cuba_sync" && c.outcome == "ok"),
        "and the successful call still has to be visible somewhere — the ring is what the panel \
         reads for recent traffic. Got: {} entries",
        ring.len()
    );

    sqlx::query("DELETE FROM brain_handler_failures")
        .execute(&pool)
        .await
        .ok();
}

#[tokio::test]
#[ignore]
async fn a_failure_carrying_a_credential_is_redacted_before_it_is_stored() {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("connect to test database");
    sqlx::query("DELETE FROM brain_handler_failures")
        .execute(&pool)
        .await
        .expect("clean");

    const TOKEN: &str = "ghp_abcdefghijklmnop";
    let failed =
        cuba_memorys::handlers::dispatch(&pool, "cuba_sync", json!({"action": TOKEN})).await;
    let message = format!("{:#}", failed.expect_err("an unknown action fails"));
    assert!(
        message.contains(TOKEN),
        "this test needs a handler that echoes its input into the error message, because that \
         is how a secret reaches the failure log at all. cuba_sync answers 'Invalid action: X' \
         with X quoted back. If that ever stops being true this test can no longer observe the \
         redaction and has to be pointed at another path rather than left passing. Got: {message}"
    );

    let stored: Vec<String> = sqlx::query_scalar("SELECT error FROM brain_handler_failures")
        .fetch_all(&pool)
        .await
        .expect("read the log");
    assert_eq!(
        stored.len(),
        1,
        "the refusal has to be recorded: {stored:?}"
    );
    assert!(
        !stored[0].contains(TOKEN),
        "the failure log would otherwise become the one place in this database where rejected \
         credentials are kept in the clear — written by the very guard that exists to keep them \
         out. Stored: {:?}",
        stored[0]
    );

    sqlx::query("DELETE FROM brain_handler_failures")
        .execute(&pool)
        .await
        .ok();
}
