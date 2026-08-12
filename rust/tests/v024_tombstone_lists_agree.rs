#[tokio::test]
#[ignore]
async fn the_import_deletes_from_exactly_the_tables_that_write_tombstones() {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("connect to test database");

    let with_triggers: Vec<String> = sqlx::query_scalar(
        "SELECT c.relname::text
         FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
         WHERE NOT t.tgisinternal AND t.tgname LIKE '%\\_tombstone'
         ORDER BY 1",
    )
    .fetch_all(&pool)
    .await
    .expect("read the tombstone triggers");

    let mut allowed: Vec<String> = cuba_memorys::handlers::sync::TOMBSTONED_TABLES
        .iter()
        .map(|(t, _)| t.to_string())
        .collect();
    allowed.sort();

    assert_eq!(
        with_triggers, allowed,
        "these two lists have to be the same set, and they drifted the moment a migration \
         added tombstone triggers for two more tables while the import's allow-list still \
         named six. The gate caught it as a refused bundle — which is the safe direction, \
         because the import refuses a tombstone for a table it was not built to delete from \
         rather than guessing. The unsafe direction is worse and this test covers it too: a \
         table that stops writing tombstones while the import still deletes from it means \
         deletions stop travelling and nothing says so"
    );

    for (table, key) in cuba_memorys::handlers::sync::TOMBSTONED_TABLES {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns
                            WHERE table_name = $1 AND column_name = $2)",
        )
        .bind(table)
        .bind(key)
        .fetch_one(&pool)
        .await
        .expect("check the key column");
        assert!(
            exists,
            "the import deletes from {table} by {key}, and that column does not exist. \
             brain_facts is keyed by fact_id, not id — assuming every table names its key \
             `id` is how the tombstone trigger itself broke before migration 0049"
        );
    }
}
