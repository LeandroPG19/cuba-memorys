use anyhow::{Context, Result};

pub async fn run_cli(args: &[String]) -> Result<()> {
    if let Some(a) = args.first() {
        if matches!(a.as_str(), "-h" | "--help") {
            eprintln!(
                "usage: cuba-memorys rem\n\n\
                 Runs one consolidation cycle now instead of waiting for the server's\n\
                 4-hour timer: stratified decay, episode decay, NPMI autolink, embedding\n\
                 backfill, PageRank. Safe to run repeatedly — every step is idempotent.\n\n\
                 CUBA_REM_AUTOLINK=0        skip the autolink step\n\
                 CUBA_REM_BACKFILL_LIMIT=N  cap embeddings recomputed per cycle (0 = skip)"
            );
            return Ok(());
        }
        anyhow::bail!("unknown rem flag: {a} (try --help)");
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for the REM cycle")?;

    let missing_before = crate::embeddings::backfill::count_missing(&pool)
        .await
        .unwrap_or(0);

    crate::protocol::run_rem_consolidation(&pool)
        .await
        .context("REM consolidation failed")?;

    let missing_after = crate::embeddings::backfill::count_missing(&pool)
        .await
        .unwrap_or(0);

    let report = serde_json::json!({
        "action": "rem",
        "embeddings_missing_before": missing_before,
        "embeddings_missing_after": missing_after,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
