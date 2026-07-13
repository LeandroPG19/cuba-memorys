pub mod datasets;
pub mod harness;
pub mod metrics;
pub mod reporters;

use anyhow::{Context, Result};

/// CLI entrypoint for the retrieval benchmark:
///   `cuba-memorys eval [--dataset PATH.jsonl] [--k N] [--json]`
///
/// Runs the production hybrid `cuba_faro` per sample and reports nDCG@k, MRR,
/// P@k, R@k. **Non-mutating**: the harness passes `track_access=false`, so the
/// Testing Effect boost that a normal search applies is skipped — running the
/// benchmark does not change importance/access of the corpus it measures, and
/// is safe against the live database. Without `--dataset` it uses the built-in
/// smoke set.
///
/// This is the entrypoint the harness lacked (the modules existed with zero
/// callers), so no mechanism in the cognitive/graph stack had a way to prove it
/// helped. Phase 1 of docs/PLAN-MEJORAS-v0.11.md: measure first.
pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut dataset_path: Option<String> = None;
    let mut json = false;
    let mut cfg = harness::EvalConfig::default();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dataset" | "-d" => dataset_path = it.next().cloned(),
            "--k" => {
                cfg.k = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--k needs an integer")?
            }
            "--json" => json = true,
            "--associative" => cfg.associative = true,
            // These two exist because the harness used to hard-code them off and
            // then report that they scored zero.
            "--abstain" => cfg.abstain = true,
            "--rerank" => cfg.rerank = true,
            "--format" => {
                cfg.format = it
                    .next()
                    .cloned()
                    .context("--format needs verbose|compact")?
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys eval [--dataset PATH.jsonl] [--k N]\n\
                     \x20                        [--associative] [--abstain] [--rerank]\n\
                     \x20                        [--format verbose|compact] [--json]\n\n\
                     --associative  multi-hop expansion (v0.11)\n\
                     --abstain      let the OOD gate fire, so abstention is actually exercised\n\
                     --rerank       run the cross-encoder reranker\n\
                     --format       response shape whose token cost is measured (default verbose)\n\n\
                     Every run reports mean/max response tokens: quality that costs twice the\n\
                     context is not free, and you cannot see that without printing both.\n\n\
                     JSONL row: {{\"query\": \"...\", \"relevant_markers\": [\"...\"], \"expected_answer\": \"...\"?}}"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown eval flag: {other} (try --help)"),
        }
    }

    let samples = match &dataset_path {
        Some(p) => {
            datasets::load_jsonl_dataset(p).with_context(|| format!("loading dataset {p}"))?
        }
        None => datasets::builtin_retrieval_set(),
    };
    if samples.is_empty() {
        anyhow::bail!("dataset is empty — nothing to evaluate");
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for eval")?;

    let report = harness::run_faro_eval(&pool, &samples, &cfg).await?;

    if json {
        println!(
            "{}",
            reporters::generate_json_report(&report, samples.len(), cfg.k)
        );
    } else {
        eprintln!(
            "eval dataset={} samples={} k={} associative={} abstain={} rerank={} format={}",
            dataset_path.as_deref().unwrap_or("<builtin>"),
            samples.len(),
            cfg.k,
            cfg.associative,
            cfg.abstain,
            cfg.rerank,
            cfg.format,
        );
        println!("{}", reporters::summary_line(&report));
    }
    Ok(())
}
