pub mod datasets;
pub mod harness;
pub mod metrics;
pub mod reporters;

use anyhow::{Context, Result};

fn per_query_ndcg(path: &str) -> Result<Vec<f64>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("{path} is not valid JSON"))?;
    let scores = doc
        .pointer("/metrics/per_query_ndcg")
        .and_then(|v| v.as_array())
        .with_context(|| {
            format!("{path} has no metrics.per_query_ndcg — rerun that arm with --json")
        })?;
    Ok(scores.iter().filter_map(|v| v.as_f64()).collect())
}

fn compare_runs(before: &str, after: &str) -> Result<()> {
    let a = per_query_ndcg(before)?;
    let b = per_query_ndcg(after)?;

    let Some((mean, lo, hi)) = metrics::paired_bootstrap(&a, &b, 2000, 0.95) else {
        anyhow::bail!(
            "cannot pair {} questions against {} — the two runs did not score the same dataset",
            a.len(),
            b.len()
        );
    };
    let mde = metrics::minimum_detectable_effect_paired(&a, &b);
    let moved = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| (*y - *x).abs() > 1e-12)
        .count();

    println!("pareado sobre n={} preguntas ({before} → {after})", a.len());
    println!("Δ nDCG@10 = {mean:+.4}  IC95 = [{lo:+.4}, {hi:+.4}]");
    println!("efecto mínimo detectable (pareado) = {mde:.4}");
    println!("preguntas cuyo nDCG cambia: {moved} de {}", a.len());
    if lo > 0.0 || hi < 0.0 {
        println!("el intervalo NO toca cero: la diferencia es real");
    } else {
        println!("el intervalo cruza cero: no se puede distinguir de ruido");
    }
    Ok(())
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut dataset_path: Option<String> = None;
    let mut json = false;
    let mut cfg = harness::EvalConfig::default();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--compare" => {
                let before = it.next().context("--compare needs two JSON reports")?;
                let after = it.next().context("--compare needs two JSON reports")?;
                return compare_runs(before, after);
            }
            "--dataset" | "-d" => dataset_path = it.next().cloned(),
            "--k" => {
                cfg.k = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--k needs an integer")?
            }
            "--json" => json = true,
            "--associative" => cfg.associative = true,
            "--abstain" => cfg.abstain = true,
            "--rerank" => cfg.rerank = true,
            "--format" => {
                cfg.format = it
                    .next()
                    .cloned()
                    .context("--format needs verbose|compact")?
            }
            "--max-tokens" => {
                cfg.max_tokens = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--max-tokens needs an integer")?
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys eval [--dataset PATH.jsonl] [--k N]\n\
                     \x20                        [--associative] [--abstain] [--rerank]\n\
                     \x20                        [--format verbose|compact] [--max-tokens N] [--json]\n\n\
                     --associative  multi-hop expansion (v0.11)\n\
                     --abstain      let the OOD gate fire, so abstention is actually exercised\n\
                     --rerank       run the cross-encoder reranker\n\
                     --format       response shape whose token cost is measured (default verbose)\n\
                     --max-tokens   response budget. Defaults to unlimited so the score measures\n\
                     \x20              the ranking; pass 5000 to reproduce what an MCP client sees\n\
                     --compare A.json B.json   paired bootstrap between two --json runs of the\n\
                     \x20              same dataset. This is the test to accept or reject a change\n\
                     \x20              with; the per-run interval printed below is not.\n\n\
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
        let budget = if cfg.max_tokens == i64::MAX {
            "unlimited".to_string()
        } else {
            cfg.max_tokens.to_string()
        };
        eprintln!(
            "eval dataset={} samples={} k={} associative={} abstain={} rerank={} format={} max_tokens={}",
            dataset_path.as_deref().unwrap_or("<builtin>"),
            samples.len(),
            cfg.k,
            cfg.associative,
            cfg.abstain,
            cfg.rerank,
            cfg.format,
            budget,
        );
        println!("{}", reporters::summary_line(&report));
    }
    Ok(())
}
