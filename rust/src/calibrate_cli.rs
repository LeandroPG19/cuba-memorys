//! `cuba-memorys calibrate` — diagnose and fix the abstention threshold.
//!
//! Reports what the OOD score actually looks like on this corpus, and computes
//! the conformal threshold from real queries instead of from a χ² assumption the
//! embeddings violate. See [`crate::search::calibrate`] for why theory fails here.

use anyhow::{Context, Result};

use crate::search::calibrate::{self, DEFAULT_ALPHA};

/// Queries used to calibrate. They must be *answerable* from the corpus: the
/// conformal guarantee is about not rejecting queries like these.
///
/// Read from the eval dataset when one is given, so the calibration set is the
/// same distribution the benchmark grades on.
fn load_queries(path: Option<&str>) -> Result<Vec<String>> {
    let Some(path) = path else {
        anyhow::bail!(
            "hace falta un dataset de calibración: --dataset rust/tests/datasets/longmemeval_abilities_es.jsonl\n\
             Deben ser consultas RESPONDIBLES: la garantía conformal es sobre no rechazarlas."
        );
    };
    let text = std::fs::read_to_string(path).with_context(|| format!("leyendo {path}"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).context("línea JSONL inválida")?;
        // Abstention samples are out-of-distribution on purpose — including them
        // in the calibration set would calibrate the threshold to accept exactly
        // what it exists to reject.
        if v.get("abstain").and_then(serde_json::Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(q) = v.get("query").and_then(serde_json::Value::as_str) {
            out.push(q.to_string());
        }
    }
    Ok(out)
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut dataset: Option<String> = None;
    let mut alpha = DEFAULT_ALPHA;
    let mut sample_limit: i64 = 5000;
    let mut json = false;
    let mut apply = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dataset" | "-d" => dataset = it.next().cloned(),
            "--alpha" => {
                alpha = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--alpha needs a float, e.g. 0.05")?
            }
            "--samples" => {
                sample_limit = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--samples needs an integer")?
            }
            "--json" => json = true,
            "--apply" => apply = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys calibrate --dataset PATH.jsonl [--alpha 0.05] [--samples N] [--json]\n\n\
                     Diagnoses the OOD abstention threshold and computes a conformal one.\n\
                     The theoretical χ² cutoff assumes Gaussian embeddings and a well-estimated\n\
                     covariance; e5 normalizes to the unit sphere and the covariance is fitted\n\
                     from 500 samples in 384 dimensions. Neither holds, and abstention rejects\n\
                     100% of answerable queries as a result.\n\n\
                     --alpha  the false-abstention rate you accept (default 0.05)\n\
                     --apply  persist the threshold so the server uses it (brain_calibration)."
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown calibrate flag: {other} (try --help)"),
        }
    }

    let queries = load_queries(dataset.as_deref())?;
    if queries.is_empty() {
        anyhow::bail!("el dataset no tiene consultas respondibles");
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for calibration")?;

    let report = calibrate::calibrate(&pool, &queries, alpha, sample_limit).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("Calibración del umbral de abstención (OOD)\n");
    println!("  dimensión del embedding : {}", report.embedding_dim);
    println!("  muestras para ajustar Σ : {}", report.fit_samples);
    println!(
        "  ratio n/d               : {:.2}  {}",
        report.fit_samples as f64 / report.embedding_dim as f64,
        if (report.fit_samples as f64) < 10.0 * report.embedding_dim as f64 {
            "← demasiado bajo: Σ mal condicionada, su inversa amplifica ruido"
        } else {
            ""
        }
    );
    println!();

    println!(
        "  umbral TEÓRICO (χ², Wilson-Hilferty) : {:.2}",
        report.theoretical_threshold
    );
    if let Some(c) = &report.corpus {
        println!(
            "\n  Distancias del CORPUS contra su propia distribución (n={}):",
            c.n
        );
        println!(
            "    min={:.1}  p50={:.1}  p90={:.1}  p95={:.1}  p99={:.1}  max={:.1}",
            c.min, c.p50, c.p90, c.p95, c.p99, c.max
        );
        println!(
            "    → el umbral teórico rechaza el {:.1}% del propio corpus (debería ser ~1%)",
            report.theoretical_rejects_corpus * 100.0
        );
    }
    if let Some(q) = &report.queries {
        println!("\n  Distancias de las CONSULTAS respondibles (n={}):", q.n);
        println!(
            "    min={:.1}  p50={:.1}  p90={:.1}  p95={:.1}  p99={:.1}  max={:.1}",
            q.min, q.p50, q.p90, q.p95, q.p99, q.max
        );
    }

    println!();
    match report.conformal_threshold {
        Some(t) => {
            println!(
                "  umbral CONFORMAL (α={:.2})           : {:.2}",
                report.alpha, t
            );
            println!(
                "\n  Garantía: como mucho el {:.0}% de las consultas respondibles futuras\n  \
                 serán rechazadas por error — sin asumir gaussianidad ni una Σ bien estimada.",
                report.alpha * 100.0
            );
            if apply {
                calibrate::store_ood_threshold(&pool, t, &report).await?;
                println!("\n  Guardado en brain_calibration. El servidor ya lo usa.");
            } else {
                println!("\n  Esto fue un diagnóstico — no se guardó nada.");
                println!(
                    "  Para que el servidor lo use:  cuba-memorys calibrate --dataset ... --apply"
                );
            }
        }
        None => println!("  no se pudo calcular el umbral conformal (sin consultas)"),
    }
    Ok(())
}
