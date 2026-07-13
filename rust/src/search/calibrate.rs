//! Conformal calibration of the OOD abstention threshold.
//!
//! ## Why the theoretical threshold cannot work here
//!
//! [`super::ood`] derives its cutoff from theory: if `x ~ N(μ, Σ)` then
//! `M²(x) ~ χ²(d)`, so a typical point sits at `M ≈ √d` and the 99th percentile
//! for `d = 384` is `τ ≈ 21.25`. That derivation needs two things this corpus
//! does not provide.
//!
//! **The covariance is not estimable at this sample size.** The REM cycle fits
//! Σ from 500 embeddings in 384 dimensions — 147,456 covariance parameters from
//! 500 samples, `n/d ≈ 1.3`. The sample covariance is then near-singular, and
//! inverting it amplifies estimation noise rather than the signal. Worse, those
//! 500 are taken `ORDER BY importance DESC`, so they are not even a random
//! sample of the distribution they claim to describe.
//!
//! **The embeddings are not Gaussian.** e5 L2-normalizes, so every vector lives
//! on the unit sphere. A distribution supported on a sphere has zero variance in
//! the radial direction; Σ is singular there by construction. The ridge term
//! makes it invertible with a tiny ε, and `1/ε` is enormous — so an infinitesimal
//! radial deviation produces an enormous Mahalanobis distance.
//!
//! Both effects push every query above any threshold derived from χ². Measured:
//! with abstention on, cuba abstained on **100% of answerable queries**.
//!
//! ## What this module does instead
//!
//! Conformal prediction (Vovk et al.; see also Angelopoulos & Bates 2021).
//! Instead of assuming a distribution and computing a quantile from theory, take
//! a calibration set of queries that are known to be in-distribution, score them,
//! and read the quantile off the empirical scores:
//!
//! ```text
//!   τ = the ⌈(n+1)(1-α)⌉-th smallest calibration score
//! ```
//!
//! This gives a finite-sample, distribution-free guarantee: the probability of
//! wrongly abstaining on a fresh exchangeable in-distribution query is at most α.
//! No Gaussianity, no well-conditioned Σ, no `n ≫ d` — the guarantee holds even
//! when the underlying score is a badly estimated Mahalanobis distance, because
//! the threshold adapts to whatever scale that score actually produces.
//!
//! The literature is explicit that a globally fixed risk level is the weak point
//! of naive CP (Conformalized Abstention Policies). We take the tractable half:
//! calibrate the threshold on real data instead of inheriting it from a theorem
//! whose assumptions are violated.

use anyhow::{Context, Result};
use sqlx::{PgPool, Row};

use super::ood::OodStats;

/// Miscoverage rate: the fraction of answerable queries we accept losing to a
/// false abstention. 5% is the conventional starting point; the CLI overrides it.
pub const DEFAULT_ALPHA: f64 = 0.05;

/// The empirical distribution of a score, for reporting.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Distribution {
    pub n: usize,
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Distribution {
    /// `scores` need not be sorted.
    fn from(mut scores: Vec<f64>) -> Option<Self> {
        if scores.is_empty() {
            return None;
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f64| -> f64 {
            let idx = ((scores.len() as f64 - 1.0) * q).round() as usize;
            scores[idx.min(scores.len() - 1)]
        };
        Some(Self {
            n: scores.len(),
            min: scores[0],
            p50: at(0.50),
            p90: at(0.90),
            p95: at(0.95),
            p99: at(0.99),
            max: scores[scores.len() - 1],
        })
    }
}

/// The conformal quantile of a calibration set.
///
/// Returns the ⌈(n+1)(1−α)⌉-th smallest score, which is the smallest threshold τ
/// such that at most an α fraction of *future* exchangeable in-distribution
/// scores exceed it. The `(n+1)` — rather than `n` — is what buys the
/// finite-sample guarantee instead of an asymptotic one.
///
/// When `⌈(n+1)(1−α)⌉ > n` the calibration set is too small to certify α at all
/// (with n=10 you cannot promise a 1% error rate), and we return the maximum
/// score: the most conservative threshold the data can justify.
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Option<f64> {
    if scores.is_empty() {
        return None;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len() as f64;
    let rank = ((n + 1.0) * (1.0 - alpha)).ceil() as usize;
    if rank == 0 {
        return Some(sorted[0]);
    }
    if rank > sorted.len() {
        // Not enough calibration points to certify this α — be conservative.
        return Some(sorted[sorted.len() - 1]);
    }
    Some(sorted[rank - 1])
}

/// Fit the OOD distribution the way the REM cycle does — but on the whole corpus
/// and without the importance bias, so the estimate at least describes the
/// distribution it is meant to describe.
async fn fit_stats(pool: &PgPool, sample_limit: i64) -> Result<(OodStats, usize)> {
    let rows = sqlx::query(
        // Identical to what check_ood fits on. Calibrating against a different
        // sample would calibrate a threshold for a distribution the server never uses.
        "SELECT embedding FROM brain_observations
         WHERE embedding IS NOT NULL AND observation_type != 'superseded'
         ORDER BY id LIMIT $1",
    )
    .bind(sample_limit)
    .fetch_all(pool)
    .await
    .context("reading embeddings for calibration")?;

    let embeddings: Vec<Vec<f32>> = rows
        .iter()
        .filter_map(|r| {
            r.try_get::<pgvector::Vector, _>("embedding")
                .ok()
                .map(|v| v.to_vec())
        })
        .collect();

    let n = embeddings.len();
    let stats = OodStats::fit(&embeddings)
        .context("could not fit the embedding distribution (too few or degenerate samples)")?;
    Ok((stats, n))
}

/// Score every stored observation against the fitted distribution.
///
/// These points are in-distribution *by construction* — they are the corpus. If
/// their own Mahalanobis distances already sit above the theoretical threshold,
/// the threshold is not measuring what it thinks it is measuring.
pub async fn corpus_distances(pool: &PgPool, stats: &OodStats, limit: i64) -> Result<Vec<f64>> {
    let rows = sqlx::query(
        "SELECT embedding FROM brain_observations
         WHERE embedding IS NOT NULL AND observation_type != 'superseded'
         ORDER BY id LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .filter_map(|r| r.try_get::<pgvector::Vector, _>("embedding").ok())
        .filter_map(|v| stats.mahalanobis(&v.to_vec()))
        .collect())
}

/// Score a set of real queries, embedded exactly as `faro` embeds them.
pub async fn query_distances(stats: &OodStats, queries: &[String]) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        // `embed_passage`, not `embed`: e5 is asymmetric, and the density was
        // estimated over passages. Scoring a "query: "-prefixed vector against a
        // passage distribution measures the prefix, not the query.
        let v = crate::embeddings::onnx::embed_passage(q)
            .await
            .with_context(|| format!("embedding calibration query: {q}"))?;
        if let Some(d) = stats.mahalanobis(&v) {
            out.push(d);
        }
    }
    Ok(out)
}

#[derive(Debug, serde::Serialize)]
pub struct CalibrationReport {
    pub embedding_dim: usize,
    pub fit_samples: usize,
    /// The threshold χ² theory prescribes — the one that abstained on everything.
    pub theoretical_threshold: f64,
    pub corpus: Option<Distribution>,
    pub queries: Option<Distribution>,
    pub alpha: f64,
    /// The threshold calibrated on real in-distribution queries.
    pub conformal_threshold: Option<f64>,
    /// Share of the corpus the theoretical threshold would reject. Should be ~1%.
    pub theoretical_rejects_corpus: f64,
}

/// Run the whole diagnosis: fit, score, and compute the conformal threshold.
pub async fn calibrate(
    pool: &PgPool,
    calibration_queries: &[String],
    alpha: f64,
    sample_limit: i64,
) -> Result<CalibrationReport> {
    let (stats, fit_samples) = fit_stats(pool, sample_limit).await?;
    let dim = crate::embeddings::onnx::embedding_dim();
    let theoretical = super::ood::default_threshold(dim);

    let corpus_d = corpus_distances(pool, &stats, sample_limit).await?;
    let rejects = if corpus_d.is_empty() {
        0.0
    } else {
        corpus_d.iter().filter(|&&d| d > theoretical).count() as f64 / corpus_d.len() as f64
    };

    let query_d = query_distances(&stats, calibration_queries).await?;
    let conformal = conformal_quantile(&query_d, alpha);

    Ok(CalibrationReport {
        embedding_dim: dim,
        fit_samples,
        theoretical_threshold: theoretical,
        corpus: Distribution::from(corpus_d),
        queries: Distribution::from(query_d),
        alpha,
        conformal_threshold: conformal,
        theoretical_rejects_corpus: rejects,
    })
}

/// Key under which the OOD threshold is stored.
pub const OOD_THRESHOLD_KEY: &str = "ood_threshold";

/// Persist a calibrated threshold, together with the context that makes it
/// interpretable. A value calibrated for 384-d embeddings is meaningless for a
/// 1024-d corpus, so the dimension travels with it and [`load_ood_threshold`]
/// refuses to hand back a threshold that was measured for a different space.
pub async fn store_ood_threshold(
    pool: &PgPool,
    threshold: f64,
    report: &CalibrationReport,
) -> Result<()> {
    let metadata = serde_json::json!({
        "embedding_dim": report.embedding_dim,
        "alpha": report.alpha,
        "fit_samples": report.fit_samples,
        "calibration_queries": report.queries.as_ref().map(|q| q.n),
        "theoretical_threshold": report.theoretical_threshold,
        "method": "conformal",
    });
    sqlx::query(
        "INSERT INTO brain_calibration (key, value, metadata, updated_at)
         VALUES ($1, $2, $3, now())
         ON CONFLICT (key) DO UPDATE
           SET value = EXCLUDED.value,
               metadata = EXCLUDED.metadata,
               updated_at = now()",
    )
    .bind(OOD_THRESHOLD_KEY)
    .bind(threshold)
    .bind(&metadata)
    .execute(pool)
    .await
    .context("persisting the calibrated OOD threshold")?;
    Ok(())
}

/// The calibrated threshold, if one was measured for *this* embedding dimension.
///
/// Returns `None` rather than a stale number when the dimension has changed —
/// after a bge-m3 migration the old 384-d threshold is not conservative, it is
/// simply about a different space, and using it would silently resurrect the
/// abstain-on-everything bug.
pub async fn load_ood_threshold(pool: &PgPool, dim: usize) -> Option<f64> {
    let row = sqlx::query(
        "SELECT value, metadata FROM brain_calibration WHERE key = $1",
    )
    .bind(OOD_THRESHOLD_KEY)
    .fetch_optional(pool)
    .await
    .ok()??;

    let value: f64 = row.try_get("value").ok()?;
    let metadata: serde_json::Value = row.try_get("metadata").ok()?;
    let stored_dim = metadata.get("embedding_dim").and_then(serde_json::Value::as_u64)? as usize;

    if stored_dim != dim {
        tracing::warn!(
            stored_dim,
            current_dim = dim,
            "calibrated OOD threshold was measured for a different embedding dimension — ignoring it; re-run `cuba-memorys calibrate`"
        );
        return None;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_conformal_quantile_is_the_rank_the_theory_asks_for() {
        // n=9, α=0.1 → ⌈10 × 0.9⌉ = 9 → the 9th smallest = the max.
        let scores: Vec<f64> = (1..=9).map(f64::from).collect();
        assert_eq!(conformal_quantile(&scores, 0.1), Some(9.0));
        // n=9, α=0.5 → ⌈10 × 0.5⌉ = 5 → the 5th smallest.
        assert_eq!(conformal_quantile(&scores, 0.5), Some(5.0));
    }

    #[test]
    fn too_few_points_to_certify_alpha_gives_the_most_conservative_threshold() {
        // With 10 points you cannot honestly promise a 1% error rate:
        // ⌈11 × 0.99⌉ = 11 > 10. Returning the max is the only threshold the
        // data supports; silently pretending otherwise is how you ship a
        // guarantee that does not hold.
        let scores: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(conformal_quantile(&scores, 0.01), Some(10.0));
    }

    #[test]
    fn an_empty_calibration_set_yields_no_threshold() {
        assert_eq!(conformal_quantile(&[], 0.05), None);
    }

    #[test]
    fn the_distribution_summary_is_ordered() {
        let d = Distribution::from(vec![5.0, 1.0, 3.0, 2.0, 4.0]).expect("non-empty");
        assert_eq!(d.n, 5);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.max, 5.0);
        assert!(d.p50 <= d.p90 && d.p90 <= d.p99);
    }
}
