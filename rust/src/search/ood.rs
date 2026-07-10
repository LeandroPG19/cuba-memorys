//! Out-of-Distribution detection via Mahalanobis distance.
//!
//! Lee, K., Lee, K., Lee, H., & Shin, J. (2018).
//! "A Simple Unified Framework for Detecting Out-of-Distribution Samples
//! and Adversarial Attacks." NeurIPS 2018.
//!
//! ## Why this matters for memory retrieval
//!
//! When a query is genuinely out-of-distribution with respect to what we
//! have stored, returning the K nearest neighbors yields garbage with low
//! confidence — encouraging hallucination. LongMemEval explicitly measures
//! the *abstention* dimension: knowing when to say "I don't have this".
//!
//! Mahalanobis(x, μ, Σ) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
//!
//! The threshold scales with dimensionality — see [`default_threshold`]. A
//! typical in-distribution point lies at `M ≈ sqrt(d)` (χ² with d dof), so a
//! fixed small τ abstains on everything.
//!
//! ## Caching
//!
//! μ and Σ⁻¹ are recomputed in the REM cycle (`cuba_zafra`) — they do not
//! change at request time. The result is a single matrix-vector product per
//! query, O(d²) — for d=384 that is ≈150K mul-adds, sub-millisecond.

use nalgebra::{DMatrix, DVector};

/// Cached statistics of the active embedding distribution.
/// Populated by `cuba_zafra` REM cycle, read by `cuba_faro` per-query.
#[derive(Debug, Clone)]
pub struct OodStats {
    pub mean: DVector<f64>,
    pub inverse_covariance: DMatrix<f64>,
    /// Number of samples used to fit μ and Σ. Caller can decide to skip
    /// OOD if too few samples (high variance estimate).
    pub n_samples: usize,
}

impl OodStats {
    /// Estimate μ and Σ⁻¹ from a sample of embeddings.
    ///
    /// `embeddings` is a flat row-major matrix: `embeddings.len() == n × d`.
    ///
    /// Returns `None` when there are fewer than 2 samples (covariance undefined),
    /// the sample is degenerate (rank-deficient covariance), or matrix inversion
    /// fails. Callers should fall back to skipping OOD checks in those cases.
    pub fn fit(embeddings: &[Vec<f32>]) -> Option<Self> {
        let n = embeddings.len();
        if n < 2 {
            return None;
        }
        let d = embeddings[0].len();
        if d == 0 || embeddings.iter().any(|e| e.len() != d) {
            return None;
        }

        // Compute mean μ
        let mut mean = DVector::<f64>::zeros(d);
        for e in embeddings {
            for (i, &v) in e.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        mean /= n as f64;

        // Compute covariance Σ with shrinkage (Ledoit-Wolf style ridge)
        // to prevent singular Σ when d > n or when features are highly correlated.
        let mut cov = DMatrix::<f64>::zeros(d, d);
        for e in embeddings {
            let mut diff = DVector::<f64>::zeros(d);
            for (i, &v) in e.iter().enumerate() {
                diff[i] = v as f64 - mean[i];
            }
            cov += &diff * diff.transpose();
        }
        cov /= (n - 1).max(1) as f64;

        // Ridge regularization: Σ + ε·I. ε scaled to trace so it survives
        // dimension changes (384 → 1024 with BGE-M3 in v1.0).
        let trace = (0..d).map(|i| cov[(i, i)]).sum::<f64>();
        let epsilon = (trace / d as f64) * 1e-3;
        for i in 0..d {
            cov[(i, i)] += epsilon;
        }

        let inverse_covariance = cov.try_inverse()?;

        Some(Self {
            mean,
            inverse_covariance,
            n_samples: n,
        })
    }

    /// Mahalanobis distance from `query` to the fitted distribution.
    /// Returns `None` if the query has the wrong dimensionality.
    pub fn mahalanobis(&self, query: &[f32]) -> Option<f64> {
        if query.len() != self.mean.len() {
            return None;
        }
        let mut diff = DVector::<f64>::zeros(query.len());
        for (i, &v) in query.iter().enumerate() {
            diff[i] = v as f64 - self.mean[i];
        }
        let m2 = (diff.transpose() * &self.inverse_covariance * &diff)[(0, 0)];
        Some(m2.max(0.0).sqrt())
    }

    /// Convenience: classify a query as OOD given a threshold.
    pub fn is_ood(&self, query: &[f32], threshold: f64) -> bool {
        self.mahalanobis(query).is_some_and(|d| d > threshold)
    }
}

/// z-score for the 99th percentile of the standard normal. Used by
/// [`default_threshold`] to pick the chi-squared quantile.
const Z_99: f64 = 2.326_347_9;

/// Default OOD threshold for a `dim`-dimensional embedding space.
///
/// # Why this is not a constant
///
/// For data drawn from a `d`-dimensional Gaussian, the *squared* Mahalanobis
/// distance follows a chi-squared distribution with `d` degrees of freedom.
/// A **typical, in-distribution** point therefore sits at `M ≈ sqrt(d)`, not
/// near zero: for `d = 384` that is `19.6`.
///
/// The previous hard-coded `5.0` — documented as "~5σ in 384-dim space" —
/// misread Mahalanobis as if it were a per-axis z-score. Measured against the
/// live corpus (n=1215, d=384), it flagged **100% of in-distribution
/// observations as OOD**: median distance 19.60, min 10.69. Abstention was
/// unconditional.
///
/// We instead return `sqrt(χ²_{0.99}(d))`, the radius enclosing 99% of the
/// distribution, via the Wilson–Hilferty cube-root approximation:
///
/// ```text
/// χ²_p(d) ≈ d · (1 − 2/(9d) + z_p · sqrt(2/(9d)))³
/// ```
///
/// It is accurate to <0.1% for `d ≥ 30`. For `d = 384` it yields `τ ≈ 21.25`,
/// matching `scipy.stats.chi2.ppf(0.99, 384)` and leaving the empirical corpus
/// with a ~10% abstention rate at the tail — the intended behavior.
pub fn default_threshold(dim: usize) -> f64 {
    if dim == 0 {
        return 0.0;
    }
    let d = dim as f64;
    let a = 2.0 / (9.0 * d);
    let chi2 = d * (1.0 - a + Z_99 * a.sqrt()).powi(3);
    chi2.max(0.0).sqrt()
}

/// Minimum samples required before OOD detection is meaningful. Below this,
/// the covariance estimate is too noisy and we skip the check entirely.
pub const MIN_SAMPLES_FOR_OOD: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;

    fn random_around(center: &[f32], n: usize, jitter: f32) -> Vec<Vec<f32>> {
        // Deterministic pseudo-random — no rand crate to keep test infra small.
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = center.to_vec();
            for (j, x) in v.iter_mut().enumerate() {
                let h = ((i * 7 + j) as u32).wrapping_mul(2654435761);
                let f = ((h % 1000) as f32 / 1000.0 - 0.5) * 2.0 * jitter;
                *x += f;
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn fit_returns_none_on_insufficient_samples() {
        assert!(OodStats::fit(&[]).is_none());
        assert!(OodStats::fit(&[vec![1.0, 2.0, 3.0]]).is_none());
    }

    #[test]
    fn fit_returns_none_on_inconsistent_dims() {
        let bad = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        assert!(OodStats::fit(&bad).is_none());
    }

    #[test]
    fn threshold_tracks_sqrt_of_dimension() {
        // Wilson–Hilferty must agree with scipy.stats.chi2.ppf(0.99, df).
        // Reference values: chi2.ppf(0.99, 384) = 451.4 -> sqrt = 21.25
        //                   chi2.ppf(0.99, 1024) = 1131.2 -> sqrt = 33.63
        let t384 = default_threshold(384);
        assert!(
            (t384 - 21.25).abs() < 0.1,
            "tau(384) should be ~21.25, got {t384}"
        );
        let t1024 = default_threshold(1024);
        assert!(
            (t1024 - 33.63).abs() < 0.15,
            "tau(1024) should be ~33.63, got {t1024}"
        );
        assert_eq!(default_threshold(0), 0.0);
    }

    /// Regression: the old fixed tau = 5.0 flagged 100% of the live corpus as
    /// OOD, because a typical point sits at M ~ sqrt(d), not near zero. The
    /// previous test only probed the mean itself (distance ~0) and so never
    /// caught it.
    #[test]
    fn typical_in_distribution_point_is_not_ood() {
        let d = 64;
        let center = vec![0.5_f32; d];
        let samples = random_around(&center, 400, 0.05);
        let stats = OodStats::fit(&samples).expect("fit");
        let tau = default_threshold(d);

        // Probe actual samples, not the centroid.
        let flagged = samples.iter().filter(|s| stats.is_ood(s, tau)).count();
        let rate = flagged as f64 / samples.len() as f64;
        assert!(
            rate < 0.05,
            "at most ~1% of in-distribution samples should be OOD at p=0.99, got {:.1}%",
            rate * 100.0
        );

        // And the old constant must reject essentially everything, which is
        // exactly why it was wrong.
        let flagged_old = samples.iter().filter(|s| stats.is_ood(s, 5.0)).count();
        assert!(
            flagged_old > samples.len() / 2,
            "sanity: the legacy tau=5.0 over-abstains (that was the bug)"
        );
    }

    #[test]
    fn out_of_distribution_query_above_threshold() {
        let d = 16;
        let center = vec![0.0_f32; d];
        let samples = random_around(&center, 200, 0.01);
        let stats = OodStats::fit(&samples).expect("fit");
        // Far from the cluster: every dim shifted well beyond the spread.
        let ood = vec![10.0_f32; d];
        let dist = stats.mahalanobis(&ood).expect("mahalanobis");
        assert!(
            dist > default_threshold(d),
            "OOD query should exceed threshold, got {dist}"
        );
    }

    #[test]
    fn wrong_dim_query_returns_none() {
        let center = vec![0.0_f32; 8];
        let samples = random_around(&center, 50, 0.1);
        let stats = OodStats::fit(&samples).expect("fit");
        assert!(stats.mahalanobis(&[0.0_f32; 16]).is_none());
    }
}
