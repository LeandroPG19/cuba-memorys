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
//! Thresholds typically calibrated on a validation set. Default τ = 5.0
//! corresponds to ~5σ in 384-dim space, conservative.
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

/// Default OOD threshold (Mahalanobis distance). Calibrated for 384-dim
/// e5-small embeddings; revisit when migrating to 1024-dim BGE-M3 in v1.0.
pub const DEFAULT_OOD_THRESHOLD: f64 = 5.0;

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
    fn in_distribution_query_below_threshold() {
        let center = vec![0.5_f32; 16];
        let samples = random_around(&center, 200, 0.05);
        let stats = OodStats::fit(&samples).expect("fit");
        let in_dist = vec![0.5_f32; 16];
        let dist = stats.mahalanobis(&in_dist).expect("mahalanobis");
        assert!(
            dist < DEFAULT_OOD_THRESHOLD,
            "in-distribution query should be below threshold, got {dist}"
        );
    }

    #[test]
    fn out_of_distribution_query_above_threshold() {
        let center = vec![0.0_f32; 16];
        let samples = random_around(&center, 200, 0.01);
        let stats = OodStats::fit(&samples).expect("fit");
        // Far from the cluster: every dim shifted by 5σ
        let ood = vec![10.0_f32; 16];
        let dist = stats.mahalanobis(&ood).expect("mahalanobis");
        assert!(
            dist > DEFAULT_OOD_THRESHOLD,
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
