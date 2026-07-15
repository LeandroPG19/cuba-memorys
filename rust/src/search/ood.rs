use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct OodStats {
    pub mean: DVector<f64>,
    pub inverse_covariance: DMatrix<f64>,
    pub n_samples: usize,
    pub shrinkage: f64,
}

impl OodStats {
    pub fn fit(embeddings: &[Vec<f32>]) -> Option<Self> {
        let n = embeddings.len();
        if n < 2 {
            return None;
        }
        let d = embeddings[0].len();
        if d == 0 || embeddings.iter().any(|e| e.len() != d) {
            return None;
        }

        let mut mean = DVector::<f64>::zeros(d);
        for e in embeddings {
            for (i, &v) in e.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        mean /= n as f64;

        let mut cov = DMatrix::<f64>::zeros(d, d);
        let mut centered: Vec<DVector<f64>> = Vec::with_capacity(n);
        for e in embeddings {
            let mut diff = DVector::<f64>::zeros(d);
            for (i, &v) in e.iter().enumerate() {
                diff[i] = v as f64 - mean[i];
            }
            cov += &diff * diff.transpose();
            centered.push(diff);
        }
        cov /= (n - 1).max(1) as f64;

        let mu = (0..d).map(|i| cov[(i, i)]).sum::<f64>() / d as f64;

        let mut target_dist = 0.0;
        for i in 0..d {
            for j in 0..d {
                let t = if i == j { mu } else { 0.0 };
                let diff = cov[(i, j)] - t;
                target_dist += diff * diff;
            }
        }
        target_dist /= d as f64;

        let mut b_bar = 0.0;
        for x in &centered {
            let mut acc = 0.0;
            for i in 0..d {
                for j in 0..d {
                    let outer = x[i] * x[j];
                    let diff = outer - cov[(i, j)];
                    acc += diff * diff;
                }
            }
            b_bar += acc / d as f64;
        }
        b_bar /= (n * n) as f64;
        let b2 = b_bar.min(target_dist);

        let intensity = if target_dist > f64::EPSILON {
            (b2 / target_dist).clamp(0.0, 1.0)
        } else {
            0.0
        };

        for i in 0..d {
            for j in 0..d {
                let target = if i == j { mu } else { 0.0 };
                cov[(i, j)] = intensity * target + (1.0 - intensity) * cov[(i, j)];
            }
        }

        let floor = (mu * 1e-6).max(f64::MIN_POSITIVE);
        for i in 0..d {
            cov[(i, i)] += floor;
        }

        let inverse_covariance = cov.try_inverse()?;

        Some(Self {
            mean,
            inverse_covariance,
            n_samples: n,
            shrinkage: intensity,
        })
    }

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

    pub fn is_ood(&self, query: &[f32], threshold: f64) -> bool {
        self.mahalanobis(query).is_some_and(|d| d > threshold)
    }
}

const Z_99: f64 = 2.326_347_9;

pub fn default_threshold(dim: usize) -> f64 {
    if dim == 0 {
        return 0.0;
    }
    let d = dim as f64;
    let a = 2.0 / (9.0 * d);
    let chi2 = d * (1.0 - a + Z_99 * a.sqrt()).powi(3);
    chi2.max(0.0).sqrt()
}

pub const MIN_SAMPLES_FOR_OOD: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;

    fn random_around(center: &[f32], n: usize, jitter: f32) -> Vec<Vec<f32>> {
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

    #[test]
    fn typical_in_distribution_point_is_not_ood() {
        let d = 64;
        let center = vec![0.5_f32; d];
        let samples = random_around(&center, 400, 0.05);
        let stats = OodStats::fit(&samples).expect("fit");
        let tau = default_threshold(d);

        let flagged = samples.iter().filter(|s| stats.is_ood(s, tau)).count();
        let rate = flagged as f64 / samples.len() as f64;
        assert!(
            rate < 0.05,
            "at most ~1% of in-distribution samples should be OOD at p=0.99, got {:.1}%",
            rate * 100.0
        );

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
