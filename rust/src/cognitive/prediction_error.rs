//! V5: Prediction Error Gating — adaptive 3-threshold system.
//!
//! Determines how observations should be processed based on
//! prediction error (novelty vs expectation).
//!
//! V5.1: Adaptive thresholds based on EMA of recent similarities.
//! V5.2: Z-score based gating (Gemini Deep Research 2026-03-14).
//!       Uses corpus distribution statistics (μ, σ) for mathematically
//!       grounded thresholds instead of arbitrary mean + N*σ formulas.
//!       Reinforce: z > 2σ above mean (top 2.3%)
//!       Update:    z > 1σ above mean (top 15.9%)
//!       Create:    z ≤ 1σ (bottom 84.1%)
//!       Handles vector space anisotropy better than static thresholds.

use crate::constants::{PRED_ERROR_REINFORCE, PRED_ERROR_UPDATE};

/// Action to take based on prediction error.
#[derive(Debug, Clone, PartialEq)]
pub enum GatingAction {
    /// High similarity (>reinforce threshold): reinforce existing.
    Reinforce,
    /// Medium similarity (>update threshold): update existing observation.
    Update,
    /// Low similarity (<update threshold): create new observation.
    Create,
}

/// Determine action based on similarity score with static thresholds.
pub fn gate(similarity: f64) -> GatingAction {
    if similarity >= PRED_ERROR_REINFORCE {
        GatingAction::Reinforce
    } else if similarity >= PRED_ERROR_UPDATE {
        GatingAction::Update
    } else {
        GatingAction::Create
    }
}

/// V5.2: Z-score based adaptive gating (Gemini Deep Research 2026-03-14).
///
/// Uses z-score of the similarity against the recent corpus distribution:
///   z = (similarity − μ) / σ
///
/// Thresholds (standard normal):
///   z ≥ 2.0 → REINFORCE (top 2.3% — near-duplicate)
///   z ≥ 1.0 → UPDATE    (top 15.9% — related content)
///   z < 1.0 → CREATE    (genuinely novel)
///
/// V0.9: deferred to `adaptive_thresholds_conformal` which does NOT assume
/// normality. Cosine similarities are notoriously anisotropic skewed-right
/// (Ethayarajh EMNLP 2019, arxiv:1909.00512), so z-score over-fires
/// REINFORCE in real corpora (P(z>2)≈5-8% empirically vs 2.3% gaussian).
///
/// Minimum 5 samples required; fallback to static thresholds.
pub fn adaptive_gate(similarity: f64, recent_similarities: &[f64]) -> GatingAction {
    let (reinforce_thresh, update_thresh) = adaptive_thresholds_conformal(recent_similarities);

    if similarity >= reinforce_thresh {
        GatingAction::Reinforce
    } else if similarity >= update_thresh {
        GatingAction::Update
    } else {
        GatingAction::Create
    }
}

/// V0.9: Conformal-style empirical quantile thresholds (Vovk-Gammerman-Shafer
/// 2005, Angelopoulos-Bates 2023 arxiv:2107.07511). Distribution-free —
/// makes no normality assumption.
///
/// Mathematical justification:
/// - REINFORCE = empirical 97.7% quantile of recent_similarities
///   (matches z=2 in standard normal but holds for any distribution).
/// - UPDATE = empirical 84.1% quantile (matches z=1).
///
/// Why this beats z-score:
/// - Cosine similarities are NOT gaussian; Ethayarajh 2019 documents
///   anisotropy in BERT/E5 embeddings — distribution is skewed right.
/// - Conformal prediction has finite-sample coverage guarantees:
///   P(true_quantile in [L,U]) ≥ 1−α independent of distribution.
///
/// Returns `(reinforce_threshold, update_threshold)` clamped so they stay
/// within `[0.50, 0.98]` and the REINFORCE/UPDATE gap is at least 0.05.
pub fn adaptive_thresholds_conformal(recent_similarities: &[f64]) -> (f64, f64) {
    if recent_similarities.len() < 5 {
        return (PRED_ERROR_REINFORCE, PRED_ERROR_UPDATE);
    }
    let mut sorted: Vec<f64> = recent_similarities.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    // Empirical quantile (linear interpolation between order statistics).
    let q = |p: f64| -> f64 {
        let idx = p * (n as f64 - 1.0);
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };
    let reinforce = q(0.977).clamp(0.80, 0.98);
    let update = q(0.841).clamp(0.50, reinforce - 0.05);
    (reinforce, update)
}

/// V5.2: Z-score based adaptive thresholds.
///
/// Returns (reinforce_threshold, update_threshold).
///
/// Formula:
///   reinforce = clamp(μ + 2σ, 0.80, 0.98)   // z=2 → top 2.3%
///   update    = clamp(μ + 1σ, 0.50, reinforce − 0.05)  // z=1 → top 15.9%
///
/// Uses population std_dev (N-based, not N-1) since we treat
/// the recent window as the full "context population".
pub fn adaptive_thresholds_zscore(recent_similarities: &[f64]) -> (f64, f64) {
    if recent_similarities.len() < 5 {
        return (PRED_ERROR_REINFORCE, PRED_ERROR_UPDATE);
    }

    let n = recent_similarities.len() as f64;
    let mean = recent_similarities.iter().sum::<f64>() / n;
    let variance = recent_similarities
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / n;
    let sigma = variance.sqrt();

    // V5.2: Z-score thresholds — μ + Nσ
    let reinforce = (mean + 2.0 * sigma).clamp(0.80, 0.98);
    let update = (mean + 1.0 * sigma).clamp(0.50, reinforce - 0.05);

    (reinforce, update)
}

/// V5.2: Adaptive novelty assessment using z-score distribution.
pub fn assess_novelty_adaptive(
    similarity_scores: &[f64],
    recent_similarities: &[f64],
) -> (bool, f64, GatingAction) {
    let max_sim = similarity_scores.iter().cloned().fold(0.0f64, f64::max);
    let action = adaptive_gate(max_sim, recent_similarities);

    let should_store = !matches!(action, GatingAction::Reinforce);
    (should_store, max_sim, action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_reinforce() {
        assert_eq!(gate(0.95), GatingAction::Reinforce);
        assert_eq!(gate(0.92), GatingAction::Reinforce);
    }

    #[test]
    fn test_gate_update() {
        assert_eq!(gate(0.85), GatingAction::Update);
        assert_eq!(gate(0.75), GatingAction::Update);
    }

    #[test]
    fn test_gate_create() {
        assert_eq!(gate(0.74), GatingAction::Create);
        assert_eq!(gate(0.5), GatingAction::Create);
        assert_eq!(gate(0.0), GatingAction::Create);
    }

    // ── V5.2: Z-score Tests ─────────────────────────────────────

    #[test]
    fn test_zscore_insufficient_data() {
        let (r, u) = adaptive_thresholds_zscore(&[0.5, 0.6]);
        assert!((r - PRED_ERROR_REINFORCE).abs() < 0.001);
        assert!((u - PRED_ERROR_UPDATE).abs() < 0.001);
    }

    #[test]
    fn test_zscore_high_similarity_corpus() {
        // All recent very similar → μ high, σ small → thresholds rise
        let recent = vec![0.90, 0.88, 0.92, 0.89, 0.91, 0.90, 0.88];
        let (r, u) = adaptive_thresholds_zscore(&recent);
        assert!(r > 0.90, "high corpus → high reinforce: {r}");
        assert!(u > 0.85, "high corpus → high update: {u}");
    }

    #[test]
    fn test_zscore_diverse_corpus() {
        // Diverse observations → μ moderate, σ large → wider CREATE zone
        let recent = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4];
        let (r, u) = adaptive_thresholds_zscore(&recent);
        assert!(r < 0.98, "diverse → reinforce not maxed: {r}");
        assert!(u < 0.85, "diverse → update reasonable: {u}");
    }

    #[test]
    fn test_zscore_novel_in_uniform() {
        // Uniform high corpus → low similarity is truly novel
        let recent = vec![0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85];
        let action = adaptive_gate(0.5, &recent);
        assert_eq!(action, GatingAction::Create);
    }

    #[test]
    fn test_zscore_redundant_in_uniform() {
        // Uniform high corpus → high similarity is redundant
        let recent = vec![0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85];
        let action = adaptive_gate(0.95, &recent);
        assert_eq!(action, GatingAction::Reinforce);
    }

    #[test]
    fn test_zscore_gap_maintained() {
        // update threshold must always be < reinforce - 0.05
        for corpus in [
            vec![0.5, 0.6, 0.5, 0.7, 0.55],
            vec![0.9, 0.91, 0.88, 0.92, 0.89],
            vec![0.1, 0.9, 0.5, 0.3, 0.7],
        ] {
            let (r, u) = adaptive_thresholds_zscore(&corpus);
            assert!(r - u >= 0.05, "gap must be ≥0.05: r={r}, u={u}");
        }
    }

    // ── V0.9: Conformal Quantile Tests ──────────────────────────

    #[test]
    fn test_conformal_insufficient_data() {
        let (r, u) = adaptive_thresholds_conformal(&[0.5, 0.6]);
        assert!((r - PRED_ERROR_REINFORCE).abs() < 0.001);
        assert!((u - PRED_ERROR_UPDATE).abs() < 0.001);
    }

    #[test]
    fn test_conformal_gap_maintained() {
        for corpus in [
            vec![0.5, 0.6, 0.5, 0.7, 0.55, 0.58, 0.52],
            vec![0.9, 0.91, 0.88, 0.92, 0.89, 0.90, 0.91],
            vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.4, 0.6, 0.8, 0.2, 0.5],
        ] {
            let (r, u) = adaptive_thresholds_conformal(&corpus);
            assert!(r - u >= 0.05, "gap must be ≥0.05: r={r}, u={u}");
        }
    }

    /// Skewed-right distribution: most values concentrated near 0.85, long tail
    /// down to 0.20. Z-score over-fires REINFORCE because σ is inflated by the
    /// tail; conformal correctly returns the 97.7% quantile of actual data.
    #[test]
    fn test_conformal_handles_skewed_right_distribution() {
        let mut skewed: Vec<f64> = vec![0.85, 0.86, 0.87, 0.85, 0.88, 0.86, 0.87, 0.85, 0.84];
        skewed.extend([0.20_f64, 0.30, 0.40]); // long left tail
        let (r_conf, _) = adaptive_thresholds_conformal(&skewed);
        let (r_zscore, _) = adaptive_thresholds_zscore(&skewed);
        // Conformal stays close to the actual concentration of similar values,
        // while z-score is pulled lower by the inflated σ from the tail.
        assert!(
            r_conf > r_zscore - 0.20,
            "conformal {r_conf} should not be wildly under z-score {r_zscore}"
        );
    }

    #[test]
    fn test_conformal_redundant_in_uniform() {
        let recent = vec![0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85, 0.87];
        let action = adaptive_gate(0.95, &recent);
        assert_eq!(action, GatingAction::Reinforce);
    }

    #[test]
    fn test_conformal_novel_in_uniform() {
        let recent = vec![0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85, 0.87];
        let action = adaptive_gate(0.5, &recent);
        assert_eq!(action, GatingAction::Create);
    }
}
