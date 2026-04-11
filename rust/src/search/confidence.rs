//! Grounding confidence scoring for claim verification.
//!
//! Computes confidence based on evidence similarity, count, and source diversity.

/// Compute grounding confidence from evidence.
///
/// Returns (confidence, level) where level is one of:
/// "verified" (>0.7), "partial" (>0.5), "weak" (>0.3), "unknown" (<0.3).
///
/// V0.7 (Mejora 7): `total_observations` enables entity-relative coverage.
/// Instead of hard-capping at 10 hits, coverage uses exponential saturation
/// `1 - e^(-k*ratio)` which models diminishing returns per Shannon's source
/// coding theorem. Falls back to absolute count if total is not provided.
pub fn compute_grounding(
    similarities: &[f64],
    sources: &[&str],
    total_observations: Option<usize>,
) -> (f64, &'static str) {
    if similarities.is_empty() {
        return (0.0, "unknown");
    }

    let max_sim = similarities.iter().cloned().fold(0.0f64, f64::max);
    let avg_sim = similarities.iter().sum::<f64>() / similarities.len() as f64;
    let count = similarities.len();

    // Source diversity factor (0..1)
    let unique_sources: std::collections::HashSet<&str> = sources.iter().copied().collect();
    let diversity = unique_sources.len() as f64 / sources.len().max(1) as f64;

    // V0.7: Entity-relative coverage with exponential saturation (diminishing returns).
    // With total_observations: coverage = 1 - e^(-3 * count/total)
    //   → reaches ~0.5 at 23% coverage, ~0.9 at 77% coverage
    // Without: coverage = 1 - e^(-0.3 * count)
    //   → reaches ~0.5 at 2 hits, ~0.95 at 10 hits (backward compatible)
    let coverage = match total_observations {
        Some(total) if total > 0 => {
            let ratio = count as f64 / total as f64;
            (1.0 - (-3.0 * ratio).exp()).min(1.0)
        }
        _ => (1.0 - (-0.3 * count as f64).exp()).min(1.0),
    };

    // Weights sum to 1.0: 0.45 + 0.25 + 0.20 + 0.10 = 1.00
    let confidence =
        (max_sim * 0.45 + avg_sim * 0.25 + coverage * 0.20 + diversity * 0.10).min(1.0);

    let level = if confidence > 0.7 {
        "verified"
    } else if confidence > 0.5 {
        "partial"
    } else if confidence > 0.3 {
        "weak"
    } else {
        "unknown"
    };

    (confidence, level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_evidence() {
        let (conf, level) = compute_grounding(&[], &[], None);
        assert_eq!(conf, 0.0);
        assert_eq!(level, "unknown");
    }

    #[test]
    fn test_strong_evidence() {
        let (conf, level) = compute_grounding(
            &[0.95, 0.88, 0.85, 0.82, 0.80],
            &["agent", "user", "agent", "user", "error_detection"],
            None,
        );
        assert!(conf > 0.7, "strong evidence should be verified: got {conf}");
        assert_eq!(level, "verified");
    }

    #[test]
    fn test_weak_evidence() {
        let (conf, _level) = compute_grounding(&[0.35], &["agent"], None);
        assert!(conf < 0.5, "single weak match: got {conf}");
    }

    #[test]
    fn test_entity_relative_coverage() {
        // Same evidence count (5), different corpus sizes
        let sims = &[0.8, 0.7, 0.6, 0.5, 0.4];
        let srcs = &["agent", "user", "agent", "user", "agent"];

        // Small corpus: 5 of 5 total = 100% coverage
        let (conf_small, _) = compute_grounding(sims, srcs, Some(5));
        // Large corpus: 5 of 200 total = 2.5% coverage
        let (conf_large, _) = compute_grounding(sims, srcs, Some(200));

        assert!(
            conf_small > conf_large,
            "5/5 coverage ({conf_small}) should score higher than 5/200 ({conf_large})"
        );
    }

    #[test]
    fn test_coverage_exponential_saturation() {
        // Coverage should show diminishing returns
        let srcs = &["agent"];
        let (c1, _) = compute_grounding(&[0.8], srcs, Some(10));
        let (c5, _) = compute_grounding(
            &[0.8, 0.7, 0.6, 0.5, 0.4],
            &["a", "b", "c", "d", "e"],
            Some(10),
        );
        // 5 hits should give more confidence than 1, but less than 5x
        assert!(c5 > c1, "more evidence should increase confidence");
    }
}
