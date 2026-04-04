//! Grounding confidence scoring for claim verification.
//!
//! Computes confidence based on evidence similarity, count, and source diversity.

/// Compute grounding confidence from evidence.
///
/// Returns (confidence, level) where level is one of:
/// "verified" (>0.7), "partial" (>0.5), "weak" (>0.3), "unknown" (<0.3).
pub fn compute_grounding(similarities: &[f64], sources: &[&str]) -> (f64, &'static str) {
    if similarities.is_empty() {
        return (0.0, "unknown");
    }

    let max_sim = similarities.iter().cloned().fold(0.0f64, f64::max);
    let avg_sim = similarities.iter().sum::<f64>() / similarities.len() as f64;
    let count = similarities.len();

    // Source diversity factor (0..1)
    let unique_sources: std::collections::HashSet<&str> = sources.iter().copied().collect();
    let diversity = unique_sources.len() as f64 / sources.len().max(1) as f64;

    // Weights sum to 1.0: 0.45 + 0.25 + 0.20 + 0.10 = 1.00
    let coverage = (count as f64 / 10.0).min(1.0);
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
        let (conf, level) = compute_grounding(&[], &[]);
        assert_eq!(conf, 0.0);
        assert_eq!(level, "unknown");
    }

    #[test]
    fn test_strong_evidence() {
        let (conf, level) = compute_grounding(
            &[0.95, 0.88, 0.85, 0.82, 0.80],
            &["agent", "user", "agent", "user", "error_detection"],
        );
        assert!(conf > 0.7, "strong evidence should be verified: got {conf}");
        assert_eq!(level, "verified");
    }

    #[test]
    fn test_weak_evidence() {
        let (conf, _level) = compute_grounding(&[0.35], &["agent"]);
        assert!(conf < 0.5, "single weak match: got {conf}");
    }
}
