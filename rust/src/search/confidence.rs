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

    let unique_sources: std::collections::HashSet<&str> = sources.iter().copied().collect();
    let diversity = unique_sources.len() as f64 / sources.len().max(1) as f64;

    let coverage = match total_observations {
        Some(total) if total > 0 => {
            let ratio = count as f64 / total as f64;
            (1.0 - (-3.0 * ratio).exp()).min(1.0)
        }
        _ => (1.0 - (-0.3 * count as f64).exp()).min(1.0),
    };

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

pub const MIN_EVIDENCE_SIMILARITY: f64 = 0.40;

pub struct JudgedEvidence {
    pub similarity: f64,
    pub verdict: String,
    pub judge_confidence: f64,
}

pub fn compute_grounding_judged(evidence: &[JudgedEvidence]) -> (f64, &'static str) {
    let mut support = 0.0f64;
    let mut contra = 0.0f64;

    for e in evidence {
        let weight = e.similarity * e.judge_confidence;
        match e.verdict.as_str() {
            "supports" => support += weight,
            "contradicts" => contra += weight,
            _ => {}
        }
    }

    if support + contra <= f64::EPSILON {
        return (0.0, "unknown");
    }

    if contra > support {
        let share = support / (support + contra);
        return (share.clamp(0.0, 1.0), "contradicted");
    }

    let share = support / (support + contra);
    let strength = 1.0 - (-2.0 * support).exp();
    let confidence = (share * strength).clamp(0.0, 1.0);

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

    fn ev(similarity: f64, verdict: &str, judge_confidence: f64) -> JudgedEvidence {
        JudgedEvidence {
            similarity,
            verdict: verdict.to_string(),
            judge_confidence,
        }
    }

    #[test]
    fn a_false_claim_no_longer_outscores_a_true_one() {
        let false_claim = compute_grounding_judged(&[
            ev(0.55, "contradicts", 0.95),
            ev(0.54, "contradicts", 0.9),
            ev(0.53, "unrelated", 0.8),
        ]);
        let true_claim = compute_grounding_judged(&[
            ev(0.57, "supports", 0.95),
            ev(0.52, "supports", 0.85),
            ev(0.50, "unrelated", 0.7),
        ]);

        assert_eq!(false_claim.1, "contradicted");
        assert!(
            true_claim.0 > false_claim.0,
            "the true claim ({:.2}) must outscore the false one ({:.2}) — under the old \
             similarity-only scoring it was 0.59 vs 0.61, the wrong way round",
            true_claim.0,
            false_claim.0
        );
    }

    #[test]
    fn on_topic_but_silent_evidence_grounds_nothing() {
        let (conf, level) = compute_grounding_judged(&[
            ev(0.59, "unrelated", 0.9),
            ev(0.55, "unrelated", 0.9),
            ev(0.51, "unrelated", 0.8),
        ]);
        assert_eq!(conf, 0.0);
        assert_eq!(
            level, "unknown",
            "three highly-similar memories that say nothing about the claim are not grounding"
        );
    }

    #[test]
    fn no_evidence_is_unknown() {
        let (conf, level) = compute_grounding_judged(&[]);
        assert_eq!(conf, 0.0);
        assert_eq!(level, "unknown");
    }

    #[test]
    fn a_contradiction_costs_the_claim() {
        let clean = compute_grounding_judged(&[ev(0.9, "supports", 1.0), ev(0.8, "supports", 1.0)]);
        let disputed = compute_grounding_judged(&[
            ev(0.9, "supports", 1.0),
            ev(0.8, "supports", 1.0),
            ev(0.85, "contradicts", 1.0),
        ]);
        assert!(
            disputed.0 < clean.0,
            "a contradiction must lower confidence: clean {:.2}, disputed {:.2}",
            clean.0,
            disputed.0
        );
        assert_eq!(clean.1, "verified");
    }

    #[test]
    fn one_hesitant_supporter_certifies_nothing() {
        let (conf, _) = compute_grounding_judged(&[ev(0.45, "supports", 0.5)]);
        assert!(
            conf < 0.5,
            "a single weak, uncertain match must not read as verified: got {conf:.2}"
        );
    }

    #[test]
    fn a_wall_of_support_reaches_verified() {
        let (conf, level) = compute_grounding_judged(&[
            ev(0.92, "supports", 0.95),
            ev(0.88, "supports", 0.9),
            ev(0.85, "supports", 0.9),
            ev(0.80, "supports", 0.85),
        ]);
        assert!(conf > 0.7, "got {conf:.2}");
        assert_eq!(level, "verified");
    }

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
        let sims = &[0.8, 0.7, 0.6, 0.5, 0.4];
        let srcs = &["agent", "user", "agent", "user", "agent"];

        let (conf_small, _) = compute_grounding(sims, srcs, Some(5));
        let (conf_large, _) = compute_grounding(sims, srcs, Some(200));

        assert!(
            conf_small > conf_large,
            "5/5 coverage ({conf_small}) should score higher than 5/200 ({conf_large})"
        );
    }

    #[test]
    fn test_coverage_exponential_saturation() {
        let srcs = &["agent"];
        let (c1, _) = compute_grounding(&[0.8], srcs, Some(10));
        let (c5, _) = compute_grounding(
            &[0.8, 0.7, 0.6, 0.5, 0.4],
            &["a", "b", "c", "d", "e"],
            Some(10),
        );
        assert!(c5 > c1, "more evidence should increase confidence");
    }
}
