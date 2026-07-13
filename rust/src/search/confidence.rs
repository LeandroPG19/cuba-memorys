//! Grounding confidence scoring for claim verification.
//!
//! ## What v0.5–v0.11.1 got wrong
//!
//! [`compute_grounding`] scores a claim from the similarity of whatever the search
//! returned: `max_sim*0.45 + avg_sim*0.25 + coverage*0.20 + diversity*0.10`. Every
//! term measures how much the retrieved text *resembles* the claim. None measures
//! whether it *supports* it.
//!
//! Cosine similarity captures what a text is ABOUT. "cuba-memorys is written in
//! Rust" and "cuba-memorys is written in Java" are nearly the same vector — same
//! subject, same shape, one word apart — so a store full of the first happily
//! "grounds" the second. Measured on the live 1,461-observation corpus:
//!
//! | claim                                  | confidence |
//! |----------------------------------------|-----------|
//! | "cuba-memorys usa RRF con k=60" (true)  | 0.59      |
//! | "cuba-memorys está escrito en Java" (false) | **0.61**  |
//! | "el reactor de Chernobyl usaba grafito" (unrelated) | 0.45, with 10 "evidence" items |
//!
//! The false claim outscored the true one. And nothing was ever unrelated enough to
//! score zero, because retrieval always returns its top-K and `coverage` +
//! `diversity` hand out 0.30 for free. The `confidence: 0.0 / "unknown"` case the
//! README advertised could essentially never fire.
//!
//! No threshold fixes this. The distributions overlap completely — true claims
//! landed at 0.43–0.57 max-similarity, false ones at 0.55–0.59. There is nothing to
//! separate, because similarity is not the quantity that answers the question.
//!
//! ## What replaces it
//!
//! [`compute_grounding_judged`] takes a verdict per piece of evidence — supports /
//! contradicts / unrelated, from [`crate::cognitive::judge`] — and derives
//! confidence from *those*, weighting each verdict by the similarity of the evidence
//! it came from. Similarity decides how much a verdict counts. It no longer decides
//! what the verdict is.

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

/// Below this cosine similarity, a retrieved memory is not evidence about the claim
/// — it is just the closest thing in a store that had to return something.
///
/// Measured on the live corpus: on-topic memories came back at 0.43–0.59, while
/// "the Chernobyl reactor used graphite as a moderator" and "the best paella
/// recipe" — with not one related observation in 1,461 — still dragged back ten
/// items at 0.32–0.38. This cut removes those. It does NOT separate true claims
/// from false ones; nothing about similarity can. That is the judge's job.
pub const MIN_EVIDENCE_SIMILARITY: f64 = 0.40;

/// One piece of evidence, after a judge has read it.
pub struct JudgedEvidence {
    /// Cosine similarity of the evidence to the claim — how much this verdict weighs.
    pub similarity: f64,
    /// `supports` | `contradicts` | `unrelated` | `unknown`
    pub verdict: String,
    /// The judge's own confidence in its verdict (0..1).
    pub judge_confidence: f64,
}

/// Confidence that the CLAIM IS TRUE, given what the judge made of the evidence.
///
/// Support and contradiction are accumulated separately, each weighted by the
/// similarity of the evidence and the judge's confidence in its own verdict:
///
/// ```text
/// S = Σ similarity·judge_conf   over verdict = supports
/// C = Σ similarity·judge_conf   over verdict = contradicts
///
/// confidence = S / (S + C)  ·  (1 − e^(−2·S))
///              └─ share ──┘    └── strength ──┘
/// ```
///
/// The **share** term is what makes a contradiction actually cost something: one
/// solid contradiction against one solid support lands at 0.5, not 0.9. The
/// **strength** term (exponential saturation, diminishing returns) stops a single
/// hesitant "supports" from certifying anything — with S≈0.3 it caps confidence
/// near 0.45 no matter how one-sided the share.
///
/// `unrelated` and `unknown` verdicts contribute to NEITHER. That is deliberate and
/// it is the whole repair: a memory that is on-topic but says nothing about what the
/// claim asserts used to be counted as grounding. It now counts as what it is —
/// nothing. When every verdict is unrelated, S = C = 0 and the answer is `unknown`
/// with confidence 0, which is the honest answer and the one the old code could
/// almost never give.
pub fn compute_grounding_judged(evidence: &[JudgedEvidence]) -> (f64, &'static str) {
    let mut support = 0.0f64;
    let mut contra = 0.0f64;

    for e in evidence {
        let weight = e.similarity * e.judge_confidence;
        match e.verdict.as_str() {
            "supports" => support += weight,
            "contradicts" => contra += weight,
            _ => {} // unrelated / unknown: not evidence for or against. Not a vote.
        }
    }

    if support + contra <= f64::EPSILON {
        return (0.0, "unknown");
    }

    if contra > support {
        // The store actively disagrees. Confidence in the CLAIM is what is left over,
        // and the caller gets told why rather than being handed a low number to guess at.
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

    /// The bug, as a test. These are the real numbers from the live corpus.
    #[test]
    fn a_false_claim_no_longer_outscores_a_true_one() {
        // "cuba-memorys está escrito en Java" — the store says Rust, loudly.
        let false_claim = compute_grounding_judged(&[
            ev(0.55, "contradicts", 0.95),
            ev(0.54, "contradicts", 0.9),
            ev(0.53, "unrelated", 0.8),
        ]);
        // "cuba-memorys usa RRF con k=60" — the store says exactly this.
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

    /// On-topic is not support. This is the conflation the whole rewrite exists to end.
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
