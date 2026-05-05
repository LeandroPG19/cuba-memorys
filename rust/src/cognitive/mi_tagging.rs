//! Mutual Information based tag selection.
//!
//! Brown, G., Pocock, A., Zhao, M., & Luján, M. (2012).
//! "Conditional Likelihood Maximisation: A Unifying Framework for
//! Information Theoretic Feature Selection." JMLR 13: 27–66.
//!
//! Replaces pure-frequency TF with `MI(tag, observation_type)`. A tag that
//! appears in 90% of observations of every type is high-frequency but
//! low-MI (no information about which type). A tag that strongly correlates
//! with one type has high MI even at low frequency.
//!
//! ## Formula
//! MI(W, C) = Σ_w Σ_c p(w, c) · log2(p(w, c) / (p(w) · p(c)))
//!
//! where W ∈ {tag, ¬tag} and C is observation_type.

use std::collections::HashMap;

/// Compute MI for every tag against the observation_type label, return
/// top-K tags ranked by MI (descending).
///
/// `corpus[i] = (tags_in_observation_i, observation_type_i)`.
pub fn rank_tags_by_mi(corpus: &[(Vec<String>, String)], top_k: usize) -> Vec<(&str, f64)> {
    if corpus.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let n = corpus.len() as f64;

    // Counts
    let mut tag_counts: HashMap<&str, usize> = HashMap::new();
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    let mut joint: HashMap<(&str, &str), usize> = HashMap::new();
    for (tags, ty) in corpus {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        *type_counts.entry(ty.as_str()).or_insert(0) += 1;
        for tag in tags {
            // Treat each tag as binary feature per observation
            if seen.insert(tag.as_str()) {
                *tag_counts.entry(tag.as_str()).or_insert(0) += 1;
                *joint.entry((tag.as_str(), ty.as_str())).or_insert(0) += 1;
            }
        }
    }

    // Compute MI per tag
    let mut mi_scores: Vec<(&str, f64)> = Vec::with_capacity(tag_counts.len());
    for (&tag, &n_w) in &tag_counts {
        let p_w = n_w as f64 / n;
        let p_not_w = 1.0 - p_w;
        let mut mi = 0.0_f64;
        for (&ty, &n_c) in &type_counts {
            let p_c = n_c as f64 / n;
            let n_wc = *joint.get(&(tag, ty)).unwrap_or(&0) as f64;
            let n_notw_c = (n_c as f64 - n_wc).max(0.0);
            // p(w, c) and p(¬w, c)
            for (n_joint, p_marginal) in [(n_wc, p_w), (n_notw_c, p_not_w)] {
                if n_joint > 0.0 && p_marginal > 0.0 && p_c > 0.0 {
                    let p_joint = n_joint / n;
                    mi += p_joint * (p_joint / (p_marginal * p_c)).log2();
                }
            }
        }
        if mi.is_finite() {
            mi_scores.push((tag, mi));
        }
    }

    // Sort descending
    mi_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    mi_scores.truncate(top_k);
    mi_scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_returns_empty() {
        let v = rank_tags_by_mi(&[], 10);
        assert!(v.is_empty());
    }

    #[test]
    fn perfectly_predictive_tag_has_high_mi() {
        // 'rust' appears only with observation_type='fact', never with 'error'
        let mut corpus: Vec<(Vec<String>, String)> = Vec::new();
        for _ in 0..5 {
            corpus.push((vec!["rust".to_string()], "fact".to_string()));
            corpus.push((vec!["bug".to_string()], "error".to_string()));
        }
        let ranked = rank_tags_by_mi(&corpus, 5);
        assert!(!ranked.is_empty());
        // Both 'rust' and 'bug' are perfectly predictive — both rank high
        let names: Vec<&str> = ranked.iter().map(|(t, _)| *t).collect();
        assert!(names.contains(&"rust") && names.contains(&"bug"));
        // Their MI should be > 0
        for (_, mi) in &ranked {
            assert!(*mi > 0.0);
        }
    }

    #[test]
    fn uninformative_tag_has_low_mi() {
        // 'common' appears in 50% of every observation type → low MI
        let mut corpus: Vec<(Vec<String>, String)> = Vec::new();
        for i in 0..10 {
            let tags = if i % 2 == 0 {
                vec!["common".to_string(), "specific".to_string()]
            } else {
                vec!["common".to_string()]
            };
            let ty = if i < 5 { "fact" } else { "error" };
            corpus.push((tags, ty.to_string()));
        }
        let ranked = rank_tags_by_mi(&corpus, 5);
        // 'specific' has higher MI than 'common' even though 'common' is more frequent
        let mi_common = ranked.iter().find(|(t, _)| *t == "common").map(|(_, m)| *m);
        let mi_specific = ranked
            .iter()
            .find(|(t, _)| *t == "specific")
            .map(|(_, m)| *m);
        if let (Some(c), Some(s)) = (mi_common, mi_specific) {
            assert!(s >= c, "specific MI {s} should be ≥ common MI {c}");
        }
    }
}
