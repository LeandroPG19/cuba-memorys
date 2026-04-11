//! §A: Weighted RRF Entropy Routing.
//!
//! Reciprocal Rank Fusion with Shannon entropy-based dynamic weighting.
//! V2: Post-fusion dedup removes semantic duplicates across signals.
//! V4: k=60 constant (Cormack et al. 2009) — eliminates V3 adaptive instability.

use std::collections::{HashMap, HashSet};

/// Fixed RRF k constant — empirical consensus (Cormack 2009, Azure AI Search, ES 8.8+).
///
/// Adaptive k (V3) was removed per Gemini Deep Research audit 2026-03-14:
/// dynamic sqrt-based k introduced non-monotonic ranking instabilities and
/// violated determinism requirements for the MCP server.
const RRF_K: f64 = 60.0;

/// A ranked search result.
#[derive(Clone, Debug)]
pub struct RankedResult {
    pub id: String,
    pub content: String,
    pub score: f64,
    pub source: String, // Which signal produced this
}

/// §A: Compute Shannon entropy of query for dynamic weight routing.
///
/// V0.7 (Mejora 8a): Uses HashMap for O(n) frequency counting instead of
/// O(n*k) nested filter per unique word.
///
/// V0.7+: Tokenizes by non-alphanumeric characters (consistent with
/// `information_density` and `text_overlap`) so that "rust!" and "rust"
/// are the same token. Avoids inflated entropy from punctuation variants
/// in multilingual queries.
pub fn query_entropy(query: &str) -> f64 {
    let words: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let total = words.len();
    if total == 0 {
        return 0.0;
    }
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &words {
        *freq.entry(w).or_default() += 1;
    }
    let mut entropy = 0.0;
    for &count in freq.values() {
        let p = count as f64 / total as f64;
        entropy -= p * p.log2();
    }
    entropy
}

/// RRF fusion across N ranked signal lists.
///
/// Each signal can have a custom weight (§A: entropy-based).
/// V4: Uses fixed k=60 constant — Cormack et al. 2009 consensus.
///
/// Score_RRF(d) = Σ weight / (60 + rank + 1)
pub fn fuse(
    signals: &[(Vec<RankedResult>, f64)], // (results, weight)
    dedup_threshold: f64,
) -> Vec<RankedResult> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut items: HashMap<String, RankedResult> = HashMap::new();

    for (results, weight) in signals {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = weight / (RRF_K + rank as f64 + 1.0);
            *scores.entry(result.id.clone()).or_default() += rrf_score;
            items
                .entry(result.id.clone())
                .or_insert_with(|| result.clone());
        }
    }

    // Sort by fused score, then by id for deterministic tie-breaking
    let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
    sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0)) // Deterministic tie-break by id
    });

    // V2: Post-fusion dedup by content overlap
    let mut unique: Vec<RankedResult> = Vec::new();
    for (id, score) in sorted {
        if let Some(mut item) = items.remove(&id) {
            let is_dup = unique
                .iter()
                .any(|existing| text_overlap(&item.content, &existing.content) > dedup_threshold);
            if !is_dup {
                item.score = score;
                unique.push(item);
            }
        }
    }

    unique
}

/// V2: Word-overlap ratio (Jaccard-like with min denominator).
///
/// V0.7 (Mejora 8b): Tokenizes by non-alphanumeric characters instead of
/// whitespace only. Fixes: "configuracion." != "configuracion" which caused
/// false negatives in multilingual dedup detection.
fn text_overlap(a: &str, b: &str) -> f64 {
    let tokenize = |s: &str| -> HashSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    };
    let words_a = tokenize(a);
    let words_b = tokenize(b);
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    intersection as f64 / words_a.len().min(words_b.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_entropy_uniform() {
        // All unique words → high entropy
        let e = query_entropy("rust is fast and safe for systems programming");
        assert!(e > 2.5, "diverse query should have high entropy: got {e}");
    }

    #[test]
    fn test_query_entropy_repetitive() {
        let e = query_entropy("hello hello hello");
        assert!(
            e < 0.01,
            "repetitive query should have near-zero entropy: got {e}"
        );
    }

    #[test]
    fn test_query_entropy_punctuation_invariant() {
        // "rust!" and "rust" should be the same token after non-alphanumeric split.
        // Previously split_whitespace() treated them as distinct, inflating entropy.
        let e_clean = query_entropy("rust is fast");
        let e_punct = query_entropy("rust! is fast.");
        assert!(
            (e_clean - e_punct).abs() < 1e-9,
            "punctuation should not affect entropy: clean={e_clean}, punct={e_punct}"
        );
    }

    #[test]
    fn test_query_entropy_multilingual_punctuation() {
        // Spanish query with punctuation — consistent tokenization
        let e1 = query_entropy("configuracion sistema");
        let e2 = query_entropy("configuracion. sistema,");
        assert!(
            (e1 - e2).abs() < 1e-9,
            "trailing punctuation should not change entropy: {e1} vs {e2}"
        );
    }

    #[test]
    fn test_text_overlap_identical() {
        assert!((text_overlap("hello world", "hello world") - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_text_overlap_disjoint() {
        assert_eq!(text_overlap("hello world", "foo bar"), 0.0);
    }

    #[test]
    fn test_rrf_fusion_basic() {
        let signal1 = vec![
            RankedResult {
                id: "a".into(),
                content: "alpha".into(),
                score: 0.0,
                source: "text".into(),
            },
            RankedResult {
                id: "b".into(),
                content: "beta".into(),
                score: 0.0,
                source: "text".into(),
            },
        ];
        let signal2 = vec![
            RankedResult {
                id: "b".into(),
                content: "beta".into(),
                score: 0.0,
                source: "vec".into(),
            },
            RankedResult {
                id: "c".into(),
                content: "gamma".into(),
                score: 0.0,
                source: "vec".into(),
            },
        ];

        let fused = fuse(&[(signal1, 0.5), (signal2, 0.5)], 0.75);
        assert!(!fused.is_empty());
        // "b" appears in both signals, should rank first
        assert_eq!(fused[0].id, "b", "item in both signals should rank first");
    }

    #[test]
    fn test_rrf_k60_deterministic() {
        // V4: k=60 always, deterministic scores
        let signal = vec![RankedResult {
            id: "a".into(),
            content: "alpha".into(),
            score: 0.0,
            source: "text".into(),
        }];

        let fused = fuse(&[(signal, 1.0)], 0.75);
        // Score should be exactly 1.0 / (60.0 + 0 + 1.0) = 1/61
        let expected = 1.0 / 61.0;
        assert!(
            (fused[0].score - expected).abs() < 1e-10,
            "k=60 fixed: expected {} got {}",
            expected,
            fused[0].score
        );
    }
}
