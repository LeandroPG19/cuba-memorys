//! Information-retrieval metrics for `cuba_faro` benchmark harness.

pub fn calculate_exact_match(predicted: &str, expected: &str) -> f32 {
    if predicted.trim() == expected.trim() {
        1.0
    } else {
        0.0
    }
}

pub fn calculate_f1_score(predicted: &str, expected: &str) -> f32 {
    let pred_tokens: std::collections::HashSet<_> = predicted
        .split_whitespace()
        .map(str::to_lowercase)
        .collect();
    let exp_tokens: std::collections::HashSet<_> =
        expected.split_whitespace().map(str::to_lowercase).collect();
    if pred_tokens.is_empty() && exp_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || exp_tokens.is_empty() {
        return 0.0;
    }
    let inter = pred_tokens.intersection(&exp_tokens).count() as f32;
    let precision = inter / pred_tokens.len() as f32;
    let recall = inter / exp_tokens.len() as f32;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// DCG@k with binary relevance (1 if doc is relevant, else 0).
pub fn dcg_at_k(relevances: &[bool], k: usize) -> f64 {
    let k = k.min(relevances.len());
    relevances[..k]
        .iter()
        .enumerate()
        .map(|(i, &rel)| {
            let gain = if rel { 1.0 } else { 0.0 };
            if i == 0 {
                gain
            } else {
                gain / (i as f64 + 1.0).log2()
            }
        })
        .sum()
}

/// nDCG@k — normalized by ideal DCG.
pub fn ndcg_at_k(relevances: &[bool], k: usize) -> f64 {
    let dcg = dcg_at_k(relevances, k);
    let mut ideal: Vec<bool> = relevances.to_vec();
    ideal.sort_by(|a, b| b.cmp(a));
    let idcg = dcg_at_k(&ideal, k);
    if idcg <= f64::EPSILON {
        0.0
    } else {
        dcg / idcg
    }
}

/// Mean reciprocal rank — 1/rank of first relevant doc, else 0.
pub fn mrr(relevance_lists: &[Vec<bool>]) -> f64 {
    if relevance_lists.is_empty() {
        return 0.0;
    }
    let sum: f64 = relevance_lists
        .iter()
        .map(|rels| {
            rels.iter()
                .position(|&r| r)
                .map(|i| 1.0 / (i as f64 + 1.0))
                .unwrap_or(0.0)
        })
        .sum();
    sum / relevance_lists.len() as f64
}

pub fn precision_at_k(relevances: &[bool], k: usize) -> f64 {
    let k = k.min(relevances.len()).max(1);
    let hits = relevances[..k].iter().filter(|&&r| r).count();
    hits as f64 / k as f64
}

pub fn recall_at_k(relevances: &[bool], total_relevant: usize, k: usize) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }
    let k = k.min(relevances.len());
    let hits = relevances[..k].iter().filter(|&&r| r).count();
    hits as f64 / total_relevant as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_perfect_ranking() {
        let rels = vec![true, true, false];
        assert!((ndcg_at_k(&rels, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mrr_first_hit() {
        let lists = vec![vec![false, true, false], vec![true, false]];
        assert!((mrr(&lists) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn precision_at_2() {
        let rels = vec![true, false, true];
        assert!((precision_at_k(&rels, 2) - 0.5).abs() < 1e-6);
    }
}
