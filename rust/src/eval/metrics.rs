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

pub fn ndcg_at_k(relevances: &[bool], k: usize, total_relevant: usize) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }
    let dcg = dcg_at_k(relevances, k);
    let ideal_hits = total_relevant.min(k);
    let ideal: Vec<bool> = (0..k).map(|i| i < ideal_hits).collect();
    let idcg = dcg_at_k(&ideal, k);
    if idcg <= f64::EPSILON {
        0.0
    } else {
        (dcg / idcg).clamp(0.0, 1.0)
    }
}

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
    if relevances.is_empty() || k == 0 {
        return 0.0;
    }
    let k = k.min(relevances.len());
    let hits = relevances[..k].iter().filter(|&&r| r).count();
    hits as f64 / k as f64
}

pub fn recall_at_k(relevances: &[bool], total_relevant: usize, k: usize) -> f64 {
    if total_relevant == 0 || relevances.is_empty() {
        return 0.0;
    }
    let k = k.min(relevances.len());
    let hits = relevances[..k].iter().filter(|&&r| r).count();
    (hits as f64 / total_relevant as f64).clamp(0.0, 1.0)
}

pub fn bootstrap_ci(scores: &[f64], iterations: usize, confidence: f64) -> (f64, f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return (mean, mean, mean);
    }

    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };

    let mut means: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += scores[next() % n];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = (1.0 - confidence) / 2.0;
    let lo_idx = ((alpha * iterations as f64) as usize).min(iterations - 1);
    let hi_idx = (((1.0 - alpha) * iterations as f64) as usize).min(iterations - 1);
    (mean, means[lo_idx], means[hi_idx])
}

pub fn minimum_detectable_effect(scores: &[f64]) -> f64 {
    let n = scores.len();
    if n < 2 {
        return f64::INFINITY;
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    2.80 * var.sqrt() * (2.0 / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_result_list_is_scored_not_a_crash() {
        let empty: Vec<bool> = vec![];
        assert_eq!(precision_at_k(&empty, 10), 0.0);
        assert_eq!(recall_at_k(&empty, 3, 10), 0.0);
        assert_eq!(ndcg_at_k(&empty, 10, 3), 0.0);
        assert_eq!(mrr(std::slice::from_ref(&empty)), 0.0);
        assert_eq!(precision_at_k(&[true, false], 0), 0.0);
    }

    #[test]
    fn missing_relevant_documents_costs_ndcg() {
        let rels = vec![
            true, true, false, false, false, false, false, false, false, false,
        ];

        let honest = ndcg_at_k(&rels, 10, 5);
        assert!(
            honest < 0.75,
            "finding 2 of 5 relevant docs must NOT score near-perfect: got {honest:.3}"
        );

        let all = vec![
            true, true, true, true, true, false, false, false, false, false,
        ];
        assert!((ndcg_at_k(&all, 10, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_perfect_ranking() {
        let rels = vec![true, true, false];
        assert!((ndcg_at_k(&rels, 3, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_cannot_exceed_one() {
        let rels = vec![true; 10];
        assert!(recall_at_k(&rels, 2, 10) <= 1.0);
        assert_eq!(recall_at_k(&[true, true, false], 4, 10), 0.5);
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

    #[test]
    fn bootstrap_brackets_the_mean_and_is_deterministic() {
        let scores: Vec<f64> = (0..40).map(|i| 0.5 + (i % 5) as f64 * 0.1).collect();
        let (m, lo, hi) = bootstrap_ci(&scores, 2000, 0.95);
        assert!(lo <= m && m <= hi, "the CI must bracket the mean");
        assert!(hi - lo > 0.0, "a 40-sample CI is not a point");

        let again = bootstrap_ci(&scores, 2000, 0.95);
        assert_eq!((m, lo, hi), again);
    }

    #[test]
    fn mde_exposes_an_underpowered_benchmark() {
        let ten: Vec<f64> = vec![0.9, 0.8, 1.0, 0.7, 1.0, 0.9, 0.6, 1.0, 0.8, 0.9];
        let mde_10 = minimum_detectable_effect(&ten);
        assert!(
            mde_10 > 0.10,
            "with n=10 the smallest detectable effect is huge ({mde_10:.3}) — the −0.03 \
             'regression' attributed to associative retrieval was inside the noise"
        );

        let many: Vec<f64> = ten.iter().cycle().take(300).copied().collect();
        assert!(
            minimum_detectable_effect(&many) < mde_10 / 3.0,
            "more samples must shrink the detectable effect"
        );
    }
}
