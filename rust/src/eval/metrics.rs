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
