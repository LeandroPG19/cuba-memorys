//! Shannon information density — FIX B5.
//!
//! FIX B5: H_max uses log2(n_unique) (vocabulary size), not log2(n_words).
//! This gives correct normalized entropy.

use std::collections::HashSet;

/// Compute normalized Shannon information density of text.
///
/// Returns value in [0, 1]:
/// - 0.0 = completely repetitive (1 unique word)
/// - 1.0 = maximum diversity (all words unique)
///
/// FIX B5: Denominator uses unique word count, not total word count.
pub fn information_density(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    let total = words.len();
    if total <= 1 {
        return 0.0;
    }

    let unique: HashSet<&str> = words.iter().copied().collect();
    let vocab_size = unique.len();

    if vocab_size <= 1 {
        return 0.0; // All same word
    }

    // Shannon entropy H = -Σ p(w) * log2(p(w))
    let mut entropy = 0.0;
    for word in &unique {
        let freq = words.iter().filter(|w| *w == word).count() as f64 / total as f64;
        if freq > 0.0 {
            entropy -= freq * freq.log2();
        }
    }

    // FIX B5: Normalize by log2(vocab_size), not log2(total_words)
    let h_max = (vocab_size as f64).log2();
    if h_max == 0.0 {
        return 0.0;
    }

    (entropy / h_max).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_diverse() {
        let d = information_density("rust is a fast safe and modern language each word differs");
        assert!(d > 0.9, "all unique words should be high density: got {d}");
    }

    #[test]
    fn test_density_repetitive() {
        let d = information_density("hello hello hello hello hello");
        assert!(d < 0.01, "same word repeated should be zero: got {d}");
    }

    #[test]
    fn test_density_empty() {
        assert_eq!(information_density(""), 0.0);
        assert_eq!(information_density("word"), 0.0);
    }

    #[test]
    fn test_density_mixed() {
        // Non-uniform distribution: "fast" appears 5x, rest 1x each → skewed entropy
        let d = information_density("fast fast fast fast fast safe modern language");
        assert!(
            d > 0.3 && d < 0.9,
            "skewed distribution should be medium: got {d}"
        );
    }
}
