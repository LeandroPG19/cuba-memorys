pub fn information_density(text: &str) -> f64 {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let total = words.len();
    if total <= 1 {
        return 0.0;
    }

    let mut freq_map: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in &words {
        *freq_map.entry(w).or_default() += 1;
    }
    let vocab_size = freq_map.len();

    if vocab_size <= 1 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &count in freq_map.values() {
        let p = count as f64 / total as f64;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

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
        let d = information_density("fast fast fast fast fast safe modern language");
        assert!(
            d > 0.3 && d < 0.9,
            "skewed distribution should be medium: got {d}"
        );
    }
}
