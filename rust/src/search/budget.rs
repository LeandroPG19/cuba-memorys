use std::sync::OnceLock;
use tiktoken_rs::{CoreBPE, cl100k_base};

fn encoder() -> &'static CoreBPE {
    static ENCODER: OnceLock<CoreBPE> = OnceLock::new();
    ENCODER.get_or_init(|| cl100k_base().expect("cl100k_base loads from embedded data"))
}

pub fn count_tokens(text: &str) -> usize {
    encoder().encode_with_special_tokens(text).len()
}

pub fn truncate_to_budget(text: &str, max_tokens: usize) -> String {
    let enc = encoder();
    let tokens = enc.encode_with_special_tokens(text);
    if tokens.len() <= max_tokens {
        return text.to_string();
    }
    let truncated = &tokens[..max_tokens];
    enc.decode(truncated.to_vec()).unwrap_or_else(|_| {
        let approx = max_tokens.saturating_mul(4);
        let mut end = approx.min(text.len());
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        text[..end].to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_zero_for_empty() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn count_short_english() {
        let n = count_tokens("Hello world!");
        assert!(
            (1..=4).contains(&n),
            "Expected 1-4 for 'Hello world!', got {n}"
        );
    }

    #[test]
    fn count_handles_spanish_correctly() {
        let s = "El sistema MCP de cuba-memorys es excelente para agentes de IA.";
        let n = count_tokens(s);
        assert!(n > 0);
        assert!(n < 200, "sanity bound, got {n}");
    }

    #[test]
    fn truncate_preserves_short_text_unchanged() {
        let s = "short";
        assert_eq!(truncate_to_budget(s, 100), s);
    }

    #[test]
    fn truncate_returns_at_most_max_tokens() {
        let long = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.".repeat(20);
        let truncated = truncate_to_budget(&long, 50);
        assert!(count_tokens(&truncated) <= 50);
    }

    #[test]
    fn truncate_preserves_utf8_boundaries() {
        let s = "café".repeat(100);
        let t = truncate_to_budget(&s, 5);
        let _ = t.chars().count();
    }
}
