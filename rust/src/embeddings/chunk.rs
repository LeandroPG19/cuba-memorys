pub const DEFAULT_CHUNK_CHARS: usize = 1400;
pub const DEFAULT_OVERLAP_CHARS: usize = 200;
pub const CHUNK_THRESHOLD_CHARS: usize = 1800;

pub fn threshold_chars() -> usize {
    parse_usize(
        std::env::var("CUBA_CHUNK_THRESHOLD_CHARS").ok().as_deref(),
        CHUNK_THRESHOLD_CHARS,
    )
}

pub fn chunk_chars() -> usize {
    parse_usize(
        std::env::var("CUBA_CHUNK_CHARS").ok().as_deref(),
        DEFAULT_CHUNK_CHARS,
    )
    .max(200)
}

fn parse_usize(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

pub fn needs_chunking(content: &str) -> bool {
    content.chars().count() > threshold_chars()
}

pub fn split(content: &str, chunk_chars: usize, overlap_chars: usize) -> Vec<String> {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= chunk_chars {
        return vec![content.to_string()];
    }
    let overlap = overlap_chars.min(chunk_chars / 2);
    let stride = chunk_chars - overlap;

    let mut out = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let hard_end = (start + chunk_chars).min(chars.len());
        let end = if hard_end == chars.len() {
            hard_end
        } else {
            boundary_before(&chars, start, hard_end)
        };
        let piece: String = chars[start..end].iter().collect();
        let piece = piece.trim().to_string();
        if !piece.is_empty() {
            out.push(piece);
        }
        if end >= chars.len() {
            break;
        }
        start = if end > start + overlap {
            end - overlap
        } else {
            start + stride
        };
    }
    out
}

fn boundary_before(chars: &[char], start: usize, hard_end: usize) -> usize {
    let min_acceptable = start + (hard_end - start) / 2;
    for i in (min_acceptable..hard_end).rev() {
        if matches!(chars[i], '.' | '!' | '?' | '\n') {
            return i + 1;
        }
    }
    for i in (min_acceptable..hard_end).rev() {
        if chars[i].is_whitespace() {
            return i + 1;
        }
    }
    hard_end
}

pub fn chunks_for(content: &str) -> Vec<String> {
    if !needs_chunking(content) {
        return Vec::new();
    }
    split(content, chunk_chars(), DEFAULT_OVERLAP_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_content_is_never_chunked() {
        assert!(chunks_for("short").is_empty());
        assert!(!needs_chunking("short"));
    }

    #[test]
    fn a_long_text_splits_into_overlapping_pieces() {
        let text = "Sentence one. ".repeat(400);
        let pieces = split(&text, 1000, 200);
        assert!(pieces.len() > 1, "must split");
        for p in &pieces {
            assert!(
                p.chars().count() <= 1000,
                "no piece may exceed the budget: {}",
                p.chars().count()
            );
        }
    }

    #[test]
    fn splitting_covers_the_whole_text() {
        let text: String = (0..300).map(|i| format!("item{i} ")).collect();
        let pieces = split(&text, 500, 100);
        let rejoined: String = pieces.join(" ");
        for needle in ["item0", "item150", "item299"] {
            assert!(
                rejoined.contains(needle),
                "{needle} was lost — chunking must not drop content"
            );
        }
    }

    #[test]
    fn splitting_terminates_on_pathological_input() {
        let text = "x".repeat(5000);
        let pieces = split(&text, 300, 250);
        assert!(!pieces.is_empty());
        assert!(
            pieces.len() < 200,
            "overlap must not stall progress: got {} pieces",
            pieces.len()
        );
    }

    #[test]
    fn multibyte_text_is_split_without_panicking() {
        let text = "Ñandú corría — 日本語のテキスト。".repeat(200);
        let pieces = split(&text, 400, 80);
        assert!(pieces.len() > 1);
        for p in &pieces {
            assert!(!p.is_empty());
        }
    }

    #[test]
    fn a_boundary_is_preferred_over_cutting_mid_word() {
        let text = format!("{}. Final sentence here.", "word ".repeat(400));
        let pieces = split(&text, 600, 100);
        assert!(pieces.len() > 1);
        assert!(
            pieces[0].ends_with('.') || pieces[0].ends_with("word"),
            "first piece should end at a clean boundary, got: {:?}",
            &pieces[0][pieces[0].len().saturating_sub(20)..]
        );
    }

    #[test]
    fn env_overrides_are_validated() {
        assert_eq!(parse_usize(None, 99), 99);
        assert_eq!(parse_usize(Some("0"), 99), 99);
        assert_eq!(parse_usize(Some("-5"), 99), 99);
        assert_eq!(parse_usize(Some("garbage"), 99), 99);
        assert_eq!(parse_usize(Some(" 250 "), 99), 250);
    }
}
