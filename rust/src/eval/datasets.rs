use serde::Deserialize;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EvaluationSample {
    pub query: String,
    /// Substrings that identify a retrieved doc as relevant (matched against content).
    pub relevant_markers: Vec<String>,
    pub expected_answer: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JsonlRow {
    query: String,
    #[serde(default)]
    relevant_markers: Vec<String>,
    #[serde(default)]
    relevant: Vec<String>,
    #[serde(default)]
    expected_answer: Option<String>,
}

/// Built-in mini corpus for unit tests and smoke benchmarks (no external files).
pub fn builtin_retrieval_set() -> Vec<EvaluationSample> {
    vec![
        EvaluationSample {
            query: "error conexión postgres".into(),
            relevant_markers: vec!["postgres".into(), "conexión".into()],
            expected_answer: None,
        },
        EvaluationSample {
            query: "decisión arquitectura MCP".into(),
            relevant_markers: vec!["MCP".into(), "arquitectura".into()],
            expected_answer: None,
        },
    ]
}

/// Load JSONL: one object per line with `query` and `relevant_markers` (or alias `relevant`).
pub fn load_jsonl_dataset(path: &str) -> Result<Vec<EvaluationSample>, io::Error> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("dataset not found: {path}"),
        ));
    }
    let file = File::open(p)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: JsonlRow = serde_json::from_str(trimmed).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: {e}", line_no + 1),
            )
        })?;
        let markers = if !row.relevant_markers.is_empty() {
            row.relevant_markers
        } else {
            row.relevant
        };
        if markers.is_empty() {
            continue;
        }
        out.push(EvaluationSample {
            query: row.query,
            relevant_markers: markers,
            expected_answer: row.expected_answer,
        });
    }
    Ok(out)
}

/// LOCOMO-style loader: accepts JSONL path or returns builtin set when path is empty.
pub fn load_locomo_dataset(path: &str) -> Result<Vec<EvaluationSample>, io::Error> {
    if path.is_empty() {
        return Ok(builtin_retrieval_set());
    }
    load_jsonl_dataset(path)
}
