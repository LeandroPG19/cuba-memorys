//! Evaluation datasets.
//!
//! ## Relevance is judged by ID, not by substring
//!
//! Until v0.12 a retrieved document counted as relevant if its *content contained
//! one of a list of marker substrings*. With markers like `["postgres", "conexión"]`,
//! every observation that so much as mentions postgres scored as a hit — whether or
//! not it answered the question. That is not relevance, it is keyword presence, and
//! it biased the whole benchmark toward the lexical branch and against the vector
//! one: the thing being optimized was "does the term appear", not "is this the
//! answer".
//!
//! It also produced impossible numbers. Recall was computed as
//! `hits / relevant_markers.len()` — the count of *marker strings*, not of relevant
//! *documents* — so a query with 2 markers and 6 substring hits reported **R@10 =
//! 3.125**. Recall cannot exceed 1. That number sat in the README.
//!
//! Ground truth is now a set of observation IDs per query (TREC-style qrels). A
//! result is relevant iff its id is in that set. Recall has a real denominator.
//!
//! `relevant_markers` still loads, for old datasets, and the harness falls back to
//! substring matching when no ids are present — but it warns, because the number it
//! produces is not comparable to an id-scored one.

use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EvaluationSample {
    pub query: String,
    /// Ground truth: ids of the observations that actually answer this query.
    /// Empty for abstention samples (nothing answers them, by construction).
    pub relevant_ids: HashSet<String>,
    /// Legacy: substrings that identify a retrieved doc as relevant. Only used when
    /// `relevant_ids` is empty and the sample is not an abstention — see module docs
    /// for why this is a worse signal.
    pub relevant_markers: Vec<String>,
    pub expected_answer: Option<String>,
    /// LongMemEval-style ability/question_type this sample exercises
    /// (information-extraction, multi-session, knowledge-update, temporal-reasoning,
    /// abstention). Used to break metrics down per ability.
    pub ability: Option<String>,
    /// True when the correct behavior is to ABSTAIN: no stored fact answers the
    /// query, so a good system retrieves nothing relevant. Success = zero relevant
    /// hits in the top-k.
    pub abstain: bool,
}

impl EvaluationSample {
    /// Whether this sample is scored by id (the good way) or by substring (legacy).
    pub fn scored_by_id(&self) -> bool {
        !self.relevant_ids.is_empty()
    }

    /// How many documents in the corpus answer this query — the true denominator of
    /// recall. Falls back to the marker count only for legacy datasets, where it is
    /// wrong but at least bounded.
    pub fn relevant_count(&self) -> usize {
        if self.scored_by_id() {
            self.relevant_ids.len()
        } else {
            self.relevant_markers.len().max(1)
        }
    }
}

#[derive(Debug, Deserialize)]
struct JsonlRow {
    query: String,
    /// v0.12: the ground truth. Ids of observations that answer this query.
    #[serde(default)]
    relevant_ids: Vec<String>,
    #[serde(default)]
    relevant_markers: Vec<String>,
    #[serde(default)]
    relevant: Vec<String>,
    #[serde(default)]
    expected_answer: Option<String>,
    /// Alias `question_type` accepted for LongMemEval parity.
    #[serde(default)]
    ability: Option<String>,
    #[serde(default)]
    question_type: Option<String>,
    #[serde(default)]
    abstain: bool,
}

/// Built-in mini corpus for unit tests and smoke benchmarks (no external files).
pub fn builtin_retrieval_set() -> Vec<EvaluationSample> {
    vec![
        EvaluationSample {
            query: "error conexión postgres".into(),
            relevant_ids: HashSet::new(),
            relevant_markers: vec!["postgres".into(), "conexión".into()],
            expected_answer: None,
            ability: Some("information-extraction".into()),
            abstain: false,
        },
        EvaluationSample {
            query: "decisión arquitectura MCP".into(),
            relevant_ids: HashSet::new(),
            relevant_markers: vec!["MCP".into(), "arquitectura".into()],
            expected_answer: None,
            ability: Some("information-extraction".into()),
            abstain: false,
        },
    ]
}

/// Load JSONL. One object per line:
///
/// ```json
/// {"query": "...", "relevant_ids": ["uuid", ...], "ability": "...", "abstain": false}
/// ```
///
/// `relevant_markers` (or its alias `relevant`) still parses, for datasets written
/// before v0.12.
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
    let mut legacy = 0usize;

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
        let ids: HashSet<String> = row.relevant_ids.into_iter().collect();

        // Abstention samples legitimately have no ground truth (nothing should
        // match). Everything else needs SOME way to be scored.
        if ids.is_empty() && markers.is_empty() && !row.abstain {
            continue;
        }
        if ids.is_empty() && !row.abstain {
            legacy += 1;
        }

        out.push(EvaluationSample {
            query: row.query,
            relevant_ids: ids,
            relevant_markers: markers,
            expected_answer: row.expected_answer,
            ability: row.ability.or(row.question_type),
            abstain: row.abstain,
        });
    }

    if legacy > 0 {
        eprintln!(
            "eval: AVISO — {legacy} muestra(s) sin `relevant_ids`, puntuadas por coincidencia \
             de substring. Ese criterio cuenta como acierto cualquier documento que MENCIONE \
             el término, responda o no, y sus cifras NO son comparables con las de un dataset \
             puntuado por id."
        );
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
