use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EvaluationSample {
    pub query: String,
    pub relevant_ids: HashSet<String>,
    pub relevant_markers: Vec<String>,
    pub expected_answer: Option<String>,
    pub ability: Option<String>,
    pub abstain: bool,
}

impl EvaluationSample {
    pub fn scored_by_id(&self) -> bool {
        !self.relevant_ids.is_empty()
    }

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
    #[serde(default)]
    relevant_ids: Vec<String>,
    #[serde(default)]
    relevant_markers: Vec<String>,
    #[serde(default)]
    relevant: Vec<String>,
    #[serde(default)]
    expected_answer: Option<String>,
    #[serde(default)]
    ability: Option<String>,
    #[serde(default)]
    question_type: Option<String>,
    #[serde(default)]
    abstain: bool,
}

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

pub fn load_locomo_dataset(path: &str) -> Result<Vec<EvaluationSample>, io::Error> {
    if path.is_empty() {
        return Ok(builtin_retrieval_set());
    }
    load_jsonl_dataset(path)
}
