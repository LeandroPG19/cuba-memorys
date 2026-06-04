use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use sqlx::PgPool;

use super::datasets::EvaluationSample;
use super::metrics::{mrr, ndcg_at_k, precision_at_k, recall_at_k};
use crate::eval::metrics::{calculate_exact_match, calculate_f1_score};

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub sample_count: usize,
    pub k: usize,
    pub ndcg_at_k: f64,
    pub mrr: f64,
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub mean_exact_match: f32,
    pub mean_f1: f32,
}

pub struct BenchmarkHarness {
    dataset: Vec<EvaluationSample>,
    k: usize,
}

impl BenchmarkHarness {
    pub fn new(dataset: Vec<EvaluationSample>) -> Self {
        Self { dataset, k: 10 }
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.clamp(1, 50);
        self
    }

    pub async fn run(&self, pool: &PgPool) -> Result<EvalReport> {
        run_faro_eval(pool, &self.dataset, self.k).await
    }
}

/// Run hybrid `cuba_faro` per sample and aggregate IR metrics.
pub async fn run_faro_eval(
    pool: &PgPool,
    samples: &[EvaluationSample],
    k: usize,
) -> Result<EvalReport> {
    let k = k.clamp(1, 50);
    if samples.is_empty() {
        return Ok(EvalReport {
            sample_count: 0,
            k,
            ndcg_at_k: 0.0,
            mrr: 0.0,
            precision_at_k: 0.0,
            recall_at_k: 0.0,
            mean_exact_match: 0.0,
            mean_f1: 0.0,
        });
    }

    let mut relevance_lists: Vec<Vec<bool>> = Vec::with_capacity(samples.len());
    let mut ndcg_sum = 0.0;
    let mut prec_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut em_sum = 0.0_f32;
    let mut f1_sum = 0.0_f32;

    for sample in samples {
        let args = serde_json::json!({
            "query": sample.query,
            "mode": "hybrid",
            "limit": k,
            "enable_bm25": true,
            "rerank": false,
            "diversify": false
        });
        let response = crate::handlers::faro::handle(pool, args)
            .await
            .context("faro handle failed during eval")?;
        let ranked = extract_ranked_contents(&response);
        let rels: Vec<bool> = ranked
            .iter()
            .map(|content| is_relevant(content, &sample.relevant_markers))
            .collect();
        let total_rel = sample.relevant_markers.len().max(1);
        ndcg_sum += ndcg_at_k(&rels, k);
        prec_sum += precision_at_k(&rels, k);
        recall_sum += recall_at_k(&rels, total_rel, k);
        relevance_lists.push(rels);

        if let Some(expected) = &sample.expected_answer {
            let top = ranked.first().map(String::as_str).unwrap_or("");
            em_sum += calculate_exact_match(top, expected);
            f1_sum += calculate_f1_score(top, expected);
        }
    }

    let n = samples.len();
    let qa_count = samples
        .iter()
        .filter(|s| s.expected_answer.is_some())
        .count();

    Ok(EvalReport {
        sample_count: n,
        k,
        ndcg_at_k: ndcg_sum / n as f64,
        mrr: mrr(&relevance_lists),
        precision_at_k: prec_sum / n as f64,
        recall_at_k: recall_sum / n as f64,
        mean_exact_match: if qa_count > 0 {
            em_sum / qa_count as f32
        } else {
            0.0
        },
        mean_f1: if qa_count > 0 {
            f1_sum / qa_count as f32
        } else {
            0.0
        },
    })
}

fn extract_ranked_contents(response: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let results = response
        .get("results")
        .or_else(|| response.get("observations"))
        .and_then(|v| v.as_array());
    if let Some(arr) = results {
        for item in arr {
            if let Some(c) = item.get("content").and_then(|v| v.as_str()) {
                out.push(c.to_string());
            } else if let Some(c) = item.get("c").and_then(|v| v.as_str()) {
                out.push(c.to_string());
            }
        }
    }
    out
}

fn is_relevant(content: &str, markers: &[String]) -> bool {
    let lower = content.to_lowercase();
    markers
        .iter()
        .any(|m| !m.is_empty() && lower.contains(&m.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relevance_marker_match() {
        assert!(is_relevant(
            "fallo de conexión postgres en docker",
            &["postgres".into()]
        ));
        assert!(!is_relevant("todo ok", &["postgres".into()]));
    }
}
