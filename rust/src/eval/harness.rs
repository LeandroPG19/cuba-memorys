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
    /// Per-ability breakdown (LongMemEval-style question types). Empty when the
    /// dataset carries no ability labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_ability: Vec<AbilityScore>,
    /// Fraction of abstention samples where the system correctly retrieved
    /// nothing relevant. None when the dataset has no abstention samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abstention_accuracy: Option<f64>,
}

/// Retrieval quality for one LongMemEval ability (question type).
#[derive(Debug, Clone, Serialize)]
pub struct AbilityScore {
    pub ability: String,
    pub count: usize,
    pub ndcg_at_k: f64,
    pub recall_at_k: f64,
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
        run_faro_eval(pool, &self.dataset, self.k, false).await
    }
}

/// Run hybrid `cuba_faro` per sample and aggregate IR metrics.
///
/// `associative` toggles the v0.11 multi-hop expansion so the two configs can
/// be compared on the same dataset.
pub async fn run_faro_eval(
    pool: &PgPool,
    samples: &[EvaluationSample],
    k: usize,
    associative: bool,
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
            per_ability: Vec::new(),
            abstention_accuracy: None,
        });
    }

    let mut relevance_lists: Vec<Vec<bool>> = Vec::with_capacity(samples.len());
    let mut ndcg_sum = 0.0;
    let mut prec_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut em_sum = 0.0_f32;
    let mut f1_sum = 0.0_f32;

    // Per-ability accumulators: (count, ndcg_sum, recall_sum).
    use std::collections::BTreeMap;
    let mut per_ability: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    // Abstention: count samples and how many correctly retrieved nothing relevant.
    let mut abstain_total = 0usize;
    let mut abstain_correct = 0usize;

    for sample in samples {
        let args = serde_json::json!({
            "query": sample.query,
            "mode": "hybrid",
            "limit": k,
            "enable_bm25": true,
            "rerank": false,
            "diversify": false,
            "associative": associative,
            // Do not apply the Testing Effect boost — the benchmark must not
            // mutate the corpus it is measuring.
            "track_access": false
        });
        let response = crate::handlers::faro::handle(pool, args)
            .await
            .context("faro handle failed during eval")?;
        let ranked = extract_ranked_contents(&response);
        let rels: Vec<bool> = ranked
            .iter()
            .map(|content| is_relevant(content, &sample.relevant_markers))
            .collect();

        // Abstention samples are scored differently: success = the system returned
        // NOTHING for an out-of-domain query (OOD gate / empty result), rather than
        // fabricating a top-k of loosely-related rows. Measures whether cuba
        // actually declines instead of always emitting k hits.
        if sample.abstain {
            abstain_total += 1;
            if ranked.is_empty() {
                abstain_correct += 1;
            }
            let e = per_ability
                .entry(sample.ability.clone().unwrap_or_else(|| "abstention".into()))
                .or_insert((0, 0.0, 0.0));
            e.0 += 1; // count only; ndcg/recall undefined for abstention
            continue;
        }

        let total_rel = sample.relevant_markers.len().max(1);
        let s_ndcg = ndcg_at_k(&rels, k);
        let s_recall = recall_at_k(&rels, total_rel, k);
        ndcg_sum += s_ndcg;
        prec_sum += precision_at_k(&rels, k);
        recall_sum += s_recall;
        relevance_lists.push(rels);

        if let Some(ability) = &sample.ability {
            let e = per_ability.entry(ability.clone()).or_insert((0, 0.0, 0.0));
            e.0 += 1;
            e.1 += s_ndcg;
            e.2 += s_recall;
        }

        if let Some(expected) = &sample.expected_answer {
            let top = ranked.first().map(String::as_str).unwrap_or("");
            em_sum += calculate_exact_match(top, expected);
            f1_sum += calculate_f1_score(top, expected);
        }
    }

    // Denominator for corpus-wide IR means excludes abstention samples (they have
    // no relevant docs to rank).
    let scored = (samples.len() - abstain_total).max(1);

    let n = samples.len();
    let qa_count = samples
        .iter()
        .filter(|s| s.expected_answer.is_some())
        .count();

    let ability_scores: Vec<AbilityScore> = per_ability
        .into_iter()
        .map(|(ability, (count, nd, rc))| {
            // Abstention rows contribute count only; guard the divisor.
            let denom = count.max(1) as f64;
            AbilityScore {
                ability,
                count,
                ndcg_at_k: nd / denom,
                recall_at_k: rc / denom,
            }
        })
        .collect();

    Ok(EvalReport {
        sample_count: n,
        k,
        ndcg_at_k: ndcg_sum / scored as f64,
        mrr: mrr(&relevance_lists),
        precision_at_k: prec_sum / scored as f64,
        recall_at_k: recall_sum / scored as f64,
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
        per_ability: ability_scores,
        abstention_accuracy: if abstain_total > 0 {
            Some(abstain_correct as f64 / abstain_total as f64)
        } else {
            None
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
