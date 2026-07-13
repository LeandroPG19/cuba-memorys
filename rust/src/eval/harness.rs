use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use sqlx::PgPool;

use super::datasets::EvaluationSample;
use super::metrics::{
    bootstrap_ci, minimum_detectable_effect, mrr, ndcg_at_k, precision_at_k, recall_at_k,
};
use crate::eval::metrics::{calculate_exact_match, calculate_f1_score};

/// Bootstrap resamples. 2000 is the usual floor for a stable 95% interval and costs
/// microseconds — the expensive part of this benchmark is the retrieval, not the
/// statistics.
const BOOTSTRAP_ITERATIONS: usize = 2000;

/// What the eval actually switches on.
///
/// This exists because the previous signature hard-coded `rerank: false` and
/// never passed `abstain_ood` at all. Two features were being measured with the
/// switch off, and then reported as scoring zero — which read as "they do not
/// work" when it meant "they never ran".
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub k: usize,
    /// Multi-hop associative expansion (v0.11).
    pub associative: bool,
    /// Let the OOD gate fire, so abstention can actually be measured.
    pub abstain: bool,
    /// Run the cross-encoder reranker.
    pub rerank: bool,
    /// Response shape — and therefore what the token cost actually is.
    pub format: String,
}

impl Default for EvalConfig {
    fn default() -> Self {
        // Everything off, and `verbose`, so the baseline stays comparable with
        // what was already measured (nDCG@10 = 0.7389). The CLI flags turn each
        // one on for an A/B.
        //
        // `verbose` specifically, even though the handler now defaults to
        // compact: relevance is judged by looking for markers in the returned
        // text, and compact truncates that text to 200 chars. Grading on the
        // truncated shape would score a marker at character 300 as a miss —
        // penalizing the *presentation* format for a retrieval that was correct.
        // `--format compact` then measures what the saving actually costs.
        Self {
            k: 10,
            associative: false,
            abstain: false,
            rerank: false,
            format: "verbose".to_string(),
        }
    }
}

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
    /// Mean tokens the response actually costs the agent's context, counted with
    /// tiktoken (cl100k_base) over the serialized payload — not the retrieved
    /// text alone. Quality per token is the axis the field competes on now
    /// (Mem0 advertises 93.4% on LongMemEval under 7k tokens per retrieval);
    /// an optimization that improves nDCG while doubling cost is not a win, and
    /// without this number you cannot tell the two apart.
    pub mean_response_tokens: f64,
    /// Worst-case cost. The mean hides the query that blows the context.
    pub max_response_tokens: usize,
    /// Per-ability breakdown (LongMemEval-style question types). Empty when the
    /// dataset carries no ability labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_ability: Vec<AbilityScore>,
    /// Fraction of abstention samples where the system correctly retrieved
    /// nothing relevant. None when the dataset has no abstention samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abstention_accuracy: Option<f64>,
    /// False abstentions: answerable queries the system wrongly declined.
    /// Abstention with no such counterweight is trivially maximized by refusing
    /// to answer anything.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub false_abstention_rate: Option<f64>,

    /// 95% bootstrap interval around `ndcg_at_k`. A mean without one is a point
    /// estimate that invites exactly the mistake this project already made twice:
    /// reading a 3-point difference as a regression when the interval was ±12.
    pub ndcg_ci95: (f64, f64),
    /// The smallest true difference this sample size can detect (80% power, α=.05).
    /// **Read this before believing any A/B on this dataset.** If a comparison shows
    /// a gap smaller than this, the honest report is "could not measure", not "no
    /// effect".
    pub minimum_detectable_effect: f64,
    /// nDCG per query, in dataset order. Kept so failures can be inspected instead
    /// of averaged away — the mean tells you the score, never which query broke.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_query_ndcg: Vec<f64>,
    /// True when every scorable sample carried `relevant_ids`. When false, some
    /// queries were graded by substring match — a laxer criterion whose numbers are
    /// NOT comparable to id-scored ones.
    pub scored_by_id: bool,
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
        let cfg = EvalConfig {
            k: self.k,
            ..EvalConfig::default()
        };
        run_faro_eval(pool, &self.dataset, &cfg).await
    }
}

/// Run hybrid `cuba_faro` per sample and aggregate IR metrics.
///
/// `associative` toggles the v0.11 multi-hop expansion so the two configs can
/// be compared on the same dataset.
pub async fn run_faro_eval(
    pool: &PgPool,
    samples: &[EvaluationSample],
    cfg: &EvalConfig,
) -> Result<EvalReport> {
    let k = cfg.k.clamp(1, 50);
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
            mean_response_tokens: 0.0,
            max_response_tokens: 0,
            per_ability: Vec::new(),
            abstention_accuracy: None,
            false_abstention_rate: None,
            ndcg_ci95: (0.0, 0.0),
            minimum_detectable_effect: f64::INFINITY,
            per_query_ndcg: Vec::new(),
            scored_by_id: false,
        });
    }

    let mut relevance_lists: Vec<Vec<bool>> = Vec::with_capacity(samples.len());
    let mut ndcg_scores: Vec<f64> = Vec::with_capacity(samples.len());
    let mut ndcg_sum = 0.0;
    let mut prec_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut em_sum = 0.0_f32;
    let mut f1_sum = 0.0_f32;

    // Every scorable sample must carry ids, or the whole run is graded on the laxer
    // substring criterion and its numbers cannot be compared with an id-scored run.
    let all_by_id = samples
        .iter()
        .filter(|s| !s.abstain)
        .all(|s| s.scored_by_id());

    // Cost, measured on the payload the agent would actually receive.
    let mut token_sum = 0usize;
    let mut token_max = 0usize;

    // Per-ability accumulators: (count, ndcg_sum, recall_sum).
    use std::collections::BTreeMap;
    let mut per_ability: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    // Abstention: count samples and how many correctly retrieved nothing relevant.
    let mut abstain_total = 0usize;
    let mut abstain_correct = 0usize;
    // The counterweight: answerable queries the system wrongly declined.
    let mut answerable_total = 0usize;
    let mut false_abstentions = 0usize;

    for sample in samples {
        let args = serde_json::json!({
            "query": sample.query,
            "mode": "hybrid",
            "limit": k,
            "enable_bm25": true,
            "rerank": cfg.rerank,
            "diversify": false,
            "associative": cfg.associative,
            "abstain_ood": cfg.abstain,
            "format": cfg.format,
            // Do not apply the Testing Effect boost — the benchmark must not
            // mutate the corpus it is measuring.
            "track_access": false
        });
        let response = crate::handlers::faro::handle(pool, args)
            .await
            .context("faro handle failed during eval")?;

        // Exactly what the response costs in the agent's context.
        let cost = crate::search::budget::count_tokens(&response.to_string());
        token_sum += cost;
        token_max = token_max.max(cost);

        let ranked = extract_ranked(&response);
        let rels: Vec<bool> = ranked.iter().map(|hit| hit.is_relevant(sample)).collect();

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
                .entry(
                    sample
                        .ability
                        .clone()
                        .unwrap_or_else(|| "abstention".into()),
                )
                .or_insert((0, 0.0, 0.0));
            e.0 += 1; // count only; ndcg/recall undefined for abstention
            continue;
        }

        // An answerable query that came back empty is a FALSE abstention. Without
        // tracking these, "abstention accuracy" is trivially maximized by a system
        // that refuses to answer anything at all.
        answerable_total += 1;
        if ranked.is_empty() {
            false_abstentions += 1;
        }

        // The denominator that makes both metrics mean what they say: how many
        // documents in the CORPUS answer this query — including the ones retrieval
        // missed. Grading against only what was found is how a system that recovered
        // 2 of 5 relevant documents used to score a perfect nDCG.
        let total_rel = sample.relevant_count();
        let s_ndcg = ndcg_at_k(&rels, k, total_rel);
        let s_recall = recall_at_k(&rels, total_rel, k);
        ndcg_sum += s_ndcg;
        ndcg_scores.push(s_ndcg);
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
            let top = ranked.first().map(|h| h.content.as_str()).unwrap_or("");
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

    let (_, lo, hi) = bootstrap_ci(&ndcg_scores, BOOTSTRAP_ITERATIONS, 0.95);
    let mde = minimum_detectable_effect(&ndcg_scores);

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
        mean_response_tokens: token_sum as f64 / n as f64,
        max_response_tokens: token_max,
        per_ability: ability_scores,
        abstention_accuracy: if abstain_total > 0 {
            Some(abstain_correct as f64 / abstain_total as f64)
        } else {
            None
        },
        false_abstention_rate: if answerable_total > 0 {
            Some(false_abstentions as f64 / answerable_total as f64)
        } else {
            None
        },
        ndcg_ci95: (lo, hi),
        minimum_detectable_effect: mde,
        per_query_ndcg: ndcg_scores,
        scored_by_id: all_by_id,
    })
}

/// One retrieved result: its id, and its text.
struct Hit {
    id: Option<String>,
    content: String,
}

impl Hit {
    /// Relevant iff the ground truth says so.
    ///
    /// By id when the dataset provides one — a result is the answer, or it is not.
    /// By substring only for pre-v0.12 datasets, where "relevant" meant "mentions
    /// the word", which counted every observation containing "postgres" as a correct
    /// answer to a question about postgres, whether it answered anything or not.
    fn is_relevant(&self, sample: &EvaluationSample) -> bool {
        if sample.scored_by_id() {
            return self
                .id
                .as_ref()
                .is_some_and(|id| sample.relevant_ids.contains(id));
        }
        is_relevant_by_marker(&self.content, &sample.relevant_markers)
    }
}

fn extract_ranked(response: &Value) -> Vec<Hit> {
    let mut out = Vec::new();
    let results = response
        .get("results")
        .or_else(|| response.get("observations"))
        .and_then(|v| v.as_array());
    if let Some(arr) = results {
        for item in arr {
            // `id` is present in both shapes. It was missing from compact until
            // v0.11.1, which is precisely why compact could not be graded by id.
            let id = item.get("id").and_then(|v| v.as_str()).map(str::to_string);
            let content = item
                .get("content")
                .or_else(|| item.get("c"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            out.push(Hit { id, content });
        }
    }
    out
}

fn is_relevant_by_marker(content: &str, markers: &[String]) -> bool {
    let lower = content.to_lowercase();
    markers
        .iter()
        .any(|m| !m.is_empty() && lower.contains(&m.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn sample_by_id(ids: &[&str]) -> EvaluationSample {
        EvaluationSample {
            query: "¿en qué lenguaje está escrito?".into(),
            relevant_ids: ids.iter().map(|s| s.to_string()).collect(),
            relevant_markers: vec![],
            expected_answer: None,
            ability: None,
            abstain: false,
        }
    }

    fn hit(id: &str, content: &str) -> Hit {
        Hit {
            id: Some(id.into()),
            content: content.into(),
        }
    }

    /// The whole point of v0.12: a document is the answer, or it is not. Mentioning
    /// the topic is not answering the question.
    #[test]
    fn id_scoring_does_not_reward_merely_being_on_topic() {
        let sample = sample_by_id(&["the-answer"]);

        assert!(hit("the-answer", "cuba-memorys está escrito en Rust").is_relevant(&sample));

        // Same subject, same vocabulary, says nothing about the language. Under the
        // old substring rule with marker "cuba-memorys" this scored as a hit.
        assert!(
            !hit("some-other-row", "cuba-memorys usa PostgreSQL con pgvector").is_relevant(&sample),
            "an on-topic document that is not the answer must not count as relevant"
        );
    }

    /// Legacy datasets still load, and still score the lax way — which is exactly
    /// why the harness reports `scored_by_id: false` and warns.
    #[test]
    fn marker_scoring_survives_for_old_datasets() {
        let legacy = EvaluationSample {
            query: "error conexión postgres".into(),
            relevant_ids: HashSet::new(),
            relevant_markers: vec!["postgres".into()],
            expected_answer: None,
            ability: None,
            abstain: false,
        };
        assert!(!legacy.scored_by_id());
        assert!(hit("x", "fallo de conexión postgres en docker").is_relevant(&legacy));
        assert!(!hit("y", "todo ok").is_relevant(&legacy));
    }

    /// Recall's denominator is the number of relevant DOCUMENTS, not of markers.
    /// The old code used `relevant_markers.len()`, which is how R@10 = 3.125 — a
    /// value a proportion cannot take — ended up in the README.
    #[test]
    fn relevant_count_is_the_size_of_the_ground_truth() {
        assert_eq!(sample_by_id(&["a", "b", "c"]).relevant_count(), 3);
    }

    /// `id` must survive extraction from BOTH response shapes. Grading by id is
    /// impossible otherwise — and compact carried no id at all until v0.11.1.
    #[test]
    fn ids_are_extracted_from_verbose_and_compact() {
        let verbose = serde_json::json!({
            "results": [{"id": "abc", "content": "texto largo", "entity_name": "e"}]
        });
        let compact = serde_json::json!({
            "results": [{"id": "abc", "c": "texto largo", "e": "e"}]
        });

        for (shape, response) in [("verbose", verbose), ("compact", compact)] {
            let hits = extract_ranked(&response);
            assert_eq!(hits.len(), 1, "{shape}");
            assert_eq!(hits[0].id.as_deref(), Some("abc"), "{shape}: falta el id");
            assert_eq!(
                hits[0].content, "texto largo",
                "{shape}: falta el contenido"
            );
        }
    }
}
