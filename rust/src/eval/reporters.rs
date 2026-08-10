use serde::Serialize;

use super::harness::EvalReport;

#[cfg(test)]
fn report_with_warmup(warmup_ms: f64) -> EvalReport {
    EvalReport {
        sample_count: 5,
        k: 10,
        ndcg_at_k: 0.5,
        mrr: 0.5,
        precision_at_k: 0.5,
        recall_at_k: 0.5,
        mean_exact_match: 0.0,
        mean_f1: 0.0,
        mean_response_tokens: 100.0,
        max_response_tokens: 200,
        per_ability: Vec::new(),
        abstention_accuracy: None,
        false_abstention_rate: None,
        ndcg_ci95: (0.4, 0.6),
        minimum_detectable_effect: 0.1,
        per_query_ndcg: Vec::new(),
        scored_by_id: true,
        latency_p50_ms: 1500.0,
        latency_p95_ms: 1700.0,
        warmup_ms,
    }
}

#[derive(Serialize)]
pub struct JsonReport<'a> {
    pub version: &'static str,
    pub samples: usize,
    pub k: usize,
    pub metrics: &'a EvalReport,
}

pub fn generate_json_report(report: &EvalReport, samples: usize, k: usize) -> String {
    let payload = JsonReport {
        version: env!("CARGO_PKG_VERSION"),
        samples,
        k,
        metrics: report,
    };
    serde_json::to_string_pretty(&payload).unwrap_or_default()
}

pub fn summary_line(report: &EvalReport) -> String {
    let (lo, hi) = report.ndcg_ci95;
    let mut s = format!(
        "nDCG@{}={:.4} [95% CI {:.3}–{:.3}] MRR={:.4} P@{}={:.4} R@{}={:.4} (n={})",
        report.k,
        report.ndcg_at_k,
        lo,
        hi,
        report.mrr,
        report.k,
        report.precision_at_k,
        report.k,
        report.recall_at_k,
        report.sample_count
    );
    s.push_str(&format!(
        " | tokens: mean={:.0} max={}",
        report.mean_response_tokens, report.max_response_tokens
    ));
    if report.latency_p50_ms > 0.0 {
        s.push_str(&format!(
            " | latencia: p50={:.0}ms p95={:.0}ms",
            report.latency_p50_ms, report.latency_p95_ms
        ));
    }
    if report.warmup_ms > 0.0 {
        s.push_str(&format!(
            " | warm-up (fuera de la latencia): {:.0}ms",
            report.warmup_ms
        ));
    }

    if report.minimum_detectable_effect.is_finite() {
        s.push_str(&format!(
            "\nefecto mínimo detectable = {:.3} nDCG (80% poder, α=.05) sobre n={} puntuadas — \
             es la COTA DEL PEOR CASO, la de dos muestras independientes. Comparando dos corridas \
             de este mismo dataset el test es pareado y resuelve diferencias mucho menores: \
             usá paired_bootstrap sobre per_query_ndcg, no este número",
            report.minimum_detectable_effect,
            report.per_query_ndcg.len(),
        ));
    }

    if !report.scored_by_id && report.sample_count > 0 {
        s.push_str(
            "\n⚠ puntuado por SUBSTRING, no por id: cuenta como acierto cualquier documento que \
             MENCIONE el término, responda o no. Las cifras no son comparables con las de un \
             dataset con `relevant_ids`.",
        );
    }
    if let Some(abst) = report.abstention_accuracy {
        s.push_str(&format!("\nabstention={:.0}%", abst * 100.0));
        if let Some(fa) = report.false_abstention_rate {
            s.push_str(&format!(
                " (falsas abstenciones sobre lo respondible={:.0}%)",
                fa * 100.0
            ));
        }
    }
    for a in &report.per_ability {
        s.push_str(&format!(
            "\n  [{}] n={} nDCG@{}={:.4} R@{}={:.4}",
            a.ability, a.count, report.k, a.ndcg_at_k, report.k, a.recall_at_k
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_line_reports_the_model_load_separately_from_the_search_latency() {
        let report = report_with_warmup(70180.0);
        let line = summary_line(&report);

        assert!(
            line.contains("warm-up (fuera de la latencia): 70180ms"),
            "the cold-start model load must be visible, not silently folded into p50/p95: {line}"
        );
    }

    #[test]
    fn summary_line_omits_the_warmup_note_when_the_models_were_already_warm() {
        let report = report_with_warmup(0.0);
        let line = summary_line(&report);

        assert!(!line.contains("warm-up"), "{line}");
    }

    #[test]
    fn json_report_exposes_warmup_ms_as_its_own_field() {
        let report = report_with_warmup(70180.0);
        let json = generate_json_report(&report, report.sample_count, report.k);

        assert!(json.contains("\"warmup_ms\": 70180.0"), "{json}");
    }
}
