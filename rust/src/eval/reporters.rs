use serde::Serialize;

use super::harness::EvalReport;

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

    if report.minimum_detectable_effect.is_finite() {
        s.push_str(&format!(
            "\nefecto mínimo detectable = {:.3} nDCG (80% poder, α=.05) — una diferencia menor \
             que esto NO es medible con n={}",
            report.minimum_detectable_effect, report.sample_count
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
