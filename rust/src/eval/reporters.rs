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
    let mut s = format!(
        "nDCG@{}={:.4} MRR={:.4} P@{}={:.4} R@{}={:.4} (n={})",
        report.k,
        report.ndcg_at_k,
        report.mrr,
        report.k,
        report.precision_at_k,
        report.k,
        report.recall_at_k,
        report.sample_count
    );
    // Cost rides next to quality, always. A retrieval that scores higher while
    // costing twice the context is not obviously a better retrieval, and the
    // only way to notice is to print both numbers side by side.
    s.push_str(&format!(
        " | tokens: mean={:.0} max={}",
        report.mean_response_tokens, report.max_response_tokens
    ));
    if let Some(abst) = report.abstention_accuracy {
        s.push_str(&format!("\nabstention={:.0}%", abst * 100.0));
        // Meaningless without its counterweight: a system that answers nothing
        // scores 100% abstention.
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
