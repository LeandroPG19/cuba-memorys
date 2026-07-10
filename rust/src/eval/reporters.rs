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
        version: "0.10.0",
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
    if let Some(abst) = report.abstention_accuracy {
        s.push_str(&format!(" | abstention={:.0}%", abst * 100.0));
    }
    for a in &report.per_ability {
        s.push_str(&format!(
            "\n  [{}] n={} nDCG@{}={:.4} R@{}={:.4}",
            a.ability, a.count, report.k, a.ndcg_at_k, report.k, a.recall_at_k
        ));
    }
    s
}
