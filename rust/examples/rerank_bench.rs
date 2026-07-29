//! Measures cross-encoder rerank latency against its own budget.
//!
//! The reranker is only worth its +93% nDCG if it finishes inside
//! `CUBA_RERANK_TIMEOUT_SECS` (20 s by default). Past that, `faro` discards the
//! scores and returns the RRF order — the inference cost is paid and thrown
//! away. This tells you where a machine actually lands.
//!
//! ```bash
//! cargo run --release --example rerank_bench                 # auto threads
//! CUBA_RERANK_INTRA_THREADS=2 cargo run --release --example rerank_bench
//! cargo run --release --example rerank_bench -- 50 3         # 50 candidates, 3 runs
//! ```

use std::time::{Duration, Instant};

const DEFAULT_CANDIDATES: usize = 50;
const DEFAULT_RUNS: usize = 3;

/// Roughly the length of a real observation, so token counts are representative.
///
/// Real candidates are not all the same size — a one-line preference sits next
/// to a full postmortem — and since a batch pads to its longest member, that
/// spread is itself a cost. `uniform` isolates the thread effect; the default
/// mixed corpus is what a live `faro` call actually hands the cross-encoder.
fn corpus(n: usize, uniform: bool) -> Vec<String> {
    let base = "The retrieval pipeline fuses lexical BM25 with pgvector cosine similarity \
                through reciprocal rank fusion, and the surviving candidates are rescored by \
                a cross-encoder that reads query and passage together instead of embedding \
                them apart. Postgres holds the graph; the reranker only reorders it.";
    (0..n)
        .map(|i| {
            let repeats = if uniform { 1 } else { 1 + (i * 7) % 9 };
            let body = base.repeat(repeats);
            format!("[doc {i}] {body} Variant {i} discusses shard {i} of the corpus.")
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CANDIDATES);
    let runs: usize = args
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RUNS);

    let budget = Duration::from_secs(
        std::env::var("CUBA_RERANK_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20),
    );

    let threads = std::env::var("CUBA_RERANK_INTRA_THREADS").unwrap_or_else(|_| "auto".to_string());
    let uniform = std::env::var("BENCH_UNIFORM").is_ok();
    let bucketing =
        std::env::var("CUBA_RERANK_LENGTH_BUCKETING").unwrap_or_else(|_| "auto".to_string());
    println!(
        "candidates={n} runs={runs} intra_threads={threads} budget={budget:?} \
         corpus={} bucketing={bucketing}",
        if uniform { "uniform" } else { "mixed" }
    );

    let docs = corpus(n, uniform);
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let query = "how are candidates rescored after fusion?";

    let load = Instant::now();
    let enabled = tokio::task::spawn_blocking(cuba_memorys::search::rerank::enabled)
        .await
        .unwrap();
    let load = load.elapsed();

    if !enabled {
        println!("reranker not available — nothing to measure");
        println!("  install it with: cuba-memorys models reranker");
        return;
    }
    println!("model load: {load:.2?}");

    let mut timings = Vec::with_capacity(runs);
    for i in 1..=runs {
        let started = Instant::now();
        let scored = cuba_memorys::search::rerank::rerank(query, &refs)
            .await
            .expect("rerank failed");
        let elapsed = started.elapsed();
        assert_eq!(scored.len(), n, "reranker dropped candidates");
        timings.push(elapsed);
        let verdict = if elapsed <= budget {
            "within budget"
        } else {
            "OVER BUDGET — scores discarded"
        };
        println!("  run {i}: {elapsed:.2?}  ({verdict})");
    }

    // The ranking itself, so a CPU run and a GPU run can be diffed. Speed is only
    // an improvement if the order it produces is the same one.
    let scored = cuba_memorys::search::rerank::rerank(query, &refs)
        .await
        .expect("rerank failed");
    println!("top-8 ranking (index:score):");
    for (idx, score) in scored.iter().take(8) {
        println!("  {idx}:{score:.6}");
    }

    let total: Duration = timings.iter().sum();
    let mean = total / runs as u32;
    let best = timings.iter().min().unwrap();
    println!(
        "mean {:.2?} | best {:.2?} | per candidate {:.1?}",
        mean,
        best,
        mean / n as u32
    );
    if mean <= budget {
        println!("VERDICT: fits in the {budget:?} budget — the rerank actually applies");
    } else {
        println!(
            "VERDICT: exceeds the {budget:?} budget by {:.2?} — faro will fall back to RRF",
            mean - budget
        );
    }
}
