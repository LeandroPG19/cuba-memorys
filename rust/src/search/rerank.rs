//! Cross-encoder reranker — bge-reranker-v2-m3 (Xiao 2023).
//!
//! Real ONNX forward pass via the `ort` crate (already used by
//! `embeddings::onnx`). Activated when `CUBA_RERANKER_PATH` points to a
//! directory containing `model.onnx` (or `model_quantized.onnx`) plus
//! `tokenizer.json`. Identity fallback when not configured.
//!
//! ## Architecture
//!
//! bge-reranker is a cross-encoder: it tokenizes the [CLS] + query +
//! [SEP] + passage + [SEP] sentence pair and emits a single relevance
//! logit per pair. We sigmoid the logit to a [0, 1] score and sort
//! descending. One forward pass per (query, candidate) pair — typically
//! 50 candidates × ~10 ms each on CPU = ~500 ms latency, paid only when
//! the user opts into reranking via `cuba_faro rerank=true`.
//!
//! ## Pipeline used by `cuba_faro`
//!
//!   top-50 from RRF → rerank() → top-K returned to client
//!
//! Expected gain: +12-25% nDCG@10 over RRF alone (Xiao 2023).
//!
//! ## Threading
//!
//! Same pattern as `embeddings::onnx`: shared `Session` behind a `Mutex`,
//! forward passes wrapped in `tokio::task::spawn_blocking`, semaphore caps
//! concurrent inference at 2 to match `with_intra_threads(2)` (Little's
//! Law — prevents threadpool starvation under load).

use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Resolved once at startup. `None` = reranker disabled (identity).
static RERANKER_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Reranker model status — loaded only after a successful init.
static RERANKER_STATUS: OnceLock<RerankerStatus> = OnceLock::new();

/// ONNX session (Mutex because `Session::run` requires `&mut self`).
static RERANKER_SESSION: OnceLock<std::sync::Mutex<Session>> = OnceLock::new();

/// Tokenizer for the reranker.
static RERANKER_TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();

/// Concurrency cap mirroring the embeddings semaphore. Same Little's Law
/// reasoning: prevent spawn_blocking pool starvation.
static RERANKER_SEMAPHORE: OnceLock<tokio::sync::Semaphore> = OnceLock::new();

enum RerankerStatus {
    /// Real ONNX cross-encoder loaded.
    Loaded,
    /// No model — identity fallback.
    Fallback,
}

fn semaphore() -> &'static tokio::sync::Semaphore {
    RERANKER_SEMAPHORE.get_or_init(|| tokio::sync::Semaphore::new(2))
}

/// Returns true if `CUBA_RERANKER_PATH` points to a valid model directory.
/// Lazy — does the resolution exactly once per process.
pub fn enabled() -> bool {
    matches!(get_status(), RerankerStatus::Loaded)
}

fn get_status() -> &'static RerankerStatus {
    RERANKER_STATUS.get_or_init(|| {
        let path = RERANKER_PATH.get_or_init(|| {
            std::env::var("CUBA_RERANKER_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|p| p.exists())
        });
        match path {
            Some(p) => match init_session(p) {
                Ok(()) => {
                    tracing::info!(path = %p.display(), "bge-reranker ONNX loaded");
                    RerankerStatus::Loaded
                }
                Err(e) => {
                    tracing::warn!(error = %e, "reranker init failed — identity fallback");
                    RerankerStatus::Fallback
                }
            },
            None => RerankerStatus::Fallback,
        }
    })
}

/// Initialize the ONNX session and tokenizer from the model directory.
/// Mirrors the loader in `embeddings::onnx::init_onnx_session`.
fn init_session(model_dir: &std::path::Path) -> Result<()> {
    // Pick model file: prefer quantized, fall back to plain `model.onnx`.
    let candidates = ["model_quantized.onnx", "model.onnx"];
    let model_file = candidates
        .iter()
        .map(|n| model_dir.join(n))
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow::anyhow!("no model.onnx / model_quantized.onnx found in {model_dir:?}")
        })?;

    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_intra_threads(2)
        .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
        .commit_from_file(&model_file)
        .map_err(|e| anyhow::anyhow!("load model: {e}"))?;
    RERANKER_SESSION
        .set(std::sync::Mutex::new(session))
        .map_err(|_| anyhow::anyhow!("session already initialized"))?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json missing at {tokenizer_path:?}");
    }
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let truncation = tokenizers::TruncationParams {
        max_length: 512,
        ..Default::default()
    };
    tokenizer
        .with_truncation(Some(truncation))
        .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
    let padding = tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    tokenizer.with_padding(Some(padding));
    RERANKER_TOKENIZER
        .set(tokenizer)
        .map_err(|_| anyhow::anyhow!("tokenizer already initialized"))?;
    Ok(())
}

/// Re-rank `candidates` against `query`. Returns `(index, score)` pairs
/// sorted by score descending.
///
/// When the reranker is disabled (no model loaded), returns identity
/// pairs preserving the upstream RRF order — production transparent.
pub async fn rerank(query: &str, candidates: &[&str]) -> Result<Vec<(usize, f64)>> {
    if !enabled() {
        return Ok(identity_pairs(candidates.len()));
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Move owned strings into the blocking task — avoids lifetime grief.
    let query_owned = query.to_string();
    let candidates_owned: Vec<String> = candidates.iter().map(|c| c.to_string()).collect();

    // Cap concurrent inference (Little's Law).
    let _permit = semaphore()
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("reranker semaphore closed"))?;

    let scored = tokio::task::spawn_blocking(move || score_pairs(&query_owned, &candidates_owned))
        .await
        .context("reranker task panicked")??;

    let mut indexed: Vec<(usize, f64)> = scored.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed)
}

/// CPU-bound scoring. Tokenizes each (query, candidate) pair and runs the
/// ONNX forward pass to extract a single relevance logit, then sigmoids
/// to [0, 1]. One inference per candidate — bge-reranker is a cross-encoder
/// (no batched single-tower trick).
fn score_pairs(query: &str, candidates: &[String]) -> Result<Vec<f64>> {
    let session_lock = RERANKER_SESSION
        .get()
        .context("reranker session not initialized")?;
    let mut session = session_lock
        .lock()
        .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
    let tokenizer = RERANKER_TOKENIZER
        .get()
        .context("reranker tokenizer not initialized")?;

    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        // bge-reranker tokenizes the pair as `[CLS] query [SEP] candidate [SEP]`.
        // Tokenizer's encode_pair handles the [SEP]/segment ids correctly.
        let encoding = tokenizer
            .encode((query, candidate.as_str()), true)
            .map_err(|e| anyhow::anyhow!("encode pair: {e}"))?;

        let ids = encoding.get_ids();
        let attn_mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len();

        let input_ids: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
        let attn: Vec<i64> = attn_mask.iter().map(|&m| m as i64).collect();
        let types: Vec<i64> = type_ids.iter().map(|&t| t as i64).collect();
        let shape = vec![1i64, seq_len as i64];

        let input_ids_t = ort::value::Tensor::from_array((shape.clone(), input_ids))
            .context("input_ids tensor")?;
        let attn_t = ort::value::Tensor::from_array((shape.clone(), attn))
            .context("attention_mask tensor")?;
        let type_t = ort::value::Tensor::from_array((shape, types))
            .context("token_type_ids tensor")?;

        let outputs = session
            .run(ort::inputs! {
                "input_ids" => input_ids_t,
                "attention_mask" => attn_t,
                "token_type_ids" => type_t,
            })
            .map_err(|e| anyhow::anyhow!("inference: {e}"))?;

        if outputs.len() == 0 {
            anyhow::bail!("reranker returned no outputs");
        }
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract logits: {e}"))?;

        // bge-reranker output: [batch=1, num_labels=1] → single logit.
        // Some checkpoints emit [1, 2] (binary classification) — pick label 1.
        let logit = match out_shape.as_ref() {
            &[_, 1] | &[1] => data[0],
            &[_, 2] | &[2] => data[1] - data[0], // log-odds of relevance
            _ => {
                tracing::warn!(?out_shape, "unexpected reranker output shape — taking [0]");
                data[0]
            }
        };
        // Sigmoid → [0, 1] relevance probability.
        let score = 1.0_f64 / (1.0 + (-logit as f64).exp());
        scores.push(score);
    }
    Ok(scores)
}

fn identity_pairs(n: usize) -> Vec<(usize, f64)> {
    (0..n).map(|i| (i, (n - i) as f64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn identity_when_disabled() {
        // Env var not set → enabled() = false → identity output.
        // SAFETY: tests in this module run sequentially; no other thread
        // mutates env. We do NOT remove CUBA_RERANKER_PATH because
        // RERANKER_STATUS is cached after first call — the OnceLock makes
        // this safe regardless.
        let pairs = rerank("anything", &["a", "b", "c"]).await.unwrap();
        assert_eq!(pairs.len(), 3);
        if !enabled() {
            // Identity preserves order
            assert_eq!(pairs[0].0, 0);
            assert!(pairs[0].1 > pairs[1].1);
        }
    }

    #[test]
    fn identity_pairs_descending() {
        let pairs = identity_pairs(5);
        for win in pairs.windows(2) {
            assert!(win[0].1 > win[1].1);
        }
    }

    #[tokio::test]
    async fn empty_candidates_returns_empty() {
        let pairs = rerank("q", &[]).await.unwrap();
        assert!(pairs.is_empty());
    }
}
