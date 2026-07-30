use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::sync::OnceLock;

static RERANKER_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

static RERANKER_STATUS: OnceLock<RerankerStatus> = OnceLock::new();

static RERANKER_SESSION: OnceLock<std::sync::Mutex<Session>> = OnceLock::new();

static RERANKER_TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();

static RERANKER_SEMAPHORE: OnceLock<tokio::sync::Semaphore> = OnceLock::new();

enum RerankerStatus {
    Loaded,
    Fallback,
}

fn semaphore() -> &'static tokio::sync::Semaphore {
    RERANKER_SEMAPHORE.get_or_init(|| tokio::sync::Semaphore::new(2))
}

/// How many threads ONNX Runtime may use inside a single inference.
///
/// This was hardcoded to 2. The cross-encoder is an XLM-RoBERTa-large — 24
/// layers, 1024 hidden — and every candidate is a full forward pass over up to
/// 512 tokens, so two threads is what pushed a 50-candidate rerank past its own
/// 20 s budget: the model ran, and the scores were thrown away for a timeout.
///
/// Physical cores, not SMT siblings: GEMM kernels saturate the vector units, so
/// hyperthreads add scheduling overhead without adding throughput.
fn intra_threads() -> usize {
    if let Some(n) = std::env::var("CUBA_RERANK_INTRA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    // On the GPU the GEMMs run as CUDA kernels and these threads only marshal
    // tensors across the boundary, so a wide pool is idle stacks contending with
    // the embedder's — and the embedder is the one that really runs on the CPU.
    if crate::gpu::wants_gpu(crate::gpu::Workload::Reranker) {
        return 2;
    }
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 8))
        .unwrap_or(2)
}

/// Whether the cross-encoder is loaded and usable.
///
/// Loading is lazy, so the first call pays for a multi-GB model read. That makes
/// this a *blocking* call: never invoke it straight from an async task — wrap it
/// in `spawn_blocking`, or one client's first search stalls the whole runtime,
/// which under the shared daemon means every other client too.
pub fn enabled() -> bool {
    matches!(get_status(), RerankerStatus::Loaded)
}

/// True once the status has been resolved, without resolving it. Lets callers
/// (and tests) tell "not loaded" from "not asked yet" without paying the load.
pub fn status_resolved() -> bool {
    RERANKER_STATUS.get().is_some()
}

pub fn is_configured() -> bool {
    if let Ok(p) = std::env::var("CUBA_RERANKER_PATH") {
        return PathBuf::from(p).join("model.onnx").exists();
    }
    std::env::var("HOME")
        .ok()
        .map(|h| {
            PathBuf::from(h)
                .join(".cache/cuba-memorys/reranker/model.onnx")
                .exists()
        })
        .unwrap_or(false)
}

const WARMUP_CANDIDATES: usize = 50;
const WARMUP_PASSAGE_CHARS: usize = 240;

pub async fn warm_up() -> bool {
    if !tokio::task::spawn_blocking(enabled).await.unwrap_or(false) {
        return false;
    }
    let passage = "the retrieval pipeline fuses lexical and vector signals before the \
                   cross-encoder rescores the surviving candidates in a single batch "
        .repeat(WARMUP_PASSAGE_CHARS / 100 + 1);
    let passages: Vec<&str> = std::iter::repeat_n(passage.as_str(), WARMUP_CANDIDATES).collect();
    rerank("which passage answers the question best", &passages)
        .await
        .is_ok()
}

fn get_status() -> &'static RerankerStatus {
    RERANKER_STATUS.get_or_init(|| {
        let path = RERANKER_PATH.get_or_init(|| {
            if let Some(p) = std::env::var("CUBA_RERANKER_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|p| p.exists())
            {
                return Some(p);
            }
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache/cuba-memorys/reranker"))
                .filter(|p| p.join("model.onnx").exists())
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

fn init_session(model_dir: &std::path::Path) -> Result<()> {
    let candidates = ["model_quantized.onnx", "model.onnx"];
    let model_file = candidates
        .iter()
        .map(|n| model_dir.join(n))
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow::anyhow!("no model.onnx / model_quantized.onnx found in {model_dir:?}")
        })?;

    let builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_intra_threads(intra_threads())
        .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?;
    let session = crate::gpu::configure(builder, crate::gpu::Workload::Reranker)?
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
        strategy: if fixed_shape() {
            tokenizers::PaddingStrategy::Fixed(RERANK_MAX_TOKENS)
        } else {
            tokenizers::PaddingStrategy::BatchLongest
        },
        ..Default::default()
    };
    tokenizer.with_padding(Some(padding));
    RERANKER_TOKENIZER
        .set(tokenizer)
        .map_err(|_| anyhow::anyhow!("tokenizer already initialized"))?;
    Ok(())
}

pub async fn rerank(query: &str, candidates: &[&str]) -> Result<Vec<(usize, f64)>> {
    // Nothing to rank: answer before touching the model. This used to sit *after*
    // the `enabled()` check, so reranking an empty list paid a multi-GB lazy load
    // in full and then returned the empty vector anyway.
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let query_owned = query.to_string();
    let candidates_owned: Vec<String> = candidates.iter().map(|c| c.to_string()).collect();

    let _permit = semaphore()
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("reranker semaphore closed"))?;

    // `enabled()` may load the model, so it belongs on the blocking pool with the
    // inference it gates — not on an async worker where it would stall every
    // other client the daemon is serving.
    let n = candidates.len();
    let scored = tokio::task::spawn_blocking(move || {
        if !enabled() {
            return Ok(None);
        }
        score_pairs(&query_owned, &candidates_owned).map(Some)
    })
    .await
    .context("reranker task panicked")??;

    let Some(scored) = scored else {
        return Ok(identity_pairs(n));
    };

    let mut indexed: Vec<(usize, f64)> = scored.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed)
}

const RERANK_CHUNK: usize = 16;
const RERANK_MAX_TOKENS: usize = 512;

/// How many candidates cross the encoder in one forward pass.
///
/// Activation memory scales with `chunk × tokens`, and under `fixed_shape` every
/// batch is padded to the full `RERANK_MAX_TOKENS`, which makes this the single
/// biggest lever on the GPU arena this session reserves. 16 keeps the device
/// busy; halving it roughly halves peak activations for the price of more
/// passes over the same total work. Tunable without a rebuild so the tradeoff
/// can be measured against `nvidia-smi` on the machine that has to live with it.
fn rerank_chunk() -> usize {
    std::env::var("CUBA_RERANK_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(RERANK_CHUNK)
}

/// Whether every batch is padded to `RERANK_MAX_TOKENS`.
///
/// Fixed shapes keep the GPU from re-planning per batch, but they also mean a
/// one-line note costs the same as a full 512-token passage — which is only
/// worth it when the device is actually running the model. Keyed off the real
/// placement rather than the compile-time feature, so pointing the reranker at
/// the CPU does not silently leave it padding everything to 512.
pub fn fixed_shape() -> bool {
    match std::env::var("CUBA_RERANK_FIXED_SHAPE").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => false,
        Ok(_) => true,
        Err(_) => crate::gpu::wants_gpu(crate::gpu::Workload::Reranker),
    }
}

/// Whether to group similar-length candidates into the same batch.
///
/// A batch pads to its longest member, and every padded position is still a full
/// column of attention and GEMM work. Real candidate sets mix one-line notes with
/// long postmortems, so an unsorted batch can spend most of its compute on
/// padding. Sorting by length first cuts that away.
///
/// Scores are unaffected: the attention mask already zeroes padded positions, so
/// each (query, passage) pair is scored on exactly its own tokens regardless of
/// who it shares a batch with. Pointless under `fixed_shape`, where every batch
/// is padded to `RERANK_MAX_TOKENS` by construction.
fn length_bucketing() -> bool {
    match std::env::var("CUBA_RERANK_LENGTH_BUCKETING").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => false,
        Ok(_) => true,
        Err(_) => !fixed_shape(),
    }
}

fn score_pairs(query: &str, candidates: &[String]) -> Result<Vec<f64>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let session_lock = RERANKER_SESSION
        .get()
        .context("reranker session not initialized")?;
    let mut session = session_lock
        .lock()
        .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
    let tokenizer = RERANKER_TOKENIZER
        .get()
        .context("reranker tokenizer not initialized")?;

    // Longest first: the biggest batch runs while the machine is coldest, and a
    // ragged final chunk carries the shortest inputs instead of the longest.
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    if length_bucketing() {
        order.sort_by_key(|&i| std::cmp::Reverse(candidates[i].len()));
    }

    let chunk_size = rerank_chunk();
    let mut scores = vec![0.0_f64; candidates.len()];
    for chunk in order.chunks(chunk_size) {
        let texts: Vec<String> = chunk.iter().map(|&i| candidates[i].clone()).collect();

        let chunk_scores = if !fixed_shape() || texts.len() == chunk_size {
            score_chunk(&mut session, tokenizer, query, &texts)?
        } else {
            let mut padded = texts.clone();
            padded.resize(chunk_size, String::new());
            let mut s = score_chunk(&mut session, tokenizer, query, &padded)?;
            s.truncate(texts.len());
            s
        };

        // Back to the caller's order — `rerank` pairs these with their indices.
        for (pos, &original) in chunk.iter().enumerate() {
            scores[original] = chunk_scores[pos];
        }
    }
    Ok(scores)
}

fn score_chunk(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    query: &str,
    candidates: &[String],
) -> Result<Vec<f64>> {
    let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query, c.as_str())).collect();
    let encodings = tokenizer
        .encode_batch(pairs, true)
        .map_err(|e| anyhow::anyhow!("encode batch: {e}"))?;

    let batch = encodings.len();
    let seq = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .context("empty batch")?;

    let mut ids = Vec::with_capacity(batch * seq);
    let mut mask = Vec::with_capacity(batch * seq);
    let mut types = Vec::with_capacity(batch * seq);
    for e in &encodings {
        let (e_ids, e_mask, e_types) = (e.get_ids(), e.get_attention_mask(), e.get_type_ids());
        for i in 0..seq {
            ids.push(*e_ids.get(i).unwrap_or(&0) as i64);
            mask.push(*e_mask.get(i).unwrap_or(&0) as i64);
            types.push(*e_types.get(i).unwrap_or(&0) as i64);
        }
    }

    let shape = vec![batch as i64, seq as i64];
    let input_ids_t =
        ort::value::Tensor::from_array((shape.clone(), ids)).context("input_ids tensor")?;
    let attn_t =
        ort::value::Tensor::from_array((shape.clone(), mask)).context("attention_mask tensor")?;

    let wants_type_ids = session
        .inputs()
        .iter()
        .any(|i| i.name() == "token_type_ids");

    let outputs = if wants_type_ids {
        let type_t =
            ort::value::Tensor::from_array((shape, types)).context("token_type_ids tensor")?;
        session.run(ort::inputs! {
            "input_ids" => input_ids_t,
            "attention_mask" => attn_t,
            "token_type_ids" => type_t,
        })
    } else {
        session.run(ort::inputs! {
            "input_ids" => input_ids_t,
            "attention_mask" => attn_t,
        })
    }
    .map_err(|e| anyhow::anyhow!("inference: {e}"))?;

    if outputs.len() == 0 {
        anyhow::bail!("reranker returned no outputs");
    }

    let (out_shape, data): (Vec<i64>, Vec<f32>) = match outputs[0].try_extract_tensor::<f32>() {
        Ok((s, d)) => (s.to_vec(), d.to_vec()),
        Err(_) => {
            let (s, d) = outputs[0]
                .try_extract_tensor::<half::f16>()
                .map_err(|e| anyhow::anyhow!("extract logits (f32 and f16): {e}"))?;
            (s.to_vec(), d.iter().map(|h| h.to_f32()).collect())
        }
    };

    let num_labels = out_shape.last().copied().unwrap_or(1).max(1) as usize;
    if data.len() < batch * num_labels {
        anyhow::bail!(
            "reranker returned {} values, expected {}×{}",
            data.len(),
            batch,
            num_labels
        );
    }

    let mut scores = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &data[b * num_labels..(b + 1) * num_labels];
        let logit = match num_labels {
            1 => row[0],
            2 => row[1] - row[0],
            _ => row[0],
        };
        scores.push(1.0_f64 / (1.0 + (-logit as f64).exp()));
    }
    Ok(scores)
}

fn identity_pairs(n: usize) -> Vec<(usize, f64)> {
    (0..n).map(|i| (i, (n - i) as f64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolves the status against a directory that exists but holds no model,
    /// so the fallback path is exercised without reading the real multi-GB one.
    /// If something already resolved it, that wins — `OnceLock` is per process.
    fn force_fallback_if_unresolved() {
        if !status_resolved() {
            // SAFETY: single-threaded setup before the status is first read; the
            // only reader of this variable is `get_status`, gated by a OnceLock.
            unsafe { std::env::set_var("CUBA_RERANKER_PATH", std::env::temp_dir()) };
        }
    }

    #[tokio::test]
    async fn identity_when_disabled() {
        force_fallback_if_unresolved();
        if enabled() {
            eprintln!("SKIP: reranker already loaded; identity path not exercised");
            return;
        }
        let pairs = rerank("anything", &["a", "b", "c"]).await.unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, 0);
        assert!(pairs[0].1 > pairs[1].1);
    }

    #[test]
    fn identity_pairs_descending() {
        let pairs = identity_pairs(5);
        for win in pairs.windows(2) {
            assert!(win[0].1 > win[1].1);
        }
    }

    /// The regression that hung this suite: reranking nothing used to run the
    /// lazy model load to completion before returning the empty vector.
    #[tokio::test]
    async fn empty_candidates_returns_empty_without_loading_the_model() {
        let resolved_before = status_resolved();
        let started = std::time::Instant::now();

        let pairs = rerank("q", &[]).await.unwrap();

        assert!(pairs.is_empty());
        assert_eq!(
            status_resolved(),
            resolved_before,
            "an empty rerank must not resolve — let alone load — the model"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "empty rerank took {:?}; it should not touch the model at all",
            started.elapsed()
        );
    }

    #[test]
    fn intra_threads_is_configurable_and_never_zero() {
        // SAFETY: no other thread reads this variable during the test.
        unsafe { std::env::set_var("CUBA_RERANK_INTRA_THREADS", "6") };
        assert_eq!(intra_threads(), 6);

        unsafe { std::env::set_var("CUBA_RERANK_INTRA_THREADS", "0") };
        assert!(intra_threads() >= 1, "0 must fall through to the default");

        unsafe { std::env::remove_var("CUBA_RERANK_INTRA_THREADS") };
        let auto = intra_threads();
        assert!((1..=8).contains(&auto), "auto value out of range: {auto}");
    }
}
