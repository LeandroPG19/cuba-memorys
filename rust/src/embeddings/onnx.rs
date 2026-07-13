//! ONNX embeddings — multilingual-e5-small via ort crate.
//!
//! Model: intfloat/multilingual-e5-small (384d, 94 languages, MIT, ~113MB quantized).
//! Replaces BGE-small-en-v1.5 (English-only) to support mixed-language knowledge graphs.
//!
//! E5 models require instruction prefixes:
//!   - embed()         → "query: " prefix   (for search queries in cuba_faro)
//!   - embed_passage() → "passage: " prefix  (for content stored in observations)
//!
//! FIX B2: All embedding computations run in spawn_blocking
//! to avoid blocking the Tokio event loop.
//!
//! Model loading: set ONNX_MODEL_PATH env var to the directory containing
//! model_quantized.onnx and tokenizer.json.
//! If not set, falls back to deterministic hash-based embeddings for testing.

use crate::search::cache::TtlLruCache;
use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Default embedding dimension (multilingual-e5-small: 384-d, same as BGE-small).
pub const EMBEDDING_DIM: usize = 384;

/// Runtime embedding dimension. Defaults to [`EMBEDDING_DIM`] but can be overridden
/// with `CUBA_EMBEDDING_DIM` (Phase 5) to run a different model — e.g. bge-m3
/// (1024-d) for stronger Spanish — without a recompile. The column type and the
/// loaded ONNX model must agree with this value.
pub fn embedding_dim() -> usize {
    std::env::var("CUBA_EMBEDDING_DIM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&d| d > 0)
        .unwrap_or(EMBEDDING_DIM)
}

/// Query instruction prefix. E5 uses "query: "; bge-m3 uses none (set
/// `CUBA_QUERY_PREFIX=""`). Kept model-agnostic so a model swap is config-only.
fn query_prefix() -> String {
    std::env::var("CUBA_QUERY_PREFIX").unwrap_or_else(|_| "query: ".to_string())
}

/// Passage instruction prefix. E5 uses "passage: "; bge-m3 uses none.
fn passage_prefix() -> String {
    std::env::var("CUBA_PASSAGE_PREFIX").unwrap_or_else(|_| "passage: ".to_string())
}

/// Pooling strategy over token embeddings: mean (E5, default) or CLS (bge-m3).
/// Set `CUBA_POOLING=cls` for bge-m3.
fn use_cls_pooling() -> bool {
    std::env::var("CUBA_POOLING")
        .map(|v| v.eq_ignore_ascii_case("cls"))
        .unwrap_or(false)
}

/// The model tag when `CUBA_EMBED_MODEL` says nothing. Private on purpose.
///
/// It used to be `pub const CURRENT_MODEL`, and a name like that is an invitation:
/// four call sites took it up, and every one of them was a site that WROTE. The
/// sites that COMPARED used [`current_model()`]. So on a bge-m3 corpus, each new
/// observation was written with a bge-m3 vector and stamped "multilingual-e5-small"
/// — permanently, silently stale to `doctor` and to `zafra reembed`, which could
/// never converge because the thing it re-encoded was re-mislabelled on the next
/// write.
///
/// The vectors were always fine (measured: cross-model cosine on same-topic pairs
/// sits in the same range as within-model, so it is one vector space, not two). Only
/// the label lied. But a label that lies about which model produced a vector is the
/// one piece of metadata you cannot afford to lose — it is what tells you, after the
/// next model change, which rows still need re-encoding.
const DEFAULT_MODEL: &str = "multilingual-e5-small";

/// The model tag to store alongside an embedding. **Always use this** — never a
/// constant. It reads `CUBA_EMBED_MODEL`, which is the only thing that knows what
/// model actually produced the vector you are about to persist.
pub fn current_model() -> String {
    std::env::var("CUBA_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// Global embedding cache (LRU with TTL — FIX B6, V7).
static CACHE: OnceLock<std::sync::Mutex<TtlLruCache<Vec<f32>>>> = OnceLock::new();

/// V0.7: Semaphore limiting concurrent ONNX inference (Little's Law — Mejora 3).
/// Permits = 2 matching `with_intra_threads(2)` in session config.
/// Prevents spawn_blocking threadpool exhaustion under concurrent load.
static ONNX_SEMAPHORE: OnceLock<tokio::sync::Semaphore> = OnceLock::new();

fn get_semaphore() -> &'static tokio::sync::Semaphore {
    ONNX_SEMAPHORE.get_or_init(|| tokio::sync::Semaphore::new(2))
}

/// Whether ONNX model is loaded (checked once at startup).
static MODEL_STATUS: OnceLock<ModelStatus> = OnceLock::new();

/// ONNX session — loaded once, shared across threads.
/// Wrapped in Mutex because Session::run requires &mut self.
static ONNX_SESSION: OnceLock<std::sync::Mutex<Session>> = OnceLock::new();

/// Tokenizer — loaded once, shared across threads.
static TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();

enum ModelStatus {
    /// Real ONNX model loaded successfully.
    Loaded,
    /// No model — using hash-based fallback.
    Fallback,
}

fn get_cache() -> &'static std::sync::Mutex<TtlLruCache<Vec<f32>>> {
    CACHE.get_or_init(|| std::sync::Mutex::new(TtlLruCache::new()))
}

/// Where the ONNX Runtime shared library actually is, or `None`.
///
/// `ort` is built with `load-dynamic`, so it opens libonnxruntime at runtime rather
/// than linking it. When it cannot find the library it does not return an error we
/// can catch and fall back from — the process simply stops answering. Set
/// ONNX_MODEL_PATH without ORT_DYLIB_PATH and the server starts, connects, migrates,
/// announces itself ready, and then hangs on the first embedding, having logged
/// nothing at all. That is the worst failure this codebase can produce, and it is
/// the one a person hits on their very first attempt to enable semantic search.
///
/// So we look for the library ourselves, before handing the problem to `ort`, and
/// degrade to hash embeddings with an explanation if it is not there. Checking the
/// same places `dlopen` would means we do not reject setups that work today: a
/// system-installed onnxruntime with no ORT_DYLIB_PATH is perfectly valid.
fn locate_onnxruntime() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(explicit);
        return p.exists().then_some(p);
    }

    let lib = if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };

    let search: Vec<PathBuf> = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .chain(
            [
                "/usr/lib",
                "/usr/local/lib",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib64",
                "/opt/homebrew/lib",
            ]
            .iter()
            .map(PathBuf::from),
        )
        .collect();

    search.into_iter().map(|d| d.join(lib)).find(|p| p.exists())
}

fn get_model_status() -> &'static ModelStatus {
    MODEL_STATUS.get_or_init(|| {
        // Before anything else: can we even load the runtime? See locate_onnxruntime.
        if std::env::var("ONNX_MODEL_PATH").is_ok_and(|p| !p.is_empty())
            && locate_onnxruntime().is_none()
        {
            tracing::error!(
                "ONNX_MODEL_PATH está definido pero no encuentro la librería de ONNX Runtime. \
                 `ort` la carga dinámicamente: sin ella el servidor NO falla — se cuelga en el \
                 primer embedding, sin decir nada. Degradando al fallback de hash (la búsqueda \
                 léxica y BM25 siguen funcionando; la semántica no).\n\
                 Definí ORT_DYLIB_PATH con la ruta a libonnxruntime.so — \
                 `cuba-memorys doctor` lo diagnostica."
            );
            return ModelStatus::Fallback;
        }

        match std::env::var("ONNX_MODEL_PATH") {
            Ok(path) if !path.is_empty() => {
                let p = PathBuf::from(&path);
                let model_file = if p.is_dir() {
                    p.join("model_quantized.onnx")
                } else {
                    p.clone()
                };
                if model_file.exists() {
                    match init_onnx_session(&model_file, &p) {
                        Ok(()) => {
                            tracing::info!(path = %model_file.display(), "ONNX model loaded successfully");
                            ModelStatus::Loaded
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to load ONNX model — using fallback");
                            ModelStatus::Fallback
                        }
                    }
                } else {
                    tracing::warn!(path = %model_file.display(), "ONNX model file not found — using fallback");
                    ModelStatus::Fallback
                }
            }
            _ => {
                tracing::info!("ONNX_MODEL_PATH not set — using hash-based fallback embeddings");
                ModelStatus::Fallback
            }
        }
    })
}

/// Initialize ONNX session and tokenizer from model directory.
fn init_onnx_session(model_file: &std::path::Path, model_dir: &std::path::Path) -> Result<()> {
    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_intra_threads(2)
        .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
        .commit_from_file(model_file)
        .map_err(|e| anyhow::anyhow!("load model: {e}"))?;

    ONNX_SESSION
        .set(std::sync::Mutex::new(session))
        .map_err(|_| anyhow::anyhow!("ONNX session already initialized"))?;

    // Load tokenizer
    let tokenizer_dir: std::path::PathBuf = if model_dir.is_dir() {
        model_dir.to_path_buf()
    } else {
        model_dir
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    };
    let tokenizer_path = tokenizer_dir.join("tokenizer.json");

    if tokenizer_path.exists() {
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let truncation = tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        };
        tokenizer
            .with_truncation(Some(truncation))
            .map_err(|e| anyhow::anyhow!("failed to set truncation: {e}"))?;

        let padding = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));

        TOKENIZER
            .set(tokenizer)
            .map_err(|_| anyhow::anyhow!("Tokenizer already initialized"))?;
    } else {
        return Err(anyhow::anyhow!(
            "tokenizer.json not found at {}",
            tokenizer_path.display()
        ));
    }

    Ok(())
}

/// Generate embedding for a search QUERY (adds "query: " prefix for E5 models).
///
/// Use this when encoding queries in cuba_faro hybrid search.
///
/// FIX B2: Uses spawn_blocking for CPU-bound ONNX inference.
/// FIX B6: Results cached in LRU with TTL.
///
/// Returns 384-dimensional f32 vector.
pub async fn embed(text: &str) -> Result<Vec<f32>> {
    embed_with_prefix(text, &query_prefix()).await
}

/// Generate embedding for a PASSAGE (adds "passage: " prefix for E5 models).
///
/// Use this when encoding content to store in observations/episodes.
/// E5 models show ~2-5% better retrieval quality when query and passage
/// use their respective prefixes (Wang et al. 2022).
///
/// Returns 384-dimensional f32 vector.
pub async fn embed_passage(text: &str) -> Result<Vec<f32>> {
    embed_with_prefix(text, &passage_prefix()).await
}

/// Embed passage with entity context prepended (Contextual Retrieval).
///
/// Improves retrieval quality by ~20% (Anthropic benchmarks) by prepending
/// `[entity_type:entity_name]` before the content. This gives the embedding
/// model richer context about what the text refers to.
pub async fn embed_passage_contextual(
    text: &str,
    entity_type: &str,
    entity_name: &str,
) -> Result<Vec<f32>> {
    let contextualized = format!("[{}:{}] {}", entity_type, entity_name, text);
    embed_with_prefix(&contextualized, &passage_prefix()).await
}

/// Internal: generate embedding with a given instruction prefix.
///
/// Cache key includes the prefix to avoid query/passage collisions.
async fn embed_with_prefix(text: &str, prefix: &str) -> Result<Vec<f32>> {
    // Cache key includes prefix to distinguish query vs passage embeddings
    let cache_key = format!("{}{}", prefix, text);
    if let Ok(mut cache) = get_cache().lock()
        && let Some(cached) = cache.get(&cache_key)
    {
        return Ok(cached);
    }

    // V0.7 (Mejora 3): Acquire semaphore BEFORE spawn_blocking to limit
    // concurrent ONNX calls (Little's Law). Converts blocking contention
    // into async waiting — zero OS thread cost while queued.
    let _permit = get_semaphore()
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("ONNX semaphore closed"))?;

    // FIX B2: spawn_blocking for CPU-bound work
    let prefixed = format!("{}{}", prefix, text);
    let embedding = tokio::task::spawn_blocking(move || compute_embedding(&prefixed))
        .await
        .context("embedding task panicked")??;

    // Store in cache
    if let Ok(mut cache) = get_cache().lock() {
        cache.put(cache_key, embedding.clone());
    }

    Ok(embedding)
}

/// CPU-bound embedding computation.
///
/// Routes to ONNX inference if model is loaded, otherwise uses
/// deterministic hash-based fallback for testing.
fn compute_embedding(text: &str) -> Result<Vec<f32>> {
    match get_model_status() {
        ModelStatus::Loaded => compute_onnx_embedding(text),
        ModelStatus::Fallback => compute_hash_embedding(text),
    }
}

/// Real ONNX inference via ort crate.
///
/// Pipeline: tokenize → Tensor::from_array → Session::run → mean pooling → L2 normalize.
/// Exact parity with Python embeddings.py implementation.
fn compute_onnx_embedding(text: &str) -> Result<Vec<f32>> {
    let session_lock = ONNX_SESSION.get().context("ONNX session not initialized")?;
    let mut session = session_lock
        .lock()
        .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
    let tokenizer = TOKENIZER.get().context("Tokenizer not initialized")?;

    // 1. Tokenize
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let token_type_ids = encoding.get_type_ids();

    let seq_len = ids.len();

    // 2. Build input tensors via ort::value::Tensor::from_array (batch_size=1)
    let input_ids: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
    let attn_mask: Vec<i64> = attention_mask.iter().map(|&m| m as i64).collect();
    let type_ids: Vec<i64> = token_type_ids.iter().map(|&t| t as i64).collect();

    let shape = vec![1i64, seq_len as i64];

    let input_ids_tensor = ort::value::Tensor::from_array((shape.clone(), input_ids))
        .context("failed to create input_ids tensor")?;

    let attn_mask_tensor = ort::value::Tensor::from_array((shape.clone(), attn_mask.clone()))
        .context("failed to create attention_mask tensor")?;

    // Some models take token_type_ids (BERT-family, e5); others don't
    // (XLM-RoBERTa / bge-m3). Passing an input the model doesn't declare makes
    // ORT fail with "Invalid input name", so gate it on the model's own inputs.
    let wants_token_type = session
        .inputs()
        .iter()
        .any(|i| i.name() == "token_type_ids");

    // 3. Run inference — inputs! returns Vec, not Result
    let outputs = if wants_token_type {
        let type_ids_tensor = ort::value::Tensor::from_array((shape, type_ids))
            .context("failed to create token_type_ids tensor")?;
        session.run(ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attn_mask_tensor,
            "token_type_ids" => type_ids_tensor,
        })
    } else {
        session.run(ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attn_mask_tensor,
        })
    }
    .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;

    // 4. Extract token embeddings (shape: [1, seq_len, 384])
    // ort outputs support index access; check non-empty first to avoid panic
    if outputs.len() == 0 {
        return Err(anyhow::anyhow!("ONNX model returned no output tensors"));
    }
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("extract tensor: {e}"))?;

    // Validate shape: [1, seq_len, dim] where dim is the runtime model dimension
    // (384 for e5-small, 1024 for bge-m3). Shape implements Deref<Target=[i64]>.
    let dim = embedding_dim();
    if shape.len() != 3 || shape[2] as usize != dim {
        return Err(anyhow::anyhow!(
            "unexpected output shape: {:?}, expected [1, {}, {}] — check CUBA_EMBEDDING_DIM matches the loaded model",
            shape,
            seq_len,
            dim
        ));
    }

    // 5. Pool token embeddings → sentence embedding.
    //    - CLS pooling (bge-m3): take token 0's vector.
    //    - Mean pooling (E5, default): attention-masked mean (parity with Python).
    let mut sum_embedding = vec![0.0f32; dim];
    if use_cls_pooling() {
        sum_embedding.copy_from_slice(&data[0..dim]);
    } else {
        let mut sum_mask = 0.0f32;
        for (t, &mask_raw) in attn_mask.iter().enumerate().take(seq_len) {
            let mask_val = mask_raw as f32;
            sum_mask += mask_val;
            let offset = t * dim;
            for d in 0..dim {
                sum_embedding[d] += data[offset + d] * mask_val;
            }
        }
        // Avoid division by zero
        let sum_mask = sum_mask.max(1e-9);
        for v in sum_embedding.iter_mut() {
            *v /= sum_mask;
        }
    }

    // 6. L2 normalize
    let norm: f32 = sum_embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-9);
    for v in sum_embedding.iter_mut() {
        *v /= norm;
    }

    Ok(sum_embedding)
}

/// Deterministic hash-based embedding for testing.
///
/// Produces consistent 384-d vectors from text content.
/// NOT suitable for production — use ONNX model instead.
fn compute_hash_embedding(text: &str) -> Result<Vec<f32>> {
    let dim = embedding_dim();
    let mut embedding = vec![0.0f32; dim];

    let text_lower = text.to_lowercase();
    let words: Vec<&str> = text_lower.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let hash = word
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        let idx = (hash as usize) % dim;
        embedding[idx] += 1.0 / (1.0 + i as f32);
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in embedding.iter_mut() {
            *v /= norm;
        }
    }

    Ok(embedding)
}

/// Compute cosine similarity between two embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Check if ONNX model is available.
pub fn is_model_loaded() -> bool {
    matches!(get_model_status(), ModelStatus::Loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let emb = compute_hash_embedding("hello world").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_embedding_normalized() {
        let emb = compute_hash_embedding("test sentence for normalization").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "should be L2 normalized: got {norm}"
        );
    }

    #[test]
    fn test_embedding_deterministic() {
        let a = compute_hash_embedding("same text").unwrap();
        let b = compute_hash_embedding("same text").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = compute_hash_embedding("hello world").unwrap();
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_different() {
        let a = compute_hash_embedding("rust programming language").unwrap();
        let b = compute_hash_embedding("cooking recipes mediterranean").unwrap();
        let sim = cosine_similarity(&a, &b);
        assert!(sim < 0.8);
    }

    #[test]
    fn test_fallback_mode() {
        // Without ONNX_MODEL_PATH, should be fallback
        let emb = compute_hash_embedding("test").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
    }
}
