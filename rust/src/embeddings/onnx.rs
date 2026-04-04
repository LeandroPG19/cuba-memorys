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

use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::sync::OnceLock;
use crate::search::cache::TtlLruCache;

/// Embedding dimension (multilingual-e5-small uses 384-d vectors, same as BGE-small).
pub const EMBEDDING_DIM: usize = 384;

/// Global embedding cache (LRU with TTL — FIX B6, V7).
static CACHE: OnceLock<std::sync::Mutex<TtlLruCache<Vec<f32>>>> = OnceLock::new();

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

fn get_model_status() -> &'static ModelStatus {
    MODEL_STATUS.get_or_init(|| {
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

    ONNX_SESSION.set(std::sync::Mutex::new(session))
        .map_err(|_| anyhow::anyhow!("ONNX session already initialized"))?;

    // Load tokenizer
    let tokenizer_dir: std::path::PathBuf = if model_dir.is_dir() {
        model_dir.to_path_buf()
    } else {
        model_dir.parent()
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
        tokenizer.with_truncation(Some(truncation))
            .map_err(|e| anyhow::anyhow!("failed to set truncation: {e}"))?;

        let padding = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));

        TOKENIZER.set(tokenizer)
            .map_err(|_| anyhow::anyhow!("Tokenizer already initialized"))?;
    } else {
        return Err(anyhow::anyhow!("tokenizer.json not found at {}", tokenizer_path.display()));
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
    embed_with_prefix(text, "query: ").await
}

/// Generate embedding for a PASSAGE (adds "passage: " prefix for E5 models).
///
/// Use this when encoding content to store in observations/episodes.
/// E5 models show ~2-5% better retrieval quality when query and passage
/// use their respective prefixes (Wang et al. 2022).
///
/// Returns 384-dimensional f32 vector.
pub async fn embed_passage(text: &str) -> Result<Vec<f32>> {
    embed_with_prefix(text, "passage: ").await
}

/// Internal: generate embedding with a given instruction prefix.
///
/// Cache key includes the prefix to avoid query/passage collisions.
async fn embed_with_prefix(text: &str, prefix: &str) -> Result<Vec<f32>> {
    // Cache key includes prefix to distinguish query vs passage embeddings
    let cache_key = format!("{}{}", prefix, text);
    if let Ok(mut cache) = get_cache().lock()
        && let Some(cached) = cache.get(&cache_key) {
            return Ok(cached);
        }

    // FIX B2: spawn_blocking for CPU-bound work
    let prefixed = format!("{}{}", prefix, text);
    let embedding = tokio::task::spawn_blocking(move || {
        compute_embedding(&prefixed)
    })
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
        ModelStatus::Loaded => {
            compute_onnx_embedding(text)
        }
        ModelStatus::Fallback => {
            compute_hash_embedding(text)
        }
    }
}

/// Real ONNX inference via ort crate.
///
/// Pipeline: tokenize → Tensor::from_array → Session::run → mean pooling → L2 normalize.
/// Exact parity with Python embeddings.py implementation.
fn compute_onnx_embedding(text: &str) -> Result<Vec<f32>> {
    let session_lock = ONNX_SESSION.get()
        .context("ONNX session not initialized")?;
    let mut session = session_lock.lock()
        .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
    let tokenizer = TOKENIZER.get()
        .context("Tokenizer not initialized")?;

    // 1. Tokenize
    let encoding = tokenizer.encode(text, true)
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

    let input_ids_tensor = ort::value::Tensor::from_array(
        (shape.clone(), input_ids)
    ).context("failed to create input_ids tensor")?;

    let attn_mask_tensor = ort::value::Tensor::from_array(
        (shape.clone(), attn_mask.clone())
    ).context("failed to create attention_mask tensor")?;

    let type_ids_tensor = ort::value::Tensor::from_array(
        (shape, type_ids)
    ).context("failed to create token_type_ids tensor")?;

    // 3. Run inference — inputs! returns Vec, not Result
    let inputs = ort::inputs! {
        "input_ids" => input_ids_tensor,
        "attention_mask" => attn_mask_tensor,
        "token_type_ids" => type_ids_tensor,
    };
    let outputs = session.run(inputs)
        .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;

    // 4. Extract token embeddings (shape: [1, seq_len, 384])
    // ort outputs support index access; check non-empty first to avoid panic
    if outputs.len() == 0 {
        return Err(anyhow::anyhow!("ONNX model returned no output tensors"));
    }
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("extract tensor: {e}"))?;

    // Validate shape: [1, seq_len, EMBEDDING_DIM]
    // Shape implements Deref<Target=[i64]>
    if shape.len() != 3 || shape[2] as usize != EMBEDDING_DIM {
        return Err(anyhow::anyhow!(
            "unexpected output shape: {:?}, expected [1, {}, {}]",
            shape, seq_len, EMBEDDING_DIM
        ));
    }

    // 5. Mean pooling with attention mask (parity with Python)
    let mut sum_embedding = vec![0.0f32; EMBEDDING_DIM];
    let mut sum_mask = 0.0f32;

    for (t, &mask_raw) in attn_mask.iter().enumerate().take(seq_len) {
        let mask_val = mask_raw as f32;
        sum_mask += mask_val;
        let offset = t * EMBEDDING_DIM;
        for d in 0..EMBEDDING_DIM {
            sum_embedding[d] += data[offset + d] * mask_val;
        }
    }

    // Avoid division by zero
    let sum_mask = sum_mask.max(1e-9);
    for v in sum_embedding.iter_mut() {
        *v /= sum_mask;
    }

    // 6. L2 normalize
    let norm: f32 = sum_embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
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
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];

    let text_lower = text.to_lowercase();
    let words: Vec<&str> = text_lower.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let hash = word.bytes().fold(0u32, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u32)
        });
        let idx = (hash as usize) % EMBEDDING_DIM;
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
        assert!((norm - 1.0).abs() < 0.01, "should be L2 normalized: got {norm}");
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
