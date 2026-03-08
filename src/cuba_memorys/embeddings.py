import sys
from pathlib import Path
from typing import Any

import numpy as np

_session: Any | None = None
_tokenizer: Any | None = None
_init_attempted: bool = False
_available: bool = False

MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_DIR = Path.home() / ".cache" / "cuba-memorys" / "models"
EMBEDDING_DIM = 384

def _ensure_model() -> bool:
    global _session, _tokenizer, _init_attempted, _available

    if _init_attempted:
        return _available

    _init_attempted = True

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError:
        print(
            "[cuba-memorys] onnxruntime not installed — "
            "embeddings disabled, using TF-IDF fallback",
            file=sys.stderr,
        )
        return False

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="model_quantized.onnx",
            cache_dir=str(MODEL_DIR),
        )
        tokenizer_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="tokenizer.json",
            cache_dir=str(MODEL_DIR),
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        _session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        _tokenizer.enable_truncation(max_length=512)
        _tokenizer.enable_padding(length=512)

        _available = True
        print(
            "[cuba-memorys] Embedding model loaded: BGE-small-en-v1.5 (384d)",
            file=sys.stderr,
        )
        return True

    except Exception as e:
        print(
            f"[cuba-memorys] Embedding model load failed: {e}",
            file=sys.stderr,
        )
        return False

def embed(texts: list[str]) -> np.ndarray:
    if not _ensure_model() or _session is None or _tokenizer is None:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    encodings = _tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array(
        [e.attention_mask for e in encodings], dtype=np.int64,
    )
    token_type_ids = np.zeros_like(input_ids)

    outputs = _session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    token_embeddings = outputs[0]
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
    embeddings = sum_embeddings / sum_mask

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    embeddings = embeddings / norms

    return embeddings.astype(np.float32)

def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def is_available() -> bool:
    return _available and _session is not None
