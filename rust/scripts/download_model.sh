#!/bin/bash
# Download multilingual-e5-small ONNX model (quantized) from HuggingFace.
#
# Model: intfloat/multilingual-e5-small (384d, 94 languages, MIT, ~113MB int8)
# Replaces BGE-small-en-v1.5 (English-only) for mixed Spanish/English support.
#
# Usage:
#   ./scripts/download_model.sh [target_dir]
#
# Default target: ~/.cache/cuba-memorys/models/

set -euo pipefail

TARGET_DIR="${1:-$HOME/.cache/cuba-memorys/models}"

# Primary: Teradata's quantized ONNX export (int8, ~113MB)
# Fallback: Xenova's ONNX conversion
REPO="Teradata/multilingual-e5-small"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

FILES=(
    "model_quantized.onnx"
    "tokenizer.json"
)

mkdir -p "$TARGET_DIR"

echo "Downloading multilingual-e5-small (quantized int8) to ${TARGET_DIR}..."

for file in "${FILES[@]}"; do
    dest="${TARGET_DIR}/${file}"
    if [ -f "$dest" ]; then
        echo "  already exists: ${file}, skipping"
    else
        echo "  downloading ${file}..."
        if ! curl -sSL --fail -o "$dest" "${BASE_URL}/${file}"; then
            # Fallback to Xenova repo
            FALLBACK_URL="https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/onnx/${file}"
            echo "  primary failed, trying fallback..."
            curl -sSL --fail -o "$dest" "$FALLBACK_URL" || {
                echo "ERROR: failed to download ${file}" >&2
                rm -f "$dest"
                exit 1
            }
        fi
        echo "  done: ${file} ($(du -h "$dest" | cut -f1))"
    fi
done

echo ""
echo "Model ready. Set env var:"
echo "   export ONNX_MODEL_PATH=\"${TARGET_DIR}\""
echo ""
echo "Also install ONNX Runtime library:"
echo "   # Ubuntu/Debian:"
echo "   wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz"
echo "   tar xzf onnxruntime-linux-x64-1.21.0.tgz"
echo "   export ORT_DYLIB_PATH=\$(pwd)/onnxruntime-linux-x64-1.21.0/lib/libonnxruntime.so"
