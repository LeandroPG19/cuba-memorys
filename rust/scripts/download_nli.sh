#!/bin/bash
# Download the multilingual NLI model that decides entailment for `cuba_faro mode=verify`.
#
# Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
#        100 languages, 87.1% on XNLI, MIT. ~1.1 GB (fp32).
#
# Why fp32 and not the 323 MB int8 export in the same repo:
#
#   The quantized checkpoint is not just less accurate — it is wrong in the one
#   direction this whole subsystem exists to prevent. Measured on identical pairs:
#
#     evidence "…the reranker is disabled by default"
#     claim    "the reranker is enabled by default"
#       int8 → SUPPORTS (0.62)   ← confirms a false claim
#       fp32 → CONTRADICTS (0.995)
#
#     evidence "cuba-memorys …escrito en Rust…"
#     claim    "cuba-memorys está escrito en Java"
#       int8 → unrelated (0.41)  ← lets the false claim through
#       fp32 → CONTRADICTS (0.999)
#
#   DeBERTa-v3's disentangled attention does not survive int8. And the quantization
#   bought nothing: 48 ms per verdict quantized, 53 ms at full precision. It was
#   paying in accuracy for a speed-up that does not exist on CPU.
#
# Usage:
#   ./scripts/download_nli.sh [target_dir]
#
# Default target: ~/.cache/cuba-memorys/models-nli/

set -euo pipefail

TARGET_DIR="${1:-$HOME/.cache/cuba-memorys/models-nli}"
REPO="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$TARGET_DIR"

echo "Downloading mDeBERTa-v3-base-xnli (fp32) to ${TARGET_DIR}..."
echo "  ~1.1 GB — this is the full-precision model. See the header for why."
echo ""

# path-in-repo : local-name
FILES=(
    "onnx/model.onnx:model.onnx"
    "tokenizer.json:tokenizer.json"
    "config.json:config.json"
    "tokenizer_config.json:tokenizer_config.json"
)

for entry in "${FILES[@]}"; do
    remote="${entry%%:*}"
    local="${entry##*:}"
    dest="${TARGET_DIR}/${local}"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  already exists: ${local}, skipping"
        continue
    fi

    echo "  downloading ${local}..."
    # To a temp file first. A half-downloaded model.onnx that `ls` reports as present
    # is worse than no model at all: the server would load it, fail deep inside ort,
    # and the only symptom would be verdicts quietly reverting to the LLM.
    if ! curl -sSL --fail --retry 3 -o "${dest}.part" "${BASE_URL}/${remote}"; then
        echo "ERROR: failed to download ${remote}" >&2
        rm -f "${dest}.part"
        exit 1
    fi
    mv "${dest}.part" "$dest"
    echo "  done: ${local} ($(du -h "$dest" | cut -f1))"
done

# The int8 export lives under the same name in some mirrors. If a previous run left
# one here, it would win nothing and lose accuracy — say so rather than let it sit.
if [ -f "${TARGET_DIR}/model_quantized.onnx" ] && [ -f "${TARGET_DIR}/model.onnx" ]; then
    echo ""
    echo "  note: model_quantized.onnx is also present and will be IGNORED (fp32 wins)."
    echo "        You can delete it: rm ${TARGET_DIR}/model_quantized.onnx"
fi

echo ""
echo "NLI ready. cuba-memorys finds it here automatically; to move it, set:"
echo "   export CUBA_NLI_PATH=\"${TARGET_DIR}\""
echo ""
echo "Verify with:  cuba-memorys doctor"
