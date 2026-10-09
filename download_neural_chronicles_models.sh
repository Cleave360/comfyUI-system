#!/usr/bin/env zsh

set -euo pipefail

ROOT_DIR="${0:A:h}"
MODELS_DIR="$ROOT_DIR/models"
HF_BIN="$ROOT_DIR/ComfyUI-source/venv/bin/hf"
REPO_ID="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
TEXT_ENCODER_FILE="umt5_xxl_fp8_e4m3fn_scaled.safetensors"

if [[ ! -x "$HF_BIN" ]]; then
  echo "hf CLI not found at $HF_BIN" >&2
  exit 1
fi

mkdir -p "$MODELS_DIR/text_encoders"

if [[ -f "$MODELS_DIR/text_encoders/$TEXT_ENCODER_FILE" ]]; then
  echo "Already present: $MODELS_DIR/text_encoders/$TEXT_ENCODER_FILE"
  exit 0
fi

echo "Downloading $TEXT_ENCODER_FILE into $MODELS_DIR/text_encoders"
"$HF_BIN" download "$REPO_ID" \
  "split_files/text_encoders/$TEXT_ENCODER_FILE" \
  --local-dir "$MODELS_DIR" \
  --local-dir-use-symlinks False

echo "Done. Restart ComfyUI after the download completes."