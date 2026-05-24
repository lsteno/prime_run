#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-../configs/rlm_sft/local_8xrtx6000ada_48gb_qwen3_4b_instruct_curated_v2.toml}"
OUTPUT_DIR="${2:-../outputs/rlm-rlvr-sft-curated-v2-qwen3-4b-instruct-8xrtx6000ada}"
HF_MODEL_REPO_ID="${3:-${HF_MODEL_REPO_ID:-lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
UV_BIN="${UV_BIN:-uv}"

"${UV_BIN}" run --extra flash-attn --extra flash-attn-3 torchrun \
  --local-ranks-filter 0 \
  --nproc-per-node "${NPROC_PER_NODE}" \
  src/prime_rl/trainer/sft/train.py \
  @ "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}"

"${UV_BIN}" run python ../scripts/rlm_sft/upload_latest_weights.py \
  --output-dir "${OUTPUT_DIR}" \
  --repo-id "${HF_MODEL_REPO_ID}"
