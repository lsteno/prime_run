#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-dataset-name> [output-dir] [wandb-run-name]"
  exit 1
fi

DATASET_NAME="$1"
OUTPUT_DIR="${2:-../outputs/rlm-rlvr-sft-local-qwen3-4b}"
RUN_NAME="${3:-rlm-rlvr-sft-local-qwen3-4b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_DIR="${REPO_ROOT}/prime-rl"
CONFIG_PATH="../configs/rlm_sft/local_h100x8_qwen3_4b.toml"

cd "${PRIME_RL_DIR}"

ulimit -n 32000 || true

uv run sft @ "${CONFIG_PATH}" \
  --data.name "${DATASET_NAME}" \
  --val.data.name "${DATASET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --wandb.name "${RUN_NAME}"
