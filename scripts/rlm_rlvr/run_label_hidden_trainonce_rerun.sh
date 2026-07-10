#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$ROOT/service_account.json}"
export WANDB_MODE="${WANDB_MODE:-online}"

CONFIG_PATH="${1:-configs/rlm_rlvr/rerun_depth1_h100_lora_label_hidden_trainonce/qwen3_4b_instruct_sanjaya_depth1_llmonly_r004_a008_lr1e-4_s150_8xh100_train-subcalls-root8-beta1-ff05-judgeguard-labelhidden-trainonceeasy-bal35f40v1.toml}"

if [[ "$CONFIG_PATH" = /* ]]; then
  CONFIG_ARG="$CONFIG_PATH"
else
  CONFIG_ARG="../$CONFIG_PATH"
fi

cd "$ROOT/prime-rl"
exec env -u VIRTUAL_ENV UV_NO_SYNC=1 uv run --no-sync rl @ "$CONFIG_ARG"
