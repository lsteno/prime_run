#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${1:-$ROOT_DIR/configs/rlm_rlvr/ablation_rank_lr/manifest.csv}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-999}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/rlm_rank_lr_ablation_launcher_logs}"
UPLOAD_ADAPTERS="${UPLOAD_ADAPTERS:-1}"
HF_LORA_REPO_PREFIX="${HF_LORA_REPO_PREFIX:-lsteno/Qwen3-4B-Instruct-2507-RLM-RL-depth1}"

mkdir -p "$LOG_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

echo "Using manifest: $MANIFEST"
echo "Index range: $START_INDEX..$END_INDEX"
echo "Dry run: $DRY_RUN"
echo "Upload adapters after successful runs: $UPLOAD_ADAPTERS"
if [[ "$UPLOAD_ADAPTERS" == "1" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be set when UPLOAD_ADAPTERS=1" >&2
  exit 1
fi

python3 - "$MANIFEST" "$START_INDEX" "$END_INDEX" <<'PY' | while IFS=$'\t' read -r index run_id rank alpha lr max_steps max_async_level max_off_policy_steps hard_cooldown_steps seed batch_size rollouts_per_example train_worker_count eval_worker_count rollout_timeout_seconds repl_timeout_seconds max_total_subcalls max_batched_subcalls experiment_depth runtime_max_depth prompt_variant config_path output_dir wandb_project wandb_name base_config status notes; do
import csv
import sys

manifest = sys.argv[1]
start_index = int(sys.argv[2])
end_index = int(sys.argv[3])
fields = [
    "index",
    "run_id",
    "rank",
    "alpha",
    "lr",
    "max_steps",
    "max_async_level",
    "max_off_policy_steps",
    "hard_cooldown_steps",
    "seed",
    "batch_size",
    "rollouts_per_example",
    "train_worker_count",
    "eval_worker_count",
    "rollout_timeout_seconds",
    "repl_timeout_seconds",
    "max_total_subcalls",
    "max_batched_subcalls",
    "experiment_depth",
    "runtime_max_depth",
    "prompt_variant",
    "config_path",
    "output_dir",
    "wandb_project",
    "wandb_name",
    "base_config",
    "status",
    "notes",
]
with open(manifest, newline="") as handle:
    for row in csv.DictReader(handle):
        index = int(row["index"])
        if start_index <= index <= end_index:
            if not row.get("hard_cooldown_steps"):
                row["hard_cooldown_steps"] = "5"
            print("\t".join(row.get(field, "") for field in fields), flush=True)
PY
  if [[ -z "$index" ]]; then
    continue
  fi
  if (( index < START_INDEX || index > END_INDEX )); then
    continue
  fi
  if [[ "$status" != "pending" ]]; then
    echo "Skipping $run_id because manifest status is '$status'"
    continue
  fi

  config_abs="$ROOT_DIR/$config_path"
  if [[ ! -f "$config_abs" ]]; then
    echo "Missing config for $run_id: $config_abs" >&2
    exit 1
  fi

  echo "[$(timestamp)] Starting $run_id (rank=$rank alpha=$alpha lr=$lr steps=$max_steps depth=$experiment_depth prompt=$prompt_variant runtime_max_depth=$runtime_max_depth async=$max_async_level off_policy_steps=$max_off_policy_steps hard_cooldown_steps=$hard_cooldown_steps train_workers=$train_worker_count rollout_timeout=${rollout_timeout_seconds}s repl_timeout=${repl_timeout_seconds}s subcalls=${max_total_subcalls}/${max_batched_subcalls})"
  echo "Config: $config_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "(cd prime-rl && uv run rl @ ../$config_path)"
    continue
  fi

  (
    cd "$ROOT_DIR/prime-rl"
    UV_NO_SYNC=1 uv run --no-sync rl @ "../$config_path"
  ) 2>&1 | tee "$LOG_DIR/${run_id}.log"
  if [[ "$UPLOAD_ADAPTERS" == "1" ]]; then
    echo "[$(timestamp)] Uploading LoRA adapter for $run_id"
    (
      cd "$ROOT_DIR"
      "$ROOT_DIR/prime-rl/.venv/bin/python" scripts/rlm_rlvr/upload_lora_adapter.py \
        --output-dir "$ROOT_DIR/$output_dir" \
        --run-id "$run_id" \
        --repo-prefix "$HF_LORA_REPO_PREFIX"
    ) 2>&1 | tee -a "$LOG_DIR/${run_id}.log"
    echo "[$(timestamp)] Uploaded LoRA adapter for $run_id"
  fi
  echo "[$(timestamp)] Finished $run_id"
done
