#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PATH="$HOME/.local/bin:$PATH"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi
MANIFEST="${1:-$ROOT_DIR/configs/rlm_rlvr/ablation_rank_lr/manifest.csv}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-999}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/rlm_rank_lr_ablation_launcher_logs}"
UPLOAD_ADAPTERS="${UPLOAD_ADAPTERS:-1}"
HF_LORA_REPO_PREFIX="${HF_LORA_REPO_PREFIX:-lsteno/qwen3-rlm-depth1}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
TRAIN_FINISH_GRACE_SECONDS="${TRAIN_FINISH_GRACE_SECONDS:-120}"
TRAIN_FINISH_POLL_SECONDS="${TRAIN_FINISH_POLL_SECONDS:-15}"
TRAIN_FINISH_TERM_GRACE_SECONDS="${TRAIN_FINISH_TERM_GRACE_SECONDS:-30}"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

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
if [[ -z "$UV_BIN" ]]; then
  echo "uv is not on PATH and $HOME/.local/bin/uv was not found" >&2
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

  run_log="$LOG_DIR/${run_id}.log"
  : > "$run_log"
  (
    tail -n +1 -F "$run_log" &
    tail_pid=$!
    trap 'kill "$tail_pid" 2>/dev/null || true' EXIT

    cd "$ROOT_DIR/prime-rl"
    setsid env UV_NO_SYNC=1 "$UV_BIN" run --no-sync rl @ "../$config_path" >> "$run_log" 2>&1 &
    rl_pid=$!
    rl_pgid="$(ps -o pgid= -p "$rl_pid" | tr -d '[:space:]')"
    echo "[$(timestamp)] Started training process pid=$rl_pid pgid=${rl_pgid:-unknown}" >> "$run_log"

    completed_by_log=0
    completed_by_files=0
    final_adapter_dir="$ROOT_DIR/$output_dir/run_default/broadcasts/step_$max_steps"
    while kill -0 "$rl_pid" 2>/dev/null; do
      if grep -q "RL trainer finished!" "$run_log"; then
        completed_by_log=1
      fi
      if [[ -f "$final_adapter_dir/STABLE" && -f "$final_adapter_dir/adapter_config.json" && -f "$final_adapter_dir/adapter_model.safetensors" ]]; then
        completed_by_files=1
      fi
      if [[ "$completed_by_log" == "1" && "$completed_by_files" == "1" ]]; then
        echo "[$(timestamp)] Final trainer checkpoint and LoRA adapter are complete for $run_id; waiting ${TRAIN_FINISH_GRACE_SECONDS}s for Prime-RL cleanup" >> "$run_log"
        sleep "$TRAIN_FINISH_GRACE_SECONDS"
        if kill -0 "$rl_pid" 2>/dev/null; then
          echo "[$(timestamp)] Prime-RL process still alive after completed training; terminating stale process group ${rl_pgid:-$rl_pid}" >> "$run_log"
          if [[ -n "${rl_pgid:-}" ]]; then
            kill -TERM "-$rl_pgid" 2>/dev/null || true
          else
            kill -TERM "$rl_pid" 2>/dev/null || true
          fi
          sleep "$TRAIN_FINISH_TERM_GRACE_SECONDS"
          if kill -0 "$rl_pid" 2>/dev/null; then
            echo "[$(timestamp)] Stale process group did not exit after TERM; sending KILL" >> "$run_log"
            if [[ -n "${rl_pgid:-}" ]]; then
              kill -KILL "-$rl_pgid" 2>/dev/null || true
            else
              kill -KILL "$rl_pid" 2>/dev/null || true
            fi
          fi
        fi
        break
      fi
      sleep "$TRAIN_FINISH_POLL_SECONDS"
    done

    set +e
    wait "$rl_pid"
    rl_status=$?
    set -e
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    if [[ "$completed_by_log" == "1" && "$completed_by_files" == "1" ]]; then
      echo "[$(timestamp)] Treating $run_id as successful: final trainer checkpoint and adapter were verified" >> "$run_log"
      exit 0
    fi
    if [[ "$rl_status" != "0" ]]; then
      echo "[$(timestamp)] Training command failed for $run_id with exit status $rl_status" >> "$run_log"
      exit "$rl_status"
    fi
  )
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
