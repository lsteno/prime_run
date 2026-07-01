#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/prime_run}"
export PATH="$HOME/.local/bin:$PATH"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/final_lora_vs_fullft_pass10_${RUN_STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-rlm-rlvr-evals}"
WANDB_MODE="${WANDB_MODE:-online}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
FULLFT_OUTPUT_DIR="${FULLFT_OUTPUT_DIR:-$ROOT_DIR/outputs/rlm-rlvr-qwen3-4b-depth1-llmonly-fullft-lr5e-6-s150-bal35f40v1-ncclfixed-allrollouts}"
FULLFT_WEIGHTS_PATH="${FULLFT_WEIGHTS_PATH:-$FULLFT_OUTPUT_DIR/weights/step_150}"
FULLFT_HF_REPO="${FULLFT_HF_REPO:-lsteno/Qwen3-4B-Instruct-2507-RLM-RLVR-FullFT-lr5e-6-depth1-v1}"
HF_RESULTS_REPO="${HF_RESULTS_REPO:-lsteno/rlm-rlvr-beeg-depth1-pass10-final-comparison-v1}"
HF_LORA_REPO_PREFIX="${HF_LORA_REPO_PREFIX:-lsteno/qwen3-rlm-depth1}"

MANIFEST="${MANIFEST:-$ROOT_DIR/configs/rlm_rlvr/ablation_rank_lr/manifest.csv}"
SELECTED_MODELS_JSON="${SELECTED_MODELS_JSON:-$WORK_DIR/selected_models.json}"
SELECTED_OVERRIDES="${SELECTED_OVERRIDES:-}"
UPLOAD_FULLFT="${UPLOAD_FULLFT:-1}"
UPLOAD_SELECTED_LORAS="${UPLOAD_SELECTED_LORAS:-1}"
UPLOAD_EVAL_RESULTS="${UPLOAD_EVAL_RESULTS:-1}"

ROLLOUTS_PER_EXAMPLE="${ROLLOUTS_PER_EXAMPLE:-10}"
FULL_EXAMPLES="${FULL_EXAMPLES:-452}"
SHARD_COUNT="${SHARD_COUNT:-8}"
EVAL_WORKERS_PER_SHARD="${EVAL_WORKERS_PER_SHARD:-4}"
MAX_CONCURRENT_PER_SHARD="${MAX_CONCURRENT_PER_SHARD:-8}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-400}"
EVAL_MAX_RETRIES="${EVAL_MAX_RETRIES:-2}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$ROOT_DIR/environments/rlm_rlvr/outputs/evals}"

PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CUDA_VISIBLE_DEVICES_ALL="${CUDA_VISIBLE_DEVICES_ALL:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-8}"
FULLFT_API_SERVER_COUNT="${FULLFT_API_SERVER_COUNT:-8}"
LORA_API_SERVER_COUNT="${LORA_API_SERVER_COUNT:-1}"
GPUS_PER_MODEL="${GPUS_PER_MODEL:-8}"

EVAL_FINISH_GRACE_SECONDS="${EVAL_FINISH_GRACE_SECONDS:-120}"
EVAL_FINISH_TERM_GRACE_SECONDS="${EVAL_FINISH_TERM_GRACE_SECONDS:-30}"
POLL_SECONDS="${POLL_SECONDS:-15}"
DRY_RUN="${DRY_RUN:-0}"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv is not on PATH and $HOME/.local/bin/uv was not found" >&2
  exit 1
fi
if [[ "$UPLOAD_FULLFT" == "1" || "$UPLOAD_SELECTED_LORAS" == "1" || "$UPLOAD_EVAL_RESULTS" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be set for uploads" >&2
    exit 1
  fi
fi

mkdir -p "$WORK_DIR"/{configs,logs,markers,shards,live_traces,results,summary,wandb_stop}

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

json_marker() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  python3 - "$path" "$@" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {"time": __import__("time").time()}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    payload[key] = value
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

process_group_for_pid() {
  local pid="$1"
  ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true
}

stop_process_group() {
  local pid="${1:-}"
  local grace="${2:-30}"
  local label="${3:-process}"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local pgid
  pgid="$(process_group_for_pid "$pid")"
  echo "[$(timestamp)] Stopping $label pid=$pid pgid=${pgid:-unknown}" | tee -a "$WORK_DIR/logs/driver.log"
  if [[ -n "$pgid" ]]; then
    kill -TERM "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  sleep "$grace"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$(timestamp)] $label still alive after TERM; sending KILL" | tee -a "$WORK_DIR/logs/driver.log"
    if [[ -n "$pgid" ]]; then
      kill -KILL "-$pgid" 2>/dev/null || true
    else
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

port_open() {
  python3 - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
}

assert_port_closed() {
  local port="$1"
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if ! python3 - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "Port $port is still open after cleanup" >&2
  return 1
}

model_slug() {
  printf 'rlm_rlvr--%s\n' "${1//\//--}"
}

write_sharded_dataset() {
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" "$FULL_EXAMPLES" "$ROLLOUTS_PER_EXAMPLE" "$SHARD_COUNT" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

work_dir = Path(sys.argv[1])
full_examples = int(sys.argv[2])
rollouts_per_example = int(sys.argv[3])
shard_count = int(sys.argv[4])

dataset = load_dataset("lsteno/BEEG-agents", split="eval")
dataset = dataset.shuffle(seed=42).select(range(min(full_examples, len(dataset))))

schema = pa.schema(
    [
        ("id", pa.large_string()),
        ("prompt", pa.large_string()),
        ("context", pa.large_string()),
        ("answer", pa.large_string()),
        ("acceptable_answers", pa.large_string()),
        ("dataset", pa.large_string()),
        ("task", pa.large_string()),
        ("answer_type", pa.large_string()),
        ("context_token_count", pa.int64()),
        ("metadata", pa.large_string()),
    ]
)

def normalize(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)

def flush(writer, rows):
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()

manifest_rows = []
shard_paths = [work_dir / "shards" / f"eval_shard_{idx:02d}.parquet" for idx in range(shard_count)]
writers = [pq.ParquetWriter(path, schema=schema) for path in shard_paths]
buffers = [[] for _ in range(shard_count)]
counts = [0 for _ in range(shard_count)]
unit_index = 0
for original_index, row in enumerate(dataset):
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    source_id = str(row.get("id") or row.get("example_id") or original_index)
    row_metadata = dict(metadata)
    row_metadata["original_source_id"] = source_id
    row_metadata["original_example_index"] = original_index
    manifest_rows.append(
        {
            "source_id": source_id,
            "original_example_index": original_index,
            "dataset": row.get("dataset"),
            "task": row.get("task"),
            "answer_type": row.get("answer_type"),
            "context_token_count": row.get("context_token_count"),
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }
    )
    base = {
        "prompt": normalize(row.get("prompt") or row.get("question") or row.get("task")),
        "context": normalize(row.get("context") or ""),
        "answer": normalize(row.get("answer")),
        "acceptable_answers": normalize(row.get("acceptable_answers") or row.get("answers") or row.get("answer")),
        "dataset": normalize(row.get("dataset")),
        "task": normalize(row.get("task")),
        "answer_type": normalize(row.get("answer_type")),
        "context_token_count": row.get("context_token_count"),
    }
    for rollout_index in range(rollouts_per_example):
        rollout_metadata = dict(row_metadata)
        rollout_metadata["original_rollout_index"] = rollout_index
        rollout = {
            "id": f"{source_id}__rollout_{rollout_index}",
            **base,
            "metadata": json.dumps(rollout_metadata, ensure_ascii=False),
        }
        shard_idx = unit_index % shard_count
        buffers[shard_idx].append(rollout)
        counts[shard_idx] += 1
        if len(buffers[shard_idx]) >= 16:
            flush(writers[shard_idx], buffers[shard_idx])
        unit_index += 1

try:
    for writer, rows in zip(writers, buffers):
        flush(writer, rows)
finally:
    for writer in writers:
        writer.close()

(work_dir / "manifest.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
    encoding="utf-8",
)
for path, count in zip(shard_paths, counts):
    path.with_suffix(".count").write_text(str(count), encoding="utf-8")
print(json.dumps({"examples": len(dataset), "rollouts_per_example": rollouts_per_example, "total_rollouts": unit_index, "shards": shard_count}))
PY
}

state_columns="used_repl,used_recursion,used_llm_subcalls,used_rlm_subcalls,num_subcalls,num_llm_subcalls,num_rlm_subcalls,max_depth_reached,cost_prompt_tokens,cost_completion_tokens,cost_total_tokens,cost_trainable_tokens,cost_plain_subcall_tokens,total_model_tokens,total_env_tokens,total_prompt_tokens,total_completion_tokens,total_rollout_tokens,judge_score,judge_raw_response,reward_correctness,rlm_trace,rlm_segments,final_answer,prompt_variant,stop_condition,used_forced_finalize_prompt,hit_max_turn_without_final,missing_final,finalized_before_forced_prompt,finalized_on_forced_prompt"

write_env_args_template() {
  cat > "$WORK_DIR/configs/env_args_template.json" <<'JSON'
{
  "data_paths": [],
  "eval_data_paths": [],
  "dataset_id": null,
  "seed": 42,
  "max_examples": -1,
  "max_eval_examples": -1,
  "prompt_variant": "sanjaya_text_depth1_llm_only_v1",
  "include_budget_reminder": false,
  "max_depth": 0,
  "max_iterations": 15,
  "turn_max_tokens": 2048,
  "subcall_max_tokens": 2048,
  "max_prompt_tokens": 47104,
  "subcall_prompt_limit_ratio": 0.85,
  "subcall_budget_enabled": true,
  "max_total_subcalls": 50,
  "max_batched_subcalls": 50,
  "subcall_batch_max_workers": 2,
  "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
  "inference_mode": "local",
  "inference_api_key": "local-vllm",
  "repl_backend": "local",
  "repl_timeout_seconds": 300,
  "repl_fast_timeout_seconds": 30,
  "judge_provider": "vertex",
  "judge_model": "gemini-3-flash-preview",
  "judge_vertex_project_env": "GOOGLE_CLOUD_PROJECT",
  "judge_vertex_location": "global",
  "judge_thinking_level": "medium",
  "efficiency_penalty_mode": "static_per_1k",
  "efficiency_penalty_coef": 0.0,
  "max_turn_penalty_enabled": true,
  "max_turn_penalty": 0.25,
  "missing_final_at_max_turn_zero_reward": true
}
JSON
}

make_env_args() {
  local parquet_path="$1"
  local live_dir="$2"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR/configs/env_args_template.json" "$parquet_path" "$live_dir" "$PORT" <<'PY'
import json
import sys
template, parquet_path, live_dir, port = sys.argv[1:]
payload = json.loads(open(template, encoding="utf-8").read())
payload["data_paths"] = [parquet_path]
payload["eval_data_paths"] = [parquet_path]
payload["live_trace_dir"] = live_dir
payload["inference_base_url"] = f"http://localhost:{port}/v1"
print(json.dumps(payload, separators=(",", ":")))
PY
}

write_inference_config() {
  local label="$1"
  local model_path="$2"
  local enable_lora="$3"
  local max_lora_rank="$4"
  local api_server_count="$5"
  local config_path="$WORK_DIR/configs/inference_${label}.toml"
  cat > "$config_path" <<TOML
output_dir = "$WORK_DIR/inference_${label}"
gpu_memory_utilization = 0.90
enable_prefix_caching = true
enable_lora = $enable_lora
api_server_count = $api_server_count
max_loras = 4
max_cpu_loras = 8
max_lora_rank = $max_lora_rank

[server]
host = "0.0.0.0"
port = $PORT

[model]
name = "$model_path"
max_model_len = $MAX_MODEL_LEN
enforce_eager = false
trust_remote_code = true

[parallel]
tp = $TP_SIZE
dp = $DP_SIZE

[deployment]
type = "single_node"
gpus_per_node = $GPUS_PER_MODEL
TOML
  printf '%s\n' "$config_path"
}

wait_for_server() {
  local deadline=$((SECONDS + 1800))
  until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"$WORK_DIR/logs/models_${PORT}.json" 2>/dev/null; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for vLLM on port ${PORT}" >&2
      return 1
    fi
    sleep 10
  done
}

load_lora_adapter() {
  local adapter_name="$1"
  local adapter_path="$2"
  local payload
  payload="$(python3 - "$adapter_name" "$adapter_path" <<'PY'
import json, sys
print(json.dumps({"lora_name": sys.argv[1], "lora_path": sys.argv[2]}))
PY
)"
  if curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/load_lora_adapter" -H "Content-Type: application/json" -d "$payload" >"$WORK_DIR/logs/load_lora_${adapter_name}.json" 2>"$WORK_DIR/logs/load_lora_${adapter_name}.err"; then
    return 0
  fi
  curl -fsS -X POST "http://127.0.0.1:${PORT}/load_lora_adapter" -H "Content-Type: application/json" -d "$payload" >"$WORK_DIR/logs/load_lora_${adapter_name}.json"
}

start_monitor() {
  local label="$1"
  local expected_rollouts="$2"
  local serve_model_name="$3"
  local started_after="$4"
  local stop_file="$WORK_DIR/wandb_stop/${label}.stop"
  local root="$EVAL_OUTPUT_ROOT/$(model_slug "$serve_model_name")"
  rm -f "$stop_file"
  "$UV_BIN" run --project "$ROOT_DIR/prime-rl" python "$ROOT_DIR/scripts/rlm_eval/monitor_eval_wandb.py" \
    --results-root "$root" \
    --aggregate-files \
    --run-label "$label" \
    --project "$WANDB_PROJECT" \
    --name "beeg-final-pass10-${label}-${RUN_STAMP}" \
    --expected-rollouts "$expected_rollouts" \
    --started-after "$started_after" \
    --stop-file "$stop_file" \
    >"$WORK_DIR/logs/${label}_wandb.log" 2>&1 &
  MONITOR_PID="$!"
}

collect_results() {
  local label="$1"
  local serve_model_name="$2"
  local started_after="$3"
  local root="$EVAL_OUTPUT_ROOT/$(model_slug "$serve_model_name")"
  local dest="$WORK_DIR/results/$label"
  mkdir -p "$dest"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -name results.jsonl -type f -newermt "@${started_after}" -print0 \
    | while IFS= read -r -d '' result_file; do
        run_dir="$(dirname "$result_file")"
        run_name="$(basename "$run_dir")"
        if [[ ! -d "$dest/$run_name" ]]; then
          cp -a "$run_dir" "$dest/$run_name.tmp"
          mv "$dest/$run_name.tmp" "$dest/$run_name"
        else
          cp -a "$run_dir/." "$dest/$run_name/"
        fi
      done
}

run_eval_shard() {
  local label="$1"
  local serve_model_name="$2"
  local shard_idx="$3"
  local parquet_path="$4"
  local count="$5"
  local dp_rank=$((shard_idx % DP_SIZE))
  local live_dir="$WORK_DIR/live_traces/${label}_shard_$(printf '%02d' "$shard_idx")"
  local output_dir="$WORK_DIR/results/${label}/shard_$(printf '%02d' "$shard_idx")"
  mkdir -p "$live_dir" "$output_dir"
  local env_args
  env_args="$(make_env_args "$parquet_path" "$live_dir")"
  LOCAL_VLLM_API_KEY=local-vllm \
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" --with prime prime eval run rlm_rlvr \
    --env-dir-path "$ROOT_DIR/environments" \
    --model "$serve_model_name" \
    --api-base-url "http://localhost:${PORT}/v1" \
    --api-key-var "LOCAL_VLLM_API_KEY" \
    --header "X-data-parallel-rank: ${dp_rank}" \
    --env-args "$env_args" \
    --num-examples "$count" \
    --rollouts-per-example 1 \
    --max-concurrent "$MAX_CONCURRENT_PER_SHARD" \
    --max-tokens 4096 \
    --temperature 0.7 \
    --sampling-args '{"extra_body":{"return_token_ids":true,"top_k":-1,"min_p":0.0}}' \
    --state-columns "$state_columns" \
    --save-results \
    --skip-upload \
    --max-retries "$EVAL_MAX_RETRIES" \
    --abbreviated-summary
}

validate_model_results() {
  local label="$1"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR/results/$label" "$FULL_EXAMPLES" "$ROLLOUTS_PER_EXAMPLE" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected_examples = int(sys.argv[2])
rollouts_per_example = int(sys.argv[3])
rows = []
for path in sorted(root.rglob("results.jsonl")):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
expected_rows = expected_examples * rollouts_per_example
if len(rows) != expected_rows:
    raise SystemExit(f"{root}: got {len(rows)} rows, expected {expected_rows}")

counts = Counter()
for row in rows:
    info = row.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    if not isinstance(info, dict):
        info = {}
    metadata = info.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    source_id = metadata.get("original_source_id") or info.get("source_id") or row.get("id") or row.get("example_id")
    source_id = str(source_id).split("__rollout_", 1)[0]
    counts[source_id] += 1
bad = {source: count for source, count in counts.items() if count != rollouts_per_example}
if len(counts) != expected_examples or bad:
    raise SystemExit(f"{root}: expected {expected_examples} sources x {rollouts_per_example}; bad={dict(list(bad.items())[:10])}")
print(json.dumps({"root": str(root), "rows": len(rows), "sources": len(counts), "rollouts_per_example": rollouts_per_example}))
PY
}

terminate_eval_pids() {
  local label="$1"
  shift
  local pids=("$@")
  local still_alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_alive=1
    fi
  done
  if [[ "$still_alive" == "1" ]]; then
    echo "[$(timestamp)] $label validated; waiting ${EVAL_FINISH_GRACE_SECONDS}s for eval shards to exit" | tee -a "$WORK_DIR/logs/driver.log"
    sleep "$EVAL_FINISH_GRACE_SECONDS"
    for pid in "${pids[@]}"; do
      stop_process_group "$pid" "$EVAL_FINISH_TERM_GRACE_SECONDS" "${label}-eval-shard"
    done
  fi
}

stop_inference() {
  local pid="${INFERENCE_PID:-}"
  if [[ -n "$pid" ]]; then
    stop_process_group "$pid" "$EVAL_FINISH_TERM_GRACE_SECONDS" "vLLM"
  fi
  pkill -f "prime_rl.*inference.*${PORT}" 2>/dev/null || true
  pkill -f "vllm.*${PORT}" 2>/dev/null || true
  sleep 5
  assert_port_closed "$PORT"
  INFERENCE_PID=""
}

run_one_model() {
  local label="$1"
  local kind="$2"
  local serve_model_name="$3"
  local model_path="$4"
  local adapter_path="$5"
  local rank="$6"
  local expected_rollouts=$((FULL_EXAMPLES * ROLLOUTS_PER_EXAMPLE))
  local model_dir="$WORK_DIR/markers/$label"
  mkdir -p "$model_dir"
  if [[ -f "$model_dir/model_process_cleanup_done.json" ]]; then
    echo "[$(timestamp)] Skipping $label because cleanup marker already exists" | tee -a "$WORK_DIR/logs/driver.log"
    return 0
  fi
  json_marker "$model_dir/model_started.json" label="$label" kind="$kind" serve_model="$serve_model_name"

  local enable_lora=false
  local max_lora_rank=1
  local api_server_count="$FULLFT_API_SERVER_COUNT"
  if [[ "$kind" == "lora" ]]; then
    enable_lora=true
    max_lora_rank="$rank"
    api_server_count="$LORA_API_SERVER_COUNT"
  fi
  local inference_config
  inference_config="$(write_inference_config "$label" "$model_path" "$enable_lora" "$max_lora_rank" "$api_server_count")"
  echo "[$(timestamp)] Starting inference for $label ($kind) with config $inference_config" | tee -a "$WORK_DIR/logs/driver.log"
  setsid bash -c '
    set -euo pipefail
    ROOT_DIR="$1"
    CUDA_VISIBLE_DEVICES_ALL="$2"
    UV_BIN="$3"
    inference_config="$4"
    cd "$ROOT_DIR/prime-rl"
    exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ALL" UV_NO_SYNC=1 "$UV_BIN" run --no-sync --project "$ROOT_DIR/prime-rl" inference @ "$inference_config"
  ' _ "$ROOT_DIR" "$CUDA_VISIBLE_DEVICES_ALL" "$UV_BIN" "$inference_config" >"$WORK_DIR/logs/${label}_vllm.log" 2>&1 &
  INFERENCE_PID="$!"
  wait_for_server
  if [[ "$kind" == "lora" ]]; then
    echo "[$(timestamp)] Loading LoRA adapter $adapter_path as $serve_model_name" | tee -a "$WORK_DIR/logs/driver.log"
    load_lora_adapter "$serve_model_name" "$adapter_path"
  fi

  local model_start_epoch
  model_start_epoch="$(date +%s)"
  echo "$model_start_epoch" > "$model_dir/model_start_epoch.txt"
  start_monitor "$label" "$expected_rollouts" "$serve_model_name" "$model_start_epoch"
  local monitor_pid="$MONITOR_PID"
  local shard_pids=()
  for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
    local shard_path="$WORK_DIR/shards/eval_shard_$(printf '%02d' "$shard_idx").parquet"
    local count
    count="$(cat "${shard_path%.parquet}.count")"
    setsid bash "$0" __run_eval_shard "$label" "$serve_model_name" "$shard_idx" "$shard_path" "$count" \
      >"$WORK_DIR/logs/${label}_shard_$(printf '%02d' "$shard_idx").log" 2>&1 &
    shard_pids+=("$!")
  done

  local deadline=$((SECONDS + 86400))
  until collect_results "$label" "$serve_model_name" "$model_start_epoch" && validate_model_results "$label" >"$WORK_DIR/logs/${label}_validation.log" 2>&1; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for $label eval validation" >&2
      cat "$WORK_DIR/logs/${label}_validation.log" >&2 || true
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
  collect_results "$label" "$serve_model_name" "$model_start_epoch"
  json_marker "$model_dir/model_validated.json" label="$label" rows="$expected_rollouts"
  touch "$WORK_DIR/wandb_stop/${label}.stop"
  wait "$monitor_pid" || true
  terminate_eval_pids "$label" "${shard_pids[@]}"
  json_marker "$model_dir/model_summary_done.json" label="$label"
  stop_inference
  json_marker "$model_dir/model_process_cleanup_done.json" label="$label"
  echo "[$(timestamp)] Completed and cleaned up $label" | tee -a "$WORK_DIR/logs/driver.log"
}

if [[ "${1:-}" == "__run_eval_shard" ]]; then
  shift
  run_eval_shard "$@"
  exit $?
fi

cleanup() {
  touch "$WORK_DIR"/wandb_stop/*.stop 2>/dev/null || true
  if [[ -n "${INFERENCE_PID:-}" ]]; then
    stop_inference || true
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"
printf 'RUN_STAMP=%s\nWORK_DIR=%s\nBASE_MODEL=%s\nROLLOUTS_PER_EXAMPLE=%s\nSHARD_COUNT=%s\nEVAL_WORKERS_PER_SHARD=%s\nMAX_CONCURRENT_PER_SHARD=%s\n' \
  "$RUN_STAMP" "$WORK_DIR" "$BASE_MODEL" "$ROLLOUTS_PER_EXAMPLE" "$SHARD_COUNT" "$EVAL_WORKERS_PER_SHARD" "$MAX_CONCURRENT_PER_SHARD" \
  > "$WORK_DIR/run_metadata.env"

write_env_args_template
write_sharded_dataset | tee "$WORK_DIR/logs/dataset.log"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete. Base model: $BASE_MODEL"
  exit 0
fi

cat > "$WORK_DIR/model_order.tsv" <<TSV
base_qwen4b	base	$BASE_MODEL	$BASE_MODEL		0
TSV

run_one_model "base_qwen4b" "base" "$BASE_MODEL" "$BASE_MODEL" "" "0"

summary_args=(--out-dir "$WORK_DIR/summary" --dataset-id "lsteno/BEEG-agents" --split "eval" --seed 42 --expected-examples "$FULL_EXAMPLES" --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE")
summary_args+=(--model-result "base_qwen4b=$WORK_DIR/results/base_qwen4b")
"$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/summarize_beeg_multi_model_passk.py" "${summary_args[@]}" | tee "$WORK_DIR/logs/summary.log"

if [[ "$UPLOAD_EVAL_RESULTS" == "1" ]]; then
  "$ROOT_DIR/prime-rl/.venv/bin/python" "$ROOT_DIR/scripts/rlm_eval/upload_eval_results_to_hf.py" \
    --work-dir "$WORK_DIR" \
    --repo-id "$HF_RESULTS_REPO" \
    --success-path "$WORK_DIR/hf_results_upload_success.json" \
    | tee "$WORK_DIR/logs/hf_results_upload.log"
fi

echo "[$(timestamp)] Done. Work dir: $WORK_DIR" | tee -a "$WORK_DIR/logs/driver.log"
