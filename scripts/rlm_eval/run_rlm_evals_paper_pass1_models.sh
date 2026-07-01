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
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/rlm_evals_paper_pass1_${RUN_STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-rlm-rlvr-evals}"
WANDB_MODE="${WANDB_MODE:-online}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$ROOT_DIR/environments/rlm_rlvr/outputs/evals}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
LORA_R4_REPO="${LORA_R4_REPO:-lsteno/qwen3-rlm-depth1-r4-a8-lr1e-4-s150-bal35f40v1-lora}"
LORA_R64_REPO="${LORA_R64_REPO:-lsteno/qwen3-rlm-depth1-r64-a128-lr1e-5-s150-bal35f40v1-lora}"
FULLFT_REPO="${FULLFT_REPO:-lsteno/Qwen3-4B-Instruct-2507-RLM-RLVR-FullFT-lr1e-5-depth1-v1}"
DEPTH2_REPO="${DEPTH2_REPO:-lsteno/Qwen3-4B-Instruct-2507-RLM-RLVR-depth2-recursive-r64-a128-lr1e-5-adapter}"
HF_RESULTS_REPO="${HF_RESULTS_REPO:-lsteno/rlm-evals-paper-pass1-traces-v2}"
HF_FIXED_DATASET_REPO="${HF_FIXED_DATASET_REPO:-lsteno/RLM-Evals-fixed-v2}"

ROLLOUTS_PER_EXAMPLE=1
SHARD_COUNT="${SHARD_COUNT:-16}"
EVAL_WORKERS_PER_SHARD="${EVAL_WORKERS_PER_SHARD:-8}"
MAX_CONCURRENT_PER_SHARD="${MAX_CONCURRENT_PER_SHARD:-16}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-400}"
EVAL_MAX_RETRIES="${EVAL_MAX_RETRIES:-2}"
DYNAMIC_EVAL_WORKERS="${DYNAMIC_EVAL_WORKERS:-32}"
DYNAMIC_EVAL_PER_RANK_CAP="${DYNAMIC_EVAL_PER_RANK_CAP:-4}"
DYNAMIC_EVAL_MAX_ATTEMPTS="${DYNAMIC_EVAL_MAX_ATTEMPTS:-3}"
DYNAMIC_EVAL_WORKER_READY_TIMEOUT_SECONDS="${DYNAMIC_EVAL_WORKER_READY_TIMEOUT_SECONDS:-900}"
DYNAMIC_EVAL_ASSIGNED_START_TIMEOUT_SECONDS="${DYNAMIC_EVAL_ASSIGNED_START_TIMEOUT_SECONDS:-180}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TP_SIZE="${TP_SIZE:-1}"
GPU_COUNT="${GPU_COUNT:-}"
DP_SIZE="${DP_SIZE:-}"
API_SERVER_COUNT_FULL="${API_SERVER_COUNT_FULL:-}"
API_SERVER_COUNT_LORA="${API_SERVER_COUNT_LORA:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
POLL_SECONDS="${POLL_SECONDS:-15}"
MODEL_TIMEOUT_SECONDS="${MODEL_TIMEOUT_SECONDS:-43200}"
FINISH_GRACE_SECONDS="${FINISH_GRACE_SECONDS:-120}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-30}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
UPLOAD_FIXED_DATASET="${UPLOAD_FIXED_DATASET:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" && -f "$ROOT_DIR/service_account.json" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$ROOT_DIR/service_account.json"
fi
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ambient-empire-492113-d7}"
export RLM_LOCAL_INFERENCE_API_KEY="${RLM_LOCAL_INFERENCE_API_KEY:-local-vllm}"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv is missing" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"/{adapters,configs,logs,markers,materialized,results,summary,wandb_stop,hf_bundle}

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

model_slug() {
  printf 'rlm_rlvr--%s\n' "${1//\//--}"
}

json_marker() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  python3 - "$path" "$@" <<'PY'
import json, time, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {"time": time.time()}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    payload[key] = value
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

process_group_for_pid() {
  ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]' || true
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
  echo "[$(timestamp)] stopping $label pid=$pid pgid=${pgid:-unknown}" | tee -a "$WORK_DIR/logs/driver.log"
  if [[ -n "$pgid" ]]; then
    kill -TERM "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  sleep "$grace"
  if kill -0 "$pid" 2>/dev/null; then
    if [[ -n "$pgid" ]]; then
      kill -KILL "-$pgid" 2>/dev/null || true
    else
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

assert_port_closed() {
  local port="$1"
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if ! python3 - "$port" <<'PY'
import socket, sys
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
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
  return 1
}

detect_gpus() {
  if [[ -z "$GPU_COUNT" ]]; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == "0" ]]; then
    GPU_COUNT=1
  fi
  if [[ -z "$DP_SIZE" ]]; then
    DP_SIZE="$GPU_COUNT"
  fi
  if [[ -z "$API_SERVER_COUNT_FULL" ]]; then
    API_SERVER_COUNT_FULL="$DP_SIZE"
  fi
  CUDA_VISIBLE_DEVICES_ALL="${CUDA_VISIBLE_DEVICES_ALL:-$(python3 - "$GPU_COUNT" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)}"
}

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    curl -LsSf https://hf.co/cli/install.sh | bash -s
    export PATH="$HOME/.local/bin:$PATH"
  fi
  hf auth whoami >/dev/null
}

preflight() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN missing" >&2
    exit 1
  fi
  if [[ -z "${WANDB_API_KEY:-}" && "$WANDB_MODE" != "offline" ]]; then
    echo "WANDB_API_KEY missing and WANDB_MODE is not offline" >&2
    exit 1
  fi
  if [[ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS does not point to a file" >&2
    exit 1
  fi
  detect_gpus
  ensure_hf_cli
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - <<'PY'
from huggingface_hub import HfApi
import os
repos = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "lsteno/qwen3-rlm-depth1-r4-a8-lr1e-4-s150-bal35f40v1-lora",
    "lsteno/qwen3-rlm-depth1-r64-a128-lr1e-5-s150-bal35f40v1-lora",
    "lsteno/Qwen3-4B-Instruct-2507-RLM-RLVR-FullFT-lr1e-5-depth1-v1",
    "lsteno/Qwen3-4B-Instruct-2507-RLM-RLVR-depth2-recursive-r64-a128-lr1e-5-adapter",
]
api = HfApi(token=os.environ["HF_TOKEN"])
for repo in repos:
    api.model_info(repo)
api.dataset_info("lsteno/RLM-Evals")
print("HF preflight ok")
PY
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - <<'PY'
import os
from google import genai
client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global")
resp = client.models.generate_content(model="gemini-3-flash-preview", contents="Return OK.")
text = getattr(resp, "text", "") or ""
if not text.strip():
    raise SystemExit("empty Vertex response")
print("Vertex preflight ok")
PY
}

download_adapters() {
  if [[ ! -f "$WORK_DIR/adapters/lora_r4/adapter_config.json" ]]; then
    hf download "$LORA_R4_REPO" --type model --local-dir "$WORK_DIR/adapters/lora_r4" --max-workers 8
  fi
  if [[ ! -f "$WORK_DIR/adapters/lora_r64/adapter_config.json" ]]; then
    hf download "$LORA_R64_REPO" --type model --local-dir "$WORK_DIR/adapters/lora_r64" --max-workers 8
  fi
  if [[ ! -f "$WORK_DIR/adapters/depth2_r64/adapter_config.json" ]]; then
    hf download "$DEPTH2_REPO" --type model --local-dir "$WORK_DIR/adapters/depth2_r64" --max-workers 8
  fi
}

materialize() {
  if [[ -f "$WORK_DIR/markers/materialized.done" ]]; then
    ensure_all_eval_parquet
    upload_fixed_dataset
    return 0
  fi
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/materialize_rlm_evals_paper.py" \
    --out-dir "$WORK_DIR/materialized" \
    --seed 42 \
    --shard-count "$SHARD_COUNT" \
    --browse-sample-size 150 \
    --browse-max-docs 1000 \
    | tee "$WORK_DIR/logs/materialize.log"
  json_marker "$WORK_DIR/markers/materialized.done" total_examples="$(python3 - "$WORK_DIR/materialized/materialized_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["total_examples"])
PY
)"
  ensure_all_eval_parquet
  upload_fixed_dataset
}

upload_fixed_dataset() {
  if [[ "$UPLOAD_FIXED_DATASET" != "1" ]]; then
    return 0
  fi
  if [[ -f "$WORK_DIR/markers/fixed_dataset_uploaded.json" ]]; then
    return 0
  fi
  local bundle="$WORK_DIR/fixed_dataset_hf_bundle"
  rm -rf "$bundle"
  mkdir -p "$bundle/data" "$bundle/meta"
  cp "$WORK_DIR/materialized/all_eval.parquet" "$bundle/data/eval.parquet"
  cp "$WORK_DIR/materialized/sample_manifest.jsonl" "$bundle/meta/sample_manifest.jsonl"
  cp "$WORK_DIR/materialized/materialized_summary.json" "$bundle/meta/materialized_summary.json"
  cat > "$bundle/README.md" <<'MD'
---
license: mit
pretty_name: RLM-Evals Fixed V2
configs:
- config_name: default
  data_files:
  - split: eval
    path: data/eval.parquet
---

# RLM-Evals Fixed V2

Materialized paper-style RLM evaluation tasks for the local RLM harness.

Important fixes:

- BrowseComp+ query, answer, and document fields are de-obfuscated from the upstream benchmark.
- OOLONG-Pairs uses full list-of-pairs gold answers from `mit-oasys/oolong-pairs`, stored as one canonical JSON-list answer per row.
- OOLONG-Pairs should be reported with pair precision/recall/F1, not ordinary binary judge accuracy.

This dataset contains plaintext BrowseComp+ benchmark content and should remain private unless benchmark-release policy is revisited.
MD
  hf repos create "$HF_FIXED_DATASET_REPO" --type dataset --private --exist-ok
  hf upload "$HF_FIXED_DATASET_REPO" "$bundle" --type dataset --private --commit-message "Upload fixed RLM eval materialization"
  json_marker "$WORK_DIR/markers/fixed_dataset_uploaded.json" repo_id="$HF_FIXED_DATASET_REPO"
}

ensure_all_eval_parquet() {
  if [[ -f "$WORK_DIR/materialized/all_eval.parquet" ]]; then
    return 0
  fi
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR/materialized" <<'PY'
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
root = Path(__import__("sys").argv[1])
tables = [pq.read_table(path) for path in sorted((root / "shards").glob("eval_shard_*.parquet"))]
if not tables:
    raise SystemExit(f"No shard parquet files under {root / 'shards'}")
pq.write_table(pa.concat_tables(tables), root / "all_eval.parquet")
PY
}

state_columns="used_repl,used_recursion,used_llm_subcalls,used_rlm_subcalls,num_subcalls,num_llm_subcalls,num_rlm_subcalls,max_depth_reached,cost_prompt_tokens,cost_completion_tokens,cost_total_tokens,cost_trainable_tokens,cost_plain_subcall_tokens,total_model_tokens,total_env_tokens,total_prompt_tokens,total_completion_tokens,total_rollout_tokens,judge_score,judge_raw_response,reward_correctness,oolong_pairs_precision,oolong_pairs_recall,oolong_pairs_f1,oolong_pairs_predicted_count,oolong_pairs_expected_count,rlm_trace,rlm_segments,final_answer,prompt_variant,stop_condition,used_forced_finalize_prompt,hit_max_turn_without_final,missing_final,finalized_before_forced_prompt,finalized_on_forced_prompt"

make_env_args() {
  local parquet_path="$1"
  local live_dir="$2"
  local mode="$3"
  python3 - "$parquet_path" "$live_dir" "$PORT" "$mode" <<'PY'
import json, sys
parquet_path, live_dir, port, mode = sys.argv[1:]
recursive = mode == "depth2"
payload = {
    "data_paths": [parquet_path],
    "eval_data_paths": [parquet_path],
    "dataset_id": None,
    "seed": 42,
    "max_examples": -1,
    "max_eval_examples": -1,
    "prompt_variant": "sanjaya_text_v1" if recursive else "sanjaya_text_depth1_llm_only_v1",
    "include_budget_reminder": False,
    "max_depth": 1 if recursive else 0,
    "recursive_cap_prompt_variant": "sanjaya_text_depth1_llm_only_v1" if recursive else None,
    "recursive_rlm_batch_mode": "serial",
    "max_iterations": 15,
    "turn_max_tokens": 2048,
    "subcall_max_tokens": 2048,
    "max_prompt_tokens": 122880,
    "subcall_prompt_limit_ratio": 0.85,
    "subcall_budget_enabled": True,
    "max_total_subcalls": 50,
    "max_batched_subcalls": 50,
    "subcall_batch_max_workers": 2,
    "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
    "inference_mode": "local",
    "inference_base_url": f"http://localhost:{port}/v1",
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
    "max_turn_penalty_enabled": True,
    "max_turn_penalty": 0.25,
    "missing_final_at_max_turn_zero_reward": True,
    "live_trace_dir": live_dir,
}
payload = {key: value for key, value in payload.items() if value is not None}
print(json.dumps(payload, separators=(",", ":")))
PY
}

write_inference_config() {
  local label="$1"
  local model_path="$2"
  local enable_lora="$3"
  local max_lora_rank="$4"
  local api_server_count="$5"
  local path="$WORK_DIR/configs/inference_${label}.toml"
  cat > "$path" <<TOML
output_dir = "$WORK_DIR/inference_${label}"
gpu_memory_utilization = $VLLM_GPU_MEMORY_UTILIZATION
enable_prefix_caching = true
enable_lora = $enable_lora
api_server_count = $api_server_count
max_loras = 8
max_cpu_loras = 16
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
gpus_per_node = $GPU_COUNT
TOML
  printf '%s\n' "$path"
}

wait_for_server() {
  local deadline=$((SECONDS + 1800))
  until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"$WORK_DIR/logs/models_${PORT}.json" 2>/dev/null; do
    if [[ -n "${INFERENCE_PID:-}" ]] && ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
      echo "Inference process exited while waiting for server on $PORT" >&2
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for inference server on $PORT" >&2
      return 1
    fi
    sleep 10
  done
}

load_lora_adapter() {
  local adapter_name="$1"
  local adapter_path="$2"
  local payload
  local status
  local deadline
  payload="$(python3 - "$adapter_name" "$adapter_path" <<'PY'
import json, sys
print(json.dumps({"lora_name": sys.argv[1], "lora_path": sys.argv[2]}))
PY
)"
  deadline=$((SECONDS + 1200))
  while true; do
    status="$(curl -sS -o "$WORK_DIR/logs/load_lora_${adapter_name}.json.tmp" -w "%{http_code}" \
      -X POST "http://127.0.0.1:${PORT}/v1/load_lora_adapter" \
      -H "Content-Type: application/json" \
      -d "$payload" || true)"
    if [[ "$status" == "200" ]]; then
      mv "$WORK_DIR/logs/load_lora_${adapter_name}.json.tmp" "$WORK_DIR/logs/load_lora_${adapter_name}.json"
      break
    fi
    cp "$WORK_DIR/logs/load_lora_${adapter_name}.json.tmp" "$WORK_DIR/logs/load_lora_${adapter_name}.last_error" 2>/dev/null || true
    if [[ -n "${INFERENCE_PID:-}" ]] && ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
      echo "Inference process exited while loading LoRA ${adapter_name}" >&2
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out loading LoRA ${adapter_name}; last status=${status}" >&2
      cat "$WORK_DIR/logs/load_lora_${adapter_name}.last_error" >&2 || true
      return 1
    fi
    sleep 10
  done
  deadline=$((SECONDS + 300))
  while true; do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"$WORK_DIR/logs/models_${PORT}.json" 2>/dev/null \
      && python3 - "$WORK_DIR/logs/models_${PORT}.json" "$adapter_name" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
ids = {item.get("id") for item in payload.get("data", [])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then
      return 0
    fi
    if (( SECONDS > deadline )); then
      echo "LoRA ${adapter_name} did not appear in /v1/models" >&2
      return 1
    fi
    sleep 5
  done
}

start_inference() {
  local label="$1"
  local kind="$2"
  local model_path="$3"
  local adapter_path="$4"
  local rank="$5"
  local enable_lora=false
  local max_rank=1
  local api_server_count="$API_SERVER_COUNT_FULL"
  if [[ "$kind" == "lora" ]]; then
    enable_lora=true
    max_rank="$rank"
    api_server_count="$API_SERVER_COUNT_LORA"
  fi
  local config
  config="$(write_inference_config "$label" "$model_path" "$enable_lora" "$max_rank" "$api_server_count")"
  setsid bash -c '
    set -euo pipefail
    cd "$1/prime-rl"
    exec env CUDA_VISIBLE_DEVICES="$2" "$3" run --project "$1/prime-rl" inference @ "$4"
  ' _ "$ROOT_DIR" "$CUDA_VISIBLE_DEVICES_ALL" "$UV_BIN" "$config" >"$WORK_DIR/logs/${label}_vllm.log" 2>&1 &
  INFERENCE_PID="$!"
  wait_for_server
  if [[ "$kind" == "lora" ]]; then
    load_lora_adapter "$label" "$adapter_path"
  fi
  json_marker "$WORK_DIR/markers/$label/server_ready.json" label="$label"
}

stop_inference() {
  if [[ -n "${INFERENCE_PID:-}" ]]; then
    stop_process_group "$INFERENCE_PID" "$TERM_GRACE_SECONDS" "inference"
  fi
  pkill -f "prime_rl.*inference.*${PORT}" 2>/dev/null || true
  pkill -f "vllm.*${PORT}" 2>/dev/null || true
  sleep 5
  assert_port_closed "$PORT" || true
  INFERENCE_PID=""
}

run_eval_shard() {
  local label="$1"
  local serve_model_name="$2"
  local shard_idx="$3"
  local parquet_path="$4"
  local count="$5"
  local mode="$6"
  local dp_rank=$((shard_idx % DP_SIZE))
  local live_dir="$WORK_DIR/live_traces/${label}/shard_$(printf '%02d' "$shard_idx")"
  mkdir -p "$live_dir"
  local env_args
  env_args="$(make_env_args "$parquet_path" "$live_dir" "$mode")"
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
    --resume \
    --max-retries "$EVAL_MAX_RETRIES" \
    --abbreviated-summary
}

if [[ "${1:-}" == "__run_eval_shard" ]]; then
  shift
  run_eval_shard "$@"
  exit $?
fi

run_dynamic_eval() {
  local label="$1"
  local serve_model_name="$2"
  local mode="$3"
  local parquet_path="$WORK_DIR/materialized/all_eval.parquet"
  local live_dir="$WORK_DIR/live_traces/${label}/dynamic"
  local output_dir="$WORK_DIR/results/${label}/dynamic"
  local env_args_file="$WORK_DIR/configs/env_args_${label}.json"
  mkdir -p "$live_dir" "$output_dir"
  make_env_args "$parquet_path" "$live_dir" "$mode" > "$env_args_file"
  LOCAL_VLLM_API_KEY=local-vllm \
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/run_dynamic_rlm_eval.py" \
    --repo-root "$ROOT_DIR" \
    --label "$label" \
    --model "$serve_model_name" \
    --api-base-url "http://localhost:${PORT}/v1" \
    --api-key-var "LOCAL_VLLM_API_KEY" \
    --env-args-file "$env_args_file" \
    --result-root "$WORK_DIR/results/$label" \
    --output-dir "$output_dir" \
    --workers "$DYNAMIC_EVAL_WORKERS" \
    --dp-size "$DP_SIZE" \
    --per-rank-cap "$DYNAMIC_EVAL_PER_RANK_CAP" \
    --rollout-timeout-seconds "$EVAL_TIMEOUT_SECONDS" \
    --worker-ready-timeout-seconds "$DYNAMIC_EVAL_WORKER_READY_TIMEOUT_SECONDS" \
    --assigned-start-timeout-seconds "$DYNAMIC_EVAL_ASSIGNED_START_TIMEOUT_SECONDS" \
    --max-attempts "$DYNAMIC_EVAL_MAX_ATTEMPTS" \
    --max-retries "$EVAL_MAX_RETRIES" \
    --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" \
    --state-columns "$state_columns" \
    --sampling-args '{"max_tokens":4096,"temperature":0.7,"extra_body":{"return_token_ids":true,"top_k":-1,"min_p":0.0}}'
}

count_result_rows() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
count = 0
for file in root.rglob("results.jsonl"):
    with file.open(encoding="utf-8") as handle:
        count += sum(1 for line in handle if line.strip())
print(count)
PY
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

validate_model_results() {
  local label="$1"
  local expected="$2"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR/results/$label" "$expected" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []
for file in sorted(root.rglob("results.jsonl")):
    with file.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
if len(rows) != expected:
    raise SystemExit(f"{root}: got {len(rows)} rows, expected {expected}")
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
    meta = info.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    sid = meta.get("original_source_id") or info.get("source_id") or row.get("id") or row.get("example_id")
    counts[str(sid).split("__rollout_", 1)[0]] += 1
bad = {key: value for key, value in counts.items() if value != 1}
if len(counts) != expected or bad:
    raise SystemExit(f"{root}: expected {expected} unique sources, bad={dict(list(bad.items())[:10])}")
print(json.dumps({"rows": len(rows), "sources": len(counts)}))
PY
}

run_one_model() {
  local label="$1"
  local kind="$2"
  local serve_model_name="$3"
  local model_path="$4"
  local adapter_path="$5"
  local rank="$6"
  local mode="$7"
  local expected
  expected="$(python3 - "$WORK_DIR/materialized/materialized_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["total_examples"])
PY
)"
  mkdir -p "$WORK_DIR/markers/$label"
  if [[ -f "$WORK_DIR/markers/$label/model_validated.json" ]]; then
    echo "[$(timestamp)] skipping already validated $label" | tee -a "$WORK_DIR/logs/driver.log"
    return 0
  fi
  json_marker "$WORK_DIR/markers/$label/model_started.json" label="$label" kind="$kind" mode="$mode"
  start_inference "$label" "$kind" "$model_path" "$adapter_path" "$rank"
  run_dynamic_eval "$label" "$serve_model_name" "$mode" >"$WORK_DIR/logs/${label}_dynamic_eval.log" 2>&1
  validate_model_results "$label" "$expected" >"$WORK_DIR/logs/${label}_validation.log" 2>&1
  json_marker "$WORK_DIR/markers/$label/model_validated.json" label="$label" rows="$expected"
  stop_inference
  json_marker "$WORK_DIR/markers/$label/cleanup_done.json" label="$label"
}

write_model_order() {
  cat > "$WORK_DIR/model_order.tsv" <<TSV
base_qwen4b	full	$BASE_MODEL	$BASE_MODEL		0	depth1
lora_r4	lora	lora_r4	$BASE_MODEL	$WORK_DIR/adapters/lora_r4	4	depth1
lora_r64	lora	lora_r64	$BASE_MODEL	$WORK_DIR/adapters/lora_r64	64	depth1
fullft_lr1e5	full	$FULLFT_REPO	$FULLFT_REPO		0	depth1
depth2_r64	lora	depth2_r64	$BASE_MODEL	$WORK_DIR/adapters/depth2_r64	64	depth2
TSV
}

summarize() {
  local expected
  expected="$(python3 - "$WORK_DIR/materialized/materialized_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["total_examples"])
PY
)"
  local args=(--manifest "$WORK_DIR/materialized/sample_manifest.jsonl" --out-dir "$WORK_DIR/summary" --expected-examples "$expected")
  while IFS=$'\t' read -r label kind serve_model model_path adapter_path rank mode; do
    args+=(--model-result "$label=$WORK_DIR/results/$label")
  done < "$WORK_DIR/model_order.tsv"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/summarize_rlm_evals_multi_model.py" "${args[@]}" | tee "$WORK_DIR/logs/summary.log"
}

build_hf_bundle() {
  local bundle="$WORK_DIR/hf_bundle"
  rm -rf "$bundle"
  mkdir -p "$bundle/data" "$bundle/traces" "$bundle/summary" "$bundle/configs"
  cp "$WORK_DIR"/summary/* "$bundle/summary/"
  cp "$WORK_DIR/materialized/sample_manifest.jsonl" "$bundle/sample_manifest.jsonl"
  cp "$WORK_DIR/materialized/materialized_summary.json" "$bundle/materialized_summary.json"
  cp "$WORK_DIR/model_order.tsv" "$bundle/model_order.tsv"
  cp "$WORK_DIR/configs"/*.toml "$bundle/configs/" 2>/dev/null || true
  while IFS=$'\t' read -r label kind serve_model model_path adapter_path rank mode; do
    mkdir -p "$bundle/data/$label" "$bundle/traces/$label"
    find "$WORK_DIR/results/$label" -name results.jsonl -print0 | sort -z | xargs -0 cat > "$bundle/data/$label/results.jsonl"
    if [[ -d "$WORK_DIR/live_traces/$label" ]]; then
      cp -a "$WORK_DIR/live_traces/$label/." "$bundle/traces/$label/"
    fi
  done < "$WORK_DIR/model_order.tsv"
  python3 - "$bundle/README.md" "$WORK_DIR/model_order.tsv" <<'PY'
import sys
from pathlib import Path
readme = Path(sys.argv[1])
rows = [line.rstrip("\n").split("\t") for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
lines = [
    "---",
    "license: mit",
    "pretty_name: RLM-Evals Paper Pass@1 Traces",
    "configs:",
]
for row in rows:
    label = row[0]
    lines.extend([
        f"- config_name: {label}",
        "  data_files:",
        "  - split: eval",
        f"    path: data/{label}/results.jsonl",
    ])
lines.extend([
    "---",
    "",
    "# RLM-Evals Paper-Style Pass@1 Traces",
    "",
    "Each dataset config is one evaluated model. Result rows include saved RLM trace/state columns; live trace JSON files are stored under `traces/<model>/`.",
])
readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

upload_results() {
  if [[ "$UPLOAD_RESULTS" != "1" ]]; then
    return 0
  fi
  build_hf_bundle
  hf repos create "$HF_RESULTS_REPO" --type dataset --private --exist-ok
  hf upload-large-folder "$HF_RESULTS_REPO" "$WORK_DIR/hf_bundle" --type dataset --num-workers 16
  json_marker "$WORK_DIR/hf_results_upload_success.json" repo_id="$HF_RESULTS_REPO"
}

cleanup() {
  if [[ -n "${INFERENCE_PID:-}" ]]; then
    stop_inference || true
  fi
}
trap cleanup EXIT

main() {
  cd "$ROOT_DIR"
  preflight
  download_adapters
  materialize
  write_model_order
  printf 'RUN_STAMP=%s\nWORK_DIR=%s\nGPU_COUNT=%s\nDP_SIZE=%s\nSHARD_COUNT=%s\nMAX_CONCURRENT_PER_SHARD=%s\nEVAL_WORKERS_PER_SHARD=%s\n' \
    "$RUN_STAMP" "$WORK_DIR" "$GPU_COUNT" "$DP_SIZE" "$SHARD_COUNT" "$MAX_CONCURRENT_PER_SHARD" "$EVAL_WORKERS_PER_SHARD" \
    > "$WORK_DIR/run_metadata.env"
  if [[ "$DRY_RUN" == "1" ]]; then
    cat "$WORK_DIR/model_order.tsv"
    exit 0
  fi
  while IFS=$'\t' read -r label kind serve_model model_path adapter_path rank mode; do
    run_one_model "$label" "$kind" "$serve_model" "$model_path" "$adapter_path" "$rank" "$mode"
  done < "$WORK_DIR/model_order.tsv"
  summarize
  upload_results
  echo "[$(timestamp)] done: $WORK_DIR" | tee -a "$WORK_DIR/logs/driver.log"
}

main "$@"
