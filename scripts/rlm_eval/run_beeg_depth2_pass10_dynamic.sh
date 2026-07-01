#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/prime_run}"
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$ROOT_DIR/service_account.json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ambient-empire-492113-d7}"
export RLM_LOCAL_INFERENCE_API_KEY="${RLM_LOCAL_INFERENCE_API_KEY:-local-vllm}"
export HF_HUB_ENABLE_HF_TRANSFER=1

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/beeg_depth2_r64_pass10_${RUN_STAMP}}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DEPTH2_REPO="${DEPTH2_REPO:-lsteno/Qwen3-4B-Instruct-2507-RLM-RLVR-depth2-recursive-r64-a128-lr1e-5-adapter}"
ADAPTER_DIR="${ADAPTER_DIR:-$WORK_DIR/adapters/depth2_r64}"
PORT="${PORT:-8000}"
GPU_COUNT="${GPU_COUNT:-8}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
DYNAMIC_EVAL_WORKERS="${DYNAMIC_EVAL_WORKERS:-64}"
DYNAMIC_EVAL_PER_RANK_CAP="${DYNAMIC_EVAL_PER_RANK_CAP:-8}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-400}"
EVAL_MAX_ATTEMPTS="${EVAL_MAX_ATTEMPTS:-3}"
EVAL_MAX_RETRIES="${EVAL_MAX_RETRIES:-2}"
FULL_EXAMPLES="${FULL_EXAMPLES:-452}"
ROLLOUTS_PER_EXAMPLE="${ROLLOUTS_PER_EXAMPLE:-10}"
CUDA_VISIBLE_DEVICES_ALL="${CUDA_VISIBLE_DEVICES_ALL:-0,1,2,3,4,5,6,7}"
UV_BIN="${UV_BIN:-$(command -v uv)}"

mkdir -p "$WORK_DIR"/{adapters,configs,logs,materialized,results/depth2_r64,live_traces,tmp,summary}

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$WORK_DIR/logs/driver.log"
}

stop_inference() {
  if [[ -n "${INFERENCE_PID:-}" ]] && kill -0 "$INFERENCE_PID" 2>/dev/null; then
    local pgid
    pgid="$(ps -o pgid= -p "$INFERENCE_PID" 2>/dev/null | tr -d '[:space:]' || true)"
    log "Stopping inference pid=$INFERENCE_PID pgid=${pgid:-unknown}"
    if [[ -n "$pgid" ]]; then
      kill -TERM "-$pgid" 2>/dev/null || true
    else
      kill -TERM "$INFERENCE_PID" 2>/dev/null || true
    fi
    sleep 20
    if kill -0 "$INFERENCE_PID" 2>/dev/null; then
      if [[ -n "$pgid" ]]; then
        kill -KILL "-$pgid" 2>/dev/null || true
      else
        kill -KILL "$INFERENCE_PID" 2>/dev/null || true
      fi
    fi
  fi
  pkill -f "prime_rl.*inference.*${PORT}" 2>/dev/null || true
  pkill -f "vllm.*${PORT}" 2>/dev/null || true
}
trap stop_inference EXIT

materialize_beeg() {
  if [[ -f "$WORK_DIR/materialized/all_eval.parquet" ]]; then
    return 0
  fi
  log "Materializing BEEG eval pass@10 rows"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" "$FULL_EXAMPLES" "$ROLLOUTS_PER_EXAMPLE" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

work_dir = Path(sys.argv[1])
full_examples = int(sys.argv[2])
rollouts_per_example = int(sys.argv[3])
out = work_dir / "materialized"
out.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("lsteno/BEEG-agents", split="eval").shuffle(seed=42).select(range(full_examples))
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


def norm(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


rows = []
manifest = []
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
    manifest.append(
        {
            "source_id": source_id,
            "original_example_index": original_index,
            "dataset": row.get("dataset"),
            "task": row.get("task"),
            "answer_type": row.get("answer_type"),
            "context_token_count": row.get("context_token_count"),
        }
    )
    base = {
        "prompt": norm(row.get("prompt") or row.get("question") or row.get("task")),
        "context": norm(row.get("context") or ""),
        "answer": norm(row.get("answer")),
        "acceptable_answers": norm(row.get("acceptable_answers") or row.get("answers") or row.get("answer")),
        "dataset": norm(row.get("dataset")),
        "task": norm(row.get("task")),
        "answer_type": norm(row.get("answer_type")),
        "context_token_count": row.get("context_token_count"),
    }
    row_metadata = dict(metadata)
    row_metadata.update({"original_source_id": source_id, "original_example_index": original_index})
    rows.append({"id": source_id, **base, "metadata": json.dumps(row_metadata, ensure_ascii=False)})

pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "all_eval.parquet")
(out / "manifest.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
    encoding="utf-8",
)
(out / "summary.json").write_text(
    json.dumps(
        {"examples": len(dataset), "rollouts_per_example": rollouts_per_example, "total_rollouts": len(rows) * rollouts_per_example},
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps({"rows": len(rows), "examples": len(dataset), "total_rollouts": len(rows) * rollouts_per_example}))
PY
}

make_env_args() {
  local live_dir="$1"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR/materialized/all_eval.parquet" "$live_dir" "$PORT" <<'PY'
import json
import sys

parquet_path, live_dir, port = sys.argv[1:]
payload = {
    "data_paths": [parquet_path],
    "eval_data_paths": [parquet_path],
    "dataset_id": None,
    "seed": 42,
    "max_examples": -1,
    "max_eval_examples": -1,
    "prompt_variant": "sanjaya_text_v1",
    "include_budget_reminder": False,
    "max_depth": 1,
    "recursive_cap_prompt_variant": "sanjaya_text_depth1_llm_only_v1",
    "recursive_rlm_batch_mode": "serial",
    "max_iterations": 15,
    "turn_max_tokens": 2048,
    "subcall_max_tokens": 2048,
    "max_prompt_tokens": 65536,
    "subcall_prompt_limit_ratio": 0.85,
    "subcall_budget_enabled": True,
    "max_total_subcalls": 50,
    "max_batched_subcalls": 50,
    "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
    "inference_mode": "local",
    "inference_base_url": f"http://localhost:{port}/v1",
    "inference_api_key": "local-vllm",
    "llm_subcall_provider": "vertex",
    "llm_subcall_model": "gemini-3.1-flash-lite",
    "llm_subcall_vertex_project_env": "GOOGLE_CLOUD_PROJECT",
    "llm_subcall_vertex_location": "global",
    "llm_subcall_thinking_level": "medium",
    "llm_subcall_empty_response_max_attempts": 3,
    "llm_subcall_empty_response_base_retry_seconds": 1.0,
    "llm_subcall_empty_response_max_retry_seconds": 10.0,
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
print(json.dumps(payload, separators=(",", ":")))
PY
}

write_inference_config() {
  cat > "$WORK_DIR/configs/inference_depth2_r64.toml" <<TOML
output_dir = "$WORK_DIR/inference_depth2_r64"
gpu_memory_utilization = $VLLM_GPU_MEMORY_UTILIZATION
enable_prefix_caching = true
enable_lora = true
api_server_count = 1
max_loras = 4
max_cpu_loras = 8
max_lora_rank = 64

[server]
host = "0.0.0.0"
port = $PORT

[model]
name = "$BASE_MODEL"
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
}

wait_for_server() {
  local deadline=$((SECONDS + 2400))
  until curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$WORK_DIR/logs/models.json" 2>/dev/null; do
    if [[ -n "${INFERENCE_PID:-}" ]] && ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
      log "Inference exited while starting"
      exit 1
    fi
    if (( SECONDS > deadline )); then
      log "Timed out waiting for vLLM"
      exit 1
    fi
    sleep 10
  done
}

load_adapter() {
  local payload
  payload="$(python3 - "$ADAPTER_DIR" <<'PY'
import json
import sys

print(json.dumps({"lora_name": "depth2_r64", "lora_path": sys.argv[1]}))
PY
)"
  log "Loading LoRA adapter"
  curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/load_lora_adapter" \
    -H "Content-Type: application/json" \
    -d "$payload" | tee "$WORK_DIR/logs/load_lora.json"
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" | tee "$WORK_DIR/logs/models_after_lora.json"
}

summarize() {
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/summarize_beeg_multi_model_passk.py" \
    --out-dir "$WORK_DIR/summary" \
    --dataset-id "lsteno/BEEG-agents" \
    --split eval \
    --seed 42 \
    --expected-examples "$FULL_EXAMPLES" \
    --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" \
    --model-result "depth2_r64=$WORK_DIR/results/depth2_r64" | tee "$WORK_DIR/logs/summary.log"
}

state_columns="used_repl,used_recursion,used_llm_subcalls,used_rlm_subcalls,num_subcalls,num_llm_subcalls,num_rlm_subcalls,max_depth_reached,cost_prompt_tokens,cost_completion_tokens,cost_total_tokens,cost_trainable_tokens,cost_plain_subcall_tokens,total_model_tokens,total_env_tokens,total_prompt_tokens,total_completion_tokens,total_rollout_tokens,judge_score,judge_raw_response,reward_correctness,segment_total_tokens,segment_trainable_tokens,segment_plain_subcall_tokens,empty_plain_subcalls,error_like_plain_subcalls,final_answer,prompt_variant,stop_condition,used_forced_finalize_prompt,hit_max_turn_without_final,missing_final,finalized_before_forced_prompt,finalized_on_forced_prompt"

log "Run dir: $WORK_DIR"
materialize_beeg
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  log "Downloading depth2 adapter from HF"
  "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" hf download "$DEPTH2_REPO" \
    --type model \
    --local-dir "$ADAPTER_DIR" \
    --max-workers 16
fi
write_inference_config
if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$WORK_DIR/logs/models.json" 2>/dev/null; then
  log "Reusing existing vLLM inference server on port $PORT"
  INFERENCE_PID=""
else
  log "Starting vLLM inference"
  setsid bash -c 'cd "$1/prime-rl" && exec env CUDA_VISIBLE_DEVICES="$2" "$3" run --project "$1/prime-rl" inference @ "$4"' \
    _ "$ROOT_DIR" "$CUDA_VISIBLE_DEVICES_ALL" "$UV_BIN" "$WORK_DIR/configs/inference_depth2_r64.toml" \
    > "$WORK_DIR/logs/vllm.log" 2>&1 &
  INFERENCE_PID=$!
  echo "$INFERENCE_PID" > "$WORK_DIR/inference.pid"
  wait_for_server
fi
load_adapter

live_dir="$WORK_DIR/live_traces/depth2_r64"
mkdir -p "$live_dir"
make_env_args "$live_dir" > "$WORK_DIR/configs/env_args_depth2_r64.json"
log "Starting dynamic eval: workers=$DYNAMIC_EVAL_WORKERS per_rank_cap=$DYNAMIC_EVAL_PER_RANK_CAP expected=$((FULL_EXAMPLES * ROLLOUTS_PER_EXAMPLE))"
LOCAL_VLLM_API_KEY=local-vllm "$UV_BIN" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/run_dynamic_rlm_eval.py" \
  --repo-root "$ROOT_DIR" \
  --label depth2_r64 \
  --model depth2_r64 \
  --api-base-url "http://localhost:${PORT}/v1" \
  --api-key-var LOCAL_VLLM_API_KEY \
  --env-args-file "$WORK_DIR/configs/env_args_depth2_r64.json" \
  --result-root "$WORK_DIR/results/depth2_r64" \
  --output-dir "$WORK_DIR/results/depth2_r64/dynamic" \
  --workers "$DYNAMIC_EVAL_WORKERS" \
  --dp-size "$DP_SIZE" \
  --per-rank-cap "$DYNAMIC_EVAL_PER_RANK_CAP" \
  --rollout-timeout-seconds "$EVAL_TIMEOUT_SECONDS" \
  --worker-ready-timeout-seconds 900 \
  --assigned-start-timeout-seconds 180 \
  --max-attempts "$EVAL_MAX_ATTEMPTS" \
  --max-retries "$EVAL_MAX_RETRIES" \
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" \
  --state-columns "$state_columns" \
  --sampling-args '{"max_tokens":4096,"temperature":0.7,"extra_body":{"return_token_ids":true,"top_k":-1,"min_p":0.0}}' \
  2>&1 | tee "$WORK_DIR/logs/dynamic_eval.log"
summarize
log "Done"
