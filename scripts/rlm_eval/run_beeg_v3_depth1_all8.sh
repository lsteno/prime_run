#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="${ROOT_DIR:-$HOME/prime_run}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/beeg_v3_depth1_pass10_${RUN_STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-rlm-rlvr-evals}"

SFT_MODEL="${SFT_MODEL:-lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v3-per-root-turn}"
ROLLOUTS_PER_EXAMPLE="${ROLLOUTS_PER_EXAMPLE:-10}"
FULL_EXAMPLES="${FULL_EXAMPLES:-452}"
SHARD_COUNT="${SHARD_COUNT:-8}"
MAX_CONCURRENT_PER_SHARD="${MAX_CONCURRENT_PER_SHARD:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
SFT_PORT="${SFT_PORT:-8001}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-8}"
API_SERVER_COUNT="${API_SERVER_COUNT:-8}"
GPUS_PER_MODEL="${GPUS_PER_MODEL:-8}"
UV="${UV:-$HOME/.local/bin/uv}"
EVAL_OUTPUT_ROOT="$ROOT_DIR/environments/rlm_rlvr/outputs/evals"

mkdir -p "$WORK_DIR"/{configs,logs,results/sft_full,wandb_stop,shards,live_traces,summary}

cd "$ROOT_DIR"
set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export PATH="$HOME/.local/bin:$PATH"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-local-vllm}"

ENV_ARGS_TEMPLATE="$WORK_DIR/configs/env_args_template.json"
cat > "$ENV_ARGS_TEMPLATE" <<'JSON'
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
  "turn_max_tokens": 4096,
  "subcall_max_tokens": 4096,
  "max_prompt_tokens": 65536,
  "subcall_prompt_limit_ratio": 0.85,
  "subcall_budget_enabled": true,
  "max_total_subcalls": 80,
  "max_batched_subcalls": 80,
  "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
  "inference_mode": "local",
  "inference_api_key": "local-vllm",
  "llm_subcall_provider": "vertex",
  "llm_subcall_model": "gemini-3.1-flash-lite",
  "llm_subcall_vertex_project_env": "GOOGLE_CLOUD_PROJECT",
  "llm_subcall_vertex_location": "global",
  "llm_subcall_thinking_level": "medium",
  "llm_subcall_empty_response_max_attempts": 3,
  "repl_backend": "local",
  "repl_timeout_seconds": 900,
  "repl_fast_timeout_seconds": 30,
  "judge_provider": "vertex",
  "judge_model": "gemini-3-flash-preview",
  "judge_vertex_project_env": "GOOGLE_CLOUD_PROJECT",
  "judge_vertex_location": "global",
  "judge_thinking_level": "medium",
  "efficiency_penalty_mode": "static_per_1k",
  "efficiency_penalty_coef": 0.0
}
JSON

state_columns="used_repl,used_recursion,used_llm_subcalls,used_rlm_subcalls,num_subcalls,num_llm_subcalls,num_rlm_subcalls,max_depth_reached,cost_prompt_tokens,cost_completion_tokens,cost_total_tokens,cost_trainable_tokens,cost_plain_subcall_tokens,total_model_tokens,total_env_tokens,total_prompt_tokens,total_completion_tokens,total_rollout_tokens,judge_score,judge_raw_response,reward_correctness,rlm_trace,final_answer,prompt_variant"

model_slug() {
  printf 'rlm_rlvr--%s\n' "${1//\//--}"
}

write_inference_config() {
  local config_path="$WORK_DIR/configs/inference_v3_all8.toml"
  cat > "$config_path" <<TOML
output_dir = "$WORK_DIR/inference_v3_all8"
gpu_memory_utilization = 0.90
enable_prefix_caching = true
api_server_count = $API_SERVER_COUNT

[server]
host = "0.0.0.0"
port = $SFT_PORT

[model]
name = "$SFT_MODEL"
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
  until curl -fsS "http://127.0.0.1:${SFT_PORT}/v1/models" >/tmp/rlm_eval_models_${SFT_PORT}.json; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for vLLM server for ${SFT_MODEL} on port ${SFT_PORT}" >&2
      return 1
    fi
    sleep 10
  done
}

stop_pid() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    sleep 10
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" || true
    fi
  fi
}

stop_inference() {
  stop_pid "${SFT_INFERENCE_PID:-}"
  pkill -f "vllm.*${SFT_PORT}" || true
  sleep 5
}

write_sharded_dataset() {
  "$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" "$FULL_EXAMPLES" "$ROLLOUTS_PER_EXAMPLE" "$SHARD_COUNT" <<'PY'
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

def normalize_text(value):
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
shard_writers = [pq.ParquetWriter(path, schema=schema) for path in shard_paths]
shard_buffers = [[] for _ in range(shard_count)]
shard_counts = [0 for _ in range(shard_count)]
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
    base_row = {
        "prompt": normalize_text(row.get("prompt") or row.get("question") or row.get("task")),
        "context": normalize_text(row.get("context") or ""),
        "answer": normalize_text(row.get("answer")),
        "acceptable_answers": normalize_text(row.get("acceptable_answers") or row.get("answers") or row.get("answer")),
        "dataset": normalize_text(row.get("dataset")),
        "task": normalize_text(row.get("task")),
        "answer_type": normalize_text(row.get("answer_type")),
        "context_token_count": row.get("context_token_count"),
    }
    for rollout_index in range(rollouts_per_example):
        rollout_metadata = dict(row_metadata)
        rollout_metadata["original_rollout_index"] = rollout_index
        rollout_row = {
            "id": f"{source_id}__rollout_{rollout_index}",
            **base_row,
            "metadata": json.dumps(rollout_metadata, ensure_ascii=False),
        }
        shard_idx = unit_index % shard_count
        shard_buffers[shard_idx].append(rollout_row)
        shard_counts[shard_idx] += 1
        if len(shard_buffers[shard_idx]) >= 16:
            flush(shard_writers[shard_idx], shard_buffers[shard_idx])
        unit_index += 1

try:
    for writer, rows in zip(shard_writers, shard_buffers):
        flush(writer, rows)
finally:
    for writer in shard_writers:
        writer.close()

(work_dir / "manifest.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
    encoding="utf-8",
)
for shard_path, count in zip(shard_paths, shard_counts):
    shard_path.with_suffix(".count").write_text(str(count))

print(json.dumps({"examples": len(dataset), "rollouts_per_example": rollouts_per_example, "total_rollouts": unit_index, "shards": shard_count}))
PY
}

make_env_args() {
  local parquet_path="$1"
  local live_dir="$2"
  "$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$ENV_ARGS_TEMPLATE" "$parquet_path" "$live_dir" "$SFT_PORT" <<'PY'
import json
import sys
path, parquet_path, live_dir, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
payload = json.loads(open(path).read())
payload["data_paths"] = [parquet_path]
payload["eval_data_paths"] = [parquet_path]
payload["live_trace_dir"] = live_dir
payload["inference_base_url"] = f"http://localhost:{port}/v1"
print(json.dumps(payload, separators=(",", ":")))
PY
}

run_eval_shard() {
  local shard_idx="$1"
  local parquet_path="$2"
  local count="$3"
  local dp_rank=$((shard_idx % DP_SIZE))
  local live_dir="$WORK_DIR/live_traces/v3_full_shard_${shard_idx}"
  local env_args
  mkdir -p "$live_dir"
  env_args="$(make_env_args "$parquet_path" "$live_dir")"
  LOCAL_VLLM_API_KEY=local-vllm \
  "$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" --with prime prime eval run rlm_rlvr \
    --env-dir-path "$ROOT_DIR/environments" \
    --model "$SFT_MODEL" \
    --api-base-url "http://localhost:${SFT_PORT}/v1" \
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
    --max-retries 2 \
    --debug
}

start_monitor() {
  local expected_rollouts="$1"
  local started_after="$2"
  local stop_file="$WORK_DIR/wandb_stop/v3_full.stop"
  local root="$EVAL_OUTPUT_ROOT/$(model_slug "$SFT_MODEL")"
  rm -f "$stop_file"
  "$UV" run --project "$ROOT_DIR/prime-rl" python "$ROOT_DIR/scripts/rlm_eval/monitor_eval_wandb.py" \
    --results-root "$root" \
    --aggregate-files \
    --run-label "v3_full" \
    --project "$WANDB_PROJECT" \
    --name "beeg-depth1-pass10-v3-${RUN_STAMP}" \
    --expected-rollouts "$expected_rollouts" \
    --started-after "$started_after" \
    --stop-file "$stop_file" \
    >"$WORK_DIR/logs/v3_full_wandb.log" 2>&1 &
  MONITOR_PID="$!"
}

collect_results() {
  local started_after="$1"
  local root="$EVAL_OUTPUT_ROOT/$(model_slug "$SFT_MODEL")"
  local dest="$WORK_DIR/results/sft_full"
  mkdir -p "$dest"
  find "$root" -name results.jsonl -type f -newermt "@${started_after}" -print0 \
    | sort -z \
    | while IFS= read -r -d '' result_file; do
        run_dir="$(dirname "$result_file")"
        run_name="$(basename "$run_dir")"
        cp -a "$run_dir" "$dest/${run_name}"
      done
  find "$dest" -name results.jsonl -type f | sort > "$dest/result_files.txt"
}

validate_counts() {
  local expected_rollouts=$((FULL_EXAMPLES * ROLLOUTS_PER_EXAMPLE))
  "$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR/results/sft_full" "$expected_rollouts" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = 0
files = 0
rlm_subcall_rows = 0
for path in sorted(root.rglob("results.jsonl")):
    files += 1
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = record.get("state") or {}
            info = record.get("info") or {}
            value = state.get("used_rlm_subcalls", info.get("used_rlm_subcalls", False))
            if value:
                rlm_subcall_rows += 1
print(f"v3: {rows}/{expected} rows across {files} files; accidental_rlm_subcall_rows={rlm_subcall_rows}")
if rows != expected:
    raise SystemExit(f"v3 result row count mismatch: got {rows}, expected {expected}")
if rlm_subcall_rows:
    print(f"WARNING: {rlm_subcall_rows} rows reported used_rlm_subcalls")
PY
}

cleanup() {
  touch "$WORK_DIR/wandb_stop/v3_full.stop" 2>/dev/null || true
  if [[ -n "${MONITOR_PID:-}" ]]; then
    wait "$MONITOR_PID" || true
  fi
  stop_inference || true
}
trap cleanup EXIT

printf 'RUN_STAMP=%s\nWORK_DIR=%s\nSFT_MODEL=%s\nROLLOUTS_PER_EXAMPLE=%s\nSHARD_COUNT=%s\nMAX_CONCURRENT_PER_SHARD=%s\nDP_SIZE=%s\nAPI_SERVER_COUNT=%s\n' \
  "$RUN_STAMP" "$WORK_DIR" "$SFT_MODEL" "$ROLLOUTS_PER_EXAMPLE" "$SHARD_COUNT" "$MAX_CONCURRENT_PER_SHARD" "$DP_SIZE" "$API_SERVER_COUNT" \
  > "$WORK_DIR/run_metadata.env"

write_sharded_dataset | tee "$WORK_DIR/logs/dataset.log"

inference_config="$(write_inference_config)"

echo "[$(date -Is)] Starting v3 inference on all 8 GPUs" | tee -a "$WORK_DIR/logs/driver.log"
cd "$ROOT_DIR/prime-rl"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" "$UV" run --project "$ROOT_DIR/prime-rl" inference @ "$inference_config" >"$WORK_DIR/logs/v3_vllm.log" 2>&1 &
SFT_INFERENCE_PID=$!
echo "$SFT_INFERENCE_PID" > "$WORK_DIR/v3_inference.pid"

wait_for_server

expected_rollouts=$((FULL_EXAMPLES * ROLLOUTS_PER_EXAMPLE))
full_start="$(date +%s)"
echo "$full_start" > "$WORK_DIR/full_start_epoch.txt"
start_monitor "$expected_rollouts" "$full_start"

pids=()
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_path="$WORK_DIR/shards/eval_shard_$(printf '%02d' "$shard_idx").parquet"
  count="$(cat "${shard_path%.parquet}.count")"
  run_eval_shard "$shard_idx" "$shard_path" "$count" >"$WORK_DIR/logs/v3_full_shard_$(printf '%02d' "$shard_idx").log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

touch "$WORK_DIR/wandb_stop/v3_full.stop"
wait "$MONITOR_PID" || true
MONITOR_PID=""

collect_results "$full_start"
validate_counts | tee "$WORK_DIR/logs/validate_counts.log"

if [[ "$failed" != "0" ]]; then
  echo "At least one eval shard failed; partial results were collected." >&2
  exit 1
fi

echo "[$(date -Is)] Done. V3 results in $WORK_DIR" | tee -a "$WORK_DIR/logs/driver.log"
