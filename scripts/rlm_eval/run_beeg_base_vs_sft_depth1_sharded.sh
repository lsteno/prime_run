#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="${ROOT_DIR:-$HOME/prime_run}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/beeg_base_vs_sft_depth1_pass10_${RUN_STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-rlm-rlvr-evals}"

ROLLOUTS_PER_EXAMPLE="${ROLLOUTS_PER_EXAMPLE:-10}"
FULL_EXAMPLES="${FULL_EXAMPLES:-452}"
SHARD_COUNT="${SHARD_COUNT:-8}"
MAX_CONCURRENT_PER_SHARD="${MAX_CONCURRENT_PER_SHARD:-3}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-3}"
SMOKE_ROLLOUTS="${SMOKE_ROLLOUTS:-1}"
RUN_SMOKE="${RUN_SMOKE:-0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
BASE_PORT="${BASE_PORT:-8000}"
SFT_PORT="${SFT_PORT:-8001}"
BASE_GPUS="${BASE_GPUS:-0,1,2,3}"
SFT_GPUS="${SFT_GPUS:-4,5,6,7}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
API_SERVER_COUNT="${API_SERVER_COUNT:-4}"
GPUS_PER_MODEL="${GPUS_PER_MODEL:-4}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SFT_MODEL="${SFT_MODEL:-lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2}"
EVAL_OUTPUT_ROOT="$ROOT_DIR/environments/rlm_rlvr/outputs/evals"

mkdir -p "$WORK_DIR"/{configs,logs,results,wandb_stop,shards,live_traces,summary}

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
  local label="$1"
  local model="$2"
  local port="$3"
  local config_path="$WORK_DIR/configs/inference_${label}.toml"
  cat > "$config_path" <<TOML
output_dir = "$WORK_DIR/inference_${label}"
gpu_memory_utilization = 0.90
enable_prefix_caching = true
api_server_count = $API_SERVER_COUNT

[server]
host = "0.0.0.0"
port = $port

[model]
name = "$model"
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
  local port="$1"
  local model="$2"
  local deadline=$((SECONDS + 1800))
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/tmp/rlm_eval_models_${port}.json; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for vLLM server for ${model} on port ${port}" >&2
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
  stop_pid "${BASE_INFERENCE_PID:-}"
  stop_pid "${SFT_INFERENCE_PID:-}"
  pkill -f "vllm.*${BASE_PORT}" || true
  pkill -f "vllm.*${SFT_PORT}" || true
  sleep 5
}

write_sharded_dataset() {
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" "$FULL_EXAMPLES" "$ROLLOUTS_PER_EXAMPLE" "$SHARD_COUNT" "$SMOKE_EXAMPLES" <<'PY'
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
smoke_rollout_units = int(sys.argv[5])

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
example_rows = []
shard_paths = [work_dir / "shards" / f"eval_shard_{idx:02d}.parquet" for idx in range(shard_count)]
shard_writers = [pq.ParquetWriter(path, schema=schema) for path in shard_paths]
smoke_path = work_dir / "shards" / "smoke.parquet"
smoke_writer = pq.ParquetWriter(smoke_path, schema=schema)
shard_buffers = [[] for _ in range(shard_count)]
shard_counts = [0 for _ in range(shard_count)]
smoke_buffer = []
smoke_count = 0
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
    example_rows.append({"id": source_id, **base_row, "metadata": json.dumps(row_metadata, ensure_ascii=False)})
    for rollout_index in range(rollouts_per_example):
        rollout_metadata = dict(row_metadata)
        rollout_metadata["original_rollout_index"] = rollout_index
        rollout_id = f"{source_id}__rollout_{rollout_index}"
        rollout_row = {
            "id": rollout_id,
            **base_row,
            "metadata": json.dumps(rollout_metadata, ensure_ascii=False),
        }
        shard_idx = unit_index % shard_count
        shard_buffers[shard_idx].append(rollout_row)
        shard_counts[shard_idx] += 1
        if len(shard_buffers[shard_idx]) >= 16:
            flush(shard_writers[shard_idx], shard_buffers[shard_idx])
        if smoke_count < smoke_rollout_units:
            smoke_buffer.append(rollout_row)
            smoke_count += 1
            if len(smoke_buffer) >= 16:
                flush(smoke_writer, smoke_buffer)
        unit_index += 1

try:
    for writer, rows in zip(shard_writers, shard_buffers):
        flush(writer, rows)
    flush(smoke_writer, smoke_buffer)
finally:
    for writer in shard_writers:
        writer.close()
    smoke_writer.close()

(work_dir / "manifest.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
    encoding="utf-8",
)
(work_dir / "manifest.csv").write_text(
    "source_id,original_example_index,dataset,task,answer_type,context_token_count,metadata\n"
    + "".join(
        ",".join(json.dumps(row[key], ensure_ascii=False) for key in ("source_id", "original_example_index", "dataset", "task", "answer_type", "context_token_count", "metadata")) + "\n"
        for row in manifest_rows
    ),
    encoding="utf-8",
)
pq.write_table(pa.Table.from_pylist(example_rows, schema=schema), work_dir / "eval_examples.parquet")
(work_dir / "shards" / "smoke.count").write_text(str(smoke_count))
for shard_path, count in zip(shard_paths, shard_counts):
    shard_path.with_suffix(".count").write_text(str(count))

print(json.dumps({"examples": len(dataset), "rollouts_per_example": rollouts_per_example, "total_rollouts": unit_index, "shards": shard_count}))
PY
}

make_env_args() {
  local parquet_path="$1"
  local live_dir="$2"
  local port="$3"
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$ENV_ARGS_TEMPLATE" "$parquet_path" "$live_dir" "$port" <<'PY'
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
  local label="$1"
  local model="$2"
  local port="$3"
  local shard_idx="$4"
  local parquet_path="$5"
  local count="$6"
  local suffix="$7"
  local rollouts="$8"
  local dp_rank=$((shard_idx % DP_SIZE))
  local live_dir="$WORK_DIR/live_traces/${label}_${suffix}_shard_${shard_idx}"
  local env_args
  mkdir -p "$live_dir"
  env_args="$(make_env_args "$parquet_path" "$live_dir" "$port")"
  LOCAL_VLLM_API_KEY=local-vllm \
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" --with prime prime eval run rlm_rlvr \
    --env-dir-path "$ROOT_DIR/environments" \
    --model "$model" \
    --api-base-url "http://localhost:${port}/v1" \
    --api-key-var "LOCAL_VLLM_API_KEY" \
    --header "X-data-parallel-rank: ${dp_rank}" \
    --env-args "$env_args" \
    --num-examples "$count" \
    --rollouts-per-example "$rollouts" \
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
  local label="$1"
  local model="$2"
  local expected_rollouts="$3"
  local started_after="$4"
  local __pid_var="$5"
  local stop_file="$WORK_DIR/wandb_stop/${label}_full.stop"
  local root="$EVAL_OUTPUT_ROOT/$(model_slug "$model")"
  rm -f "$stop_file"
  uv run --project "$ROOT_DIR/prime-rl" python "$ROOT_DIR/scripts/rlm_eval/monitor_eval_wandb.py" \
    --results-root "$root" \
    --aggregate-files \
    --run-label "${label}_full" \
    --project "$WANDB_PROJECT" \
    --name "beeg-depth1-pass10-${label}-${RUN_STAMP}" \
    --expected-rollouts "$expected_rollouts" \
    --started-after "$started_after" \
    --stop-file "$stop_file" \
    >"$WORK_DIR/logs/${label}_full_wandb.log" 2>&1 &
  printf -v "$__pid_var" '%s' "$!"
}

collect_results() {
  local label="$1"
  local model="$2"
  local started_after="$3"
  local root="$EVAL_OUTPUT_ROOT/$(model_slug "$model")"
  local dest="$WORK_DIR/results/${label}_full"
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

run_smoke_pair() {
  local smoke_path="$WORK_DIR/shards/smoke.parquet"
  local count="$SMOKE_EXAMPLES"
  echo "[$(date -Is)] Smoke evals" | tee -a "$WORK_DIR/logs/driver.log"
  run_eval_shard "base" "$BASE_MODEL" "$BASE_PORT" 0 "$smoke_path" "$count" "smoke" "$SMOKE_ROLLOUTS" >"$WORK_DIR/logs/base_smoke_eval.log" 2>&1 &
  local base_smoke_pid=$!
  run_eval_shard "sft" "$SFT_MODEL" "$SFT_PORT" 0 "$smoke_path" "$count" "smoke" "$SMOKE_ROLLOUTS" >"$WORK_DIR/logs/sft_smoke_eval.log" 2>&1 &
  local sft_smoke_pid=$!
  wait "$base_smoke_pid"
  wait "$sft_smoke_pid"
}

run_full_pair() {
  local expected_rollouts=$((FULL_EXAMPLES * ROLLOUTS_PER_EXAMPLE))
  local full_start
  full_start="$(date +%s)"
  echo "$full_start" > "$WORK_DIR/full_start_epoch.txt"
  local base_monitor_pid
  local sft_monitor_pid
  start_monitor "base" "$BASE_MODEL" "$expected_rollouts" "$full_start" base_monitor_pid
  start_monitor "sft" "$SFT_MODEL" "$expected_rollouts" "$full_start" sft_monitor_pid

  pids=()
  for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
    shard_path="$WORK_DIR/shards/eval_shard_$(printf '%02d' "$shard_idx").parquet"
    count="$(cat "${shard_path%.parquet}.count")"
    run_eval_shard "base" "$BASE_MODEL" "$BASE_PORT" "$shard_idx" "$shard_path" "$count" "full" \
      "1" >"$WORK_DIR/logs/base_full_shard_$(printf '%02d' "$shard_idx").log" 2>&1 &
    pids+=("$!")
    run_eval_shard "sft" "$SFT_MODEL" "$SFT_PORT" "$shard_idx" "$shard_path" "$count" "full" \
      "1" >"$WORK_DIR/logs/sft_full_shard_$(printf '%02d' "$shard_idx").log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  touch "$WORK_DIR/wandb_stop/base_full.stop" "$WORK_DIR/wandb_stop/sft_full.stop"
  wait "$base_monitor_pid" || true
  wait "$sft_monitor_pid" || true

  collect_results "base" "$BASE_MODEL" "$full_start"
  collect_results "sft" "$SFT_MODEL" "$full_start"

  if [[ "$failed" != "0" ]]; then
    echo "At least one full eval shard failed; partial results were collected." >&2
    return 1
  fi
}

validate_counts() {
  local expected_rollouts=$((FULL_EXAMPLES * ROLLOUTS_PER_EXAMPLE))
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" "$expected_rollouts" <<'PY'
import json
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
for label in ("base", "sft"):
    files = sorted((work_dir / "results" / f"{label}_full").rglob("results.jsonl"))
    rows = 0
    for path in files:
        with path.open() as handle:
            rows += sum(1 for line in handle if line.strip())
    print(f"{label}: {rows}/{expected} rows across {len(files)} files")
    if rows != expected:
        raise SystemExit(f"{label} result row count mismatch: got {rows}, expected {expected}")
PY
}

cleanup() {
  stop_inference || true
}
trap cleanup EXIT

cd "$ROOT_DIR"
printf 'RUN_STAMP=%s\nWORK_DIR=%s\nBASE_MODEL=%s\nSFT_MODEL=%s\nROLLOUTS_PER_EXAMPLE=%s\nSHARD_COUNT=%s\nMAX_CONCURRENT_PER_SHARD=%s\n' \
  "$RUN_STAMP" "$WORK_DIR" "$BASE_MODEL" "$SFT_MODEL" "$ROLLOUTS_PER_EXAMPLE" "$SHARD_COUNT" "$MAX_CONCURRENT_PER_SHARD" \
  > "$WORK_DIR/run_metadata.env"

write_sharded_dataset | tee "$WORK_DIR/logs/dataset.log"

base_config="$(write_inference_config base "$BASE_MODEL" "$BASE_PORT")"
sft_config="$(write_inference_config sft "$SFT_MODEL" "$SFT_PORT")"

echo "[$(date -Is)] Starting base inference on GPUs ${BASE_GPUS}" | tee -a "$WORK_DIR/logs/driver.log"
cd "$ROOT_DIR/prime-rl"
CUDA_VISIBLE_DEVICES="$BASE_GPUS" uv run --project "$ROOT_DIR/prime-rl" inference @ "$base_config" >"$WORK_DIR/logs/base_vllm.log" 2>&1 &
BASE_INFERENCE_PID=$!

echo "[$(date -Is)] Starting SFT inference on GPUs ${SFT_GPUS}" | tee -a "$WORK_DIR/logs/driver.log"
CUDA_VISIBLE_DEVICES="$SFT_GPUS" uv run --project "$ROOT_DIR/prime-rl" inference @ "$sft_config" >"$WORK_DIR/logs/sft_vllm.log" 2>&1 &
SFT_INFERENCE_PID=$!

wait_for_server "$BASE_PORT" "$BASE_MODEL"
wait_for_server "$SFT_PORT" "$SFT_MODEL"

if [[ "$RUN_SMOKE" == "1" ]]; then
  run_smoke_pair
else
  echo "[$(date -Is)] Skipping smoke evals because RUN_SMOKE=${RUN_SMOKE}" | tee -a "$WORK_DIR/logs/driver.log"
fi
run_full_pair
validate_counts

uv run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/summarize_beeg_base_vs_sft.py" \
  --base "$WORK_DIR/results/base_full" \
  --sft "$WORK_DIR/results/sft_full" \
  --out-dir "$WORK_DIR/summary" \
  --dataset-id "lsteno/BEEG-agents" \
  --split "eval" \
  --seed 42 \
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" | tee "$WORK_DIR/logs/summary.log"

echo "[$(date -Is)] Done. Results in $WORK_DIR" | tee -a "$WORK_DIR/logs/driver.log"
