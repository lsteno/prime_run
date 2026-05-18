#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/prime_run}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/beeg_base_vs_sft_depth1_pass10_20260516T200137Z}"
EVAL_OUTPUT_ROOT="$ROOT_DIR/environments/rlm_rlvr/outputs/evals"
SFT_MODEL="${SFT_MODEL:-lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2}"
BASE_ROOT="$EVAL_OUTPUT_ROOT/rlm_rlvr--Qwen--Qwen3-4B-Instruct-2507"
SFT_ROOT="$EVAL_OUTPUT_ROOT/rlm_rlvr--lsteno--Qwen3-4B-Instruct-2507-RLM-SFT-v2"
ACCEL_DIR="${ACCEL_DIR:-$WORK_DIR/sft_accelerated_all8_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$WORK_DIR/logs"
UV="${UV:-$HOME/.local/bin/uv}"

mkdir -p "$ACCEL_DIR"/{configs,shards,live_traces,logs,wandb_stop} "$LOG_DIR" "$WORK_DIR/results/base_full" "$WORK_DIR/results/sft_full" "$WORK_DIR/summary"

cd "$ROOT_DIR"
set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export PATH="$HOME/.local/bin:$PATH"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ambient-empire-492113-d7}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-local-vllm}"

FULL_START="$(cat "$WORK_DIR/full_start_epoch.txt" 2>/dev/null || echo 1778962236)"

echo "[$(date -Is)] Accelerating remaining SFT eval into $ACCEL_DIR" | tee -a "$LOG_DIR/accelerate_sft.log"

old_pgid="$(
  ps -eo pid,pgid,cmd \
    | awk '/run_beeg_base_vs_sft_depth1_sharded[.]sh/ {print $2; exit}'
)"
if [[ -n "${old_pgid:-}" ]]; then
  echo "[$(date -Is)] Stopping old process group $old_pgid" | tee -a "$LOG_DIR/accelerate_sft.log"
  pkill -TERM -g "$old_pgid" || true
  sleep 20
  pkill -KILL -g "$old_pgid" || true
fi

pkill -f "prime-rl inference.*inference_base.toml" || true
pkill -f "prime-rl inference.*inference_sft.toml" || true
pkill -f "vllm.*8000" || true
pkill -f "vllm.*8001" || true
sleep 10

cat > "$ACCEL_DIR/configs/inference_sft_all8.toml" <<TOML
output_dir = "$ACCEL_DIR/inference_sft_all8"
gpu_memory_utilization = 0.90
enable_prefix_caching = true
api_server_count = 8

[server]
host = "0.0.0.0"
port = 8001

[model]
name = "$SFT_MODEL"
max_model_len = 65536
enforce_eager = false
trust_remote_code = true

[parallel]
tp = 1
dp = 8

[deployment]
type = "single_node"
gpus_per_node = 8
TOML

cat > "$ACCEL_DIR/configs/env_args_template.json" <<'JSON'
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

"$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" "$ACCEL_DIR" <<'PY'
from __future__ import annotations

from pathlib import Path
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

work = Path(sys.argv[1])
accel = Path(sys.argv[2])

completed: set[str] = set()
current_dirs: set[Path] = set()
for log in sorted((work / "logs").glob("sft_full_shard_*.log")):
    for line in log.read_text(errors="ignore").splitlines():
        marker = "Saving results to "
        if marker in line:
            current_dirs.add(Path(line.split(marker, 1)[1].strip()))

for run_dir in current_dirs:
    result_file = run_dir / "results.jsonl"
    if not result_file.exists():
        continue
    with result_file.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = (obj.get("info") or {}).get("source_id")
            if source_id:
                completed.add(str(source_id))

schema = pq.read_schema(work / "shards" / "eval_shard_00.parquet")
writers = [
    pq.ParquetWriter(accel / "shards" / f"sft_remaining_shard_{idx:02d}.parquet", schema=schema)
    for idx in range(8)
]
buffers: list[list[dict]] = [[] for _ in range(8)]
counts = [0] * 8

def flush(idx: int) -> None:
    if buffers[idx]:
        writers[idx].write_table(pa.Table.from_pylist(buffers[idx], schema=schema))
        buffers[idx].clear()

remaining = 0
for shard in sorted((work / "shards").glob("eval_shard_*.parquet")):
    for row in pq.read_table(shard).to_pylist():
        if str(row.get("id")) in completed:
            continue
        idx = remaining % 8
        buffers[idx].append(row)
        counts[idx] += 1
        remaining += 1
        if len(buffers[idx]) >= 16:
            flush(idx)

for idx, writer in enumerate(writers):
    flush(idx)
    writer.close()
    (accel / "shards" / f"sft_remaining_shard_{idx:02d}.count").write_text(str(counts[idx]))

manifest = {
    "completed_rows": len(completed),
    "remaining_rows": remaining,
    "counts": counts,
    "current_result_dirs": [str(path) for path in sorted(current_dirs)],
}
(accel / "remaining_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

remaining="$(
  python3 - "$ACCEL_DIR/remaining_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["remaining_rows"])
PY
)"
echo "[$(date -Is)] Remaining SFT rows: $remaining" | tee -a "$LOG_DIR/accelerate_sft.log"
if [[ "$remaining" == "0" ]]; then
  echo "[$(date -Is)] No remaining SFT rows." | tee -a "$LOG_DIR/accelerate_sft.log"
  exit 0
fi

echo "[$(date -Is)] Starting SFT inference on all 8 GPUs" | tee -a "$LOG_DIR/accelerate_sft.log"
cd "$ROOT_DIR/prime-rl"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" "$UV" run --project "$ROOT_DIR/prime-rl" inference @ "$ACCEL_DIR/configs/inference_sft_all8.toml" >"$ACCEL_DIR/logs/sft_vllm_all8.log" 2>&1 &
sft_pid=$!
echo "$sft_pid" > "$ACCEL_DIR/sft_inference.pid"

for attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:8001/v1/models" >/tmp/rlm_sft_all8_models.json 2>/dev/null; then
    break
  fi
  if ! kill -0 "$sft_pid" 2>/dev/null; then
    echo "SFT inference died during startup" >&2
    tail -200 "$ACCEL_DIR/logs/sft_vllm_all8.log" >&2 || true
    exit 1
  fi
  if [[ "$attempt" == "180" ]]; then
    echo "Timed out waiting for all8 SFT server" >&2
    exit 1
  fi
  sleep 10
done

rm -f "$ACCEL_DIR/wandb_stop/sft_accelerated.stop"
cd "$ROOT_DIR"
"$UV" run --project "$ROOT_DIR/prime-rl" python "$ROOT_DIR/scripts/rlm_eval/monitor_eval_wandb.py" \
  --results-root "$SFT_ROOT" \
  --aggregate-files \
  --run-label sft_full_accelerated \
  --project rlm-rlvr-evals \
  --name "beeg-depth1-pass10-sft-accelerated-$(basename "$ACCEL_DIR")" \
  --expected-rollouts 4520 \
  --started-after "$FULL_START" \
  --stop-file "$ACCEL_DIR/wandb_stop/sft_accelerated.stop" \
  >"$ACCEL_DIR/logs/sft_accelerated_wandb.log" 2>&1 &
monitor_pid=$!
echo "$monitor_pid" > "$ACCEL_DIR/wandb_monitor.pid"

state_columns="used_repl,used_recursion,used_llm_subcalls,used_rlm_subcalls,num_subcalls,num_llm_subcalls,num_rlm_subcalls,max_depth_reached,cost_prompt_tokens,cost_completion_tokens,cost_total_tokens,cost_trainable_tokens,cost_plain_subcall_tokens,total_model_tokens,total_env_tokens,total_prompt_tokens,total_completion_tokens,total_rollout_tokens,judge_score,judge_raw_response,reward_correctness,rlm_trace,final_answer,prompt_variant"

make_env_args() {
  local parquet_path="$1"
  local live_dir="$2"
  "$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$ACCEL_DIR/configs/env_args_template.json" "$parquet_path" "$live_dir" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
payload["data_paths"] = [sys.argv[2]]
payload["eval_data_paths"] = [sys.argv[2]]
payload["live_trace_dir"] = sys.argv[3]
payload["inference_base_url"] = "http://localhost:8001/v1"
print(json.dumps(payload, separators=(",", ":")))
PY
}

pids=()
for shard_idx in $(seq 0 7); do
  shard_path="$ACCEL_DIR/shards/sft_remaining_shard_$(printf '%02d' "$shard_idx").parquet"
  count="$(cat "${shard_path%.parquet}.count")"
  if [[ "$count" == "0" ]]; then
    continue
  fi
  live_dir="$ACCEL_DIR/live_traces/sft_remaining_shard_${shard_idx}"
  mkdir -p "$live_dir"
  env_args="$(make_env_args "$shard_path" "$live_dir")"
  (
    LOCAL_VLLM_API_KEY=local-vllm "$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" --with prime prime eval run rlm_rlvr \
      --env-dir-path "$ROOT_DIR/environments" \
      --model "$SFT_MODEL" \
      --api-base-url "http://localhost:8001/v1" \
      --api-key-var LOCAL_VLLM_API_KEY \
      --header "X-data-parallel-rank: ${shard_idx}" \
      --env-args "$env_args" \
      --num-examples "$count" \
      --rollouts-per-example 1 \
      --max-concurrent 4 \
      --max-tokens 4096 \
      --temperature 0.7 \
      --sampling-args '{"extra_body":{"return_token_ids":true,"top_k":-1,"min_p":0.0}}' \
      --state-columns "$state_columns" \
      --save-results \
      --skip-upload \
      --max-retries 2 \
      --debug
  ) >"$ACCEL_DIR/logs/sft_remaining_shard_$(printf '%02d' "$shard_idx").log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

touch "$ACCEL_DIR/wandb_stop/sft_accelerated.stop"
wait "$monitor_pid" || true

collect_model() {
  local root="$1"
  local dest="$2"
  rm -rf "$dest"
  mkdir -p "$dest"
  find "$root" -name results.jsonl -type f -newermt "@$FULL_START" -print0 \
    | sort -z \
    | while IFS= read -r -d '' result_file; do
        run_dir="$(dirname "$result_file")"
        run_name="$(basename "$run_dir")"
        cp -a "$run_dir" "$dest/$run_name"
      done
  find "$dest" -name results.jsonl -type f | sort > "$dest/result_files.txt"
}

collect_model "$BASE_ROOT" "$WORK_DIR/results/base_full"
collect_model "$SFT_ROOT" "$WORK_DIR/results/sft_full"

"$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$WORK_DIR" <<'PY' | tee "$ACCEL_DIR/logs/final_counts.log"
from pathlib import Path
import sys

work = Path(sys.argv[1])
for label in ("base", "sft"):
    rows = 0
    files = 0
    for path in (work / "results" / f"{label}_full").rglob("results.jsonl"):
        files += 1
        rows += sum(1 for line in path.open() if line.strip())
    print(f"{label}: {rows}/4520 rows across {files} files")
PY

"$UV" run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/summarize_beeg_base_vs_sft.py" \
  --base "$WORK_DIR/results/base_full" \
  --sft "$WORK_DIR/results/sft_full" \
  --out-dir "$WORK_DIR/summary" \
  --dataset-id "lsteno/BEEG-agents" \
  --split eval \
  --seed 42 \
  --rollouts-per-example 10 \
  | tee "$WORK_DIR/logs/summary.log" || true

if [[ "$failed" != "0" ]]; then
  echo "[$(date -Is)] One or more accelerated SFT shards failed; partial results collected." | tee -a "$LOG_DIR/accelerate_sft.log"
  exit 1
fi

echo "[$(date -Is)] Accelerated SFT eval complete." | tee -a "$LOG_DIR/accelerate_sft.log"
