#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="${ROOT_DIR:-$HOME/prime_run}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/evals/beeg_base_vs_sft_depth1_${RUN_STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-rlm-rlvr-evals}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
FULL_EXAMPLES="${FULL_EXAMPLES:-452}"
ROLLOUTS_PER_EXAMPLE="${ROLLOUTS_PER_EXAMPLE:-5}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-3}"
SMOKE_ROLLOUTS="${SMOKE_ROLLOUTS:-1}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SFT_MODEL="${SFT_MODEL:-lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2}"

mkdir -p "$WORK_DIR"/{configs,logs,results,wandb_stop}

ENV_ARGS_TEMPLATE="$WORK_DIR/configs/env_args_template.json"
cat > "$ENV_ARGS_TEMPLATE" <<'JSON'
{
  "dataset_id": "lsteno/BEEG-agents",
  "dataset_eval_split": "eval",
  "seed": 42,
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
  "inference_base_url": "http://localhost:8000/v1",
  "inference_api_key": "local-vllm",
  "llm_subcall_provider": "vertex",
  "llm_subcall_model": "gemini-3.1-flash-lite",
  "llm_subcall_vertex_project_env": "GOOGLE_CLOUD_PROJECT",
  "llm_subcall_vertex_location": "global",
  "llm_subcall_thinking_level": "medium",
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

write_inference_config() {
  local label="$1"
  local model="$2"
  local config_path="$WORK_DIR/configs/inference_${label}.toml"
  cat > "$config_path" <<TOML
output_dir = "$WORK_DIR/inference_${label}"
gpu_memory_utilization = 0.90
enable_prefix_caching = true
api_server_count = 1

[server]
host = "0.0.0.0"
port = $INFERENCE_PORT

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
gpus_per_node = 2
TOML
  printf '%s\n' "$config_path"
}

wait_for_server() {
  local model="$1"
  local deadline=$((SECONDS + 1800))
  until curl -fsS "http://127.0.0.1:${INFERENCE_PORT}/v1/models" >/tmp/rlm_eval_models.json; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for vLLM server for ${model}" >&2
      return 1
    fi
    sleep 10
  done
}

stop_inference() {
  if [[ -n "${INFERENCE_PID:-}" ]] && kill -0 "$INFERENCE_PID" 2>/dev/null; then
    kill "$INFERENCE_PID" || true
    sleep 10
    if kill -0 "$INFERENCE_PID" 2>/dev/null; then
      kill -9 "$INFERENCE_PID" || true
    fi
  fi
  pkill -f "vllm.*${INFERENCE_PORT}" || true
  sleep 5
}

make_env_args() {
  local label="$1"
  local live_dir="$2"
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$ENV_ARGS_TEMPLATE" "$live_dir" <<'PY'
import json
import sys
path, live_dir = sys.argv[1], sys.argv[2]
payload = json.loads(open(path).read())
payload["live_trace_dir"] = live_dir
print(json.dumps(payload, separators=(",", ":")))
PY
}

run_prime_eval() {
  local label="$1"
  local model="$2"
  local examples="$3"
  local rollouts="$4"
  local suffix="$5"
  local live_dir="$WORK_DIR/live_traces/${label}_${suffix}"
  local env_args
  mkdir -p "$live_dir"
  env_args="$(make_env_args "$label" "$live_dir")"
  local log_path="$WORK_DIR/logs/${label}_${suffix}_eval.log"
  local stop_file="$WORK_DIR/wandb_stop/${label}_${suffix}.stop"
  rm -f "$stop_file"
  uv run --project "$ROOT_DIR/prime-rl" python "$ROOT_DIR/scripts/rlm_eval/monitor_eval_wandb.py" \
    --results-root "$ROOT_DIR/environments/rlm_rlvr/outputs/evals" \
    --run-label "${label}_${suffix}" \
    --project "$WANDB_PROJECT" \
    --name "beeg-depth1-${label}-${suffix}-${RUN_STAMP}" \
    --expected-rollouts "$((examples * rollouts))" \
    --started-after "$(date +%s)" \
    --stop-file "$stop_file" \
    >"$WORK_DIR/logs/${label}_${suffix}_wandb.log" 2>&1 &
  local monitor_pid=$!
  set +e
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" --with prime prime eval run rlm_rlvr \
    --env-dir-path "$ROOT_DIR/environments" \
    --model "$model" \
    --api-base-url "http://localhost:${INFERENCE_PORT}/v1" \
    --api-key-var "LOCAL_VLLM_API_KEY" \
    --env-args "$env_args" \
    --num-examples "$examples" \
    --rollouts-per-example "$rollouts" \
    --max-concurrent "$MAX_CONCURRENT" \
    --max-tokens 4096 \
    --temperature 0.7 \
    --sampling-args '{"extra_body":{"return_token_ids":true,"top_k":-1,"min_p":0.0}}' \
    --state-columns "used_repl,used_recursion,used_llm_subcalls,used_rlm_subcalls,num_subcalls,num_llm_subcalls,num_rlm_subcalls,max_depth_reached,cost_prompt_tokens,cost_completion_tokens,cost_total_tokens,cost_trainable_tokens,cost_plain_subcall_tokens,total_model_tokens,total_env_tokens,total_prompt_tokens,total_completion_tokens,total_rollout_tokens,rlm_trace,final_answer,prompt_variant" \
    --save-results \
    --max-retries 2 \
    --debug \
    2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  touch "$stop_file"
  wait "$monitor_pid" || true
  local latest
  latest="$(find "$ROOT_DIR/environments/rlm_rlvr/outputs/evals" -name results.jsonl -newermt "@$(($(date +%s) - 86400))" -print0 | xargs -0 ls -t 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" ]]; then
    mkdir -p "$WORK_DIR/results/${label}_${suffix}"
    cp -a "$(dirname "$latest")" "$WORK_DIR/results/${label}_${suffix}/run_dir"
    printf '%s\n' "$latest" > "$WORK_DIR/results/${label}_${suffix}/source_results_path.txt"
  fi
  return "$status"
}

check_smoke_clean() {
  local label="$1"
  local result_path="$WORK_DIR/results/${label}_smoke/run_dir/results.jsonl"
  uv run --project "$ROOT_DIR/environments/rlm_rlvr" python - "$result_path" <<'PY'
import json
import sys
path = sys.argv[1]
rows = [json.loads(line) for line in open(path) if line.strip()]
if not rows:
    raise SystemExit(f"smoke produced no rows: {path}")
errored = [row for row in rows if row.get("has_error") or row.get("error")]
if errored:
    first = errored[0]
    raise SystemExit(f"smoke had {len(errored)}/{len(rows)} errored rows; first={first.get('error') or first}")
print(f"smoke clean: {len(rows)} rows")
PY
}

run_model() {
  local label="$1"
  local model="$2"
  local config_path
  config_path="$(write_inference_config "$label" "$model")"
  echo "[$(date -Is)] Starting inference for ${label}: ${model}" | tee -a "$WORK_DIR/logs/driver.log"
  stop_inference
  cd "$ROOT_DIR/prime-rl"
  uv run --project "$ROOT_DIR/prime-rl" inference @ "$config_path" >"$WORK_DIR/logs/${label}_vllm.log" 2>&1 &
  INFERENCE_PID=$!
  wait_for_server "$model"
  echo "[$(date -Is)] Smoke eval for ${label}" | tee -a "$WORK_DIR/logs/driver.log"
  run_prime_eval "$label" "$model" "$SMOKE_EXAMPLES" "$SMOKE_ROLLOUTS" "smoke"
  check_smoke_clean "$label"
  echo "[$(date -Is)] Full eval for ${label}" | tee -a "$WORK_DIR/logs/driver.log"
  run_prime_eval "$label" "$model" "$FULL_EXAMPLES" "$ROLLOUTS_PER_EXAMPLE" "full"
  stop_inference
}

cleanup() {
  stop_inference || true
}
trap cleanup EXIT

cd "$ROOT_DIR"
printf 'RUN_STAMP=%s\nWORK_DIR=%s\nBASE_MODEL=%s\nSFT_MODEL=%s\nMAX_CONCURRENT=%s\n' \
  "$RUN_STAMP" "$WORK_DIR" "$BASE_MODEL" "$SFT_MODEL" "$MAX_CONCURRENT" > "$WORK_DIR/run_metadata.env"

run_model "base" "$BASE_MODEL"
run_model "sft" "$SFT_MODEL"

uv run --project "$ROOT_DIR/environments/rlm_rlvr" python "$ROOT_DIR/scripts/rlm_eval/summarize_beeg_base_vs_sft.py" \
  --base "$WORK_DIR/results/base_full/run_dir" \
  --sft "$WORK_DIR/results/sft_full/run_dir" \
  --out-dir "$WORK_DIR/summary" \
  --dataset-id "lsteno/BEEG-agents" \
  --split "eval" \
  --seed 42 \
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" | tee "$WORK_DIR/logs/summary.log"

echo "[$(date -Is)] Done. Results in $WORK_DIR" | tee -a "$WORK_DIR/logs/driver.log"
