#!/usr/bin/env bash
set -euo pipefail

# Wait for a remote Prime-RL SFT run to finish, verify that the upload helper ran
# successfully, copy audit/debug artifacts back locally, then terminate the pod.
#
# This intentionally skips checkpoint weight tensors by default. The model
# weights are expected to be uploaded to Hugging Face before pod termination; the
# copied local artifacts are the pieces useful for thesis/debug provenance.

POD_HOST="${POD_HOST:-ubuntu@216.81.248.132}"
POD_ID="${POD_ID:-b805f622186545068c7c06b45193dfd0}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/home/ubuntu/prime_run/outputs/rlm-rlvr-sft-curated-v2-qwen3-4b-instruct-8xrtx6000ada-cat-cp4}"
REMOTE_LOG="${REMOTE_LOG:-/home/ubuntu/prime_run/outputs/run_logs/sft_curated_v2_cat_cp4_full.log}"
REMOTE_CONFIG="${REMOTE_CONFIG:-/home/ubuntu/prime_run/configs/rlm_sft/local_8xrtx6000ada_48gb_qwen3_4b_instruct_curated_v2.toml}"
LOCAL_DEST="${LOCAL_DEST:-outputs/pod_artifacts/sft_curated_v2_cat_cp4_$(date -u +%Y%m%dT%H%M%SZ)}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TERMINATE_POD="${TERMINATE_POD:-1}"
HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2}"

mkdir -p "${LOCAL_DEST}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOCAL_DEST}/watcher.log"
}

remote() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${POD_HOST}" "$@"
}

log "Watching ${POD_HOST}"
log "Remote output: ${REMOTE_OUTPUT_DIR}"
log "Remote log: ${REMOTE_LOG}"
log "Local destination: ${LOCAL_DEST}"

while true; do
  if remote "pgrep -af 'trainer/sft/train.py|torchrun|run_curated_v2_with_upload|upload_latest_weights.py' >/dev/null"; then
    remote "tail -n 5 '${REMOTE_LOG}' 2>/dev/null || true" | sed 's/^/[remote] /' | tee -a "${LOCAL_DEST}/watcher.log" >/dev/null
    sleep "${POLL_SECONDS}"
    continue
  fi
  break
done

log "No active remote SFT/upload process detected; checking upload result."

if ! remote "grep -q 'Selected weights directory:' '${REMOTE_LOG}'"; then
  log "HF upload success marker was not found. Leaving pod running for manual inspection."
  remote "tail -n 80 '${REMOTE_LOG}' 2>/dev/null || true" > "${LOCAL_DEST}/remote_log_tail_on_failure.txt" || true
  exit 2
fi

log "HF upload success marker found. Verifying files on Hugging Face before copying or terminating."

REMOTE_SELECTED_WEIGHTS="$(
  remote "grep 'Selected weights directory:' '${REMOTE_LOG}' | tail -1 | sed 's/.*Selected weights directory: //'"
)"
if [[ -z "${REMOTE_SELECTED_WEIGHTS}" ]]; then
  log "Could not parse selected weights directory from upload log. Leaving pod running."
  exit 3
fi

if ! remote "cd /home/ubuntu/prime_run && set -a && source .env && set +a && prime-rl/.venv/bin/python - '${HF_MODEL_REPO_ID}' '${REMOTE_SELECTED_WEIGHTS}' <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id = sys.argv[1]
weights_path = Path(sys.argv[2])
if not os.environ.get('HF_TOKEN'):
    raise SystemExit('HF_TOKEN is not set on pod')
expected = sorted(
    path.relative_to(weights_path).as_posix()
    for path in weights_path.rglob('*')
    if path.is_file()
)
if not expected:
    raise SystemExit(f'No local weight files found under {weights_path}')
if not any(path.endswith('.safetensors') for path in expected):
    raise SystemExit(f'No .safetensors files found under {weights_path}')
api = HfApi(token=os.environ['HF_TOKEN'])
uploaded = set(api.list_repo_files(repo_id=repo_id, repo_type='model'))
missing = [path for path in expected if path not in uploaded]
if missing:
    raise SystemExit(f'Missing {len(missing)} files on HF: {missing[:10]}')
print(f'Verified {len(expected)} files on HF repo {repo_id}')
PY"; then
  log "HF file verification failed. Leaving pod running for manual inspection."
  exit 4
fi

log "HF file verification passed. Copying artifacts."

mkdir -p "${LOCAL_DEST}/run_logs" "${LOCAL_DEST}/configs" "${LOCAL_DEST}/remote_output"

rsync -az --progress "${POD_HOST}:${REMOTE_LOG}" "${LOCAL_DEST}/run_logs/" | tee -a "${LOCAL_DEST}/watcher.log"
rsync -az --progress "${POD_HOST}:${REMOTE_CONFIG}" "${LOCAL_DEST}/configs/" | tee -a "${LOCAL_DEST}/watcher.log"

# Copy logs, W&B metadata, progress/checkpoint metadata, and lightweight files,
# but not the large checkpoint tensors already uploaded to Hugging Face.
rsync -az --progress \
  --exclude 'weights/**' \
  --exclude 'ckpt/**' \
  --exclude '*.safetensors' \
  --exclude '*.bin' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  "${POD_HOST}:${REMOTE_OUTPUT_DIR}/" \
  "${LOCAL_DEST}/remote_output/" | tee -a "${LOCAL_DEST}/watcher.log"

remote "find '${REMOTE_OUTPUT_DIR}/weights' -maxdepth 2 -type f -printf '%P\t%s bytes\n' 2>/dev/null || true" \
  > "${LOCAL_DEST}/weights_manifest.txt" || true

remote "grep -E 'View run at|Step [0-9]+ \\||Validation \\| Step|Selected weights directory:' '${REMOTE_LOG}' 2>/dev/null || true" \
  > "${LOCAL_DEST}/run_summary_lines.txt" || true

log "Artifacts copied."

if [[ "${TERMINATE_POD}" == "1" ]]; then
  log "Terminating pod ${POD_ID}."
  prime --plain pods terminate --yes "${POD_ID}" | tee -a "${LOCAL_DEST}/watcher.log"
  log "Terminate command issued."
else
  log "TERMINATE_POD=${TERMINATE_POD}; pod left running."
fi
