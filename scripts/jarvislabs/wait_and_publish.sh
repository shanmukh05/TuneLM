#!/usr/bin/env bash
# Wait for a local cloud_train.py run PID, then merge/publish to Hugging Face.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PID="${1:?usage: wait_and_publish.sh TRAIN_PID}"
CONFIG="${TUNELM_SFT_RESUME_CONFIG:-configs/gemma4_4b/sft_resume.yaml}"
LOG_DIR="${TUNELM_LOG_DIR:-results/jarvislabs}"
LOG_FILE="${LOG_DIR}/publish-after-train.log"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

PY="${ROOT}/.venv/bin/python"
mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

log "waiting for training pid $PID"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done

if ! kill -0 "$PID" 2>/dev/null; then
  # Allow a moment for the process to flush logs/exit cleanly.
  sleep 2
fi
if kill -0 "$PID" 2>/dev/null; then
  log "training process $PID still running; skipping publish"
  exit 1
fi

log "training finished; publishing best + final checkpoints"
PUBLISH_ARGS=(publish --config "$CONFIG" --scope best-and-final --gpu "${TUNELM_GPU:-A100-80GB}" --storage "${TUNELM_STORAGE:-100}" --destroy --name tunelm-sft-publish)
if [[ -n "${TUNELM_REGION:-}" ]]; then
  PUBLISH_ARGS+=(--region "$TUNELM_REGION")
fi

"$PY" scripts/jarvislabs/cloud_train.py "${PUBLISH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
log "publish complete"
