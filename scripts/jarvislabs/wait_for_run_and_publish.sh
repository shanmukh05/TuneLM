#!/usr/bin/env bash
# Poll a JarvisLabs managed run, then fetch checkpoints and publish to Hugging Face.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${1:?usage: wait_for_run_and_publish.sh RUN_ID}"
CONFIG="${TUNELM_SFT_RESUME_CONFIG:-configs/gemma4_4b/sft_resume.yaml}"
MACHINE_ID="${TUNELM_MACHINE_ID:-}"
LOG_DIR="${TUNELM_LOG_DIR:-results/jarvislabs}"
LOG_FILE="${LOG_DIR}/publish-after-run.log"

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

state_for_run() {
  jl run status "$RUN_ID" --json 2>/dev/null | "$PY" -c "import json,sys; print(json.load(sys.stdin).get('state',''))"
}

log "waiting for run $RUN_ID"
while [[ "$(state_for_run)" == "running" ]]; do
  sleep 120
done

state="$(state_for_run)"
log "run $RUN_ID finished with state=$state"
if [[ "$state" != "succeeded" ]]; then
  log "run did not succeed; skipping fetch/publish"
  exit 1
fi

FETCH_ARGS=(fetch --config "$CONFIG")
PUBLISH_ARGS=(publish --config "$CONFIG" --scope best-and-final --gpu "${TUNELM_GPU:-A100-80GB}" --storage "${TUNELM_STORAGE:-100}" --destroy --name tunelm-sft-publish)
if [[ -n "$MACHINE_ID" ]]; then
  FETCH_ARGS+=(--on "$MACHINE_ID")
  PUBLISH_ARGS+=(--on "$MACHINE_ID")
fi
if [[ -n "${TUNELM_REGION:-}" ]]; then
  PUBLISH_ARGS+=(--region "$TUNELM_REGION")
fi

log "fetching checkpoints"
"$PY" scripts/jarvislabs/cloud_train.py "${FETCH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
log "publishing merged models"
"$PY" scripts/jarvislabs/cloud_train.py "${PUBLISH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
log "done"
