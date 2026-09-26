#!/usr/bin/env bash
# Resume SFT from a local checkpoint, train additional epochs on JarvisLabs,
# download the best checkpoint, merge on GPU, publish to Hugging Face, and
# destroy the JarvisLabs instances when each phase completes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${TUNELM_SFT_RESUME_CONFIG:-configs/gemma4_4b/sft_resume.yaml}"
GPU="${TUNELM_GPU:-A100-80GB}"
STORAGE="${TUNELM_STORAGE:-100}"
REGION="${TUNELM_REGION:-}"
LOG_DIR="${TUNELM_LOG_DIR:-results/jarvislabs}"
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/resume_sft_publish-${TIMESTAMP}.log"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

require_cmd jl

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  log "WANDB_API_KEY is required (set in .env.local)"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  log "HF_TOKEN is required to publish merged models (set in .env.local)"
  exit 1
fi

RESUME_CKPT="$(
  "$PY" - <<PY
from tunelm.config import load_config, repo_root

config = load_config(repo_root() / "${CONFIG}")
print(config.get("training", {}).get("resume_from_checkpoint", ""))
PY
)"

if [[ -z "$RESUME_CKPT" || ! -d "$ROOT/$RESUME_CKPT" ]]; then
  log "resume checkpoint missing: ${RESUME_CKPT:-<unset>}"
  exit 1
fi

RUN_ARGS=(run --config "$CONFIG" --gpu "$GPU" --storage "$STORAGE" --destroy --name tunelm-sft-resume)
PUBLISH_ARGS=(publish --config "$CONFIG" --scope best-and-final --gpu "$GPU" --storage "$STORAGE" --destroy --name tunelm-sft-publish)
if [[ -n "${TUNELM_MACHINE_ID:-}" ]]; then
  RUN_ARGS+=(--on "$TUNELM_MACHINE_ID")
  PUBLISH_ARGS+=(--on "$TUNELM_MACHINE_ID")
fi
if [[ -n "$REGION" ]]; then
  RUN_ARGS+=(--region "$REGION")
  PUBLISH_ARGS+=(--region "$REGION")
fi

log "logging to $LOG_FILE"
log "resume checkpoint: $RESUME_CKPT"
log "config: $CONFIG"

log "step 1/2: resume SFT training on JarvisLabs"
if ! "$PY" scripts/jarvislabs/cloud_train.py "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  log "training failed; see $LOG_FILE"
  exit 1
fi

log "step 2/2: merge best checkpoint and publish to Hugging Face"
if ! "$PY" scripts/jarvislabs/cloud_train.py "${PUBLISH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  log "publish failed; see $LOG_FILE"
  exit 1
fi

if [[ -f results/gemma4-4b-sft/best_manifest.json ]]; then
  log "best checkpoint manifest:"
  cat results/gemma4-4b-sft/best_manifest.json | tee -a "$LOG_FILE"
fi

if [[ -f results/published_models.jsonl ]]; then
  log "latest published model:"
  tail -n 1 results/published_models.jsonl | tee -a "$LOG_FILE"
fi

log "pipeline complete"
