#!/usr/bin/env bash
# Install Python 3.11, TuneLM training extras, Node.js Strudel deps on JarvisLabs.
#
# Lives next to cloud_train.py and is uploaded to the instance. Re-run is safe.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:-configs/gemma4_4b/smoke_sft.yaml}"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install 3.11
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.11 --seed
fi

uv pip install --python .venv/bin/python -e '.[train]'
uv pip install --python .venv/bin/python torch

if command -v npm >/dev/null 2>&1; then
  npm install
else
  echo "warning: npm not found; Strudel node backend may be unavailable"
fi

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
fi
.venv/bin/python scripts/train.py --config "$CONFIG" --dry-run
echo "remote setup ok"
