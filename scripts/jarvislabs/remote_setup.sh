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

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 ||
   ! node -e 'if (Number(process.versions.node.split(".")[0]) < 18) process.exit(1)'; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Node.js/npm is required; install Node 22 before training" >&2
    exit 1
  fi
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi
node -e 'if (Number(process.versions.node.split(".")[0]) < 18) process.exit(1)'
npm ci

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
fi
.venv/bin/python scripts/train.py --config "$CONFIG" --dry-run
echo "remote setup ok"
