# TuneLM

TuneLM trains small language models to **compose, edit, and repair** executable
[Strudel](https://strudel.cc/) music programs. Completions are JSON: a music
plan, inspectable controls, and Strudel code that a verifier can run. Audio
rendering and subjective quality rewards are out of scope for this milestone.

Data creation, SFT, RL task banks, GRPO, and evaluation are separate pipelines.
YAML configs select the model and stage; Python owns schemas, execution, and
rewards.

| Doc | Contents |
|-----|----------|
| [docs/usage.md](docs/usage.md) | Setup, generation, training, eval, JarvisLabs |
| [docs/workflows.md](docs/workflows.md) | Staged SFT/RL diagrams |
| [docs/implementation.md](docs/implementation.md) | Architecture and contracts |

## What the model emits

Every completion is one JSON object (`src/tunelm/schemas.py`). Missing or extra
fields are rejected.

```json
{
  "plan": {
    "tempo": 120,
    "instruments": ["bd", "sd", "bass"],
    "structure": "intro, build, ending",
    "musical_intent": "driving electronic groove"
  },
  "controls": {
    "bpm": 120,
    "key": "C:minor",
    "mood": ["driving"],
    "layers": [
      {"id": "drums", "role": "rhythm", "sounds": ["bd", "sd"]},
      {"id": "bass", "role": "bass", "sounds": ["bass"]}
    ]
  },
  "strudel_code": "setcpm(120/4)\nstack(s(\"bd sd\"), note(\"c2 g2\").s(\"gm_electric_bass_finger\"))"
}
```

SFT labels, GRPO rollouts, and inference share this contract and the same
system prompt.

## Tasks

| Type | What the model does | How success is scored |
|------|---------------------|------------------------|
| **Compose** | Write a new pattern from a natural-language request | Execution + optional tempo/instrument/layer constraints |
| **Edit** | Change existing Strudel in a requested direction | Execution + directional edit consistency |
| **Repair** | Fix broken Strudel | Execution + overlap with a valid reference |

User-facing prompts stay natural language. Machine-verifiable constraints live
on the task row for validation and RL rewards; they are not stuffed into the
training user message.

## Pipeline

```text
vocab + teachers  →  SFT JSONL  →  SFT LoRA
procedural tasks  →  RL JSONL   →  GRPO  →  benchmark eval
```

1. Generate teacher-validated SFT rows (`configs/data/sft.yaml`).
2. Generate an RL task bank with no solutions (`configs/data/rl_tasks.yaml`).
3. Supervised-fine-tune Gemma 4 2B (default) or Qwen3 0.6B.
4. GRPO against format, execution, constraint, edit, and repair rewards.
5. Score a fixed 500-task benchmark (`configs/data/benchmark.yaml`).

Teacher solutions are never stored in RL files. Reward functions do not import
trainer internals.

## Setup

Python 3.11+, Node.js 18+, and (for the supplied 4-bit configs) NVIDIA CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
npm install
pytest
```

Install extras when needed:

```bash
pip install -e '.[providers,train]'   # teacher APIs + SFT/GRPO
pip install -e '.[unsloth]'           # optional loader; set training.backend: unsloth
pip install torch bitsandbytes        # quantized GPU training
```

Confirm the Strudel executor (after `npm install`):

```bash
python - <<'PY'
from tunelm.strudel import StrudelExecutor
print(StrudelExecutor(backend="node").run('setcpm(120/4)\ns("bd sd")'))
PY
```

`backend="static"` is for schema and reward work without Node. Training configs
can be checked without loading weights:

```bash
python scripts/train.py --config configs/gemma4_2b/smoke_sft.yaml --dry-run
```

## Generate data

Export API keys in the shell, never in YAML. Supported backends: Gemini, OpenAI,
Anthropic, OpenRouter, DeepSeek.

```bash
export GEMINI_API_KEY='...'

# SFT (5k default; staged prompts → samples → validate → regenerate)
python scripts/generate.py --config configs/data/sft.yaml

# RL task bank (3k default; no teacher solutions)
python scripts/generate.py --config configs/data/rl_tasks.yaml

# Held-out benchmark (500 tasks, disjoint from the RL train set)
python scripts/generate.py --config configs/data/benchmark.yaml
```

`scripts/validate_data.py` checks JSONL schema, mix, and optional execution.
Smoke training without keys uses `datasets/sft/sample.jsonl` and
`datasets/rl/sample.jsonl`.

## Train and evaluate

Configs live under `configs/<model>/` and `extends: model.yaml`.

| Model | Hugging Face id | Role |
|-------|-----------------|------|
| Gemma 4 2B | `google/gemma-4-E2B-it` | Default target |
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | Smaller text-only baseline |

```bash
python scripts/train.py --config configs/gemma4_2b/sft.yaml
python scripts/train.py --config configs/gemma4_2b/grpo.yaml
python scripts/train.py --config configs/eval/default.yaml \
  --checkpoint checkpoints/gemma4-2b-grpo
python scripts/infer.py "Create a slow piano piece with strings" \
  --checkpoint checkpoints/gemma4-2b-grpo
```

`generate.py` and `train.py` route from the YAML shape (`tunelm-generate` /
`tunelm-train` are the same entry points). Cloud GPU: see the JarvisLabs
section in [docs/usage.md](docs/usage.md).

Default GRPO weights: format 0.1, execution 0.4, constraints 0.5, plus 0.5 edit
or repair consistency on those task types.

## Layout

```text
configs/          per-model training YAML and data/eval configs
datasets/         generated JSONL (smoke fixtures are committed)
src/tunelm/       schemas, Strudel executor, rewards, SFT/RL/train
scripts/          generate.py, train.py, validate_data.py, infer.py
tests/            verifier contract (parse, execute, reward)
```

## Security

Candidate Strudel runs in a short-lived Node process with a timeout and obvious
I/O blocked. That is not a security boundary. Treat model output as untrusted
code; use an OS/container sandbox for production or third-party inputs.

## License

AGPL-3.0-or-later.
