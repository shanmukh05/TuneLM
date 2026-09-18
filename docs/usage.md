# TuneLM usage

See [workflows.md](workflows.md) for Mermaid diagrams of SFT/RL data prep and training.

## Requirements

- Python 3.11+
- Node.js 18+
- An NVIDIA CUDA environment for the supplied 4-bit training configs
- One or more provider API keys only when generating SFT teacher solutions

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
npm install
```

### Local verification (no GPU, no model weights)

Confirm a training config without loading Gemma/Qwen checkpoints:

```bash
python scripts/train.py --config configs/gemma4_4b/smoke_sft.yaml --dry-run
python scripts/train.py --config configs/gemma4_4b/smoke_grpo.yaml --dry-run
pytest
```

Smoke configs are validated strictly; full SFT/GRPO configs warn when teacher
data or SFT adapters are not generated yet. Run real training on a cloud GPU
(see JarvisLabs section below).

Install provider and training dependencies when needed:

```bash
pip install -e '.[providers,train]'
```

Quantized training also needs a CUDA-compatible `torch` build. `bitsandbytes`
is included in the training extra:

```bash
pip install torch
```

## Scripts

| Script / entry point | Purpose |
|--------|---------|
| `scripts/generate.py` / `tunelm-generate` | RL tasks, benchmark, or SFT data |
| `scripts/train.py` / `tunelm-train` | SFT, GRPO, or evaluation |
| `scripts/validate_data.py` | JSONL schema and diversity checks |
| `scripts/infer.py` | Interactive checkpoint inference |

Both `generate.py` and `train.py` delegate to `src/tunelm/cli.py` and infer the
pipeline from the loaded YAML:

| Config shape | Routed to |
|--------------|-----------|
| `providers` present | SFT data generation |
| otherwise | RL / benchmark task generation |
| `tasks` + `predictions` present | Evaluation |
| `rewards` + `data.train_file` | GRPO training |
| `data.train_file` only | SFT training |

## Configuration layout

Training configs are grouped by model under `configs/<model>/`:

```
configs/
  gemma4_4b/          # default
    model.yaml        # shared HF name, quantization, LoRA defaults
    sft.yaml
    grpo.yaml
    smoke_sft.yaml
    smoke_grpo.yaml
  qwen3_06b/
    model.yaml
    sft.yaml
    grpo.yaml
    smoke_sft.yaml
    smoke_grpo.yaml
  data/               # dataset generation configs
  eval/
```

Each training config uses `extends: model.yaml`. SFT and GRPO configs are run
independently — inspect checkpoints between stages rather than chaining them
automatically.

| Model | HF checkpoint | Notes |
|-------|---------------|-------|
| Gemma 4 E4B (default) | `google/gemma-4-E4B-it` | About 8B total/4B effective parameters; text-only causal path |
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | Smaller text-only baseline |

## 1. Verify Strudel execution

```bash
python - <<'PY'
from tunelm.strudel import StrudelExecutor

result = StrudelExecutor(backend="node").run(
    'setcpm(120/4)\nstack(s("bd sd"), s("hh*8").gain(0.35))'
)
print(result.model_dump_json(indent=2))
PY
```

`valid: true`, non-empty `events`, and `backend: node` show that the real fast
executor is available. If npm dependencies cannot be installed, use
`backend="static"` for schema/reward development only.

## 2. Generate RL tasks

Task vocabularies and templates live in `configs/data/vocab/`. This includes
instruments and aliases, ensembles, moods, genres, rhythms, structures,
textures, tempo ranges, compose prompts, verifier-supported edit operations,
and executable repair templates. Each generated row stores the final varied
musician request directly in `prompt`.

Each canonical instrument has a `sounds` entry. Use the natural canonical name
in prompts and constraints; TuneLM supplies the mapped Strudel.cc identifier
(for example, `violin: gm_violin`, `shaker: sh`, or
`synth: gm_lead_2_sawtooth`) to the model. Add a vocabulary instrument only
when a built-in sample, registered GM SoundFont, or synthesizer sound can
implement it.

The authoritative naming references are Strudel's
[Samples](https://strudel.cc/learn/samples/) and
[Synths](https://strudel.cc/learn/synths/) documentation. Strudel.cc registers
its GM SoundFonts automatically; another host embedding Strudel must register
the same soundfonts before playing `gm_*` identifiers.

Generate the default RL task bank (3,000 tasks):

```bash
python scripts/generate.py --config configs/data/rl_tasks.yaml
python scripts/validate_data.py \
  --input datasets/rl/train.jsonl \
  --kind rl \
  --diversity-config configs/data/diversity_thresholds.json
```

### Staged RL generation (review prompts first)

When `batch.enabled: true` (default) and `prompt_author` is configured, styled
prompts are authored through the Gemini Batch API at 50% of standard pricing.
Set `batch.enabled: false` to use synchronous prompt authoring.

Set `stage` in `configs/data/rl_tasks.yaml`:

| `stage` | What it does |
|---|---|
| `prompts` | Batch prompt authoring → `prompts_file` |
| `tasks` | Copy curated `prompts_file` to `output` |
| `full` | Batch prompt authoring directly to `output` (default) |

```bash
# 1. Generate and inspect styled prompts
#    Set stage: prompts in configs/data/rl_tasks.yaml
python scripts/generate.py --config configs/data/rl_tasks.yaml
python scripts/validate_data.py \
  --input datasets/rl/prompts.jsonl \
  --kind rl

# Edit datasets/rl/prompts.jsonl: delete or edit rows as needed

# 2. Finalize the task bank
#    Set stage: tasks in configs/data/rl_tasks.yaml
python scripts/generate.py --config configs/data/rl_tasks.yaml
```

When `prompt_author` is configured, procedural skeletons (constraints, code,
verifier metadata) are generated first, then each row gets a randomly chosen
**style** prompt:

| Task type | Styles |
|---|---|
| compose | `scene`, `musician` |
| edit | `imperative`, `conversational` |
| repair | `direct`, `collaborative`, `diagnostic` |

Prompt authoring always calls a configured model; it does not fall back to
template text. Edit tasks keep `instruction` unchanged for rewards; only the
musician-facing `prompt` is rewritten.

Without `prompt_author`, legacy `mode: hybrid` still applies (`llm_fraction`
generates full task specs via LLM). The committed `datasets/rl/train.jsonl`
remains the smoke/reference bank for local tests.

Regenerate after changing vocab files under `configs/data/vocab/`.

Generate the 500-task benchmark after the train set:

```bash
python scripts/generate.py --config configs/data/benchmark.yaml
python scripts/validate_data.py \
  --input datasets/evaluation/benchmark.jsonl \
  --kind rl \
  --disjoint-from datasets/rl/train.jsonl \
  --min-unique-prompts 0.95 \
  --diversity-config configs/data/diversity_thresholds.json \
  --check-execution \
  --backend node
```

## 3. Generate validated SFT data

Set keys in the shell, never in YAML or committed files:

```bash
export GEMINI_API_KEY='...'
export OPENROUTER_API_KEY='...'   # optional
export DEEPSEEK_API_KEY='...'     # optional
```

Configure models in `configs/data/sft.yaml` under `models` (each YAML key is the
model id stored in `metadata.teacher`):

```yaml
models:
  gemini-flash:
    enabled: true
    provider: gemini
    model: gemini-2.5-flash
  deepseek-chat:
    enabled: true
    provider: deepseek
    model: deepseek-chat
  openrouter-gemini:
    enabled: true
    provider: openrouter
    model: google/gemini-2.5-flash
parallel_models: true
parallel_workers: 4
```

Batch mode (`batch.enabled: true`, default) uses the Gemini Batch API for prompt
authoring and teacher generation at 50% of standard pricing. Batch mode requires
exactly one enabled Gemini model for each stage (`compose.prompt_author.models`
for prompts, top-level `models` for samples/regenerate).

Then run:

```bash
python scripts/generate.py --config configs/data/sft.yaml
python scripts/validate_data.py \
  --input datasets/sft/train.jsonl \
  --kind sft \
  --check-execution \
  --backend node \
  --validation-config configs/data/sft.yaml
```

### Batch staged workflow

Set `stage` in `configs/data/sft.yaml`:

| `stage` | What it does |
|---|---|
| `prompts` | Batch prompt authoring → `prompts_file` |
| `samples` | Batch teacher generation → `samples_file` (`metadata.valid` unset) |
| `validate` | Strudel validation; sets `metadata.valid`; exports valid rows to `output` |
| `regenerate` | Batch retry rows where `metadata.valid: false` (up to `max_attempts`) |
| `full` | Run prompts → samples → validate → regenerate loop (default) |

`responses` is an alias for `samples`.

```bash
# 1. Batch-generate prompts
#    stage: prompts
python scripts/generate.py --config configs/data/sft.yaml

# Optional: review datasets/sft/prompts.jsonl

# 2. Batch-generate teacher samples (no Strudel check yet)
#    stage: samples
python scripts/generate.py --config configs/data/sft.yaml

# 3. Validate every sample and export passing rows
#    stage: validate
python scripts/generate.py --config configs/data/sft.yaml

# 4. Retry failed samples, then re-validate
#    stage: regenerate
python scripts/generate.py --config configs/data/sft.yaml
```

`append: true` resumes into an existing `prompts_file`, `samples_file`, or
`output` JSONL: only the remaining rows up to `count` are generated, and task
ids already on disk are skipped.

Set `batch.enabled: false` to use synchronous generation. With
`parallel_models: true`, teacher generation shards tasks across all enabled
models and runs them concurrently.

Each RL task row stores `prompt_author` (`model_id`, `provider`, `model`,
optional `style`). Each SFT row stores the same under `metadata.prompt_author`
and teacher provenance under `metadata.teacher`, `metadata.teacher_provider`,
and `metadata.teacher_model`.

Compose SFT acceptance requires executable `strudel_code` and `controls`
consistent with execution. Hidden brief constraints are recorded in
`constraint_score` but do not block teacher rows; RL/GRPO still scores
constraints.

SFT compose prompts use a two-step pattern:

1. **Brief** — each compose row samples level 1–4 (biased toward vaguer levels
   via `compose.prompt_level_weights`), mood, instruments, tempo, and optional
   hints procedurally. Tempo and instruments feed **constraints** for validation
   and rewards only.
2. **Prompt author** — models under `compose.prompt_author.models` turn scene
   hints (mood, atmosphere, creative hints) into a short user-style request.
   The author never sees BPM, instrument lists, or constraint JSON. Training
   messages match inference: the response model sees only the natural-language
   prompt (plus edit/repair code context when applicable).

Prompt authoring can use a separate `compose.prompt_author.models` router from
teacher `models`. Legacy `providers:` blocks are still accepted and are converted
to model ids automatically.

For a local SFT smoke test without API keys, use `datasets/sft/sample.jsonl` with
`configs/gemma4_4b/smoke_sft.yaml` or `configs/qwen3_06b/smoke_sft.yaml`.

## 4. Run SFT

Dry-run locally before launching on a GPU:

```bash
python scripts/train.py --config configs/gemma4_4b/smoke_sft.yaml --dry-run
```

```bash
python scripts/train.py --config configs/gemma4_4b/sft.yaml
```

Checkpoints land in `checkpoints/gemma4-4b-sft/`. For a quick smoke run:

```bash
python scripts/train.py --config configs/gemma4_4b/smoke_sft.yaml
```

## 5. Run GRPO

Generate the RL task bank and an SFT checkpoint first. Dry-run locally:

```bash
python scripts/train.py --config configs/gemma4_4b/smoke_grpo.yaml --dry-run
```

Then on a GPU:

```bash
python scripts/train.py --config configs/gemma4_4b/grpo.yaml
```

Smoke test:

```bash
python scripts/train.py --config configs/gemma4_4b/smoke_grpo.yaml
```

Default rewards:

```text
0.1 × format
0.4 × execution
0.5 × constraints
0.5 × edit consistency (edit examples only)
0.5 × repair consistency (repair examples only)
```

## 6. Evaluate a checkpoint

```bash
python scripts/train.py \
  --config configs/eval/default.yaml \
  --checkpoint checkpoints/gemma4-4b-grpo
```

Use `--limit 10` for a smoke test. Outputs:

- `results/evaluation.jsonl`: per-task reward breakdown
- `results/evaluation.summary.json`: metric means and task-type slices

## 7. Interactive checkpoint inference

```bash
python scripts/infer.py \
  "Create a slow emotional piano piece with subtle strings" \
  --checkpoint checkpoints/gemma4-4b-grpo
```

## JarvisLabs training

Run SFT and GRPO as explicit, inspectable stages. Each run uploads the dataset
selected by its config; GRPO also uploads its local SFT adapter when present.

Prerequisites:

- [JarvisLabs CLI](https://docs.jarvislabs.ai/): `uv tool install jarvislabs` then `jl setup`
- SSH key: `jl ssh-key add ~/.ssh/id_ed25519.pub`
- `.env` / `.env.local` with `WANDB_API_KEY`; optionally set `WANDB_PROJECT`
  (defaults to `tunelm`), `WANDB_ENTITY`, `WANDB_RUN_GROUP`, and `HF_TOKEN`

```bash
python3 scripts/jarvislabs/cloud_train.py self-check \
  --config configs/gemma4_4b/sft.yaml

# SFT smoke on L4
python3 scripts/jarvislabs/cloud_train.py run \
  --gpu L4 \
  --config configs/gemma4_4b/smoke_sft.yaml

# Full SFT; successful runs download to checkpoints/ and results/
python3 scripts/jarvislabs/cloud_train.py run \
  --gpu L4 \
  --config configs/gemma4_4b/sft.yaml

# Full GRPO after inspecting the downloaded SFT adapter
python3 scripts/jarvislabs/cloud_train.py run \
  --gpu A100 \
  --config configs/gemma4_4b/grpo.yaml \
  --keep
```

Training routes from config contents, not the filename. A fresh GRPO instance
requires the SFT adapter under the configured local `checkpoints/` path. Reusing
the SFT machine with `--on MACHINE_ID` also preserves its remote checkpoint.
Metrics and sampled GRPO completions go to W&B for the full configs. Set
`WANDB_LOG_MODEL=checkpoint` only when checkpoint artifact uploads are desired;
the launcher always downloads checkpoint files directly. Other commands:

| Command | Purpose |
|---------|---------|
| `fetch --on MACHINE_ID` | Download into local `checkpoints/` and `results/` |
| `status --on MACHINE_ID` | Show saved session and instance status |
| `down --on MACHINE_ID` | Pause instance (`--destroy` to delete) |

Session state is stored in `results/.jarvislabs_session.json`.

## Troubleshooting

`Cannot find package '@strudel/core'`
: Run `npm install` from the repository root.

`..._API_KEY is required`
: Export the key or disable that provider in the data config.

CUDA/bitsandbytes errors
: Install CUDA-matched `torch` and `bitsandbytes` on the GPU machine.

Many valid-syntax programs score zero constraints
: Inspect returned `features`. The analyzer only verifies explicit tempo,
  instruments/samples, and approximate layers.
