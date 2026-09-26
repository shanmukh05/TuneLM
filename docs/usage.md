# TuneLM usage

See [workflows.md](workflows.md) for Mermaid diagrams of SFT/RL data prep and training.
See [training-io.md](training-io.md) for the exact model-visible inputs and
assistant response contract for compose, edit, and repair.

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
| compose | `scene`, `musician`, `brief` |
| edit | `imperative`, `conversational` |
| repair | `direct`, `collaborative`, `diagnostic` |

Prompt authoring always calls a configured model; it does not fall back to
template text. Edit tasks keep `instruction` unchanged for rewards; only the
musician-facing `prompt` is rewritten.

`scene` and `musician` prompts are deliberately vague: the author never sees
tempo, instruments, or layer counts, so GRPO does not score those constraints
for them (see "Run GRPO"). `brief` is the only compose style that receives
`tempo_bpm`, `instruments`, `avoid_instruments`, and `min_layers`, and must
state them in a short user-style sentence ("Funky clap and slap bass groove
around 105 BPM, at least 3 layers"). Keep `brief` in `compose_styles` if you
want the constraint reward to train instruction following. The current
`datasets/rl/train.jsonl` was generated before `brief` existed; regenerate it to
get compose tasks with visible constraints.

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
0.4 × execution            (executes AND produces at least one event)
0.5 × constraints          (only constraints the prompt states)
0.5 × edit consistency     (edit examples only)
0.5 × repair consistency   (repair examples only)
0.2 × richness             (compose only: sound in every cycle + variation across cycles)
```

How the rewards stay honest:

- **Stated constraints only.** Before training, `prepare_rl_row` drops any
  constraint the prompt does not state: tempo needs a number followed by "BPM",
  instruments must be named (aliases and plurals count, e.g. "hi-hats", "claps"),
  and layer limits need an explicit count ("at least 3 layers"). Vague `scene` /
  `musician` prompts are therefore scored on execution and richness only. The
  task bank keeps the full verifier fields; only the reward view is pruned.
- **Playable, not just valid.** A program that runs but is silent earns no
  execution, constraint, edit, repair, or richness reward.
- **Features come from what plays.** With the Node backend, `instruments` lists
  sounds that produce audible events, so unused `const`s or `gain(0)` layers do
  not satisfy an instrument constraint. `layer_count` is the widest `stack(...)`
  argument count, not the number of quoted strings.
- **Richness** (`src/tunelm/rewards/richness.py`) averages coverage (share of the
  queried cycles with sound) and variation (distinct per-cycle patterns, full
  credit at 4). It counters the pull toward short one-bar loops that execution
  alone creates. Keep its weight small; random functions can inflate variation.

The full GRPO configs query **8 cycles** (`strudel.cycles: 8`,
`max_events: 8192`) so later `arrange`/`<...>` sections are executed and
scored, and sample **8 completions for each of 4 prompts** per optimizer step
(`per_device_train_batch_size: 4`, `gradient_accumulation_steps: 8`,
`num_generations: 8`). TRL's default `beta: 0` (no KL term) is kept on purpose:
with a PEFT adapter, TRL's reference model is the base model with the adapter
disabled, so a KL term would pull toward base Gemma rather than the SFT policy.

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
- `.env` / `.env.local` with `WANDB_API_KEY` for full runs; optionally set
  `WANDB_ENTITY`, `WANDB_RUN_GROUP`, and `HF_TOKEN`. Full configs set
  `training.wandb_project` (default `tunelm`) and `training.run_name` for the
  W&B project and run title. Shell env vars override the YAML when both are set.
- A registered SSH key is needed for JarvisLabs VMs, not the default PyTorch
  container. The `jl` CLI handles SSH authentication for containers.

```bash
python3 scripts/jarvislabs/cloud_train.py self-check \
  --config configs/gemma4_4b/sft.yaml

# SFT smoke on the default A100-80GB
python3 scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/smoke_sft.yaml

# Full SFT; on success downloads checkpoints/, promotes the lowest-loss
# checkpoint to results/<run_name>/best/, then pauses the instance
python3 scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/sft.yaml

# Same run, destroy the instance only after a successful download
python3 scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/sft.yaml \
  --destroy

# Full GRPO after inspecting the downloaded SFT adapter
python3 scripts/jarvislabs/cloud_train.py run \
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
| `fetch --on MACHINE_ID` | Download `checkpoints/` and copy best adapter to `results/<run_name>/best/` |
| `publish --scope best-and-final` | Merge best + final adapters on JarvisLabs and push to private Hugging Face repos |
| `status --on MACHINE_ID` | Show saved session and instance status |
| `down --on MACHINE_ID` | Pause instance (`--destroy` to delete) |

Session state is stored in `results/.jarvislabs_session.json`.

The default new instance is one on-demand `A100-80GB` with 100 GB storage.
The 80 GB headroom is intentional for Gemma E4B's 8B total weights, 8,192-token
SFT examples, and eight-way GRPO generation for four prompts per step. Run `jl gpus` before provisioning
to check current availability and your account's local-currency price. As of
September 2026, [JarvisLabs pricing](https://jarvislabs.ai/pricing) lists
$1.49 or [₹141](https://jarvislabs.ai/in) per GPU-hour, plus storage at
$0.10/GB-month or ₹0.013/GB-hour. A preliminary **20–80 GPU-hour** allowance
for both stages is about **$30–$120 or ₹2,820–₹11,280 compute**, before storage,
taxes, or extra experiments. This is a planning range, not a measured runtime;
use the smoke run's tokens/second and GRPO seconds/step to revise it. A paused
100 GB instance continues to cost about $10/month or ₹31/day until destroyed.
`--destroy` is honored only after a successful run and checkpoint download;
otherwise the instance is paused so artifacts remain recoverable.

The 5,000-row SFT dataset was measured with the Gemma E4B tokenizer and chat
template: maximum prompt + completion is 6,513 tokens. All ten smoke SFT
examples exceed 2,048 tokens; the 4,096-token smoke limit now fits all ten.
The 8,192-token full SFT limit fits all 5,000. RL prompts peak at 2,303 tokens;
the 2,048-token GRPO completion cap covers 97.8% of the SFT compose response
lengths and keeps the maximum prompt + completion at 4,351, far below Gemma's
native 128K context. `model.max_seq_length` is SFT-only; current TRL GRPO
uses `training.max_completion_length` and the model's native context. Longer sampled
GRPO completions are masked rather than optimized as truncated outputs.
Provider `max_tokens` in `configs/data/` controls generation during dataset
creation and is not a model training context limit.
The Qwen3-0.6B tokenizer independently measured a 6,427-token SFT maximum;
its full and smoke SFT limits were raised to 8,192 and 4,096 as well.
The densest SFT example has 270 events over two verifier cycles, so the eight
cycles used by GRPO and evaluation stay under the 8,192-event cap. The GRPO
Node verifier timeout is five seconds; cloud timings still need checking.

## Publish and Modal inference

Keep merged Gemma weights off your laptop. Download adapters only (~35–83 MB),
merge on a JarvisLabs GPU, publish each meaningful checkpoint to its own
**private** Hugging Face repo, then point Modal at that repo id.

Prerequisites:

- `HF_TOKEN` with write access in `.env` / `.env.local` (JarvisLabs publish +
  Modal private-repo download)
- Optional `HF_USERNAME` when your token namespace differs from the repo owner
- [Modal](https://modal.com/) account with GPU billing: `pip install -e '.[modal]'`
  then `modal setup`. Add a payment method in Modal settings before deploying GPU
  endpoints (`A10G` is used by default).
- Modal secret named `tunelm` containing at least `HF_TOKEN`

### 1. Publish merged checkpoints from JarvisLabs

After SFT finishes and adapters are fetched locally, publish the best-by-eval
and final checkpoints (at most two repos per run):

```bash
python3 scripts/jarvislabs/cloud_train.py publish \
  --scope best-and-final \
  --config configs/gemma4_4b/sft.yaml \
  --on MACHINE_ID
```

Or publish one adapter directory:

```bash
python3 scripts/jarvislabs/cloud_train.py publish \
  --scope single \
  --adapter results/gemma4-4b-sft/best \
  --tag best \
  --step 564 \
  --on MACHINE_ID
```

Each publish job uploads only the adapter (~100 MB) to the instance, downloads
the base model to ephemeral disk, merges, and pushes ~16 GB to Hugging Face.
Repo ids default to `{username}/tunelm-{run_name}-YYYYMMDD-{step}-{best|final}`.
Successful publishes append one JSON line per repo to
`results/published_models.jsonl`.

GRPO still uses the local adapter path from `configs/gemma4_4b/grpo.yaml`.
Published HF repos are for merged inference and Modal serving only.

### 2. Deploy Modal endpoint from Hugging Face

Set the repo id from the publish output (example):

```bash
export TUNELM_HF_REPO=your-username/tunelm-gemma4-4b-sft-20250920-564-best
# optional: export TUNELM_HF_REVISION=abc123

modal deploy scripts/modal/app.py
```

The Modal class loads `TUNELM_HF_REPO` at container start via
`AutoModelForCausalLM.from_pretrained`. Redeploy with a different repo id to
switch checkpoints. Strudel validation in the endpoint defaults off; pass
`"run_validation": true` in the POST body or set `TUNELM_VALIDATE=1` when deploying.
Default generation uses `7168` new tokens (`8192` SFT max seq length minus
prompt headroom); retries once at `8192` when JSON parsing fails.

### 3. Call the endpoint locally

```bash
python3 scripts/modal/infer.py "slow piano at dawn" \
  --url https://your-workspace--tunelm-tunelminference-generate.modal.run

# Batch from a TuneLM JSONL file (defaults to datasets/sft/sample.jsonl).
# Writes train.jsonl-compatible rows to results/modal_inference.jsonl.
python3 scripts/modal/infer.py --samples --url https://...

# Or edit MANUAL_PROMPTS at the top of scripts/modal/infer.py and run:
python3 scripts/modal/infer.py --url https://...
```

Response shape matches [`tunelm.inference.generate`](src/tunelm/inference.py):
parsed Strudel JSON plus optional static validation when requested.

## Troubleshooting

`Cannot find package '@strudel/core'`
: Run `npm install` from the repository root.

`..._API_KEY is required`
: Export the key or disable that provider in the data config.

CUDA/bitsandbytes errors
: Install CUDA-matched `torch` and `bitsandbytes` on the GPU machine.

Many valid-syntax programs score zero constraints
: Inspect returned `features`. The analyzer only verifies explicit tempo,
  instruments/samples that actually sound, and `stack(...)` layer count.

Compose tasks never get a constraint reward during GRPO
: Their prompts do not state the constraints (vague `scene`/`musician` styles).
  Regenerate the RL bank with the `brief` compose style enabled.
