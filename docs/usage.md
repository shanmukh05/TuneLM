# TuneLM command guide

Run commands from the repository root unless noted. This is the command reference
for data, SFT, GRPO, evaluation, publishing, and inference. The model input/output contract is in
[training-io.md](training-io.md), and the pipeline diagrams are in
[workflows.md](workflows.md).

## Setup

Python 3.11+, Node.js 18+, and npm are required. Training and model generation
need a CUDA GPU. Dataset checks and scoring saved predictions can run locally
without one. Keep provider and Hugging Face tokens in environment variables or
local `.env`/`.env.local`, never in config files.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train]'
npm install
```

Install `pip install -e '.[providers]'` for
API-backed dataset generation and `pip install -e '.[modal]'` for Modal
deployment. Use a CUDA-compatible PyTorch build on a GPU host. `npm install` provides
the Node Strudel verifier; its execution/playability check is not an audio
render or a listening-quality judgment. `python scripts/train.py --config
CONFIG --dry-run` checks training inputs without loading weights or starting a
GPU run.

| Entry point | Commands |
|---|---|
| `scripts/generate.py --config CONFIG` | Create RL tasks, benchmark tasks, or SFT teacher data, routed by config |
| `scripts/prepare_quality_data.py sft/rl` | Refilter SFT examples or add visible RL briefs |
| `scripts/build_quality_benchmark.py` | Add fixed user-style requests to the procedural benchmark |
| `scripts/validate_data.py` | Check dataset schema, diversity, disjointness, and optional execution |
| `scripts/train.py --config CONFIG` | Train SFT/GRPO or evaluate predictions, routed by config |
| `scripts/infer.py PROMPT --checkpoint MODEL` | Local/GPU inference with Node compile-and-retry |
| `scripts/jarvislabs/cloud_train.py` | Cloud train, benchmark, publish, fetch, status, and down |
| `scripts/modal/app.py` and `scripts/modal/infer.py` | Deploy a Hub model and call its endpoint |

After installation, `tunelm-generate --config CONFIG` and
`tunelm-train --config CONFIG` are equivalent to `scripts/generate.py` and
`scripts/train.py` respectively. The YAML determines which pipeline runs.

## Prepare data

The current quality artifacts already exist: `datasets/sft/quality_train.jsonl`
(3,914 rows), `datasets/sft/quality_val.jsonl` (398),
`datasets/rl/quality_train.jsonl` (3,000), and
`datasets/evaluation/benchmark_ready.jsonl` (220). Do not regenerate or edit
the frozen benchmark while comparing checkpoints. Check the prepared files:

```bash
python scripts/validate_data.py --input datasets/sft/quality_train.jsonl --kind sft
python scripts/validate_data.py --input datasets/sft/quality_val.jsonl --kind sft
python scripts/validate_data.py --input datasets/evaluation/benchmark_ready.jsonl \
  --kind rl --disjoint-from datasets/rl/quality_train.jsonl \
  --min-unique-prompts 0.95
```

### RL task banks

The authored task configs require `GEMINI_API_KEY`. `rl_tasks.yaml` is the
original bank; `rl_quality_tasks.yaml` is a separate, more brief-heavy bank.
The quality-training config currently uses the curated bank below, not the
newly generated one.

```bash
export GEMINI_API_KEY='...'
python scripts/generate.py --config configs/data/rl_tasks.yaml
python scripts/validate_data.py --input datasets/rl/train.jsonl --kind rl \
  --diversity-config configs/data/diversity_thresholds.json

# No provider call: add explicit targets to a seeded share of the existing bank.
python scripts/prepare_quality_data.py rl \
  --source datasets/rl/train.jsonl --output datasets/rl/quality_train.jsonl

# Optional fresh bank. Inspect its prompts before selecting it for GRPO.
python scripts/generate.py --config configs/data/rl_quality_tasks.yaml
```

These commands refuse to replace prepared quality output unless you pass
`--overwrite`. RL generation supports `stage: prompts`, `tasks`, or `full` in
its YAML; `prompts` lets you review `prompts_file` before `tasks` finalizes it.
The current `rl_tasks.yaml` defaults to `full` and `append: true`.

### SFT teacher data

`configs/data/sft.yaml` currently defaults to `stage: samples` and synchronous
generation, so it expects an existing `prompts_file`. For a fresh staged run,
use `configs/data/sft_quality.yaml` and set its `stage` successively to
`prompts`, `samples`, `validate`, and `regenerate`. The last two stages run the
Node verifier. Teacher/provider settings live in that YAML.

```bash
export GEMINI_API_KEY='...'
# With stage: prompts in configs/data/sft_quality.yaml
python scripts/generate.py --config configs/data/sft_quality.yaml
# Change stage to samples, then run the same command.
python scripts/generate.py --config configs/data/sft_quality.yaml
# Change stage to validate, then run the same command.
python scripts/generate.py --config configs/data/sft_quality.yaml
# Optional: change stage to regenerate and run it again for failed rows.
python scripts/generate.py --config configs/data/sft_quality.yaml
```

To recheck the original SFT rows under the stronger quality rules, use separate
accepted and rejected outputs. The current quality files were produced this
way, so use `--overwrite` only when intentionally rebuilding them:

```bash
python scripts/prepare_quality_data.py sft --source datasets/sft/train.jsonl \
  --output datasets/sft/quality_train.jsonl \
  --rejected results/sft_quality_rejected.jsonl
python scripts/prepare_quality_data.py sft --source datasets/sft/val.jsonl \
  --output datasets/sft/quality_val.jsonl \
  --rejected results/sft_quality_val_rejected.jsonl
```

For an additional behavioral check of any SFT file, add `--check-execution
--backend node --validation-config configs/data/sft.yaml` to `validate_data.py`.
That CLI currently uses the executor's default two-cycle window; use
`prepare_quality_data.py sft` for the stricter eight-cycle quality pass. Either
way, re-executing every row can take several minutes.

### Frozen benchmark

Only run these when creating the benchmark for the first time. The first
command generates 200 procedural tasks; the second adds 20 fixed user-style
requests. Keep `benchmark_ready.jsonl` unchanged for checkpoint comparisons.

```bash
python scripts/generate.py --config configs/data/benchmark.yaml
python scripts/build_quality_benchmark.py
python scripts/validate_data.py --input datasets/evaluation/benchmark_ready.jsonl \
  --kind rl --disjoint-from datasets/rl/quality_train.jsonl \
  --min-unique-prompts 0.95
```

## Train on a local GPU

The quality SFT config loads the existing `checkpoints/gemma4-4b-sft` adapter
as trainable weights and starts a new optimizer run on the curated data. It is
not a resume of the old optimizer/scheduler state. Gemma GRPO instead starts
from the public, merged quality SFT model named in `configs/gemma4_4b/grpo.yaml`.
It downloads that model from Hugging Face and trains a new LoRA adapter on top;
the local SFT adapter is not needed for GRPO. Change `model.name` in that config
to evaluate another merged SFT checkpoint as the RL starting point.

```bash
# Small integration checks (CUDA required for actual training).
python scripts/train.py --config configs/gemma4_4b/smoke_sft.yaml --dry-run
python scripts/train.py --config configs/gemma4_4b/smoke_grpo.yaml --dry-run

# Original SFT, if starting from the base model.
python scripts/train.py --config configs/gemma4_4b/sft.yaml --dry-run
python scripts/train.py --config configs/gemma4_4b/sft.yaml

# Continue from the existing SFT adapter on the curated set.
python scripts/train.py --config configs/gemma4_4b/sft_quality.yaml --dry-run
python scripts/train.py --config configs/gemma4_4b/sft_quality.yaml

# Run a 60-step GRPO pilot; inspect task-specific validation before a full run.
python scripts/train.py --config configs/gemma4_4b/grpo_pilot.yaml --dry-run
python scripts/train.py --config configs/gemma4_4b/grpo_pilot.yaml
python scripts/train.py --config configs/gemma4_4b/grpo.yaml
```

GRPO holds out 30 of the 3,000 RL tasks using the configured seed, stratified
to 18 compose, 6 edit, and 6 repair tasks. Held-out rows are excluded before
the remaining compose rows are repeated twice for training (75% compose
exposure). The 60-step pilot validates and saves at steps 12, 24, 36, 48, and
60; the full 200-step run validates and saves at steps 40, 80, 120, 160,
and 200. Both use two generations per validation prompt. Preflight requires
`save_steps == eval_steps` and enough checkpoint slots to retain every
validation point for best-checkpoint selection.

In W&B, TRL's `reward`/`eval_reward` is the **unweighted** total. The
counts-aware metrics `objective/weighted_reward` and
`eval_objective/weighted_reward` reflect the configured GRPO reward weights;
JarvisLabs selects the best new GRPO checkpoint by the latter. Individual
reward terms appear under `rewards/<function_name>/mean` and `/std` (with
`eval_` prefixes during validation), with `reward_terms/<name>/applicable_count`
showing how many completions each term scored. `task/<type>/...` metrics
separate compose, edit, and repair playability and reward; validation keys
begin with `eval_task/`. These counters replace TRL's `NaN` aggregates for
conditionally applicable rewards without treating inapplicable tasks as zero.
Older GRPO runs without a weighted metric still use `eval_reward` when fetched.
Do not compare the completed 2026-09-28 pilot's reward curve directly with a
run using the revised sampling mix. Review task-specific validation and the
frozen benchmark before scaling to the full run. For Qwen3
0.6B, use the matching `configs/qwen3_06b/{smoke_sft,sft,smoke_grpo,grpo}.yaml`
configs. To resume an *interrupted run of the same config and dataset*, set
`training.resume_from_checkpoint` to that run's checkpoint in YAML; do not set
it when beginning the new quality SFT run.

```bash
# Qwen3 baseline, if using that model instead of Gemma 4.
python scripts/train.py --config configs/qwen3_06b/sft.yaml
python scripts/train.py --config configs/qwen3_06b/grpo.yaml
```

## Evaluate locally or from Hugging Face

`configs/eval/default.yaml` uses the frozen 220-task benchmark and Node
execution. Use unique `--predictions` and `--output` paths per checkpoint and
decoding setting. `--checkpoint` accepts a local adapter directory or a merged
Hugging Face model repo ID. For private Hub repos, set `HF_TOKEN`. `--revision`
pins a Hub branch, tag, or commit for both model and tokenizer.

```bash
# Fast smoke run; then remove --limit for the full benchmark.
python scripts/train.py --config configs/eval/default.yaml \
  --checkpoint checkpoints/gemma4-4b-sft --limit 20 \
  --temperature 0 --max-new-tokens 3072 \
  --predictions results/pred_sft_smoke.jsonl \
  --output results/eval_sft_smoke.jsonl

# Same benchmark directly from a merged model uploaded to Hugging Face.
python scripts/train.py --config configs/eval/default.yaml \
  --checkpoint OWNER/MERGED_MODEL --revision COMMIT_OR_TAG \
  --temperature 0 --max-new-tokens 3072 \
  --predictions results/pred_hf_t0.jsonl \
  --output results/eval_hf_t0.jsonl

# Plan/code ablation with the same model; use new output paths.
python scripts/train.py --config configs/eval/default.yaml \
  --checkpoint OWNER/MERGED_MODEL --mode code-first \
  --predictions results/pred_hf_code_first.jsonl \
  --output results/eval_hf_code_first.jsonl
```

The scorer writes `OUTPUT.summary.json` and `OUTPUT.review.csv` alongside
`OUTPUT.jsonl` (for example, `results/eval_hf_t0.summary.json`). Fill the review
CSV's `audible` (0/1), `instruction_match` (1-5), and `musical_quality` (1-5)
columns after listening, then score the saved predictions again **without**
`--checkpoint` to aggregate manual scores:

```bash
python scripts/train.py --config configs/eval/default.yaml \
  --predictions results/pred_hf_t0.jsonl --output results/eval_hf_t0.jsonl
```

Automatic execution and form metrics are proxies, not listening
tests. Compare temperatures (`0`, `0.2`, `0.7`) and `combined` versus
`code-first` using distinct paths; avoid choosing by reward alone.

To listen to saved predictions, open `tools/sft_player.html` and select a
benchmark `predictions.jsonl` with **Load JSONL**. When opening the HTML file
directly, select `datasets/evaluation/benchmark_ready.jsonl` with **Load
benchmark tasks** to show prompts and edit/repair context. When served from
the repository root, the player loads that task file automatically. Malformed
completions remain visible as raw text for inspection.

Score already saved Modal JSONL without loading any model or GPU:

```bash
python scripts/train.py --config configs/eval/default.yaml \
  --tasks results/modal_inference_songs.jsonl \
  --predictions results/modal_inference_songs.jsonl \
  --output results/audit_songs.jsonl
```

## Infer locally

`scripts/infer.py` accepts a local checkpoint or merged Hub repo ID. It runs
Node Strudel verification and retries a parse/compile failure twice by default.
`--mode code-first` is an ablation using the same model, not a separately
trained planner/code model.

```bash
python scripts/infer.py 'Gentle piano that gradually builds' \
  --checkpoint checkpoints/gemma4-4b-sft-quality \
  --temperature 0.2 --compile-retries 2
python scripts/infer.py 'A short drum loop' \
  --checkpoint OWNER/MERGED_MODEL --mode code-first
```

`--no-validate` skips execution and retries only parsing failures. Local
inference needs a GPU for the supplied Gemma checkpoint; a Hugging Face model
still downloads its weights to the machine running this command.

## JarvisLabs: train and manage instances

Install the JarvisLabs CLI (`uv tool install jarvislabs`, then `jl setup`) and
check GPU availability/pricing with `jl gpus` before provisioning. The helper
stages only the selected data/adapter, runs remote setup, follows training,
downloads checkpoints/results, and pauses the GPU by default. Full configs use
W&B, so set `WANDB_API_KEY`; publishing models needs an `HF_TOKEN` with write access.

```bash
python scripts/jarvislabs/cloud_train.py self-check \
  --config configs/gemma4_4b/sft_quality.yaml
# Original SFT only when starting from the base model.
python scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/sft.yaml
python scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/sft_quality.yaml
python scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/grpo_pilot.yaml
# Only after the pilot improves held-out results:
python scripts/jarvislabs/cloud_train.py run \
  --config configs/gemma4_4b/grpo.yaml
```

The default is an A100-80GB with 100 GB storage. Use `--gpu GPU`, `--storage
GB`, or `--region REGION` to change provisioning. `--on MACHINE_ID` resumes a
paused machine; `--skip-setup` is for a machine already set up with the current
dependencies. `--keep` leaves the GPU running and accruing charges. Without
`--keep`, successful runs pause it after results are downloaded. `--destroy`
deletes the instance only after a successful training download.

| Command | Purpose |
|---|---|
| `cloud_train.py status --on MACHINE_ID` | Show instance and saved-run status |
| `cloud_train.py fetch --on MACHINE_ID --config CONFIG` | Fetch training checkpoints and promote the best adapter |
| `cloud_train.py down --on MACHINE_ID` | Pause the instance |
| `cloud_train.py down --on MACHINE_ID --destroy` | Permanently remove the instance |
| `cloud_train.py self-check --config CONFIG` | Verify the staged training payload locally |

Training fetches the highest weighted-validation-reward GRPO adapter (or
lowest-eval-loss SFT adapter) to
`results/<run_name>/best/`; the final adapter is under its configured
`checkpoints/` output. Session metadata is in
`results/.jarvislabs_session.json`.

## Publish a merged model to Hugging Face

The training artifact is a LoRA adapter. `publish` merges it with the base
model on a JarvisLabs GPU and uploads only the merged model plus tokenizer to
Hugging Face, not the adapter. Repos are private unless `--public` is set. The
publish manifest is appended to
`results/published_models.jsonl`. Set `HF_TOKEN` with write access first.

```bash
export HF_TOKEN='...'
# Publish one selected adapter under a chosen repo ID.
python scripts/jarvislabs/cloud_train.py publish --scope single \
  --adapter results/gemma4-4b-sft-quality/best \
  --repo-id OWNER/tunelm-sft-quality --public

# Or publish both best and final from a completed training run.
python scripts/jarvislabs/cloud_train.py publish --scope best-and-final \
  --config configs/gemma4_4b/sft_quality.yaml --public
```

`--on MACHINE_ID` reuses a GPU; otherwise `publish` creates and later pauses
one. `--scope single` may also point at `checkpoints/gemma4-4b-sft` or a
`checkpoint-N` adapter directory. On an already prepared GPU shell, the
lower-level equivalent is:

```bash
python scripts/jarvislabs/publish_model.py \
  --adapter checkpoints/gemma4-4b-sft-quality \
  --repo-id OWNER/tunelm-sft-quality --public
```

Do not pass a bare LoRA-only repo to the merged-model examples below.

## JarvisLabs: benchmark Hub checkpoints

`benchmark` stages the fixed evaluation set and code, not local weights. It
loads the merged model from Hugging Face on the GPU, scores predictions with
the Node verifier, downloads results, and pauses the instance. Give every
model/temperature/mode a unique label. For private or gated repos, `HF_TOKEN`
is forwarded from the shell or local `.env` to the machine.

```bash
# New machine and a 20-task smoke check.
python scripts/jarvislabs/cloud_train.py benchmark \
  --checkpoint OWNER/tunelm-sft --label sft-smoke-t0 --limit 20

# Reuse the printed ID. Default: full 220-task benchmark at temperature 0.
python scripts/jarvislabs/cloud_train.py benchmark --on MACHINE_ID --skip-setup \
  --checkpoint OWNER/tunelm-sft --revision COMMIT_OR_TAG --label sft-full-t0
python scripts/jarvislabs/cloud_train.py benchmark --on MACHINE_ID --skip-setup \
  --checkpoint OWNER/tunelm-sft-quality --label quality-full-t0 \
  --results-dir results/gemma4-4b-sft/benchmarks

# Same checkpoint, different decoding; keep outputs separate.
python scripts/jarvislabs/cloud_train.py benchmark --on MACHINE_ID --skip-setup \
  --checkpoint OWNER/tunelm-sft-quality --label quality-code-first \
  --temperature 0.2 --mode code-first --max-new-tokens 3072
```

Outputs are in `results/benchmarks/LABEL/` by default, or under
`--results-dir/LABEL/`: `predictions.jsonl`,
`evaluation.jsonl`, `evaluation.summary.json`, `evaluation.review.csv`, and
`run.json` (repo, revision, decoding settings, run ID). `--config` selects a
different evaluation YAML. `--detach` returns the run ID while leaving the GPU
running; unlike an attached run, it does not download results or pause the
instance automatically. `--keep` leaves the GPU running after an attached run,
and `--overwrite`
explicitly permits reusing a local result label. Do not use `--skip-setup` on a
new machine. A paused instance may still incur storage charges; use `down
--destroy` after its artifacts have been downloaded.
Each full run generates responses sequentially for 220 tasks; SFT eval loss
does not generate responses. Several benchmarks can share an A100-80GB, but
four merged E4B models use roughly 65 GB before generation overhead. Avoid
loading a fifth concurrently, and keep `--keep` on every shared-instance run
until all results have been downloaded.

After filling the downloaded review CSV, aggregate its human ratings locally
without renting the GPU again:

```bash
python scripts/train.py --config configs/eval/default.yaml \
  --predictions results/benchmarks/sft-full-t0/predictions.jsonl \
  --output results/benchmarks/sft-full-t0/evaluation.jsonl
```

## Modal: serve and inspect a Hub model

Modal serves a merged Hugging Face model, not the local LoRA adapter. Install
the Modal extra (see Setup), run `modal setup`, and create a Modal secret named
`tunelm` with `HF_TOKEN` for private repos. The deploy output gives the endpoint
URL. Set `TUNELM_HF_REVISION` to pin a model version.

```bash
export TUNELM_HF_REPO=OWNER/tunelm-sft-quality
# Optional: export TUNELM_HF_REVISION=COMMIT_OR_TAG
modal deploy scripts/modal/app.py

# One prompt, with client-side Node validation and compile retries.
python scripts/modal/infer.py 'Slow piano at dawn' --url https://YOUR-ENDPOINT \
  --output results/modal_one.jsonl --temperature 0.2 --compile-retries 2

# Batch from a TuneLM JSONL file; --samples alone uses datasets/sft/sample.jsonl.
python scripts/modal/infer.py --samples datasets/evaluation/benchmark_ready.jsonl \
  --url https://YOUR-ENDPOINT --output results/modal_benchmark.jsonl

# With no prompt or --samples, run the fixed prompts in eval/manual_prompts.py.
python scripts/modal/infer.py --url https://YOUR-ENDPOINT \
  --output results/modal_manual.jsonl
```

`--limit N`, `--max-new-tokens N`, and `--timeout SECONDS` control the client.
`--validate` also requests the endpoint's static check; the client's Node
result is the authoritative execution result. The saved rows include fresh
`metadata.valid`, validation details, and inference attempt counts. Score a
saved Modal batch with the `--tasks FILE --predictions FILE` evaluation command
above. The Modal endpoint does not render audio.

## Troubleshooting

- `Cannot find package '@strudel/core'`: run `npm install` at the repo root.
- Missing API key: export the provider key for data generation, `WANDB_API_KEY`
  for W&B-backed cloud training, or `HF_TOKEN` for private Hub models.
- CUDA/bitsandbytes failure: install a CUDA-compatible PyTorch stack on the GPU
  machine; local dataset checks do not require CUDA.
- Good automatic scores but weak music: review the generated Strudel in a
  browser player and fill the listening-review CSV. Execution and form scores
  do not measure taste or instruction fit reliably.
