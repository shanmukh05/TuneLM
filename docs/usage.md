# Running TuneLM

Use Python 3.11+, Node 22+, and a repository checkout. All commands below are
explicit actions; installation/verification does not start training or paid APIs.
The legacy assets live under ignored `archive/` and are never pipeline outputs.

## Dataset creation in plain language

We start with **working music, then turn it into teaching examples**:

1. Run the seed collector to download the configured source examples. With
   `--ingest`, it checks the code and extracts parts such as drums, bass, and
   chords. You can also supply your own source JSONL to `tunelm ingest`.
2. Run `build-memory` to make those parts searchable. This is a reference library
   for generation, not another training dataset to edit by hand.
3. Choose teachers, their shares, and the total sample count in
   `configs/generation.yaml`, then run `generate`. It either takes a working song
   as-is or combines verified parts into a new composition. The teacher writes
   a matching user request and five short musical decisions; it preserves the
   original code for existing songs and may adapt code for new combinations.
4. TuneLM checks each answer. Passing examples are saved; failed answers get
   correction retries and may be rejected. Your model shares count passing
   examples: 200 samples at 40/30/30 means targets of 80/60/60.
5. The result is plain JSONL: SFT gets requests plus answers; RL gets requests
   and reference context so the model can practise producing its own answers.
   Separate held-out rows are reserved for evaluation.

For example, a working bass-and-drums song becomes a row containing “Make a
120 BPM bass-and-drums groove,” a short musical explanation, and its Strudel
code. It is saved only if its checks pass.

The normal command sequence is:

```bash
python scripts/collect_seeds.py --config configs/sources.yaml --ingest
tunelm build-memory
tunelm generate --config configs/generation.yaml
```

Set model IDs and API keys before the last command. If generation stops, run
that same command again: completed examples are kept and unfinished work resumes.
YAML controls the process; JSONL stores the examples. The sections below explain
setup, optional checks, and the files produced by each command.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,memory,teacher,dev]'
npm ci
tunelm --help
```

On the future Vast.ai host, install the appropriate CUDA PyTorch build first.
Training library pins are in `pyproject.toml`: Transformers 5.12.1, TRL 1.12.0,
PEFT 0.20.0, datasets 5.0.1, accelerate 1.12.0. GPU memory requirements and CUDA
compatibility have not been measured. No instance has been selected or launched.
The initial training runner uses one CUDA process; distributed/DDP launch is
rejected until checkpoint-merge synchronization is added.

## Collect seeds (Step 1)

```bash
python scripts/collect_seeds.py --config configs/sources.yaml --ingest
```

This downloads only the reviewed files listed in `configs/sources.yaml`, extracts
their examples, includes the ten local starter compositions, removes exact code
duplicates, and runs the existing verifier/stem extractor. No LLM or API key is
needed. GitHub and Codeberg downloads use immutable commits and SHA-256 checksums.
Downloads are temporary: only extracted JSONL, provenance, and license notices
are retained. The allowlist now contains 19 source scopes under MIT, CC0, GPL,
and AGPL. It includes Strudel's explicitly GPL-credited drum-pattern module,
independent community patches, JSON examples, and the original example library.
See [source research](source-research.md) for selections and exclusions. This is
the configured snapshot, not all Strudel music; authorship methods are unverified.

The collector supports standalone files, independent programs in codebooks,
labeled or unlabeled Markdown code fences, JSON `code`/`pattern` records, static
`engine: 'strudel'` object records, and exported literal program templates. The
Node helper parses data without importing or executing downloaded modules. It
also reads appended records and skips other engines. Independent top-level
patterns retain their preceding setup; named track labels can become `$:` and
underscore-muted tracks stay omitted. Other programs remain intact for the
verifier to accept or reject. Expected counts prevent incomplete extraction.
A missing tempo gets an explicit 30 CPM default. Unsupported syntax,
browser/MIDI dependencies, unavailable sounds, and silent patterns are rejected.

| Output | Purpose |
| --- | --- |
| `datasets/sources/seeds.jsonl` | Deduplicated candidates, including later rejections |
| `datasets/sources/provenance.jsonl` | External source URLs/lines, revisions, hashes, license evidence, normalization |
| `datasets/sources/licenses/` | License notices and attribution, without raw code files |
| `datasets/sources/collection.json` | Configuration, hashes, counts, ingestion status |
| `datasets/seeds/compositions.jsonl` | Accepted seed list for generation |
| `datasets/stems/verified.jsonl` | Verified reusable musical parts |
| `results/ingestion/rejected.jsonl` | Failed candidates and verifier diagnostics |

The initial 2026-11-06 snapshot had 103 accepted compositions and 185 stems.
Research expanded the configured inputs to 699 candidates before exact code
deduplication, yielding 679 candidates and 20 duplicate records. Current accepted
counts and per-source results are in [source research](source-research.md) and
`datasets/sources/collection.json`. No teacher APIs were called.

Online reruns download sources temporarily and rebuild without appending duplicate
seeds. To rerun using the saved parsed JSONL, with no raw files or network:

```bash
python scripts/collect_seeds.py --config configs/sources.yaml --offline --ingest
```

The offline snapshot verifies configuration, local inputs, candidates, provenance,
and duplicate-report hashes. Changes require an online refresh. Collection and
ingestion rebuild manifests; they do not resume halfway through a verifier call.
Omit `--ingest` to collect without rebuilding accepted seeds. To add sources,
append pinned paths, hashes, license evidence, and expected counts to the YAML;
keep previous entries to preserve the existing corpus. Optional `host: codeberg`
selects Codeberg; `family` groups related material across repositories. Evidence
can use a pinned public `url`. Set `retain_notice: false` for a whole source file
used as license evidence so it is not saved as a notice.

GPL/AGPL sources are supported. The Strudel rhythm file explicitly credits its
GPL origin; that file-level declaration and the original GPL notice are retained.
This does not infer rights for every contributed track from the application
license: the [official FAQ](https://strudel.cc/learn/faq/) distinguishes music
rights. Other separately licensed flagship/community tracks were not imported.
Each external collection shares one family, with an explicit common origin family
for the rhythm bank. The small held-out set remains insufficient for production
evaluation; broader independent held-out material and listening review are needed.

For a starter-only demonstration, use `tunelm ingest examples/seeds.jsonl`.
That rebuilds the corpus from only those ten rows, replacing the larger accepted
collection. For your own manifest, use `tunelm ingest path/to/sources.jsonl`.

## Verify and prepare memory (Step 2)

```bash
tunelm verify examples/groove.js --cycles 32 --output results/verification/groove.json
tunelm decompose examples/groove.js --output results/verification/stems.json
tunelm build-memory
```

The last command downloads the configured semantic embedding weights on first
use. For an explicit offline baseline, use `tunelm build-memory --lexical`.
The local implementation checks used lexical mode; semantic model inference has
not run. Run seed collection/ingestion above before building memory. Memory
building does not require OpenAI, Gemini, or Claude.

Ingestion writes `datasets/seeds/compositions.jsonl`,
`datasets/stems/verified.jsonl`, manifests, and rejection diagnostics under
`results/ingestion/`. The index lives in `datasets/memory/`. Reingesting is a
corpus rebuild; include all desired sources in the manifest. Family split
assignments already recorded in `datasets/manifests/splits.json` cannot silently
change. Accepted license identifiers are listed in `corpus.py`.

```bash
tunelm plan 'A 120 BPM groove in C minor with drums, bass and pad' \
  --output results/plans/groove.json
```

Planning downloads only the pinned tokenizer on first use. `--offline` requires
it to be cached. Use `--plan path/to/plan.yaml` to provide exact voices/densities,
required techniques, category, or original co-creation code; see
[training I/O](training-io.md). The global tempo convention is 4 beats per cycle.

## Grounded teacher generation

Configure the complete model mix in `configs/generation.yaml`. Each entry has a
unique ID, provider, model ID (or `model_env`), and fraction of accepted samples.
Multiple models from the same provider are supported. Install SDKs with
`pip install -e '.[teacher]'`.

```yaml
count: 200
models:
  - id: openai-main
    provider: openai
    model_env: OPENAI_TEACHER_MODEL
    fraction: 0.40
  - id: gemini-main
    provider: gemini
    model_env: GEMINI_TEACHER_MODEL
    fraction: 0.30
  - id: claude-main
    provider: anthropic
    model_env: CLAUDE_TEACHER_MODEL
    fraction: 0.30
```

This is the model-selection portion of the supplied full YAML config. Set a
literal `model: <provider-model-id>` instead of `model_env` when preferred. No
specific paid model is chosen automatically. Fractions must sum to 1.0; zero
fractions disable entries. For 200 samples, this mix targets 80/60/60 **accepted**
examples. Largest-remainder rounding gives exact integer totals for smaller runs,
with YAML order breaking ties. The shares apply to the complete accepted corpus,
not separately to every train/validation/benchmark split.

| Provider | `provider` | Default key environment variable |
| --- | --- | --- |
| OpenAI or compatible endpoint | `openai` | `OPENAI_API_KEY` |
| Gemini Developer API | `gemini` | `GEMINI_API_KEY` |
| Claude Messages API | `anthropic` (alias `claude`) | `ANTHROPIC_API_KEY` |

Optional per-model fields: `key_env`, `base_url_env`, and `max_tokens` (4096 by
default). OpenAI honors `OPENAI_BASE_URL`; Claude honors `ANTHROPIC_BASE_URL`.
Gemini uses API keys rather than Vertex ADC. Credentials stay in the environment.
Gemini and Claude use native structured JSON responses, then the same local
schema, verifier, and musical gates as OpenAI. See the [Google SDK docs](https://googleapis.github.io/python-genai/)
and [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

```bash
# No tokenizer, API clients, state writes, or paid requests:
tunelm generate --config configs/generation.yaml --dry-run

# Starts paid generation after the corpus, memory, models, and keys are ready:
tunelm generate --config configs/generation.yaml

# Resume after interruption: run the exact same command/config again.
tunelm generate --config configs/generation.yaml
```

Each model runs sequentially against the frozen verified candidates and context.
Accepted examples count toward its own quota. A failed candidate gets at most two
correction retries; exhausted candidates advance to replacements. The per-model
candidate budget is target × `max_candidates_per_sample` (default 5). If the bank
or budget cannot fill a model's target, the run saves partial exports, reports its
shortfall with status `exhausted`, and exits unsuccessfully. Its share is never
silently given to another model.

Run state is under `results/generation/<run_id>/`:

- `run.json`: frozen input/model/tokenizer identity and target allocation.
- `jobs.jsonl`: immutable source jobs and packed contexts, prepared before API calls.
- `attempts.jsonl`: per-model retry records and completed response recovery.
- `accepted.jsonl`: accepted examples, with `teacher_id` and `provider:model` provenance.
- `progress.json`: per-model accepted counts, next jobs, budgets, and shortfalls.

Reruns restore accepted rows, unfinished retries, and each model's cursor from
the journals. Completed models need no client or API key on resume. Changing a
configured mix, resolved model, source corpus, tokenizer, or memory snapshot
requires a new run ID and output directory. Missing model environment variables
may reuse that run's already-frozen model IDs. An in-flight request interrupted
before a complete journal record may need to be sent again. One writer per run
is enforced with an OS file lock on macOS/Linux.

`--count`, `--run-id`, and `--dataset-dir` override YAML settings. To temporarily
select one model, supply both `--provider` and `--teacher-model`; that creates a
single profile with fraction 1.0. Use distinct run/output paths for comparisons.

Final training exports remain plain JSONL:
`datasets/sft/{train,val}.jsonl`, `datasets/rl/{train,val}.jsonl`, and
`datasets/evaluation/benchmark.jsonl`. RL/benchmark records contain no solutions.
Configuration files are YAML; saved data and run state remain JSON/JSONL.

Generated datasets, memory indexes, and ingestion/generation state were cleared
after the YAML migration. Seed collection/ingestion was subsequently completed;
memory building and teacher generation remain separate next steps. No teacher
calls or training runs were performed here.

## SFT, then GRPO on Vast.ai

Run these only after corpus/generation review and transfer of the frozen memory
and dataset snapshots to the CUDA host. Each stage is required.

```bash
tunelm train-sft --config configs/sft-warmup.yaml --dry-run --tokenize
tunelm train-sft --config configs/sft-warmup.yaml

tunelm train-sft --config configs/sft.yaml --dry-run --tokenize
tunelm train-sft --config configs/sft.yaml

tunelm train-grpo --config configs/grpo.yaml --dry-run --tokenize
tunelm train-grpo --config configs/grpo.yaml
```

Dry runs validate real rows, family separation, memory versions, native trainer
fields, and preceding adapter checkpoints. `--tokenize` also checks exact masks
and context lengths, without loading weights. A missing teacher dataset or
preceding checkpoint is an error; dry runs do not invent placeholder inputs.
Stage commands use `configs/model.yaml` unless a `model` object is embedded in
the stage config. Use `--model-config` to select another model settings file.

Start the first GPU run with a copied config using a small `max_steps` and
frequent `save_steps`/`eval_steps`, before the full schedule. Both SFT and GRPO
GPU behavior remains unverified; this is the first required hardware check.

Full SFT continues `checkpoints/lfm3-warmup`. GRPO requires
`checkpoints/lfm3-sft`, creates `checkpoints/lfm3-sft-merged`, then saves its fresh
adapter to `checkpoints/lfm3-grpo`. Preserve the merged base and the RL adapter.
Reward/component/diagnostic traces are JSONL under `results/lfm3-grpo/`.

```bash
tunelm train-grpo --config configs/grpo.yaml --resume
```

Resume uses the last native trainer checkpoint, or an explicit checkpoint path
after `--resume`. Configuration, data, and memory identities must match the
original run. Native periodic checkpoints include optimizer/RNG state; the final
adapter is also usable for inference. A run shorter than the configured save
interval may have no periodic checkpoint to resume. Choose a new output directory
for a deliberately changed training run.

## Inference and evaluation

These commands load model weights and generate music; they have not run here.

```bash
tunelm infer 'A 120 BPM groove in C minor with bass and drums' \
  --checkpoint checkpoints/lfm3-grpo --output results/inference/groove.json
tunelm predict --checkpoint checkpoints/lfm3-sft \
  --output results/benchmark-sft/predictions.jsonl
tunelm predict --checkpoint checkpoints/lfm3-grpo \
  --output results/benchmark-grpo/predictions.jsonl
tunelm evaluate --predictions results/benchmark-sft/predictions.jsonl \
  --output results/benchmark-sft
tunelm evaluate --predictions results/benchmark-grpo/predictions.jsonl \
  --output results/benchmark-grpo
```

Omit `--checkpoint` on `predict` for the pinned base model. Use the same frozen
benchmark/context across base, SFT, and GRPO. `evaluate` only scores saved JSONL;
it reports missing/extra IDs and writes per-task scores plus a JSON summary.
Copy valid Strudel into the Strudel REPL for listening and record human ratings
separately. Automated reward is not a substitute for audibility or musical review.
