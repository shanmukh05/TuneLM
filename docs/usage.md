# Running TuneLM

Use Python 3.11+, Node 22+, and a repository checkout. All commands below are
explicit actions; installation/verification does not start training or paid APIs.
The legacy assets live under ignored `archive/` and are never pipeline outputs.

## Dataset creation in plain language

The current pipeline starts with independent listener requests, not source songs:

1. Configure styles and writing examples in `configs/prompts.yaml`, then author
   separate SFT and RL input banks with `create-prompts`. No source corpus, memory
   index, tokenizer, or GPU is required for this stage.
2. Inspect/edit the input JSONL. These are requests, with no finished answer or
   composition plan. Co-creation alone can include an original starting pattern.
3. Run `generate` to compose original SFT answers. The composer returns its own
   plan, prose musical thinking, and executable Strudel code. It cannot
   rewrite the input request. Retrieved examples are optional reference material.
4. RL inputs are exported as answer-free tasks; no teacher composition is needed
   for them. They are distinct requests, not copies derived from accepted SFT rows.
5. Review execution diagnostics and listen before scaling or SFT/GRPO training.

Author and compose with the real configs (`.env.local` loads automatically):

```bash
tunelm create-prompts --dry-run
tunelm create-prompts
# Review datasets/prompts/generated/{sft,rl}/*.jsonl before an answer run.
tunelm generate --dry-run
tunelm generate --prepare-only
tunelm generate
```

Authoring size is the `counts` block in `configs/prompts.yaml`. Answer size is
`count` in `configs/generation.yaml`, or `--count` on `generate`; omit it to
answer every eligible SFT input. `--prepare-only` freezes contexts and exports
RL tasks without answer calls. Both stages resume their journals on the same
unchanged command. Editing reviewed input JSONL is supported before answer
generation; afterwards choose a new answer run and output path. Rerunning the
author refuses to overwrite edited prompt files.

Seed collection and `build-memory` remain available to build an optional reference
library. Neither is a prerequisite for authoring prompts or original compositions.
YAML controls the process; data, attempts, and accepted records remain JSON/JSONL.

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

## Collect the optional reference corpus

```bash
python scripts/collect_seeds.py --config configs/registry/sources.yaml --ingest
```

This downloads only the reviewed files listed in `configs/registry/sources.yaml`, extracts
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
| `datasets/sources/compositions.jsonl` | Accepted compositions |
| `datasets/sources/verified.jsonl` | Verified reusable musical parts |
| `datasets/sources/splits.json` | Source-family split assignments |
| `datasets/sources/corpus.json` | Ingestion counts for this snapshot |
| `results/ingestion/rejected.jsonl` | Failed candidates and verifier diagnostics |

The initial 2026-11-06 snapshot had 103 accepted compositions and 185 stems.
Research expanded the configured inputs to 699 candidates before exact code
deduplication, yielding 679 candidates and 20 duplicate records. Current accepted
counts and per-source results are in [source research](source-research.md) and
`datasets/sources/collection.json`. No teacher APIs were called.

Online reruns download sources temporarily and rebuild without appending duplicate
seeds. To rerun using the saved parsed JSONL, with no raw files or network:

```bash
python scripts/collect_seeds.py --config configs/registry/sources.yaml --offline --ingest
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

## Verify and prepare optional memory

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

Ingestion writes `datasets/sources/compositions.jsonl`,
`datasets/sources/verified.jsonl`, `datasets/sources/splits.json`, and
`datasets/sources/corpus.json`. Rejection diagnostics stay under
`results/ingestion/`. The index lives in `datasets/memory/`. Reingesting is a
corpus rebuild; include all desired sources in the manifest. Family split
assignments already recorded in `datasets/sources/splits.json` cannot silently
change. Accepted license identifiers are listed in `corpus.py`.

```bash
tunelm plan 'A 120 BPM groove in C minor with drums, bass and pad' \
  --output results/plans/groove.json
```

Planning downloads only the pinned tokenizer on first use. `--offline` requires
it to be cached. Use `--plan path/to/plan.yaml` to provide exact voices/densities,
required techniques, category, or original co-creation code; see
[training I/O](training-io.md). The global tempo convention is 4 beats per cycle.

## Independent prompt authoring and composition

`configs/prompts.yaml` holds the style palette plus input author model fractions, stage/split counts,
category scheduling, retries, and the output bank. `configs/generation.yaml`
separately controls composer model fractions, the optional answer count, and retries. Both support
OpenAI, Gemini, Anthropic, DeepSeek, and OpenRouter. DeepSeek and OpenRouter
use the OpenAI-compatible client. Multiple models per provider are allowed.
Model fractions sum to one and use largest-remainder allocation.
Each SFT input is assigned to one composer. Exhausted inputs report shortfalls;
failed answers never cause the input request to be rewritten.

Provider keys are `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`,
`DEEPSEEK_API_KEY`, and `OPENROUTER_API_KEY`.
Write each provider model id on that profile's `model` field. Credentials are
loaded from the shell or `.env.local` and never written to journals or the viewer.
Native token usage is saved with attempts. An interrupted in-flight request may
be billed again on resume; request counts are not a hard dollar spending cap.

Composition reference modes:

- `none`: no reference context.
- `api`: explained API, musical concepts, and playable instrument guide.
- `optional`: the guide plus retrieved, annotated examples when memory exists.

References never require a particular source composition, scale, tempo, voice
inventory, sound, or arrange template. The composer chooses unspecified details.
A supplied co-creation pattern remains an actual user input to preserve. Runtime
API/sound validity is distinct from artistic choice: supported methods can be
used even if absent from the guide, and the catalog includes all 125 active GM
names in the pinned official soundfont package as well as drums and synths.

Registry entries explain meanings, musical uses, and pitfalls. Retrieved examples
also explain their operations, measured sound sources, density, register, and
classifier uncertainty. These are symbolic descriptions, not listening reviews.

Author state lives under `results/prompt-generation/<run_id>/`; answer state is
under `results/generation/<run_id>/`. Each has immutable jobs, run identity,
accepted and attempt JSONL. Answer state also freezes independent RL/evaluation
contexts in `auxiliary.jsonl`. Adding prompts and running `generate` again appends their answers to the existing
dataset. A prompt that already has an answer is left as it is; add a new prompt id
for another answer. Changing the model mix or reference snapshot still requires a
new run ID. One writer per run is enforced.

Production exports are `datasets/creative/sft/{train,val}.jsonl`,
`datasets/creative/rl/{train,val}.jsonl`, and
`datasets/creative/evaluation/benchmark.jsonl`. Warmup, SFT, and GRPO configs point
to these paths. LFM training sends the listener request, so the training host does
not need the teacher reference index. SFT/validation inputs must have distinct IDs and exact prompt text;
sharing optional reference examples is not source-target leakage.

The existing HTML viewer accepts both input banks and output datasets. Start it
locally, then choose JSONL using its file picker or type a prompt:

```bash
tunelm view
```

Open `http://127.0.0.1:8765/`. Compose uses the same teacher prompt, reference
context, checks, and retries as `tunelm generate`, including the settings in
`configs/generation.yaml`. The reply stays on the Compose tab and is not written
into the training files. The first query loads the cached tokenizer. Prompt banks
have no answer to play; accepted compositions do. Playback uses the official
Strudel editor and requires internet for the runtime and audio assets.

Each author call is shown a random sample of requests already accepted, capped by
`avoid_sample` in the prompt config. Below that cap the call sees the whole bank.
The same `run_id` resumes after the author wording or palette changes. A different
`run_id` can use the same `output_dir`; its requests are added to the bank already
there, and the runs can write at the same time.

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
