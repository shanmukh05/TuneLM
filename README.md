# TuneLM

TuneLM generates executable [Strudel](https://strudel.cc/) music using compositional
memory and [`LiquidAI/LFM3-1.7B-Base`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base).
Runtime settings live in YAML under `configs/`; datasets remain plain JSONL.
The implementation includes pinned licensed-source collection, ingestion, AST stem extraction,
headless verification, independent prompt authoring, optional musical references, original composition with OpenAI,
Gemini, Claude, DeepSeek, or OpenRouter, **SFT and GRPO RL**, inference, and offline evaluation.

**Status (2026-11-09): prompt-first SFT/RL data pipeline implemented locally.**
Earlier Gemini pilots produced 13 accepted examples. The new author/composer split
was checked with real pilot inputs and no new paid calls. GPU training, model
inference, and Vast.ai jobs remain unrun; listening quality remains unverified.
The LFM3 1.7B checkpoint is pinned by revision.

The optional reference corpus has **516 compositions and 662 verified stems**.
It is reference material, not a mandatory composition factory. Registry entries
and retrieved examples explain musical behavior; unspecified choices belong to
the composer. Author SFT inputs with `configs/prompts-sft.yaml`, inspect them, then
compose with `configs/samples-sft.yaml`. RL inputs come from
`configs/prompts-rl.yaml` and `tunelm export-rl`. A smaller run lowers `counts`
there, or sets `count` on generation.
The 19-source allowlist includes MIT, CC0, GPL, and AGPL scopes; source downloads
are temporary. See [source research](docs/source-research.md) for per-source results.

| Document | Contents |
| --- | --- |
| [Usage](docs/usage.md) | Setup, corpus, generation, SFT/GRPO, inference, future Vast.ai run |
| [Implementation](docs/implementation.md) | Actual architecture, limits, and design choices |
| [Training I/O](docs/training-io.md) | JSON/JSONL, model contract, masking, and reward records |
| [Verification](docs/verification.md) | Checks performed and what remains unrun |
| [Data layout](datasets/README.md) | Canonical data and derived artifacts |
| [Original plan](design_docs/implementation-plan.md) | Requirements, source proposals, and implementation sequence |

## Local start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,memory,teacher,dev]'
npm ci
tunelm verify examples/groove.js --cycles 32
python scripts/collect_seeds.py --config configs/registry/sources.yaml --ingest
tunelm build-memory --lexical
```

Use `tunelm build-memory` for the configured semantic model instead of the explicit
offline lexical baseline. Planning uses the pinned LFM tokenizer. Teacher calls
and model execution require separate commands; see usage before launching them.
Preview input authoring without paid calls using
`tunelm create-prompts --dry-run`.
The complete two-stage run commands are in [Usage](docs/usage.md).

Required training order:

```bash
tunelm train-sft --config configs/sft-warmup.yaml
tunelm train-sft --config configs/sft.yaml
tunelm train-grpo --config configs/grpo.yaml
```

These require reviewed teacher-generated data and a CUDA host. Add `--dry-run
--tokenize` to validate prerequisites without loading weights. No training jobs
have been launched. `examples/seeds.jsonl` keeps the ten original starter inputs;
the larger accepted collection is in `datasets/sources/compositions.jsonl` after
running the collector. Its small held-out set is not a production quality benchmark.

New data uses `datasets/`, run outputs use `results/`, and adapters use
`checkpoints/`. Legacy datasets, checkpoints, and results remain intact under
ignored `contents_v1/`; see its [original dataset card](contents_v1/datasets/README.md).
Code never writes into that archive. `.cursor/` guidance and local credentials
remain in place. No unit tests were created; verification uses real inputs and
manual smoke runs. Candidate-code execution needs an OS sandbox for hostile input.

License: AGPL-3.0-or-later. Model and source-corpus licenses remain separate.
