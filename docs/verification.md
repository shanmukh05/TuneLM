# Implementation verification — 2026-11-06

The user requested local implementation only, with Vast.ai reserved for future
compute. No teacher calls, model-weight downloads, model inference, training
steps, paid services, or cloud instances were run. No unit tests, test files,
test-only dependencies, or test fixtures were created.

## Completed local checks

| Check | Observed result |
| --- | --- |
| Editable package and CLI | Installed; all explicit subcommands appear in help |
| Python static checks | Ruff formatting/lint/complexity checks passed; imports/byte-compilation passed |
| Node syntax | AST and runtime modules passed `node --check` |
| Starter corpus | 10 original licensed compositions accepted, 10 verified stems, zero ingestion rejections |
| Frozen memory | Only 6 training stems indexed; 2 validation and 2 benchmark sources excluded |
| Real-pattern smoke run | 50 valid variations across starter sources and five tempos executed |
| Invalid-program smoke run | 20 invalid programs rejected across syntax, evaluation, arc-query, and catalog stages |
| Arrangement coverage | A failure after cycle 20 was detected when requesting only 8 cycles; runtime expanded to 24 cycles |
| Indeterminate/oversized arrangements | Rejected rather than claiming complete section coverage |
| Resource limits | Event overflow rejected; a parent deadline terminated a worker |
| Timing and pitch | Sustained note yielded 2 onsets across 8 cycles; MIDI 0 retained |
| Randomness | Repeated seeded degradation yielded identical Hap streams |
| Grounded assembly | 40 combinations verified: 10 atomic, 19 stacks, 6 arrangements, 5 co-creations |
| Co-creation and splits | Original code retained in checked assemblies; no cross-split stem tuples |
| Actual LFM tokenizer | Three-voice context packed to 438 tokens; full source example 867 tokens, 222 supervised tokens |
| Loss mask/template | Prompt token prefix matched exactly, prompt labels masked, one opening thinking tag |
| Persistence boundaries | Incomplete final journal line recovered; archive output refused |
| Missing-data dry run | SFT dry run refused missing teacher exports without loading weights |
| Dependency extraction | Grouped const declarations and closure dependencies reverified as two independent stems |
| Training API/config inspection | Installed SFT/GRPO APIs expose required arguments; all three configs use known native fields; nine LoRA target names match inspected model code |
| Archive | Legacy asset metadata and dataset hashes preserved; archive is Git-ignored |

The measured local verifier median was about 82 ms and p95 about 85 ms in the
50-input run. These include fresh-process overhead and do not support the
proposal's 1–5 ms latency assumption. They are local measurements, not a GPU
throughput estimate.

Local evidence is saved under ignored `results/verification/`:
`manual-smoke.json`, `assembly.json`, `contracts.json`, `groove.json`, and
`js-complexity.json`. The starter input is committed as ordinary project data in
`examples/seeds.jsonl`, with a playable source program in `examples/groove.js`.
These checks are not a model-quality benchmark and produced no teacher training
exports or artificial checkpoint files.

## Complexity report

| Function | Before | After |
| --- | --- | --- |
| AST validation callback | 18 | 2 |
| Node `execute` | 26 | 7 |
| Python `_candidates` | 12 | 7 |

AST validation was split by node type; runtime execution now delegates request
checks, engine setup, pattern checks, and arc queries. Candidate generation uses
separate source/selection/planning helpers. JavaScript counts use an Acorn AST
walk; Python uses Ruff C901. All functions meet the threshold of 10 (JavaScript
maximum 10, Python maximum 9). The stricter threshold-5 scan is informational.

## Native teacher-provider follow-up — 2026-11-06

Added native Gemini and Anthropic Claude adapters alongside the existing
OpenAI-compatible backend. All providers share the response schema, correction
loop, verifier, and JSONL export format. The CLI accepts provider/model/count/run
overrides; provider defaults choose `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`ANTHROPIC_API_KEY`. Provider/model identifiers distinguish data provenance.

Manual offline checks instantiated and closed all three real SDK clients with
temporary inspection credentials, checked the required SDK method arguments,
validated provider selection/Claude alias/CLI overrides, and round-tripped a
starter-source response through the common schema. An invalid decision count
was rejected. Ruff and byte-compilation passed. No mock APIs, unit tests, paid
requests, or generated teacher datasets were introduced.

The generation orchestration's cyclomatic complexity changed from 6 to 2 by
moving run orchestration into `_generate_run` (5) and provider calls into focused
adapters (maximum 3). Live authentication, provider feature availability for the
chosen models, and output quality remain unverified until actual API runs.

## YAML, model mixes, and generation resume — 2026-11-06

All eight runtime configuration/registry files under `configs/` now use YAML.
Datasets, frozen jobs, journals, progress, and manifests remain JSON/JSONL;
native tokenizer, package, and adapter manifests retain their required formats.
`generation.yaml` lists multiple named models with fractions of accepted rows.
Largest-remainder allocation targets exactly 80/60/60 for the default 200-row
40/30/30 mix, and 3/2/2 for seven rows. Fractions and duplicate model IDs are
validated. No paid model IDs were selected automatically.

Manual offline replay used the original starter corpus, verified source code,
lexical memory, and the actual pinned LFM tokenizer in a temporary directory.
The replay supplied manually reviewed source records, not API-generated outputs.
It verified frozen-job reuse, independent model quotas/cursors, recovery of a
completed attempt before its accepted-row append, unfinished retry recovery,
advancement after three failed attempts, partial JSONL exports, and refusal of
changed configuration or concurrent writers. A final focused check confirmed
that tokenizer changes also prevent resume. Evidence is recorded in
`results/verification/generation-resume.json`; API requests and teacher-generated
sample counts are both zero. The native verifier also passed with YAML registry
files, and Ruff formatting/lint/complexity plus byte-compilation passed.

The generation module now delegates frozen jobs, quota allocation, journal
recovery, and per-model cursors to `generation_state.py`. All Python functions
remain within the configured complexity threshold of 10.

After these checks, current generated corpus files, memory indexes, and
ingestion/generation state were cleared for a fresh run. The original sources in
`examples/`, documentation, verification evidence, and ignored legacy archive
were preserved. The corpus/memory counts above describe the earlier local
verification snapshot, not currently populated datasets. Ingestion and memory
building must run again before paid generation.

## Seed collection and Step 1 completion — 2026-11-06

Added `scripts/collect_seeds.py`, its Acorn literal-extraction helper, and the
reviewed `configs/sources.yaml` allowlist. The real run downloaded nine unique
files from two pinned MIT-licensed scopes, retaining raw bytes, notices, source
URLs/lines, content checksums, and source-family grouping. It extracted all 130
static Strudel records from the selected example library, including appended
records, plus six standalone pattern files, and included the ten starter sources.
Downloaded JS/TS modules were read as data, never imported or executed.

Official contributed Strudel examples/tracks were excluded because the
[official FAQ](https://strudel.cc/learn/faq/) identifies separate training rights.
The selected example-library file explicitly declares MIT in its SPDX header;
its application's root license was not assumed to cover music. Source authorship
methods and perceptual quality are not established by these automated checks.

The first ingestion exposed missing default drum names in the verifier catalog.
Added the 16 names from the immutable Uzu kit manifest, with its source URL/hash
in `configs/registry/sounds.yaml`; no sample audio was downloaded. Verification
still rejects unknown names, browser/MIDI dependencies, unsupported syntax, and
silent programs. No programs were rewritten to bypass these gates.

| Result | Count |
| --- | --- |
| Collected candidates | 146 |
| Accepted compositions | 103 |
| Verified stems | 185 |
| Rejected candidates | 43 |
| Seed train / validation / benchmark | 99 / 2 / 2 |
| Stem train / validation / benchmark | 181 / 2 / 2 |

All six Strudel-TUI candidates were rejected for unsupported assignment/operator
syntax. Final rejection stages were 27 syntax, 8 evaluation, 4 arc-query, and 4
sound-catalog failures. Diagnostics remain in `results/ingestion/rejected.jsonl`.
The accepted list is `datasets/seeds/compositions.jsonl`; candidate inputs and
licensing evidence remain under `datasets/sources/`.

The final cached/offline collector run reproduced the candidate manifest hash
and accepted/stem/rejection counts. Manual inspection of the real persisted data
verified source hashes, accepted/rejected partitioning, stem lineage/splits,
refusal of a mismatched checksum, and preservation of all 468 archived files and
14 dataset hashes. Ruff lint/format/complexity, Python compilation, Node syntax,
and diff whitespace checks passed. The new Python functions have maximum
complexity 5; JavaScript functions have maximum 8 (threshold 10). No unit tests or
fixtures were added. Evidence is `results/verification/seed-collection.json`.

Step 1 is complete for this initial allowlisted snapshot. The proposed 1,500-stem
scale has not been reached; broader independent sources and held-out listening
evaluation remain needed. Both external source families map to training, leaving
only the four starter held-out compositions. Memory was not built for this
snapshot. Teacher generation, SFT/GRPO, and cloud jobs were not run.

## Source expansion and temporary downloads — 2026-11-06

The user authorized more sources including Strudel-licensed collections and asked
not to retain raw files. Added 17 scopes to the original two, including explicitly
GPL/AGPL-licensed material. Strudel's drum-pattern module exports 492 literal
program strings with GPL credit to `lvm/tidal-drum-patterns`; the original GPL
notice and per-program credit comments are retained. Four independently credited
AGPL JSON examples were added from Live Coding Music MCP. Full selections and
exclusions, with pinned sources and per-source results, are in
[source research](source-research.md).

Collection extracted 699 candidates, removed 20 exact duplicates, and ingested
679 distinct records: **516 accepted compositions, 662 stems, 163 rejections**.
This adds 413 compositions and 477 stems. All 146 previous candidates, 103 previous
accepted compositions, and 185 previous stems were preserved exactly. The GPL
rhythm source supplied 391 accepted patterns. No verifier gates were weakened.

Added Codeberg support, JSON/example-template readers, unlabeled Markdown fences,
and independent program extraction. Named channels can become canonical `$:`
tracks while underscore-muted tracks remain omitted; statement order is preserved.
Downloaded modules are parsed as data, never imported or executed. The original
runtime still rejects unknown sounds, browser-only controls, loaders/extensions,
and unsupported syntax.

Downloads now use temporary directories. Only parsed JSONL, provenance hashes,
and license/attribution notices persist; the old `datasets/sources/raw/` cache was
removed. The completed ingestion ran in offline mode using the saved parsed
snapshot, with no raw files or downloads. Snapshot configuration, local-source,
candidate, provenance, and duplicate hashes were checked; changed configurations
were refused. License notice hashes, GPL credit comments, source/stem lineage and
splits, and preservation of 468 archive files and 14 dataset hashes were checked
against real persisted artifacts. Evidence: `results/verification/source-expansion.json`.

Ruff formatting/lint/complexity, Python compilation, Node syntax, and diff checks
passed. New/changed functions remain below the threshold of 10; JavaScript maximum
is 9. `fetch_file` complexity changed from 5 to 4 and `collect` from 4 to 3 through
separate snapshot/input/notice helpers. No unit tests or fixtures were added.

The bank is rhythm-heavy and has only two validation and two benchmark source
compositions. More harmonic/arrangement sources, independent held-out material,
and listening review remain needed. The 1,500-stem target is not yet reached.
Memory, teacher generation, SFT/GRPO, and cloud jobs were not run.

## Pending model and hardware validation

- Run semantic embeddings with the pinned MiniLM revision; local checks used
  the explicit lexical baseline without downloading embedding weights.
- Choose a versioned teacher/provider, generate a reviewed pilot, inspect
  realized task/retrieval distributions, and verify prompt/summary/code fidelity.
- On Vast.ai, load the actual pinned LFM weights; measure memory, perform SFT
  optimizer steps, save/reload an adapter, and check exact training masks there.
- Perform the SFT-to-merged-base-to-GRPO handoff, actual policy updates, native
  checkpoint resume, and reload of the final RL adapter with its merged SFT base.
- Compare base/SFT/GRPO on the same frozen, source-disjoint benchmark and listen
  to representative outputs. The 10 starter compositions do not replace the
  intended diverse corpus or 200-prompt benchmark.
- Profile verifier throughput during actual rollouts before adding a worker
  pool, quantized training, or optional audio-semantic tooling.

SFT and GRPO implementation are both present. Their hardware execution remains
unverified because those runs were explicitly deferred, not because RL is optional.
