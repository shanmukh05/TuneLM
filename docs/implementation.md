# TuneLM implementation

Implemented 2026-11-06 for `LiquidAI/LFM3-1.7B-Base`. Both supervised fine-tuning
and GRPO reinforcement learning are implemented. GPU training, teacher calls,
model inference, and Vast.ai jobs remain unrun at the user's request.

## Flow and ownership

```mermaid
flowchart LR
  A[Pinned licensed source files] --> SC[Collection script]
  SC --> S
  S[Licensed source JSONL] --> I[Verify + AST decomposition]
  I --> C[Canonical seed/stem JSONL]
  C --> M[Training-only memory index]
  P[Prompt or explicit plan] --> R[Filter/rank + bounded context]
  M --> R
  C --> G[Grounded teacher generation]
  R --> G
  G --> V[Format + execution + musical gates]
  V --> D[SFT examples + answer-free RL tasks]
  D --> W[SFT warmup]
  W --> F[Full-mixture SFT]
  F --> B[Saved merged SFT base]
  B --> RL[Fresh GRPO adapter + verifier rewards]
  RL --> E[Frozen benchmark + listening review]
  R --> N[Checkpoint inference]
  N --> V
```

| Module | Owns |
| --- | --- |
| `schemas.py`, `io.py` | Versioned records, safe YAML configs, strict JSON data, atomic files, append journals, run locks, archive write protection |
| `verifier/ast.mjs` | Accepted JS subset, `$:` normalization, dependencies, extraction, literal-scale transposition, bounded arrangement spans |
| `verifier/runtime.mjs`, `verifier.py` | Native Strudel Haps, process deadlines, heap/event/output limits, catalog checking |
| `telemetry.py` | Onset counts, pitches/register, syncopation, structural/control fingerprints |
| `corpus.py` | Source/license manifests, stable family splits, seed and stem verification |
| `scripts/collect_seeds.py`, `scripts/extract_seed_objects.mjs` | Source allowlist, pinned downloads/cache, literal extraction, provenance, deduplication, optional ingestion |
| `memory.py`, `planner.py` | SQLite/NumPy indexes, embeddings, metadata filtering, ranking, typed plans, bounded XML context |
| `synthesis.py`, `generation_state.py`, `responses.py`, `teachers.py` | Grounded generation, four personas, retrieval regimes, retries, shared summary/code contract, native provider adapters |
| `training.py`, `rewards.py` | Completion-masked SFT, SFT-to-GRPO handoff, reward components, checkpoint/resume, preflight |
| `inference.py`, `cli.py` | Explicit commands, prediction journals, offline scoring, missing/extra-ID reporting |

These are intentionally ordinary Python modules, rather than many subpackages.
All paths are relative to the repository root, and output helpers reject
destinations resolving into `archive/`.

## Verification boundary

Each request starts a fresh Node process with only PATH in its environment and
a temporary working directory. Defaults: 5-second wall deadline, 256 MiB V8 heap,
20,000 Haps, 100 KB code, 128 KiB serialized request, and 32 MiB combined pipe
output. Python drains bounded pipes and kills a timed-out worker. There is no
static fallback. Logs go to stderr; stdout carries one versioned JSON response.

Strudel 1.2.6 is pinned. The installed `@kabelsalat/web` 0.4.1 package needs its
published ESM entry selected; the version-checked npm postinstall script does so.
Runtime calls use the installed native evaluator, not the proposal's incorrect
transpiler return-value assumptions. Browser visualizers are identity stubs.

AST validation accepts immutable declarations, expressions, and bounded musical
function chains. Imports, side-effect capabilities, loops, constructors, mutable
declarations, and dynamic property access are rejected. `$:` tracks require setup
before tracks. Extraction resolves named voices and copies their required global
declarations, without treating every nested stack as a separate voice. Some
complex programs are intentionally rejected rather than guessed at.

Direct `arrange` expressions, const references, stacks, and literal slow/fast
wrappers have their duration inspected. Verification expands to cover every
declared section, up to 64 cycles; indeterminate or longer arrangements fail.
General JavaScript cannot be proven to have a finite musical period. Absolute
pitch and dynamic-scale transposition are not implemented; literal scale changes
preserve octave and are reverified before use.

VM checks/process limits are **not an OS security sandbox**. Run hostile or
third-party programs in an isolated, network-disabled worker container without
credentials or writable project mounts. No public execution service is provided.

## Musical features and gates

Hap whole/part spans and onset flags are preserved. Sustained events crossing a
cycle boundary count once, MIDI pitch 0 is retained, and numeric controls are
checked before JSON serialization. Sound and bank controls are combined for
catalog validation; drum-machine/default Uzu drum-kit snapshots and a synth allowlist
are checked in. Symbolic validity does not prove samples were downloaded or audio
played. General MIDI/custom sounds require explicit catalog additions.

Telemetry separates note/rhythm fingerprints from control automation. Evolution
is evaluated in 8-cycle windows; the total queried window is stored. Rewards
hard-gate malformed responses and failed execution, then average applicable
instruction, scale, register/sound role, evolution, and gain proxies. Atomic or
explicit static loops omit evolution rewards. Accepted teacher rows must satisfy
explicit constraints and task-specific quality gates.

Role coverage and gain checks are deliberately simple proxies. They cannot prove
independent voice count, perceptual balance, artistic quality, or correspondence
of every natural-language claim. Review teacher examples and listen before scaling.

## Memory and data

`configs/sources.yaml` lists reviewed source files, immutable revisions, checksums,
and license evidence across GitHub and Codeberg. Downloads are temporary; parsed
JSONL and notices are retained. Literal objects, exported templates, JSON examples,
Markdown fences, and independent programs are extracted without executing modules.
Named tracks can be canonicalized to `$:` while muted tracks remain omitted.
Verified snapshot hashes support offline ingestion without raw files. Each external
collection stays in one family; related upstream material can share an explicit
origin family. MIT/CC0/GPL/AGPL scopes are supported, including Strudel's explicitly
GPL-credited rhythm file. Source licensing is recorded separately from the
surrounding application's license. See usage and source research for selections.

Canonical records are UTF-8 JSON/JSONL. SQLite and normalized NumPy embeddings
are rebuildable indexes. Default semantic embeddings use a pinned MiniLM model;
`--lexical` explicitly selects a local bag-of-words baseline. Only training stems
enter memory. Hard role/density filtering precedes semantic/method/evolution
ranking at 0.65/0.25/0.10, with stable ID tie breaking. Missing roles are explicit.

Basic prompt rules recognize common tempo, scale, voice, and arrangement cues.
Use explicit Plan JSON for precise requirements or co-creation. This is not a
general natural-language music planner. Context packing uses the LFM tokenizer,
keeps whole XML/code items, and reports omissions at the 550-token budget.

Ingestion assigns a source-family split before extraction; explicit split
conflicts and exact duplicate compositions fail. Descendants inherit the split,
and recombination stays within a split. Held-out targets never enter memory.
Near-duplicate musical families still need corpus review; exact hashes do not
establish semantic independence.

Generation YAML lists all teacher models, providers, and accepted-sample fractions.
Largest-remainder allocation produces exact per-model targets. Teacher calls use
OpenAI-compatible, native Gemini, or native Anthropic Claude backends through one
typed interface and explicit registry. SDK imports are lazy,
provider-specific API-key defaults are used, and clients close at run completion.
Gemini and Claude request structured JSON; all replies pass the same local schema
and musical gates. Provider/model identities are saved with each teacher example
and included in run fingerprints. Back-translation
preserves verified code; recombination uses verified, transposed stems and native
execution before teacher adaptation. High/partial/empty retrieval targets a
40/40/20 mix; absent alternatives can increase the realized empty share. Accepted
rows and attempts are separate, and retry history limits each model/job to three
attempts across restarts. Frozen jobs, per-model cursors, crash-gap recovery, and
exclusive run locks support resumption. Bounded replacement candidates fill each
model quota; shortfalls are reported without reallocating the share. Input/model/
tokenizer identities prevent mixing runs. See usage for the persisted state files.
The 20/45/15/20 task mixture and 30,000 examples remain scaling targets, not claims
about the tiny starter corpus.

## Required training stages

The checkpoint and modeling code are pinned to
`7bca5038823c0d0f14efe06439109354a10e014c`. Reviewed architecture modules include
attention projections, convolution input/output projections, and MLP `w1/w2/w3`;
the LoRA defaults target those actual module names. Model loading uses the pinned
custom code, BF16, and SDPA. Quantized training was not added without a verified
compatibility need.

SFT tokenizes the complete checkpoint template and requires an exact prompt token
prefix. It masks that prefix with -100 and rejects overlength examples. The
generation prompt already opens `<think>`; there is no duplicate tag or empty
thinking prefill. Warmup filters atomic/high-match examples; full SFT continues
the warmup adapter. GRPO merges and saves SFT first, then trains a new adapter on
that saved base. With nonzero KL, disabling the new adapter exposes the SFT policy.
Retain the merged base with the RL adapter when transferring artifacts.

Training uses TRL, saves native trainer checkpoints and final adapter/tokenizer
artifacts, and exposes resume. GPU updates and checkpoint reloads remain
unverified until the first Vast.ai run. See [usage](usage.md),
[training I/O](training-io.md), and [verification evidence](verification.md).
