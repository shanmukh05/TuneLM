# TuneLM implementation

Implemented 2026-11-06 for `LiquidAI/LFM3-1.7B-Base`. Both supervised fine-tuning
and GRPO reinforcement learning are implemented. Earlier teacher pilots ran; the prompt-first redesign was checked locally without
new paid calls. GPU training, model inference, and Vast.ai jobs remain unrun.

## Flow and ownership

`prompts-sft.yaml` / `prompts-rl.yaml` → independent SFT and RL request authoring → reviewed JSONL inputs.
RL requests → `export-rl` → answer-free GRPO tasks with contracts.
SFT requests → original composition + optional references → executable answers.
RL requests → frozen answer-free tasks. Both feed the existing SFT/GRPO stages.
Source collection and memory are optional reference-library branches.

| Module | Owns |
| --- | --- |
| `schemas.py`, `io.py` | Versioned records, safe YAML configs, strict JSON data, atomic files, append journals, run locks, archive write protection |
| `verifier/ast.mjs` | Accepted JS subset, `$:` normalization, dependencies, extraction, literal-scale transposition, bounded arrangement spans |
| `verifier/runtime.mjs`, `verifier.py` | Native Strudel Haps, process deadlines, heap/event/output limits, catalog checking |
| `telemetry.py` | Onset counts, pitches/register, syncopation, structural/control fingerprints |
| `corpus.py` | Source/license manifests, stable family splits, seed and stem verification |
| `scripts/collect_seeds.py`, `scripts/extract_seed_objects.mjs` | Source allowlist, pinned downloads/cache, literal extraction, provenance, deduplication, optional ingestion |
| `memory.py`, `planner.py` | SQLite/NumPy indexes, embeddings, metadata filtering, ranking, typed plans, bounded XML context |
| `prompt_generation.py`, `prompting.py` | Independent resumable SFT/RL request banks, author quotas, styles, reviewable JSONL |
| `synthesis.py`, `generation_state.py`, `responses.py`, `teachers.py` | Original composition for fixed requests, optional references, independent RL exports, retries, native structured provider adapters |
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
played. All 125 active GM instrument names from the pinned official soundfont package are cataloged; custom runtime additions still need verification.

Telemetry separates note/rhythm fingerprints from control automation. Evolution
is evaluated in 8-cycle windows; the total queried window is stored. Rewards
hard-gate malformed responses and failed execution, then average applicable
instruction, scale, register/sound role, evolution, and gain proxies. Atomic or
explicit static loops omit evolution rewards. Prompt-first rows enforce execution and explicit request constraints. Inferred
key/role defaults and a fixed structural-variation threshold are not acceptance
requirements. Symbolic proxies remain useful diagnostics, not creativity scores.

Role coverage and gain checks are deliberately simple proxies. They cannot prove
independent voice count, perceptual balance, artistic quality, or correspondence
of every natural-language claim. Review teacher examples and listen before scaling.

## Memory and data

`configs/registry/sources.yaml` lists reviewed source files, immutable revisions, checksums,
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

Input authoring assigns a writing style and lets the author invent the situation.
There is no fixed scene list. Separate stage/split banks reject exact duplicate prompt text and IDs across
SFT, RL, and evaluation. Reviewed input edits are protected from author reruns.
Answer generation snapshots these banks and optional contexts, chooses its own
composition plan, and never generates or rewrites the user prompt. Each input is
assigned to one composer using exact per-model targets. Failed jobs exhaust their
bounded correction attempts and report shortfalls rather than rewriting requests.

Reference modes are none, API-only, or optional annotated retrieval. Registry
entries explain behavior, musical uses, and pitfalls; snippets explain measured
sources, density, pitch register, and operation meanings. No reference tempo,
scale, instrument mapping, source code, or section template is compulsory. The
old back-translation/recombination factory has been removed from the active path.
Only explicitly supplied co-creation code must be preserved. Source-family splits
remain relevant to corpus-derived legacy targets; independent prompt inputs may
share optional reference examples. The teacher that writes the answer receives the
packed reference context, including API notes and sound descriptions. LFM training
and inference receive the listener request only, plus co-creation code when the
listener supplied it. Stored context is a record of what the teacher saw.

Provider adapters support separate prompt and composition response schemas across
OpenAI, Gemini, and Claude. Input stage replies contain a request and optional
co-creation starting code; answer replies contain a composer-chosen plan, prose
musical thinking, and code. Requests, journals, usage, snapshots, and exact model
identities stay local JSON/JSONL. RL banks have no teacher answer fields. Both
stages use locks, journal recovery, and stable identities for resume.

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
thinking prefill. Warmup uses atomic or small layered examples without requiring retrieval matches; full SFT continues
the warmup adapter. GRPO merges and saves SFT first, then trains a new adapter on
that saved base. With nonzero KL, disabling the new adapter exposes the SFT policy.
Retain the merged base with the RL adapter when transferring artifacts.

Training uses TRL, saves native trainer checkpoints and final adapter/tokenizer
artifacts, and exposes resume. A configured `hub` block also uploads the merged
full model, best and last, using `HF_TOKEN`. GPU updates and checkpoint reloads remain
unverified until the first Vast.ai run. See [usage](usage.md),
[training I/O](training-io.md), and [verification evidence](verification.md).
