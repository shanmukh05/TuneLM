# Data and model contracts

All records use `schema_version: 2`, strict Pydantic schemas, and UTF-8 JSON.
JSONL stores one complete object per line. Code and summaries remain strings with
preserved newlines. JSON writes reject non-finite numbers; manifests are replaced
atomically. Journals recover only an interrupted final line and reject corruption
in earlier lines. New writes resolving into `archive/` fail.

## Canonical data and derived files

| Artifact | Format |
| --- | --- |
| Source and stem records; SFT/GRPO/benchmark tasks | JSONL |
| Runtime configs, explicit plans, API/sound registry | YAML |
| Split, corpus, run, and summary manifests | JSON |
| Dense embedding matrix | NumPy `.npy`, no pickle; row IDs/model/revision in JSON |
| Retrieval index | Rebuildable SQLite, with a matching memory version |
| Model/adapter weights | safetensors plus native tokenizer/config JSON and custom code files |
| Native periodic training checkpoints | Framework optimizer/RNG/trainer state |

See [datasets/README](../datasets/README.md) for output paths. Legacy data is under
`archive/datasets/`; it is not accepted as the new schema implicitly.

## Sources and stems

Input Source fields: `id`, `family`, `source`, `license`, `code`, `description`,
optional `role`, optional `split`, and `tags`. Use a source URL or a repository
path with enough provenance to recover the original. A family groups all related
versions and derivatives of a composition; split the family together.

Source splits are `train`, `val`, or `benchmark`. An omitted split is assigned
deterministically by family hash (80/10/10); very small corpora should use explicit
splits. Source date is not inferred from collection time. Ingested seeds also save
code hash, verifier version, tempo/key facts, category, roles, and telemetry.

The collector writes candidate Sources to `datasets/sources/seeds.jsonl` and
external provenance to `datasets/sources/provenance.jsonl`: source ID, pinned URL
and line, repository/revision, upstream source path/hash, license scope/evidence,
and applied normalization. Local rows keep existing provenance. Collection time
and candidate/provenance/duplicate/local-input hashes are in `collection.json`.
Downloads are temporary; notices are under `datasets/sources/licenses/`, with
evidence URLs/checksums in provenance. Only accepted rows become canonical
`datasets/seeds/compositions.jsonl`. Candidates include later rejections.

Stem records retain original expression/setup, source ID/family/license/hash,
role/confidence, description/tags, methods, telemetry, and verifier version.
Roles: `drums_core`, `perc_tops`, `bass`, `chords_pad`, `arp_lead`, `fx_texture`.
Only verified training stems enter memory.

## Composition plan

Example explicit plan (JSON notation is also valid YAML):

```yaml
{
  "schema_version": 2,
  "bpm": 120,
  "key": "C:minor",
  "category": "stack",
  "voices": [
    {"role": "drums_core", "query": "909 kick and backbeat", "density": [2, 24]},
    {"role": "bass", "query": "low minor saw bass", "density": [1, 16]}
  ],
  "explicit": ["bpm", "key", "roles"],
  "short_loop": false,
  "cycles": 16,
  "original_code": null,
  "preserve": []
}
```

Categories: `atomic`, `stack`, `arrange`, `cocreate`. There are 1–6 voices and
8–64 requested evaluation cycles. `key` may be null for unpitched material. CPM is
derived as BPM/4. `original_code` is an immutable supplied voice during co-creation;
the current preservation gate requires its exact code substring. Fine-grained
semantic editing/preservation is not implemented. The only supported `preserve`
entry is `original_code`, and it requires that code to be supplied.

## Assistant completion and SFT masking

````text
<think>
1. 120 BPM, C minor, a layered groove.
2. A 909 kick and backbeat.
3. Low minor saw bass.
4. Slow filter motion on the bass.
5. Moderate gains with separated registers.
</think>
```strudel
setcpm(120/4);
stack(s("bd*4, [~ sd]*2").bank("RolandTR909").gain(0.4),
      n("<0 3 5 7>").scale("C2:minor").s("sawtooth").lpf(sine.range(300,1500).slow(8)).gain(0.4))
```
````

The outer text fence above illustrates the serialized completion. It contains
exactly five numbered musical decisions and one executable code block, without
additional prose. These summaries describe musical choices, not a claim to have
captured a model's private internal reasoning.

Task rows contain prompt, plan, packed context, source families, split, memory
version, retrieval regime, and retrieved stem IDs. SFT Example adds summary/code,
teacher identifier (`provider:model`), model-profile `teacher_id`, seed ID, pathway/persona, verified telemetry, and verifier
version. The checkpoint template is used consistently across SFT, GRPO, and
inference. Its opening `<think>` belongs to the generation prefix. Tokenization
requires the full conversation to begin with exactly those prompt token IDs,
masks that prefix, and supervises the remaining assistant content including EOS.
Overlength targets are rejected rather than truncated.

RL and benchmark rows are strict Task records with no Example solution fields.
GRPO receives rendered prompts plus plan JSON/task IDs for reward computation.
Native trainer callbacks receive only the generated continuation; the reward
adapter restores the already-prefilled opening tag before strict parsing.
Scoring logs completion, total/component rewards, and verifier diagnostics.

## Verification response

Requests specify schema version, operation, code, cycles, and event limits.
Operations are `verify`, `decompose`, and supported literal-scale `transpose`.
Responses include validity, failure stage/code/message, runtime version, elapsed
time, actual cycle window, and bounded events/telemetry when successful.
Failure stages distinguish syntax, runtime setup, evaluation, arc querying,
sound catalog, timeout, and protocol limits. No downstream control flow parses
human-readable error messages.
