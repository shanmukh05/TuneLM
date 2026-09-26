# Training input and output formats

TuneLM has three task types—`compose`, `edit`, and `repair`—and two training
stages. Both stages show the model the same shared system instruction (defined
in `src/tunelm/models/templates.py`) followed by one task-specific user
message. The model is instructed to return **one complete JSON object**, not audio, a
code diff, Markdown, or a multi-message conversation.

## What each stage stores and trains on

| Stage | JSONL input | Model input | Target / output |
|---|---|---|---|
| SFT | `datasets/sft/train.jsonl`: task fields, a teacher `response`, and `metadata` | System instruction + formatted task user message | The serialized `response` JSON followed by the tokenizer's end-of-sequence token. Loss applies only to this completion. |
| GRPO (RL) | `datasets/rl/train.jsonl`: task fields, **no** `response` | The same system instruction + formatted task user message | The policy samples complete JSON responses. Format, execution, prompt-stated constraints, edit, repair, and compose richness rewards score them; there is no gold completion in this file. |

Each JSONL file has one JSON object per line. The examples below show the
relevant fields, not every optional/default field that a generated row may
contain. In particular, serialized SFT rows may contain `null` fields for
other task types. `id`, `task_type`, and the verifier fields select/score the
task; they are **not** automatically pasted into the model's user message.
Only requirements expressed in natural-language `prompt` (and, for edit, its
`instruction`/`preserve` context) are visible to the model.

### Compose

Stored task fields (the SFT row also has `response` and `metadata`):

```json
{
  "id": "compose-001",
  "task_type": "compose",
  "prompt": "Compose a mellow 90 BPM piano piece that gradually adds a soft bass.",
  "constraints": {
    "tempo": 90,
    "required_instruments": ["gm_piano", "sine"],
    "min_layers": 2
  }
}
```

The model-visible **user** message is exactly the `prompt` text:

```text
Compose a mellow 90 BPM piano piece that gradually adds a soft bass.
```

The output is a complete composition with a plan, controls, and executable
Strudel. Machine-readable `constraints` stay on the dataset row for validation
and rewards; the 90 BPM requirement is visible here because the prompt itself
says it. GRPO scores only constraints the prompt states: had this prompt said
"a mellow piano piece", the tempo and `min_layers` targets would be dropped
from the reward (piano is still named, so that instrument check would remain).

### Edit

Stored task fields (again, SFT additionally stores `response` and `metadata`):

```json
{
  "id": "edit-001",
  "task_type": "edit",
  "prompt": "Make the bass busier without changing the drums.",
  "instruction": "Increase bass activity and preserve the drum pattern.",
  "original_code": "setcpm(120/4)\nconst drums = s('bd sd').bank('RolandTR909')\nconst bass = note('c2 ~ g2 ~').s('sawtooth')\nstack(drums, bass)",
  "preserve": ["tempo", "drums"],
  "expected_changes": {"drum_density": "unchanged", "layer_count": "unchanged"},
  "constraints": {}
}
```

The model-visible user message is assembled with blank lines between parts:

```text
Make the bass busier without changing the drums.

Edit instruction: Increase bass activity and preserve the drum pattern.

Original Strudel:
setcpm(120/4)
const drums = s('bd sd').bank('RolandTR909')
const bass = note('c2 ~ g2 ~').s('sawtooth')
stack(drums, bass)

Preserve: tempo, drums
```

If `instruction` equals `prompt`, the extra `Edit instruction:` paragraph is
omitted. An empty preserve list appears as `Preserve: nothing specified`.
`expected_changes` is **not** shown to the model; the reward uses it to check
the edit. The output must contain the **entire revised Strudel program** and
updated plan/controls, not just changed lines.

### Repair

Stored task fields (SFT also has `response` and `metadata`):

```json
{
  "id": "repair-001",
  "task_type": "repair",
  "prompt": "Fix the syntax while keeping the same groove.",
  "broken_code": "setcpm(110/4)\ns('bd sd'",
  "reference_code": "setcpm(110/4)\ns('bd sd').bank('RolandTR909')",
  "corruption": "missing_closing_parenthesis",
  "constraints": {}
}
```

The model-visible user message is:

```text
Fix the syntax while keeping the same groove.

Repair this Strudel program:

setcpm(110/4)
s('bd sd'
```

`reference_code` and `corruption` remain hidden. The reference, when present,
helps score repair consistency. The output is the **complete fixed program**
inside the same JSON contract; it should retain the musical idea.

## Shared assistant response

All three tasks use this shape for an SFT teacher label, a GRPO candidate, and
an inference result. This is an illustrative valid-shape response to the edit
example above:

```json
{
  "plan": {
    "tempo": 120,
    "instruments": ["RolandTR909", "bd", "sd", "sawtooth"],
    "structure": "Drum groove with a more active bass line",
    "musical_intent": "Keep the beat steady while the bass adds motion"
  },
  "controls": {
    "bpm": 120,
    "key": "C:minor",
    "mood": [],
    "layers": [
      {"id": "drums", "role": "rhythm", "sounds": ["bd", "sd"]},
      {"id": "bass", "role": "bass", "sounds": ["sawtooth"]}
    ]
  },
  "strudel_code": "setcpm(120/4)\nconst drums = s('bd sd').bank('RolandTR909')\nconst bass = note('c2 g2 eb2 g2').s('sawtooth')\nstack(drums, bass)"
}
```

`plan` is the human-readable summary; `controls` is the machine-editable
description and must agree with `strudel_code`. Required top-level fields are
exactly `plan`, `controls`, and `strudel_code`; the strict schema rejects extra
or missing fields. `controls.bpm` is 20–400, `key` may be `null`, `mood` is an
array, and `layers` contains at least one `{id, role, sounds}` object. The
program's final expression must evaluate to the composition. Compose outputs
should have adequate musical form/length; edit and repair outputs should
preserve unaffected parts. See `src/tunelm/schemas.py` and the shared system
instruction in `src/tunelm/models/templates.py` for the exact contract.

## Conversion into trainer inputs

- SFT converts a row to a chat-templated prompt (`system` + `user` + assistant
  generation prefix) and a serialized assistant JSON completion. The teacher
  `response` is never included in the prompt. The training set's original
  columns are removed after conversion.
- GRPO (`tunelm.rl.trainer.prepare_rl_row`) replaces the stored text `prompt`
  with a conversational `prompt` containing `system` and `user` messages and
  keeps a fixed set of reward columns on **every** row: `task_type`,
  `constraints` (pruned to what the prompt states), `original_code`,
  `expected_changes`, `preserve`, and `reference_code` (`null` when not
  applicable). Uniform columns are required because Arrow cannot store a mixed
  task bank whose rows have different keys. Inapplicable rewards return `None`
  rather than rewarding a missing constraint.
- The current training format is **single-turn**. Edit provides the current
  Strudel source, not a serialized history of earlier conversation turns. A
  conversational application must supply the latest code in each edit request;
  multi-turn history training would require an explicit dataset/formatter change.

For the pipeline diagrams, see [workflows.md](workflows.md). For commands and
configuration, see [usage.md](usage.md).
