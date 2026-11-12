# RL reward design: request contracts

This is the GRPO reward and the RL input bank it needs. It replaces the
two-objective idea in [RL rewards](rl-rewards.md). Code: `rewards.contract_score`,
`rewards.check_contract`, `training.MusicalReward`, and
`configs/registry/sound_families.yaml`.

## Summary

Each RL request carries a **request contract**: the things the listener literally
said that one verifier run can check. These are named instruments, refusals ("no
drums"), a part count ("solo", "two-part"), a stated tempo or key, a change over
time ("then it thins out"), sparse or busy, and kept code for co-creation. Every
claim quotes the words in the request that justify it. The prompt author writes
the contract in the same call as the request, and local code rejects any claim
whose quote is not in the request.

```
R = gate × 0.5^(failed craft guards) × (0.8·contract + 0.2·honesty) − overlong
```

Mood, genre, and "make it feel like X" are never scored. They stay in the request
as context. No symbolic check can tell whether a program sounds tired or pleased,
and paying for a guess trains the guess.

## Why not the earlier proposals

- **The plan is the teacher's answer, not the request.** Every SFT row has
  `intent_only: true`, and its plan records what the teacher chose. Scoring
  rollouts against a plan rewards copying one teacher's arrangement.
- **Literal-only prompts saturate.** "Use euclid at 92 bpm" is easy after SFT.
  When all four samples score 1, GRPO learns nothing from that prompt. These
  prompts also read nothing like the SFT prompts.
- **Scene cards guess.** "Density: low" from "keep the dishes company" is an
  opinion, and nothing catches a wrong card.
- **The old proxies can be gamed.** A linear `evolution` term pays for
  alternation and randomness. `roles` pays for notes in every register.

## The reward

### Gate

`R = 0` unless the reply is one non-empty `<think>` block followed by one
`strudel` block, and the verifier accepts the program. The program must run in
time, stay under the event cap, use only catalog sounds, and have audible onsets.
Truncated completions are already dropped from the loss by
`mask_truncated_completions: true`.

### Contract

| Field | Filled when the request says | Measured | Weight |
|---|---|---|---|
| `require` | an instrument or family by name | an executed sound in that family | 1 each |
| `forbid` | a refusal, such as "no drums" or "no piano" | no executed sound in that family | 2 each |
| `forbid: vocals` | "instrumental", "no singing" | no choir or voice sound | 0.5 |
| `voices` | a count: "solo", "two-part", "all three at once" | see below | 1 |
| `tempo` | digits plus `bpm` | `setcpm × 4` within 1 BPM | 1 |
| `key` | a written root and mode | ≥ 90% of pitched onsets in the scale | 1 |
| `shape` | `build`, `thin_out`, or `contrast` | a later quarter of the piece has ≥ 1.5× the onsets or more sounds than an earlier one, has fewer, or differs | 0.5 |
| `space` | `sparse` or `busy` | ≤ 8 or ≥ 12 events per cycle | 0.5 |
| `loop` | "a short loop" | turns off the variation guard | none |
| preserve | co-creation `original_code` | original kept verbatim **and** the result's telemetry differs from the original's | 2 |

`contract` is the weighted fraction of checks that pass. A refusal weighs double
because breaking "no drums" is the failure a listener notices first. Preserving
code requires an audible change too, so pasting the original back scores nothing.

**Voices.** Sounds inside one named family count as one part, so slap plus
fretless for "bass alone" is one bass. The drum kit counts as one part toward the
maximum, while each drum sound counts separately toward the minimum, so "three
dry percussion hits" can be three kit pieces. Any other extra sound counts on its
own.

**Families.** `sound_families.yaml` is a closed table of 72 families. Each family
lists exact sound names, drum-kit suffixes (so `bd` and `RolandTR909_bd` are both
kick), and for `bass`, any pitched note at or below C3. The loader rejects any
name that is missing from `sounds.yaml`. A word with no family, such as "clay pot"
or "sledgehammer", stays out of the contract, and the model may voice it however
it likes.

### Craft guards

Each failed guard halves the reward. A degenerate program that ticks every
contract box still ranks below a real attempt. Every band passes at least 98% of
verified SFT answers, so the guards remove junk and do not push pieces toward one
arrangement.

| Guard | Passes when |
|---|---|
| Density | 0.5 ≤ events per cycle ≤ 64 |
| Variation | structural variation ≥ 0.125, skipped for `loop` |
| Not stuffed | parts ≤ max(10, `voices` maximum) |
| Gain | 0 < gain max ≤ 1.0 |
| Pitch | best-fit diatonic coverage ≥ 0.6, skipped with fewer than 4 pitches or when `key` is scored |

### Honesty

The thinking is shown to people, and RL tends to let it drift from the code.
A BPM written in the thinking must match `setcpm`, and a root plus mode written
in the thinking must cover ≥ 80% of pitched onsets. Honesty is 1 when the
thinking makes no such claim.

### Length

Masked truncation is unchanged. On top of it, a DAPO-style penalty rises
linearly from 0 at `max_completion_length − 256` tokens to 0.2 at the limit.

### Ordering check

These are hand-written rollouts for "No drums. A cello and a harp trading
phrases, and let it thin out at the end":

| Rollout | R |
|---|---|
| Follows the request | 1.00 |
| Same program, but the thinking claims 120 bpm in E major | 0.80 |
| Adds a drum loop | 0.56 |
| A static two-note loop that still uses cello and harp | 0.46 |
| Co-creation: original code pasted back unchanged | 0.10 |
| Co-creation: original kept and a bass line added | 1.00 |
| Does not parse | 0 minus the length penalty |

## Calibration on SFT answers

The scorer ran on all 2,016 verified SFT teacher answers, using rough regex
contracts as a stand-in for author-written ones. All 2,016 passed the gate.

| Check | Teacher pass rate |
|---|---|
| Craft guards | 0.98–1.00 each |
| Honesty: tempo, key | 1.00, 0.95 |
| Preserve (243 co-creations) | 0.99 |
| Forbid drums, pads | 0.97, 1.00 |
| Named instruments (tuba, cello, clarinet, banjo, and others) | mostly 0.93–1.00 |
| Voices: solo, two parts, three parts | 0.88, 0.47, 0.80 |
| Space: sparse | 0.85 |
| Shape: build, thin out, contrast | about 0.6, 0.6, 0.89 |

Low instrument rates such as "harp" or "strings" came from the regex stand-in.
It read "mouth harp" as harp and "avoid sweeping strings" as a requirement, so
those rows say nothing about the table. The voice and shape rows are different.
In the failing rows we inspected, the teacher really did add a drum bed to a
"two-part" piece, or barely thinned a section the listener asked to drop. Those
checks are stricter than the teacher on purpose, because that is the behavior
RL should fix. Recheck these rates on the real contract bank before training
(see the last section).

## The RL bank

### What to generate

RL needs **requests and contracts only**. No teacher answers, plans, or reasoning
are generated for RL. Each row is a `MusicRequest` with `stage: rl` and a
`contract`. `tunelm export-rl` turns those rows into answer-free `Task` rows that
keep the contract. RL authoring and export live in `src/tunelm/rl_prompts.py`;
SFT answer generation never reads the RL bank.

| Split | Count | Used for |
|---|---|---|
| train | 2,000 | GRPO prompts |
| val | 150 | reward tracking during training |
| benchmark | 150 | base, SFT, and GRPO comparison, plus listening |

With 4 generations and 4 prompts per optimizer step, 2,000 prompts make one
500-step epoch, or 8,000 rollouts. Expect about 10–20% of authoring attempts to be
rejected by the contract check and retried. 1,500 train prompts is a reasonable
floor. Prefer more distinct prompts over a second epoch.

The bank should sound like the SFT bank: the same spoken voice, situations, and
styles. Each request should carry at least two concrete asks, and one of them
must be an instrument, a refusal, a part count, a tempo, a key, or supplied code.
`configs/prompts-rl.yaml` sets this up:

- Its style schedule drops "vague", because a request with nothing checkable
  cannot be scored. Prompts written after the first part of the bank also state a tempo and a key.
- Its focuses point at checkable asks: one named instrument alone, two that
  answer each other, three that fit, a refusal plus what they want, sparse then
  full, busy then nearly empty, rhythm with named percussion, and a short loop.
- Categories are 3/7 stack, 2/7 arrange, 1/7 atomic, and 1/7 co-creation.
- It is standalone, with its own authors, openings, and style texts. The style
  texts match `prompts-sft.yaml` except that none ask to leave tempo and key open.

Good RL requests, with their contract fields:

- "No drums, please. Just a cello and a harp trading short phrases while I fold
  laundry, and let it thin out at the end." Contract: forbid drums, require cello
  and harp, voices 2–2, shape thin_out.
- "Solo kalimba for the bus ride, nothing under it." Contract: require kalimba,
  voices 1–1.
- "Kick, snare, and a shaker for my rowing warmup at 128 bpm." Contract: require
  kick, snare, and shaker, tempo 128.
- "Can you keep this beat and add a low bass line under it?" with
  `original_code`. Contract: require bass, plus preserve.

A request like "Something silly for my goldfish" is rejected, because nothing in
it can be checked.

### Commands

```bash
tunelm create-prompts --config configs/prompts-rl.yaml --dry-run
tunelm create-prompts --config configs/prompts-rl.yaml
# Review datasets/prompts/generated/rl/*.jsonl: read about 50 contracts.
tunelm export-rl                      # datasets/creative/rl/{train,val,benchmark}.jsonl
tunelm train-grpo --config configs/grpo.yaml --dry-run --tokenize
```

`train-grpo` refuses RL rows without a contract. Point `train_file` and
`val_file` in `configs/grpo.yaml` at the `rl/` export of the dataset directory
you used.

### Reviewing contracts

The authoring check guarantees that quotes are real and that family names exist.
It cannot tell whether a quote supports its claim. Read about 50 rows and look
for these mistakes:
- an implied instrument written as `require`, such as "a jazz trio" written as
  piano, bass, and drums
- a count that is really a style name, such as "a two-step groove" written as
  voices 2
- "drops" in a register sense, such as "the melody drops lower", written as
  `thin_out`

Delete bad rows rather than editing them. The author refuses edited JSONL.

## GRPO settings

- Keep `num_generations: 4`. Four is enough once saturated prompts are removed.
  Keep TRL's default `loss_type: dapo` and a small `beta`, as now.
- The current config trains 1 prompt per optimizer step:
  `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, and
  `generation_batch_size: 4`. For steadier steps, use 4 prompts per step:
  per-device 4, accumulation 4, and generation batch 16. This depends on GPU
  memory, which has not been measured.
- One verify of a realistic SFT program takes about 0.24 s, or about 1 s per
  group of 4. This is small next to generating up to 1,536 tokens.

## Monitoring

- TRL logs `frac_reward_zero_std`. If it stays above about 0.4 after the first 50
  steps, many prompts are too easy or too hard. Drop the task IDs whose rollouts
  all scored the same in `results/lfm3-grpo/rollouts-*.jsonl`, then resume on the
  smaller bank.
- The rollout JSONL logs every component: `require.cello`, `forbid.drums`,
  `voices`, `craft.*`, `honesty.*`, and `overlong`. Watch for one check that
  climbs to 100% while the others stall. Watch for a rising part count or rising
  events per cycle, which means the model is stuffing the mix.
- Every 50 steps, listen to the five highest-reward and five lowest-reward
  rollouts. The reward cannot tell whether music is good. Only listening can.

## Evaluation

`tunelm evaluate` scores any benchmark task that carries a contract with
`contract_score`. Other tasks keep the SFT plan score. Compare gate rate and
contract rate for base, SFT, and GRPO on the RL benchmark split. Then run a blind
A/B listening test of SFT against GRPO on about 30 prompts. GRPO should win on
"did what I asked" without losing on "sounds good". CLAP or an LLM judge, if
added, belongs here and not in the training reward.

## Before the first run

- Calibrate on real contracts. Run `tunelm predict` with the SFT checkpoint on the
  RL benchmark tasks, then run `tunelm evaluate`. `scores.jsonl` gives per-check
  pass rates for the SFT policy. A check that the SFT model almost always passes
  adds little signal. A check it almost never passes, when the listener said it
  plainly, points to a measurement bug. The thresholds marked `ponytail:` in
  `rewards.py` are first values, not tuned constants.
