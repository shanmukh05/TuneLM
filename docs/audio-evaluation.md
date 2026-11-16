# MusicPrefs audio evaluation

The benchmark scores saved audio for a shared set of prompts. The default
comparison includes the seven archived MusicPrefs systems and both saved TuneLM
checkpoints.

## Saved benchmark assets

The benchmark lives at `datasets/audio-benchmarks/musicprefs`. Its `prompts.csv`
contains 120 stable prompt IDs and their exact text. The seven archived model
directories contain one WAV per prompt, with selection provenance in
`musicprefs-takes.csv`. These clips come from the pinned
[MusicPrefs snapshot](https://huggingface.co/datasets/i-need-sleep/musicprefs/tree/84b577959a31b2488dcf179c2f49d19ad41a32e5).
Copy the prepared dataset directories when setting up another host. Generated
audio, references, and results are Git-ignored.

## Generate WAVs with an LLM provider

Use Node 22 or newer and install the existing teacher dependencies in the
project environment. Set the provider key in the shell or `.env.local`.

```bash
npm ci
uv pip install --python .venv/bin/python -e '.[teacher]'
.venv/bin/python scripts/generate_provider_audio.py --provider gemini --model 'MODEL_ID'
```

The supported providers are `gemini`, `openai`, `anthropic` (or `claude`),
`deepseek`, and `openrouter`. Use the exact model ID for that provider. The
script prints its experiment name and saves 30-second stereo WAVs at
`datasets/audio-benchmarks/musicprefs/audio/<experiment>/<prompt_id>.wav`.
It records each provider response, token usage when reported, and request time
in `completions/`. A rerun skips saved WAVs and reuses saved completions. The
default allows one request per missing prompt. Pass `--attempts 3` to allow two
repair attempts with the preceding verifier or renderer error as feedback. The
script supplies the project's exact sound IDs and Strudel API reference. Models
listed in the Song Describer profile inherit its token budget and thinking level.
Remaining failures appear in `failures.json`. Pass `--resume` to use updated
settings in an existing run. The script records the settings history, keeps
saved WAVs, and checks saved completions before any new request. Set `--name`
when you want a separate experiment directory.

## Select experiments and score

Install FFmpeg and the separate Python 3.12 metrics environment.

```bash
uv venv --python 3.12 .venv/audio-eval
uv pip install --python .venv/audio-eval/bin/python -e . -r scripts/requirements/audio-eval.txt
.venv/audio-eval/bin/tunelm evaluate-audio --config configs/eval-musicprefs-sft.yaml
```

Edit `experiments` in the YAML to choose which saved runs to score. A string
selects `<benchmark>/audio/<name>`. An entry with `name` and `audio_dir` can refer
to saved audio elsewhere, as the GRPO entry does. Add a provider experiment by
placing the printed name in the list. To select runs for a single invocation,
pass their names with `--experiments`.

```bash
.venv/audio-eval/bin/tunelm evaluate-audio \
  --config configs/eval-musicprefs-sft.yaml \
  --experiments tunelm-grpo-step150 PROVIDER_EXPERIMENT_NAME \
  --output results/audio-eval/provider-comparison
```

This mode reads saved WAVs only. With `match_rendered: true`, it scores the
prompt IDs shared by the selected experiments and reports missing IDs in
`metrics.json`. Device `auto` uses CUDA when available, and you can explicitly
choose `--device cpu` or `--device cuda` on Vast.ai. The checkpoint generation
script remains available for creating missing TuneLM WAVs. The offline renderer
also needs the JACK runtime library on Linux (`libjack-jackd2-0` on Ubuntu).

For a prepared CSV manifest, use `tunelm score-audio --input INPUT_CSV` with
`--output`, `--seconds`, and `--mad-reference`. The CSV columns are
`model_name`, `audio_path`, `prompt`, and optional `prompt_id`. Paths may be
absolute or relative to the CSV. Longer windows require saved clips of that
duration. The archived 10- and 20-second take selections are under
`datasets/audio-benchmarks/musicprefs-10s` and `musicprefs-20s`.

## Reuse FMA-Pop and clip embeddings

The shared `datasets/audio-benchmarks/fma-pop` directory contains
`mad-5s.npz`, `mad-10s.npz`, and `mad-20s.npz` from completed runs. The
scorer picks the file for the requested duration, verifies its MERT revision and
layer, and records its checksum. `references.json` preserves the source run and
track count for each archive. Copy this directory to another host to score
without the FMA audio. A direct NPZ path is also supported by `--mad-reference`.
FAD and FAD-CLAP use the checksum-pinned statistics bundled with FADtk.

The default shared clip cache is `results/audio-eval/embeddings`. Audio
embeddings are reused across result directories when the audio bytes, device,
and scoring protocol match. The same model may be rescored on a different
common prompt set using its cached features.

## Scoring and artifacts

The default scores the first **five seconds** so all seven archived systems can
be included. The evaluator rejects undecodable, shorter-than-window, and silent
inputs. Each scoring window uses its own saved MAD reference. The CLAP cosine
uses the LAION AudioSet model and the original
prompt. Standard FAD uses VGGish and LAION-CLAP Audio with fixed FMA-Pop
statistics. MAD uses max-pooled layer 24 MERT features and negative log MAUVE.

- `inputs.csv` records the files scored by `evaluate-audio`.
- `metrics.csv` has one row per model with prompt counts and the four metrics.
- `metrics.json` includes model scores, the decoding and encoder protocol,
  and checksums of the references.
- `clap.csv` has each file's prompt ID and CLAP score.
- Cached clip embeddings are keyed by audio content and scoring protocol.
  The configured shared cache works across output directories.

The common prompt set and scoring settings make the newly scored rows directly
comparable within this run. The [published MusicPrefs Table 2](https://arxiv.org/html/2503.16669v1#S4.T2)
used a larger generation pool and describes an FMA-Pop reference of 4,230 songs.
The bundled FAD reference and the five-second protocol have not been verified
against that table's exact inputs. Label these results as the TuneLM evaluation
on the public MusicPrefs snapshot.
