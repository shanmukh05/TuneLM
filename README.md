# TuneLM

TuneLM turns a music description into executable [Strudel](https://strudel.cc/) code. Strudel is a JavaScript live-coding environment for music. The browser plays the program. You can edit the source to change the music, run it in Strudel, or export the rendered audio.

This repository contains the web app, the dataset generation pipeline, the Strudel verifier, and the code for training and evaluating the model.

[Try TuneLM](https://www.tensorwrites.com/tunelm) · [View a Strudel example](examples/groove.js)

## Use the web app

1. Open **Try TuneLM** in the app. Enter a prompt such as “Make a 120 BPM groove in C minor with drums, bass, and soft chords.”
2. Read **Composition notes** to see how the model interpreted the request. Open **Code** to inspect or copy the program.
3. Open **Music** to play the full mix or an individual layer. Click or drag on the waveform timeline to change the playback position. Use **Download** to export a WAV file.

Keep the chat open to revise the composition with a follow-up prompt. The app compiles the generated code and checks that it produces audio before it shows the result. If a check fails, the app can ask the model to repair the code.

## Dataset

We used Gemini 4 Argon and Carbon to create synthetic training examples. Each supervised example pairs a listener request with composition notes and a Strudel program. The verifier runs the program and checks for playable events and supported sounds before the pipeline accepts the example.

Reinforcement learning uses a separate set of requests. Each request includes checks for the musical constraints it specifies, such as tempo, key, or instruments. The model generates its own answer during training. The pipeline stores training, validation, and benchmark records as JSON Lines. The repository includes [starter patterns](examples/seeds.jsonl). The [dataset guide](docs/usage.md#dataset-creation-in-plain-language) shows how to generate the training records.

## Training and evaluation

1. **Supervised fine-tuning (SFT).** We fine-tuned [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) with low-rank adaptation (LoRA). This stage teaches the model to write Strudel programs and describe its musical choices.
2. **Reinforcement learning (RL).** We continued training with Group Relative Policy Optimization (GRPO). The model generates several responses per prompt. Invalid programs receive zero reward. The [reward function](docs/rl-reward-design.md) scores valid programs on how well they meet the requested constraints, including rules such as “no drums.”

The repository provides two ways to evaluate a model. The [code benchmark](docs/usage.md#local-inference-and-symbolic-benchmark-evaluation) checks whether generated programs run and follow held-out requests. The [MusicPrefs evaluation](docs/audio-evaluation.md) compares rendered audio across models on the same 120 prompts.

## Run locally

Install Node.js 22 or later. Run these commands from the repository root.

```sh
npm ci
npm run dev:web
```

Open `http://127.0.0.1:4173/tunelm`. Play the example on the **Overview** tab to try playback. Live generation requires a running inference service. Use Python 3.11 or later for the [dataset, verification, and training commands](docs/usage.md).

## Deployment

Vercel builds the web app with `npm run build:web` and serves it at `/tunelm`. The browser starts a composition job and polls for the result through the Vercel API. The API forwards these requests to Modal. Modal runs the model and returns the composition notes and Strudel code. The browser renders the audio.

Use the [Vercel configuration](vercel.json) for the app and API routes. Use the [Modal entry point](scripts/modal_inference.py) to deploy the inference service.

## Repository layout

| Path | Contents |
| --- | --- |
| [`web/`](web/) and [`api/`](api/) | Chat UI, browser audio playback, and the composition API |
| [`src/tunelm/`](src/tunelm/) | Python CLI, dataset generation, training, and evaluation |
| [`verifier/`](verifier/) | Strudel syntax and execution checks |
| [`configs/`](configs/) | Settings for dataset generation, training, inference, and evaluation |
| [`docs/`](docs/) | Pipeline commands, data formats, and reward design |

The code uses the [AGPL-3.0-or-later license](LICENSE).
