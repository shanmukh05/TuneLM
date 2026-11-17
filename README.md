# TuneLM

TuneLM generates executable [Strudel](https://strudel.cc/) code from music descriptions.
The browser runs the code to produce audio.

[Try TuneLM](https://www.tensorwrites.com/tunelm) · [Documentation](docs/README.md)

## Use the web app

1. Enter a music description.
2. Open **Composition notes** to read the model's notes.
3. Select **Code** to inspect or copy the Strudel program.
4. Select **Music** to play the full mix or an individual layer.
5. Select **Download** to save the audio as a WAV file.

Send a follow-up message to revise the composition. The app compiles the code and renders the full mix before it shows a result. If validation fails, the app can request up to two repairs.

## How we trained the model

1. **Dataset creation.** We used Gemini 4 Argon and Carbon to create pairs of music descriptions and Strudel programs. The verifier checked the programs before training.
2. **Supervised fine-tuning (SFT).** We used low-rank adaptation (LoRA) to teach [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) to write Strudel programs from music descriptions.
3. **Reinforcement learning.** We used Group Relative Policy Optimization (GRPO). The reward function scored programs on execution and how well they met the musical constraints.

The [inference configuration](configs/inference.yaml) selects the [GRPO step 250 checkpoint](https://huggingface.co/tensorfiend/tunelm-v2-lfm3-1.7B-grpo-step250). The checkpoint requires Hugging Face access.

## Run the web app locally

Use Node.js 22 or later. Run these commands from the repository root.

```sh
npm ci
npm run dev:web
```

Open [the local app](http://127.0.0.1:4173/tunelm). Play the example on the **Overview** tab to check audio playback.

Vercel hosts the live web app. Modal hosts the [inference service](scripts/modal_inference.py).
For live composition from the local app, connect to a deployed Modal service. For an HTTP connection, set these variables in `.env.local`.

| Variable | Value |
| --- | --- |
| `TUNELM_MODAL_URL` | Base URL of the Modal `web` function |
| `TUNELM_MODAL_KEY` | Modal proxy token ID |
| `TUNELM_MODAL_SECRET` | Modal proxy token secret |

## Project guides

- [Usage](docs/usage.md) covers data preparation, training commands, and inference.
- [Training records](docs/training-io.md) describes the data and model response formats.
- [GRPO rewards](docs/rl-reward-design.md) explains the execution checks and musical scores.
- [MusicPrefs evaluation](docs/audio-evaluation.md) explains how to compare rendered audio.

The code uses the [AGPL-3.0-or-later license](LICENSE).
