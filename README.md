# TuneLM

TuneLM generates executable [Strudel](https://strudel.cc/) code from music descriptions.
The browser runs the code to produce audio.

[Try TuneLM](https://www.tensorwrites.com/tunelm)

## Use the web app

Enter a music description. Read the composition notes or open **Code** to copy the Strudel program. In **Music**, play the full mix or an individual layer. Use **Download** to save the audio as a WAV file.

Send a follow-up message to revise the composition. The app checks that the code compiles and produces audio before it shows a result.

## Dataset

We used Gemini 4 Argon and Carbon to create synthetic pairs of music descriptions and Strudel programs. We checked the programs with the Strudel verifier before training. The reinforcement learning stage uses separate prompts with musical constraints. See the [dataset format](docs/training-io.md).

## How we trained the model

1. **Supervised fine-tuning (SFT).** We used low-rank adaptation (LoRA) to teach [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) to write Strudel programs from music descriptions.
2. **Reinforcement learning (RL).** We used Group Relative Policy Optimization (GRPO). The [reward function](docs/rl-reward-design.md) checks whether a program runs and scores how well it meets the requested musical constraints, such as tempo, key, and instrument roles.

## Deployment

Vercel hosts the web app and its API. The API sends composition requests to Modal. Modal runs the model and returns Strudel code. See the [Vercel configuration](vercel.json) and the [Modal inference service](scripts/modal_inference.py).

## Further reading

- [Pipeline usage](docs/usage.md)
- [Audio evaluation](docs/audio-evaluation.md)

The code uses the [AGPL-3.0-or-later license](LICENSE).
