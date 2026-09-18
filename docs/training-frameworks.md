# Training framework decision

This decision is specific to TuneLM: 5,000 SFT rows, 3,000 verifier-driven RL
tasks, text-only inputs, custom executable Strudel rewards, and LoRA/QLoRA
checkpoints.

## Recommendation

| Scale | SFT | RL | Why |
|---|---|---|---|
| Current 0.6B–2B models | TRL `SFTTrainer` + PEFT | TRL `GRPOTrainer` + PEFT | Smallest maintained stack that accepts conversational data and in-process custom rewards. |
| Qwen3.8-27B SFT | Axolotl + PEFT/QLoRA, after an architecture smoke | — | Better operational FSDP2/DeepSpeed, preprocessing, masking checks, and checkpoint controls at this scale. |
| Qwen3.8-27B, serious RL | — | prime-rl or verl | Both separate fast rollout inference from distributed training; this is the point where the single-process TRL path stops being the right production architecture. |

PEFT is the adapter implementation, not the overall training framework. Keeping
TRL for the present models therefore does not mean PEFT was selected as the
framework.

## Why not migrate now

TRL already supports completion-only SFT, PEFT adapters, custom GRPO reward
functions, and vLLM acceleration. Migrating the current small-model pipeline
would replace working interfaces without solving a present bottleneck. The
current code should move to TRL's vLLM server mode only after the ordinary GRPO
smoke run is correct and generation is measured as the bottleneck.

[TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer) describes
prompt-completion loss masking and packing. [TRL GRPO/vLLM
documentation](https://huggingface.co/docs/trl/vllm_integration) describes the
separate rollout server.

For 27B SFT, Axolotl becomes the better training application rather than a
better adapter library: it wraps PEFT but adds validated YAML, dataset
preprocessing/debugging, FSDP2/DeepSpeed, masking, W&B, and checkpoint/export
workflows. Its own guide recommends QLoRA for 30–70B models. Qwen3.8 is newer
than the examples reviewed here, so support must be proven with a one-step
smoke before selecting hardware for a full run.

- [Axolotl SFT reference](https://docs.axolotl.ai/docs/agents/sft.html)
- [Axolotl method and hardware guide](https://docs.axolotl.ai/docs/choosing_method.html)

## Prime-RL versus verl for 27B RL

Prime-RL is a strong match when TuneLM becomes an asynchronous environment:
an inference fleet produces rollouts, an orchestrator runs isolated verifiers,
and one or more trainer processes update the policy. Its normal full RL setup
uses separate inference and trainer GPUs, so it is not a drop-in library swap
for the current single-GPU Jarvis command. Porting requires a TuneLM verifier
environment and new deployment/configuration code.

Verl is the safer alternative when the immediate goal is conventional,
large-scale GRPO. It has established FSDP/FSDP2 and Megatron training backends,
vLLM/SGLang rollouts, and LoRA support. It also requires a separate training
integration rather than reusing `src/tunelm/rl/trainer.py`.

- [prime-rl overview](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/overview.md)
- [prime-rl scaling](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/scaling.md)
- [verl GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [verl LoRA](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)

Qwen3.8-27B is a native vision-language 27B model, although TuneLM needs only
its language path. Its model card states Transformers and vLLM compatibility.
Before committing a long run, perform three hardware smokes: causal-model load,
one QLoRA optimizer step, and one rollout/reward/update step. Do not assume that
support for an earlier Qwen3 model proves support for Qwen3.8's architecture.

[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
