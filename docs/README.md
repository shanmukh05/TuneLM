# TuneLM documentation

TuneLM builds Strudel datasets, trains SFT and GRPO models, and serves live SFT
predictions through Modal.

- [Usage](usage.md): plain-language dataset workflow, setup, and pipeline commands.
- [Live SFT inference on Modal](usage.md#live-sft-inference-on-modal) covers deployment and the dataset viewer.
- [MusicPrefs audio evaluation](audio-evaluation.md) covers saved audio, TuneLM rendering, and four audio metrics.
- [Implementation](implementation.md): actual architecture and limitations.
- [Training I/O](training-io.md): records, formatting, and loss masks.
- [RL reward design](rl-reward-design.md): the GRPO request-contract reward and how to build the RL bank.
- [Eval bench design](../design_docs/eval-bench-design.md): proposed evaluation of any LLM on the Strudel task.
- [RL rewards](../design_docs/rl-rewards.md): earlier two-objective proposal, superseded by RL reward design.
- [Verification](verification.md): evidence and outstanding hardware checks.
- [Source research](source-research.md): selected collections, licenses, import results, and exclusions.
- [Data layout](../datasets/README.md): JSON/JSONL and native artifact formats.
- [Implementation plan](../design_docs/implementation-plan.md): requirements and planned milestones.
- [Source proposals](../design_docs/): unchanged references.
