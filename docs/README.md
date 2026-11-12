# TuneLM documentation

The LFM rebuild implements both SFT and GRPO. Local verification is complete;
GPU training, model inference, and paid teacher runs are pending by request.

- [Usage](usage.md): plain-language dataset workflow, setup, and pipeline commands.
- [Implementation](implementation.md): actual architecture and limitations.
- [Training I/O](training-io.md): records, formatting, and loss masks.
- [RL reward design](rl-reward-design.md): the GRPO request-contract reward and how to build the RL bank.
- [Eval bench design](eval-bench-design.md): proposed evaluation of any LLM on the Strudel task.
- [RL rewards](rl-rewards.md): earlier two-objective proposal, superseded by RL reward design.
- [Verification](verification.md): evidence and outstanding hardware checks.
- [Source research](source-research.md): selected collections, licenses, import results, and exclusions.
- [Data layout](../datasets/README.md): JSON/JSONL and native artifact formats.
- [Implementation plan](../design_docs/implementation-plan.md): requirements and planned milestones.
- [Source proposals](../design_docs/): unchanged references.
