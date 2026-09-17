# TuneLM workflows

End-to-end diagrams for data preparation and training. Staged paths let you
review prompt banks before expensive teacher generation (SFT) or before locking
an RL task bank.

Committed smoke fixtures:

- SFT: `datasets/sft/sample.jsonl`
- RL: `datasets/rl/train.jsonl` (procedural/hybrid reference bank; regenerate for production)

## Model providers

Configured under `models` (preferred) or legacy `providers`. Each entry uses the
YAML key as the **model id** stored in dataset metadata:

```yaml
models:
  gemini-flash:
    enabled: true
    provider: gemini          # gemini | openai | anthropic | openrouter | deepseek
    model: gemini-2.5-flash   # API model slug (alias: model_name)
```

API keys (set in the shell, never committed):

| Provider | Environment variable |
|---|---|
| Gemini | `GEMINI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

When `batch.enabled: true` (default in `configs/data/sft.yaml`), SFT generation
uses the Gemini Batch API at 50% of standard pricing. Set `batch.enabled: false`
to fall back to synchronous generation with optional `parallel_models`.

## SFT data preparation

### Batch workflow (`batch.enabled: true`)

```mermaid
flowchart TD
  A[configs/data/sft.yaml] --> B{stage}
  B -->|prompts| C[Plan procedural task specs]
  C --> D[Gemini Batch API<br/>prompt authoring]
  D --> E[datasets/sft/prompts.jsonl]
  E --> F[Human review optional]
  F --> G[stage: samples]
  G --> H[Gemini Batch API<br/>teacher generation]
  H --> I[datasets/sft/samples.jsonl<br/>metadata.valid unset]
  I --> J[stage: validate]
  J --> K[Strudel executor + validator]
  K --> L[Set metadata.valid on each row]
  L --> M[datasets/sft/train.jsonl<br/>valid rows only]
  K -->|invalid| N[stage: regenerate]
  N --> O[Batch retry with validation error context]
  O --> J
```

| `stage` | Output | Notes |
|---|---|---|
| `prompts` | `prompts_file` | Batch prompt authoring only |
| `samples` | `samples_file` | Batch teacher responses; no Strudel check yet |
| `validate` | updates `samples_file` + `output` | Sets `metadata.valid` per row |
| `regenerate` | updates failed rows in `samples_file` | Re-validates when `batch.regenerate_validate: true` |
| `full` | all of the above | Runs prompts → samples → validate → regenerate loop |

`stage: responses` is an alias for `samples`.

### Legacy sync (`batch.enabled: false`)

```mermaid
flowchart TD
  A[configs/data/sft.yaml] --> B[Generate task specs + compose prompts]
  B --> C{parallel_models?}
  C -->|yes| D[Parallel teacher shards<br/>per enabled model id]
  C -->|no| E[Weighted teacher router]
  D --> F[Validate execution + constraints inline]
  E --> F
  F --> G[datasets/sft/train.jsonl]
```

## SFT training

```mermaid
flowchart TD
  A[datasets/sft/train.jsonl] --> B[format_training_text<br/>messages_for_task + response]
  B --> C[Load base model + optional SFT adapter]
  C --> D[SFTTrainer / LoRA]
  D --> E[checkpoints/*-sft]
  E --> F[experiment.json metadata]
```

## RL data preparation

### Staged (`stage: prompts` → review → `stage: tasks`)

```mermaid
flowchart TD
  A[configs/data/rl_tasks.yaml] --> B{stage}
  B -->|prompts| C[Procedural task skeletons<br/>constraints + code fields]
  C --> D[Random style per row]
  D --> E[Gemini Batch API<br/>prompt_author.models]
  E --> F[datasets/rl/prompts.jsonl]
  F --> G[Human review<br/>edit / delete rows]
  G --> H[configs/data/rl_tasks.yaml<br/>stage: tasks]
  H --> I[Schema validate rows]
  I --> J[datasets/rl/train.jsonl]
```

### Full (`stage: full`)

```mermaid
flowchart TD
  A[configs/data/rl_tasks.yaml<br/>stage: full] --> B{batch + prompt_author?}
  B -->|yes| C[Procedural skeletons + batch styled prompts]
  B -->|no| D[Legacy sync mode<br/>procedural or hybrid llm_fraction]
  C --> E[datasets/rl/train.jsonl]
  D --> E
```

## GRPO training

```mermaid
flowchart TD
  A[datasets/rl/train.jsonl] --> B[prepare_rl_row<br/>messages_for_task]
  B --> C[Load SFT checkpoint + LoRA]
  C --> D[GRPOTrainer<br/>num_generations per prompt]
  D --> E[Sample N completions from policy]
  E --> F[RewardAdapter<br/>format · execution · constraints<br/>edit/repair consistency]
  F --> G[Strudel executor cache]
  G --> H[Group-relative advantage + update]
  H --> I[checkpoints/*-grpo]
  I --> J[experiment.json metadata]
```

## Full pipeline (typical)

```mermaid
flowchart LR
  subgraph prep [Data prep]
    SFT_P[SFT prompts bank]
    SFT_T[SFT train.jsonl]
    RL_P[RL prompts bank]
    RL_T[RL train.jsonl]
    SFT_P --> SFT_T
    RL_P --> RL_T
  end
  subgraph train [Training]
    SFT[SFTTrainer]
    GRPO[GRPOTrainer]
    SFT_T --> SFT
    SFT --> GRPO
    RL_T --> GRPO
  end
  subgraph eval [Evaluation]
    BENCH[benchmark.jsonl]
    EVAL[offline evaluator]
    GRPO --> EVAL
    BENCH --> EVAL
  end
```
