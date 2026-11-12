"""RL input bank: the contract-writing author stage and answer-free task export.

`RL_STAGE` plugs into the shared authoring engine in `prompt_generation`; every RL
request carries a quoted `RequestContract` that `rewards.contract_score` grades.
`export_rl_tasks` turns the reviewed bank into GRPO and benchmark `Task` rows. RL
never needs teacher answers, reference memory, or the SFT answer run.
"""

from pathlib import Path

from tunelm.io import content_hash, write_jsonl
from tunelm.prompting import PROMPT_SYSTEM, AuthorStage, PromptReply
from tunelm.rewards import check_contract, sound_families
from tunelm.schemas import MusicRequest, RequestContract, Task
from tunelm.synthesis import load_prompt_banks, request_plan

RL_PROMPT_SYSTEM = (
    PROMPT_SYSTEM
    + """
This is an RL request. The reward can only check what the request literally says,
so write a request in which a real listener naturally says at least two concrete
things: an instrument by name, how many parts, a change over time (builds up,
thins out, two contrasting sections), sparse or busy, or a short loop. Every
request also states a tempo as digits and bpm, and a key as a root and mode,
such as 96 bpm in C minor. Mood and genre are fine as color, but they are never
part of the contract.
The focus is the idea, not a sentence to copy. Say it the way this person would,
and change the kind of ask: a complaint, a correction, a one-line order, a question,
or who comes in when. Do not keep writing "answer each other", "drops to almost
nothing", "short loop on", or "lock into one pocket". If already_written uses a
sentence shape, write a different one.
Also return contract: what the request literally asks for, nothing it only implies.
- require / forbid: families named or refused in the request. Use only names from
  families. A word with no matching family (clay pot, sledgehammer) stays out.
  "Instrumental" or "no singing" is forbid vocals.
- voices: min and max parts when the request gives a count. "Solo" is 1 to 1.
- tempo: required, digits followed by bpm. key: required, a written root and mode, as C:minor.
- shape: build, thin_out, or contrast. space: sparse or busy. loop: a short loop.
Every entry has quote: the exact words copied from your prompt that say it.
Leave a field empty when the request does not say it.
"""
)


class RLPromptReply(PromptReply):
    contract: RequestContract


def check_rl_request(row: MusicRequest) -> None:
    """A contract with quoted tempo and key, valid families, and enough checks."""
    if row.contract is None:
        raise ValueError("RL requests need a contract")
    if row.contract.tempo is None or row.contract.key is None:
        raise ValueError(
            "State a tempo in bpm and a key, such as 96 bpm in C minor, "
            "and put both in the contract"
        )
    check_contract(row.prompt, row.contract, row.original_code)


RL_STAGE = AuthorStage(
    name="rl",
    system=RL_PROMPT_SYSTEM,
    reply=RLPromptReply,
    allow_tempo_key=True,
    payload=lambda: {"families": sorted(sound_families())},
    check=check_rl_request,
)


def export_rl_tasks(prompts_dir: str, dataset_dir: str) -> dict:
    """Write `<dataset_dir>/rl/{train,val,benchmark}.jsonl` from the reviewed RL bank.

    The files are rebuilt from the bank on every run, so deleting a bad request and
    exporting again removes its task. Contracts are rechecked because the bank is
    hand-editable. The tempo-and-key rule is authoring policy only; rows written
    before it existed still export.
    """
    requests = load_prompt_banks(prompts_dir)["rl"]
    if not requests:
        raise ValueError(f"No RL requests under {prompts_dir}/rl; run create-prompts first")
    for request in requests:
        try:
            if request.contract is None:
                raise ValueError("RL requests need a contract")
            check_contract(request.prompt, request.contract, request.original_code)
        except ValueError as error:
            raise ValueError(f"RL request {request.id}: {error}") from error
    tasks = [_rl_task(request) for request in requests]
    counts = {}
    for split in ("train", "val", "benchmark"):
        rows = [task.model_dump() for task in tasks if task.split == split]
        write_jsonl(Path(dataset_dir) / "rl" / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    return {"dataset_dir": dataset_dir, "rl_tasks": counts}


def _rl_task(request: MusicRequest) -> Task:
    """GRPO renders only the request and original_code, so no reference context is packed."""
    return Task(
        id=content_hash(["rl", request.id])[:24],
        prompt=request.prompt,
        prompt_id=request.id,
        plan=request_plan(request),
        context="",
        split=request.split,
        source_families=[],
        memory_version="none",
        retrieval_regime="empty",
        prompt_style=request.prompt_style,
        prompt_scenario=request.prompt_scenario,
        contract=request.contract,
    )
