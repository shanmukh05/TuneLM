"""Parse, execute, and score one valid Strudel completion.

This is the verifier contract the rest of TuneLM depends on. Task generation,
CLI glue, and prompt wording are not covered here.
"""

import json
from pathlib import Path
from shutil import which

import pytest

from tunelm.rewards import RewardEngine
from tunelm.rl.reward_adapter import RewardAdapter
from tunelm.schemas import ComposeTask, ModelResponse, TaskConstraints, TempoRange, parse_task
from tunelm.models.templates import training_record
from tunelm.sft_data.validator import validate_solution
from tunelm.strudel.controls import infer_controls_from_code
from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import ResponseParseError, parse_model_response


def _completion(code: str) -> str:
    controls = infer_controls_from_code(code)
    instruments = sorted({sound for layer in controls.layers for sound in layer.sounds})
    return json.dumps(
        {
            "plan": {"tempo": controls.bpm, "instruments": instruments, "structure": "loop"},
            "controls": controls.model_dump(),
            "strudel_code": code,
        }
    )


def test_parse_execute_and_reward():
    code = 'setcpm(120/4)\ns("bd sd")'
    raw = _completion(code)
    parsed = parse_model_response(raw)
    assert parsed.plan.tempo == 120
    assert parse_model_response(f"```json\n{raw}\n```").strudel_code == code
    with pytest.raises(ResponseParseError):
        parse_model_response('{"strudel_code": "s(\\"bd\\")"}')

    executed = StrudelExecutor(backend="static").run(code)
    assert executed.valid
    assert executed.features["tempo"] == 120
    assert not StrudelExecutor(backend="static").run('s("bd"').valid
    assert StrudelExecutor(backend="static").run('// car\'s pulse\ns("bd")').valid

    task = parse_task(
        {
            "id": "compose",
            "task_type": "compose",
            "prompt": "a 120 BPM beat with kick and snares",
            "constraints": {"tempo": 120, "required_instruments": ["bd", "sd"]},
        }
    )
    result = RewardEngine(StrudelExecutor(backend="static")).score(raw, task)
    assert result.components["constraints"] == 1
    assert result.total == 1

    adapter = RewardAdapter(StrudelExecutor(backend="static"))
    assert adapter.constraint_reward(["not json", raw], constraints=[{}, {}]) == [None, None]


def test_rl_rows_and_reward_features():
    from tunelm.rewards.constraints import stated_constraints
    from tunelm.rewards.richness import score_richness
    from tunelm.rl.trainer import prepare_rl_row
    from tunelm.strudel.analyzer import analyze_code

    hidden = {"tempo": 110, "required_instruments": ["cello", "cp"], "min_layers": 3}
    compose = prepare_rl_row(
        {"id": "c", "task_type": "compose", "prompt": "vault heist, add claps", "constraints": hidden}
    )
    repair = prepare_rl_row(
        {
            "id": "r",
            "task_type": "repair",
            "prompt": "fix it",
            "broken_code": 's("bd"',
            "reference_code": 's("bd")',
            "corruption": "unclosed",
        }
    )
    assert compose.keys() == repair.keys()
    assert compose["constraints"]["tempo"] is None
    assert compose["constraints"]["required_instruments"] == ["cp"]
    assert compose["constraints"]["min_layers"] is None
    stated = stated_constraints("hi-hats at 90 BPM, at least 3 layers", {**hidden, "tempo": 90})
    assert stated.tempo == 90 and stated.min_layers == 3

    code = 'const unused = s("trumpet")\nstack(s("bd*4"), note("c3 e3").s("sine").gain(0), s("hh*8"),)'
    assert analyze_code(code)["layer_count"] == 3
    events = [
        {"begin": 0.0, "onset": True, "value": {"s": "bd"}},
        {"begin": 0.0, "onset": True, "value": {"s": "sine", "note": "c3", "gain": 0}},
        {"begin": 1.5, "onset": True, "value": {"s": "hh"}},
    ]
    features = analyze_code(code, events)
    assert features["instruments"] == ["bd", "hh"]
    assert (features["active_cycles"], features["distinct_cycles"]) == (2, 2)
    assert score_richness({"active_cycles": 8, "distinct_cycles": 4}, 8) == 1
    assert score_richness({"active_cycles": 8, "distinct_cycles": 1}, 8) == 0.5


def test_compose_sft_ignores_hidden_constraints():
    task = ComposeTask(
        id="compose",
        prompt="quiet suspense at dawn",
        constraints=TaskConstraints(
            tempo=TempoRange(min=200, max=240),
            required_instruments=["pluck", "strings"],
            min_layers=2,
        ),
    )
    response = ModelResponse.model_validate(
        json.loads(_completion('setcpm(60/4)\nstack(s("bd sd"), note("c4 e4").s("piano"))'))
    )
    report = validate_solution(task, response, StrudelExecutor(backend="static"))
    assert report.valid
    assert report.constraint_score is not None and report.constraint_score < 0.75


def test_sft_record_masks_prompt_and_disables_thinking():
    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "rendered prompt"

    row = {
        "id": "compose",
        "task_type": "compose",
        "prompt": "quiet piano",
        "constraints": {},
        "response": json.loads(_completion('setcpm(60/4)\nnote("c4").s("gm_piano")')),
    }
    record = training_record(row, Tokenizer())
    assert record["prompt"] == "rendered prompt"
    assert record["completion"].endswith("<eos>")
    assert "quiet piano" not in record["completion"]


def test_load_sft_datasets_excludes_val_ids(tmp_path):
    from tunelm.data.datasets import load_sft_datasets

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "task_type": "compose", "prompt": "one"}),
                json.dumps({"id": "b", "task_type": "compose", "prompt": "two"}),
                json.dumps({"id": "c", "task_type": "compose", "prompt": "three"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        json.dumps({"id": "b", "task_type": "compose", "prompt": "two"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "train_file": str(train_path),
            "val_file": str(val_path),
        }
    }
    train_dataset, eval_dataset = load_sft_datasets(config)
    train_ids = {json.loads(row["json"])["id"] for row in train_dataset}
    eval_ids = {json.loads(row["json"])["id"] for row in eval_dataset}
    assert train_ids == {"a", "c"}
    assert eval_ids == {"b"}


@pytest.mark.skipif(
    not which("node") or not Path("node_modules/@strudel/core").exists(),
    reason="Node/Strudel packages are not installed",
)
def test_node_executor():
    result = StrudelExecutor(backend="node").run('setcpm(120/4)\ns("bd sd hh*2")')
    assert result.valid, result.error
    assert result.playable
    assert result.features["event_count"] == 8
    assert not StrudelExecutor(backend="node").run("silence").playable
    blocked = StrudelExecutor(backend="node").run('fetch("https://example.com")')
    assert not blocked.valid
    assert "blocked" in (blocked.error or "")
