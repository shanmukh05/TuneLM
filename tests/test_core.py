"""Parse, execute, and score one valid Strudel completion.

This is the verifier contract the rest of TuneLM depends on. Task generation,
CLI glue, and prompt wording are not covered here.
"""

import json
from pathlib import Path
from shutil import which

import pytest

from tunelm.rewards import RewardEngine
from tunelm.rl.trainer import RewardAdapter
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
            "prompt": "beat",
            "constraints": {"tempo": 120, "required_instruments": ["bd", "sd"]},
        }
    )
    result = RewardEngine(StrudelExecutor(backend="static")).score(raw, task)
    assert result.total == 1

    adapter = RewardAdapter(StrudelExecutor(backend="static"))
    assert adapter.constraint_reward(["not json", raw], constraints=[{}, {}]) == [None, None]


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


@pytest.mark.skipif(
    not which("node") or not Path("node_modules/@strudel/core").exists(),
    reason="Node/Strudel packages are not installed",
)
def test_node_executor():
    result = StrudelExecutor(backend="node").run('setcpm(120/4)\ns("bd sd hh*2")')
    assert result.valid, result.error
    assert result.features["event_count"] == 8
    blocked = StrudelExecutor(backend="node").run('fetch("https://example.com")')
    assert not blocked.valid
    assert "blocked" in (blocked.error or "")
