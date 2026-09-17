from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskType(StrEnum):
    COMPOSE = "compose"
    EDIT = "edit"
    REPAIR = "repair"


class TempoRange(StrictModel):
    min: float = Field(ge=20, le=400)
    max: float = Field(ge=20, le=400)

    @model_validator(mode="after")
    def ordered(self) -> "TempoRange":
        if self.min > self.max:
            raise ValueError("tempo min must not exceed max")
        return self


class TaskConstraints(StrictModel):
    tempo: float | TempoRange | None = None
    required_instruments: list[str] = Field(default_factory=list)
    forbidden_instruments: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    scale: str | None = None
    min_layers: int | None = Field(default=None, ge=1, le=32)
    max_layers: int | None = Field(default=None, ge=1, le=32)

    @model_validator(mode="after")
    def coherent(self) -> "TaskConstraints":
        required = {item.casefold() for item in self.required_instruments}
        forbidden = {item.casefold() for item in self.forbidden_instruments}
        overlap = required & forbidden
        if overlap:
            raise ValueError(f"instruments cannot be both required and forbidden: {overlap}")
        if self.min_layers and self.max_layers and self.min_layers > self.max_layers:
            raise ValueError("min_layers must not exceed max_layers")
        return self


class ExpectedChanges(StrictModel):
    tempo: Literal["increase", "decrease", "unchanged"] | None = None
    drum_density: Literal["increase", "decrease", "unchanged"] | None = None
    layer_count: Literal["increase", "decrease", "unchanged"] | None = None
    add_instruments: list[str] = Field(default_factory=list)
    remove_instruments: list[str] = Field(default_factory=list)


class MusicPlan(StrictModel):
    tempo: str | float
    instruments: list[str] = Field(min_length=1)
    structure: str
    musical_intent: str | None = None


class LayerControl(StrictModel):
    """One editable layer for future UI controls (mute, swap sounds, etc.)."""

    id: str = Field(min_length=1)
    role: Literal[
        "rhythm",
        "bass",
        "melody",
        "harmony",
        "pads",
        "percussion",
        "texture",
        "fx",
        "other",
    ]
    sounds: list[str] = Field(min_length=1)


class MusicControls(StrictModel):
    """Machine-editable arrangement metadata aligned with `strudel_code`."""

    bpm: float = Field(ge=20, le=400)
    key: str | None = None
    mood: list[str] = Field(default_factory=list)
    layers: list[LayerControl] = Field(min_length=1)


class ModelResponse(StrictModel):
    plan: MusicPlan
    controls: MusicControls
    strudel_code: str = Field(min_length=1)


class ModelProvenance(StrictModel):
    """Records which provider/model authored a prompt or teacher response."""

    model_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    style: str | None = None


class BaseTask(StrictModel):
    id: str = Field(min_length=1)
    task_type: TaskType
    prompt: str = Field(min_length=1)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    difficulty: int = Field(default=1, ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    prompt_author: ModelProvenance | None = None


class ComposeTask(BaseTask):
    task_type: Literal[TaskType.COMPOSE] = TaskType.COMPOSE


class EditTask(BaseTask):
    task_type: Literal[TaskType.EDIT] = TaskType.EDIT
    original_code: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    expected_changes: ExpectedChanges
    preserve: list[str] = Field(default_factory=list)


class RepairTask(BaseTask):
    task_type: Literal[TaskType.REPAIR] = TaskType.REPAIR
    broken_code: str = Field(min_length=1)
    reference_code: str | None = None
    corruption: str | None = None


RLTask = ComposeTask | EditTask | RepairTask


class SFTMetadata(StrictModel):
    teacher: str
    teacher_provider: str = ""
    teacher_model: str
    prompt_author: ModelProvenance | None = None
    valid: bool | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    generation_attempts: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SFTExample(StrictModel):
    id: str
    task_type: TaskType
    prompt: str
    response: ModelResponse
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    original_code: str | None = None
    instruction: str | None = None
    expected_changes: ExpectedChanges | None = None
    preserve: list[str] = Field(default_factory=list)
    broken_code: str | None = None
    reference_code: str | None = None
    corruption: str | None = None
    metadata: SFTMetadata

    @model_validator(mode="after")
    def has_task_context(self) -> "SFTExample":
        if self.task_type == TaskType.EDIT:
            if not self.original_code or not self.instruction or self.expected_changes is None:
                raise ValueError(
                    "edit SFT examples require original_code, instruction, and expected_changes"
                )
        if self.task_type == TaskType.REPAIR and not self.broken_code:
            raise ValueError("repair SFT examples require broken_code")
        return self


class ExecutionResult(StrictModel):
    valid: bool
    error: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    audio_path: str | None = None
    backend: str = "unknown"
    duration_ms: float = 0.0


def parse_task(value: dict[str, Any]) -> RLTask:
    task_type = value.get("task_type")
    classes = {
        TaskType.COMPOSE: ComposeTask,
        TaskType.EDIT: EditTask,
        TaskType.REPAIR: RepairTask,
        "compose": ComposeTask,
        "edit": EditTask,
        "repair": RepairTask,
    }
    try:
        return classes[task_type].model_validate(value)
    except KeyError as exc:
        raise ValueError(f"Unknown task_type: {task_type!r}") from exc
