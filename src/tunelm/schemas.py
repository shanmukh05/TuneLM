"""Versioned JSON contracts shared by corpus, planning, and training.

Unknown fields and non-finite numbers fail at ingestion rather than disappearing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["drums_core", "perc_tops", "bass", "chords_pad", "arp_lead", "fx_texture"]
Split = Literal["train", "val", "benchmark"]
Category = Literal["atomic", "stack", "arrange", "cocreate"]
PromptStyle = Literal[
    "vague", "mood", "genre", "activity", "blend", "reference", "musical", "technical"
]
ROLES = ("drums_core", "perc_tops", "bass", "chords_pad", "arp_lead", "fx_texture")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[2] = 2


class Source(Record):
    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    code: str = Field(min_length=1, max_length=100000)
    description: str = Field(min_length=1)
    role: Role | None = None
    split: Split | None = None
    tags: list[str] = Field(default_factory=list)


class Voice(Record):
    role: Role
    query: str = Field(min_length=1)
    density: tuple[float, float] = (0, 128)
    techniques: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_density(self):
        if not 0 <= self.density[0] <= self.density[1] <= 10000:
            raise ValueError("Density must be an ordered nonnegative range")
        return self


class Plan(Record):
    bpm: float = Field(default=120, ge=30, le=300)
    key: str | None = Field(
        default="C:minor",
        pattern=r"^[A-G](?:#|b)?:(?:major|minor|dorian|mixolydian|phrygian|lydian|locrian)$",
    )
    category: Category = "stack"
    voices: list[Voice] = Field(min_length=1, max_length=6)
    explicit: list[Literal["bpm", "key", "roles"]] = Field(default_factory=list)
    short_loop: bool = False
    cycles: int = Field(default=16, ge=8, le=64)
    original_code: str | None = None
    preserve: list[Literal["original_code"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_preservation(self):
        if (self.category == "cocreate" or self.preserve) and not self.original_code:
            raise ValueError("Co-creation and preservation require original_code")
        return self

    @property
    def cpm(self) -> float:
        return self.bpm / 4


class Telemetry(Record):
    cycles: int
    event_count: int
    events_per_cycle: float
    sound_sources: list[str]
    pitches: list[float]
    pitch_range: tuple[float, float] | None
    syncopation: float
    structural_variation: float
    control_variation: float
    active_cycles: int
    gain_min: float
    gain_max: float


class Stem(Record):
    id: str
    source_id: str
    family: str
    split: Split
    source_hash: str
    license: str
    code: str
    setup: str = ""
    role: Role
    role_confidence: float = Field(ge=0, le=1)
    description: str
    tags: list[str] = Field(default_factory=list)
    methods: list[str]
    telemetry: Telemetry
    verifier_version: str


class Task(Record):
    id: str
    prompt: str = Field(min_length=1)
    plan: Plan
    context: str
    source_families: list[str]
    split: Split
    memory_version: str
    retrieval_regime: Literal["high", "partial", "empty"] = "high"
    retrieved_ids: list[str] = Field(default_factory=list)
    prompt_style: PromptStyle | None = None
    prompt_scenario: str | None = None


class Example(Task):
    summary: str
    code: str
    teacher: str
    teacher_id: str | None = None
    seed_id: str
    pathway: Literal["backtranslate", "recombine"]
    persona: Literal["vibe", "producer", "algorithmic", "cocreate"]
    verification: Telemetry
    verifier_version: str
