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
    "vague",
    "mood",
    "genre",
    "activity",
    "blend",
    "reference",
    "musical",
    "technical",
    "groove",
    "dialogue",
    "restraint",
    "solo",
    "contrast",
    "dislike",
    "dance",
]
ROLES = ("drums_core", "perc_tops", "bass", "chords_pad", "arp_lead", "fx_texture")
KEY_PATTERN = r"^[A-G](?:#|b)?:(?:major|minor|dorian|mixolydian|phrygian|lydian|locrian)$"


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
    key: str | None = Field(default="C:minor", pattern=KEY_PATTERN)
    category: Category = "stack"
    voices: list[Voice] = Field(min_length=1, max_length=6)
    explicit: list[Literal["bpm", "key", "roles"]] = Field(default_factory=list)
    short_loop: bool = False
    cycles: int = Field(default=16, ge=8, le=64)
    original_code: str | None = None
    preserve: list[Literal["original_code"]] = Field(default_factory=list)
    intent_only: bool = False

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


class FamilyClaim(Record):
    """A sound family from `configs/registry/sound_families.yaml`, quoted from the request."""

    family: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class VoiceClaim(Record):
    min: int = Field(ge=1, le=8)
    max: int = Field(ge=1, le=8)
    quote: str = Field(min_length=1)


class TempoClaim(Record):
    bpm: float = Field(ge=30, le=300)
    quote: str = Field(min_length=1)


class KeyClaim(Record):
    key: str = Field(pattern=KEY_PATTERN)
    quote: str = Field(min_length=1)


class ShapeClaim(Record):
    shape: Literal["build", "thin_out", "contrast"]
    quote: str = Field(min_length=1)


class SpaceClaim(Record):
    space: Literal["sparse", "busy"]
    quote: str = Field(min_length=1)


class LoopClaim(Record):
    quote: str = Field(min_length=1)


class RequestContract(Record):
    """What an RL request literally asks for that one verify can check.

    Every claim quotes the request. Unfilled fields are not scored. Mood and genre
    never appear here. `rewards.check_contract` enforces quotes and family names.
    """

    require: list[FamilyClaim] = Field(default_factory=list)
    forbid: list[FamilyClaim] = Field(default_factory=list)
    voices: VoiceClaim | None = None
    tempo: TempoClaim | None = None
    key: KeyClaim | None = None
    shape: ShapeClaim | None = None
    space: SpaceClaim | None = None
    loop: LoopClaim | None = None


def _absent(value) -> bool:
    return value is None


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
    prompt_id: str | None = None
    contract: RequestContract | None = Field(default=None, exclude_if=_absent)


class Example(Task):
    summary: str
    code: str
    teacher: str
    teacher_id: str | None = None
    seed_id: str
    pathway: Literal["backtranslate", "recombine", "compose"]
    persona: Literal["vibe", "producer", "algorithmic", "cocreate"]
    verification: Telemetry
    verifier_version: str


class MusicRequest(Record):
    """An independently authored input, never a teacher answer or reasoning target."""

    id: str
    stage: Literal["sft", "rl"]
    split: Split = "train"
    prompt: str = Field(min_length=1)
    category: Category
    prompt_style: PromptStyle
    prompt_scenario: str | None = None
    original_code: str | None = None
    author: str
    author_id: str
    contract: RequestContract | None = Field(default=None, exclude_if=_absent)

    @model_validator(mode="after")
    def valid_original(self):
        if self.category == "cocreate" and not self.original_code:
            raise ValueError("Co-creation inputs need an original starting pattern")
        if self.category != "cocreate" and self.original_code:
            raise ValueError("Original code is only an input for co-creation")
        if self.stage == "sft" and self.contract:
            raise ValueError("Only RL inputs carry a request contract")
        return self
