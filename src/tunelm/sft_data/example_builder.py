"""Build validated SFT rows with teacher and prompt provenance."""

from __future__ import annotations

from datetime import UTC, datetime

from tunelm.schemas import ModelProvenance, RLTask, SFTExample, SFTMetadata
from tunelm.sft_data.validator import ValidationReport


def build_sft_sample(
    task: RLTask,
    response,
    teacher: ModelProvenance,
    *,
    valid: bool | None = None,
    report: ValidationReport | None = None,
    generation_attempts: int = 1,
) -> SFTExample:
    """Build one SFT row from a teacher response before or after Strudel validation."""
    return SFTExample(
        id=task.id,
        task_type=task.task_type,
        prompt=task.prompt,
        response=response,
        constraints=task.constraints,
        original_code=getattr(task, "original_code", None),
        instruction=getattr(task, "instruction", None),
        expected_changes=getattr(task, "expected_changes", None),
        preserve=getattr(task, "preserve", []),
        broken_code=getattr(task, "broken_code", None),
        reference_code=getattr(task, "reference_code", None),
        corruption=getattr(task, "corruption", None),
        metadata=SFTMetadata(
            teacher=teacher.model_id,
            teacher_provider=teacher.provider,
            teacher_model=teacher.model,
            prompt_author=task.prompt_author,
            valid=valid,
            validation=report.as_dict() if report else {},
            generation_attempts=generation_attempts,
            created_at=datetime.now(UTC),
        ),
    )


def build_sft_example(
    task: RLTask,
    response,
    teacher: ModelProvenance,
    report: ValidationReport,
) -> SFTExample:
    return build_sft_sample(task, response, teacher, valid=True, report=report)
