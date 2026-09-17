"""Validate teacher responses before they enter the supervised dataset.

Compose rows accept executable code with consistent controls only; hidden brief
constraints are scored for reporting but not gated (RL rewards still use them).
Edit and repair rows gate on execution plus task-consistency only; plan, controls,
and constraint scores are reported but not required for SFT acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tunelm.rewards.constraints import has_verifiable_constraints, score_constraints
from tunelm.rewards.edit_consistency import score_edit_consistency
from tunelm.rewards.repair_consistency import score_repair_consistency
from tunelm.schemas import ComposeTask, EditTask, ModelResponse, RLTask, RepairTask
from tunelm.strudel.analyzer import DRUM_NAMES, canonical_instrument
from tunelm.strudel.controls import controls_match_code
from tunelm.strudel.executor import StrudelExecutor


@dataclass(slots=True)
class ValidationReport:
    """Detailed acceptance result for one task and model response."""

    valid: bool
    execution_valid: bool
    constraint_score: float | None
    plan_consistent: bool
    controls_consistent: bool
    task_consistency_score: float
    error: str | None
    features: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _plan_matches(response: ModelResponse, features: dict) -> bool:
    actual = {canonical_instrument(name) for name in features.get("instruments", [])}
    planned = {canonical_instrument(name) for name in response.plan.instruments}
    return (
        not actual or bool(actual & planned) or ("drums" in planned and bool(actual & DRUM_NAMES))
    )


def _task_consistency(task: RLTask, code: str) -> float:
    if isinstance(task, EditTask):
        return score_edit_consistency(code, task)
    if isinstance(task, RepairTask):
        return score_repair_consistency(code, task.reference_code)
    return 1.0


def _validation_error(
    task: RLTask,
    execution_error: str | None,
    constraint_score: float | None,
    minimum_constraint_score: float,
    plan_consistent: bool,
    controls_consistent: bool,
    task_consistency_score: float,
    minimum_task_consistency_score: float,
) -> str | None:
    reasons = [execution_error] if execution_error else []
    if isinstance(task, ComposeTask):
        if not controls_consistent:
            reasons.append("controls do not match strudel_code features")
        return "; ".join(reasons) or None
    if constraint_score is not None and constraint_score < minimum_constraint_score:
        reasons.append(
            f"constraint score {constraint_score:.3f} below {minimum_constraint_score:.3f}"
        )
    if not plan_consistent:
        reasons.append("plan instruments do not overlap with code instruments")
    if not controls_consistent:
        reasons.append("controls do not match strudel_code features")
    if task_consistency_score < minimum_task_consistency_score:
        reasons.append(
            f"task consistency {task_consistency_score:.3f} below "
            f"{minimum_task_consistency_score:.3f}"
        )
    return "; ".join(reasons) or None


def _accept_solution(
    task: RLTask,
    *,
    execution_valid: bool,
    constraint_score: float | None,
    minimum_constraint_score: float,
    plan_consistent: bool,
    controls_consistent: bool,
    task_consistency_score: float,
    minimum_task_consistency_score: float,
) -> bool:
    if isinstance(task, ComposeTask):
        return execution_valid and controls_consistent
    return execution_valid and task_consistency_score >= minimum_task_consistency_score


def minimum_task_consistency_score(config: dict, task: RLTask) -> float:
    """Resolve per-task-type consistency threshold from flat or nested config."""
    raw = config.get("minimum_task_consistency_score", 0.8)
    if isinstance(raw, dict):
        return float(raw.get(task.task_type.value, raw.get("default", 0.8)))
    return float(raw)


def classify_failure(
    *,
    report: ValidationReport | None = None,
    exc: BaseException | str | None = None,
) -> str:
    """Short failure bucket for progress reporting during sample generation."""
    if exc is not None:
        from tunelm.strudel.parser import ResponseParseError

        if isinstance(exc, ResponseParseError):
            return "parse"
        message = str(exc).casefold()
        if "parse" in message or "json object" in message or "invalid tunelm response" in message:
            return "parse"
        return "api"
    if report is None:
        return "unknown"
    if not report.execution_valid:
        return "exec"
    if not report.controls_consistent:
        return "controls"
    error = (report.error or "").casefold()
    if "constraint score" in error:
        return "constraint"
    if "plan instruments" in error:
        return "plan"
    if "task consistency" in error:
        return "consistency"
    return "validation"


def validate_solution(
    task: RLTask,
    response: ModelResponse,
    executor: StrudelExecutor,
    *,
    minimum_constraint_score: float = 0.75,
    minimum_task_consistency_score: float = 0.8,
) -> ValidationReport:
    """Execute and verify one response, returning all acceptance components."""
    execution = executor.run(response.strudel_code)
    constraint_score = None
    if has_verifiable_constraints(task.constraints):
        constraint_score = (
            score_constraints(execution.features, task.constraints) if execution.valid else 0.0
        )
    plan_consistent = _plan_matches(response, execution.features)
    controls_consistent = controls_match_code(response.controls, execution.features)
    task_consistency_score = _task_consistency(task, response.strudel_code)
    valid = _accept_solution(
        task,
        execution_valid=execution.valid,
        constraint_score=constraint_score,
        minimum_constraint_score=minimum_constraint_score,
        plan_consistent=plan_consistent,
        controls_consistent=controls_consistent,
        task_consistency_score=task_consistency_score,
        minimum_task_consistency_score=minimum_task_consistency_score,
    )
    return ValidationReport(
        valid=valid,
        execution_valid=execution.valid,
        constraint_score=constraint_score,
        plan_consistent=plan_consistent,
        controls_consistent=controls_consistent,
        task_consistency_score=task_consistency_score,
        error=_validation_error(
            task,
            execution.error,
            constraint_score,
            minimum_constraint_score,
            plan_consistent,
            controls_consistent,
            task_consistency_score,
            minimum_task_consistency_score,
        ),
        features=execution.features,
    )
