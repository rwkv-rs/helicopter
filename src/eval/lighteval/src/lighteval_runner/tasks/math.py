from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..task_runtime import ScoringTransform


class MathRepairStrategy(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class RepairAction(StrEnum):
    NONE = "none"
    CLOSE_THINK_AND_THEREFORE = "close-think-and-therefore"
    APPEND_THEREFORE = "append-therefore"


@dataclass(frozen=True, slots=True)
class MathScoringInput:
    raw_completion: str
    scored_completion: str
    repair_strategy: MathRepairStrategy
    repair_action: RepairAction


def _has_unclosed_think(prompt: str, completion: str) -> bool:
    visible = f"{prompt}{completion}"
    return visible.count("<think>") > visible.count("</think>")


def repair_math_completion(
    *,
    prompt: str,
    raw_completion: str,
    truncated: bool,
    strategy: MathRepairStrategy,
) -> MathScoringInput:
    scored = raw_completion
    action = RepairAction.NONE
    if strategy in {MathRepairStrategy.B, MathRepairStrategy.C} and _has_unclosed_think(
        prompt, raw_completion
    ):
        scored = f"{raw_completion}</think>\nTherefore..."
        action = RepairAction.CLOSE_THINK_AND_THEREFORE
    elif strategy is MathRepairStrategy.C and truncated:
        scored = f"{raw_completion}\nTherefore..."
        action = RepairAction.APPEND_THEREFORE
    return MathScoringInput(
        raw_completion=raw_completion,
        scored_completion=scored,
        repair_strategy=strategy,
        repair_action=action,
    )


def transform_math_scoring_text(
    prompt: str, raw_completion: str, truncated: bool, requested_strategy: str
) -> ScoringTransform:
    repaired = repair_math_completion(
        prompt=prompt,
        raw_completion=raw_completion,
        truncated=truncated,
        strategy=MathRepairStrategy(requested_strategy),
    )
    return ScoringTransform(
        completion=repaired.scored_completion,
        strategy=repaired.repair_strategy.value,
        action=repaired.repair_action.value,
    )
