from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


def upstream_task_candidates(benchmark: str) -> tuple[str, ...]:
    """Return fixed-LightEval task names for a public math benchmark alias."""

    aliases = {
        "aime24": ("aime24",),
        "aime25": ("aime25",),
        "asdiv": ("asdiv",),
        "gsm8k": ("gsm8k",),
        "gsm-plus": ("gsm_plus",),
        "gsm_plus": ("gsm_plus",),
        "math-500": ("math_500",),
        "math_500": ("math_500",),
        "olympiad-bench": ("olympiad_bench:OE_TO_maths_en_COMP",),
        "olympiadbench": ("olympiad_bench:OE_TO_maths_en_COMP",),
    }
    normalized = benchmark.strip()
    return aliases.get(normalized, ())


class MathRepairStrategy(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class RepairAction(StrEnum):
    NONE = "none"
    CLOSE_THINK_AND_THEREFORE = "close-think-and-therefore"
    APPEND_THEREFORE = "append-therefore"


@dataclass(frozen=True, slots=True)
class MathScoringText:
    raw_completion: str
    scored_completion: str
    strategy: MathRepairStrategy
    action: RepairAction


def repair_completion(
    *,
    prompt: str,
    raw_completion: str,
    truncated: bool,
    strategy: MathRepairStrategy | str,
) -> MathScoringText:
    """Apply the signed A/B/C transform without generating a second answer."""

    selected = MathRepairStrategy(strategy)
    scored = raw_completion
    action = RepairAction.NONE

    if selected in {MathRepairStrategy.B, MathRepairStrategy.C} and _has_open_think(
        prompt, raw_completion
    ):
        scored = f"{raw_completion}</think>\nTherefore..."
        action = RepairAction.CLOSE_THINK_AND_THEREFORE
    elif selected is MathRepairStrategy.C and truncated:
        scored = f"{raw_completion}\nTherefore..."
        action = RepairAction.APPEND_THEREFORE

    return MathScoringText(raw_completion, scored, selected, action)


def _has_open_think(prompt: str, completion: str) -> bool:
    visible = f"{prompt}{completion}"
    return visible.count("<think>") > visible.count("</think>")
