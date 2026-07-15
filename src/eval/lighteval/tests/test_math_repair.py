from __future__ import annotations

import pytest

from lighteval_runner.tasks.math import (
    MathRepairStrategy,
    RepairAction,
    repair_math_completion,
)


@pytest.mark.parametrize("strategy", list(MathRepairStrategy))
def test_normal_completion_is_never_rewritten(strategy):
    raw = "<think>work</think>answer"
    result = repair_math_completion(
        prompt="question", raw_completion=raw, truncated=False, strategy=strategy
    )

    assert result.raw_completion == raw
    assert result.scored_completion == raw
    assert result.repair_action is RepairAction.NONE


def test_strategy_a_never_repairs_truncated_open_think():
    raw = "<think>unfinished"
    result = repair_math_completion(
        prompt="question",
        raw_completion=raw,
        truncated=True,
        strategy=MathRepairStrategy.A,
    )

    assert result.scored_completion == raw
    assert result.repair_action is RepairAction.NONE


@pytest.mark.parametrize("strategy", [MathRepairStrategy.B, MathRepairStrategy.C])
def test_open_think_is_closed_for_repair_strategies(strategy):
    raw = "unfinished reasoning"
    result = repair_math_completion(
        prompt="question<think>", raw_completion=raw, truncated=True, strategy=strategy
    )

    assert result.raw_completion == raw
    assert result.scored_completion == "unfinished reasoning</think>\nTherefore..."
    assert result.repair_action is RepairAction.CLOSE_THINK_AND_THEREFORE


def test_strategy_c_repairs_truncated_answer_after_closed_think():
    raw = "<think>work</think>partial answer"
    result = repair_math_completion(
        prompt="question",
        raw_completion=raw,
        truncated=True,
        strategy=MathRepairStrategy.C,
    )

    assert result.raw_completion == raw
    assert result.scored_completion == f"{raw}\nTherefore..."
    assert result.repair_action is RepairAction.APPEND_THEREFORE


def test_strategy_b_does_not_repair_closed_think_answer_truncation():
    raw = "<think>work</think>partial answer"
    result = repair_math_completion(
        prompt="question",
        raw_completion=raw,
        truncated=True,
        strategy=MathRepairStrategy.B,
    )

    assert result.scored_completion == raw


def test_open_think_split_across_prompt_completion_is_detected():
    result = repair_math_completion(
        prompt="question<think",
        raw_completion=">unfinished",
        truncated=True,
        strategy=MathRepairStrategy.B,
    )

    assert result.scored_completion == ">unfinished</think>\nTherefore..."
    assert result.repair_action is RepairAction.CLOSE_THINK_AND_THEREFORE


@pytest.mark.parametrize(
    ("strategy", "truncated", "raw", "expected_action"),
    [
        (MathRepairStrategy.A, False, "plain answer", RepairAction.NONE),
        (MathRepairStrategy.A, True, "<think>open", RepairAction.NONE),
        (
            MathRepairStrategy.B,
            False,
            "<think>open",
            RepairAction.CLOSE_THINK_AND_THEREFORE,
        ),
        (
            MathRepairStrategy.B,
            True,
            "<think>open",
            RepairAction.CLOSE_THINK_AND_THEREFORE,
        ),
        (MathRepairStrategy.B, True, "<think>x</think>partial", RepairAction.NONE),
        (
            MathRepairStrategy.C,
            False,
            "<think>open",
            RepairAction.CLOSE_THINK_AND_THEREFORE,
        ),
        (
            MathRepairStrategy.C,
            True,
            "<think>open",
            RepairAction.CLOSE_THINK_AND_THEREFORE,
        ),
        (
            MathRepairStrategy.C,
            True,
            "<think>x</think>partial",
            RepairAction.APPEND_THEREFORE,
        ),
        (MathRepairStrategy.C, False, "<think>x</think>answer", RepairAction.NONE),
    ],
)
def test_a_b_c_decision_table_preserves_raw_completion(
    strategy, truncated, raw, expected_action
):
    result = repair_math_completion(
        prompt="question", raw_completion=raw, truncated=truncated, strategy=strategy
    )

    assert result.raw_completion == raw
    assert result.repair_action is expected_action
