import pytest

from helicopter_lighteval.datasets.math import RepairAction, repair_completion


def test_strategy_a_preserves_the_completion() -> None:
    result = repair_completion(
        prompt="<think>", raw_completion="reasoning", truncated=True, strategy="A"
    )
    assert result.scored_completion == "reasoning"
    assert result.action is RepairAction.NONE


@pytest.mark.parametrize("truncated", [False, True])
def test_strategy_a_never_inserts_therefore(truncated: bool) -> None:
    result = repair_completion(
        prompt="<think>",
        raw_completion="reasoning</think>",
        truncated=truncated,
        strategy="A",
    )
    assert result.scored_completion == "reasoning</think>"
    assert result.action is RepairAction.NONE


def test_strategy_b_closes_an_open_think_without_second_generation() -> None:
    result = repair_completion(
        prompt="<think>", raw_completion="reasoning", truncated=False, strategy="B"
    )
    assert result.scored_completion == "reasoning</think>\nTherefore..."
    assert result.action is RepairAction.CLOSE_THINK_AND_THEREFORE


def test_strategy_b_preserves_closed_think() -> None:
    result = repair_completion(
        prompt="<think>",
        raw_completion="reasoning</think>",
        truncated=True,
        strategy="B",
    )
    assert result.scored_completion == "reasoning</think>"
    assert result.action is RepairAction.NONE


def test_strategy_c_appends_therefore_only_after_completed_think() -> None:
    result = repair_completion(
        prompt="question", raw_completion="answer", truncated=True, strategy="C"
    )
    assert result.scored_completion == "answer\nTherefore..."
    assert result.action is RepairAction.APPEND_THEREFORE


def test_strategy_c_preserves_a_complete_non_truncated_answer() -> None:
    result = repair_completion(
        prompt="question", raw_completion="answer", truncated=False, strategy="C"
    )
    assert result.scored_completion == "answer"
    assert result.action is RepairAction.NONE


def test_strategy_c_prefers_open_think_repair() -> None:
    result = repair_completion(
        prompt="<think>", raw_completion="reasoning", truncated=True, strategy="C"
    )
    assert result.action is RepairAction.CLOSE_THINK_AND_THEREFORE
