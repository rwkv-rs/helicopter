from types import SimpleNamespace

import pytest

from helicopter_lighteval.datasets.knowledge.gpqa.diamond import GpqaDiamond
from helicopter_lighteval.datasets.knowledge.mmlu import Mmlu
from helicopter_lighteval.datasets.knowledge.mmlu_pro import MmluPro


@pytest.mark.parametrize(
    ("benchmark", "row", "document", "expected"),
    [
        (
            Mmlu(),
            {"question": "Which value?", "choices": ["one", "two", "three", "four"]},
            SimpleNamespace(),
            "Which value?\n\nA. one\nB. two\nC. three\nD. four",
        ),
        (
            MmluPro(),
            {"question": "Choose one.", "options": ["alpha", "beta"]},
            SimpleNamespace(),
            "Choose one.\n\nA. alpha\nB. beta",
        ),
        (
            GpqaDiamond(),
            {
                "Question": "What follows?",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "wrong 1",
                "Incorrect Answer 2": "wrong 2",
                "Incorrect Answer 3": "wrong 3",
            },
            SimpleNamespace(gold_index=2),
            "What follows?\n\nA. wrong 1\nB. wrong 2\nC. correct\nD. wrong 3",
        ),
    ],
)
def test_knowledge_benchmarks_render_only_question_and_choices(
    benchmark: object,
    row: dict[str, object],
    document: SimpleNamespace,
    expected: str,
) -> None:
    query = benchmark.get_query(row, document)

    assert query == expected
    assert "Question:" not in query
    assert "Answer:" not in query
    assert "Think step by step" not in query
